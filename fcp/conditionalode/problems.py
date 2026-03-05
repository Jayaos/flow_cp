import torch
from .solver import ConditionalDormandPrince45
from .utils import *
from tqdm import tqdm
import numpy as np


class CFGFlowODE(torch.nn.Module):
    """
    class to compute the backward and forward ode of the flow using the trained vector field
    solver is fixed to DormandPrince45
    """

    def __init__(self, CFGFlow, atol, rtol):
        super().__init__()
        self.vector_field = CFGFlow.vector_field
        self.solver = ConditionalDormandPrince45()
        self.null_condition_embedding = CFGFlow.null_condition_embedding
        self.atol = atol
        self.rtol = rtol

    def forward(self, x, h, t_span, guidance_scale, device="cpu"):
        """
        args
        ----
            x:
            h:
            t_span:
            guidance_scale:

        returns
        -------
        """

        x, h, t_span = self.solver.sync_device_dtype(x, h, t_span, device=device)

        if t_span[1] < t_span[0]: 
            # reverse time ode
            f_ = lambda x, h, h_null, t, guidance_scale: -self.vector_field.cfg_forward(x, 
                                                                                        h, 
                                                                                        h_null, 
                                                                                        -t, 
                                                                                        guidance_scale)
            t_span = -t_span
        else: 
            f_ = self.vector_field.cfg_forward
        
        # h_null.shape == h.shape
        h_null = torch.clone(self.null_condition_embedding).repeat(h.size(0)).view(h.size(0),-1).to(device)
		
        # obtain the initial step size
        k1 = f_(x, h, h_null, t_span[:1], guidance_scale)
        dt = init_step(f_, k1, x, h, h_null, t_span[:1], guidance_scale, 
                       self.solver.order, self.atol, self.rtol) # (batch_size, x_dim)
        dt = dt.to(device)

        if dt == 0:
            raise ValueError("initial step size is zero")
        
        t_eval, x_sol = self.adaptive_odeint(f_, k1, x, h, h_null, dt, t_span,guidance_scale,
                                             self.solver, self.atol, self.rtol)
        
        return t_eval.cpu().detach(), x_sol.cpu().detach()    

    @staticmethod
    def adaptive_odeint(f, k1, x, h, h_null, dt, t_span, guidance_scale, 
                                    solver, atol=1e-4, rtol=1e-4, args=None, 
                                    interpolator=None, return_all_eval=False, seminorm=(False, None)):
            
        t_eval, t, T = t_span[1:], t_span[:1], t_span[-1]
        ckpt_counter, ckpt_flag = 0, False
        eval_times, sol = [t], [x]

        while t < T:
            if t + dt > T:
                dt = T - t

            # Handle checkpoint interpolation (for t_eval grid)
            if t_eval is not None and ckpt_counter < len(t_eval) and t + dt > t_eval[ckpt_counter]:
                if interpolator is None:
                    dt_old, ckpt_flag = dt, True
                    dt = t_eval[ckpt_counter] - t

            f_new, x_new, x_err, stages = solver.step(f, x, h, h_null, t, guidance_scale, dt, 
                                                      k1=k1, args=args)

            # Error control
            if seminorm[0]:
                state_dim = seminorm[1]
                error = x_err[:state_dim]
                error_scaled = error / (atol + rtol * torch.max(x[:state_dim].abs(), x_new[:state_dim].abs()))
            else:
                error = x_err
                error_scaled = error / (atol + rtol * torch.max(x.abs(), x_new.abs()))
            error_ratio = hairer_norm(error_scaled)
            accept_step = error_ratio <= 1

            if accept_step:
                # Handle interpolation to hit exact t_eval checkpoints
                if t_eval is not None and interpolator is not None:
                    coefs = None
                    while ckpt_counter < len(t_eval) and t + dt > t_eval[ckpt_counter]:
                        t0, t1 = t, t + dt
                        x_mid = x + dt * sum(interpolator.bmid[i] * stages[i] for i in range(len(stages)))
                        f0, f1, x0, x1 = k1, f_new, x, x_new
                        if coefs is None:
                            coefs = interpolator.fit(dt, f0, f1, x0, x1, x_mid)
                        x_in = interpolator.evaluate(coefs, t0, t1, t_eval[ckpt_counter])
                        sol.append(x_in)
                        eval_times.append(t_eval[ckpt_counter][None])
                        ckpt_counter += 1

                if return_all_eval or (ckpt_counter < len(t_eval) and t + dt == t_eval[ckpt_counter]):
                    sol.append(x_new)
                    eval_times.append(t + dt)
                    if t + dt == t_eval[ckpt_counter]:
                        ckpt_counter += 1

                # Update for next step
                t, x = t + dt, x_new
                k1 = f_new

            # Reset step size after non-interpolated checkpoint
            if ckpt_flag:
                dt = dt_old - dt
                ckpt_flag = False

            # Update dt using adaptive rule
            dt = adapt_step(dt, error_ratio,
                            solver.safety,
                            solver.min_factor,
                            solver.max_factor,
                            solver.order)

        return torch.cat(eval_times), torch.stack(sol)


class CombinedODE(torch.nn.Module):  
    """
    class to compute the flow and the log determinant of the Jacobian of the flow 
        using the trained vector field
    
    backward flow is not implemented why?
    """

    def __init__(self, CFGFlow, atol, rtol):
        super().__init__()
        self.combined_vector_field = CombinedVectorField(CFGFlow.vector_field)
        self.solver = ConditionalDormandPrince45()
        self.null_condition_embedding = CFGFlow.null_condition_embedding
        self.atol = atol
        self.rtol = rtol

    def forward(self, x, h, t_span, guidance_scale, device="cpu"):

        x, h, t_span = self.solver.sync_device_dtype(x, h, t_span, device=device)

        # initial 
        initial_state = init_state(x) # (batch_size, x_dim+1)

        # h_null.shape == h.shape
        h_null = torch.clone(self.null_condition_embedding).repeat(h.size(0)).view(h.size(0),-1).to(device)
        
        k1 = self.combined_vector_field(initial_state, 
                                                    h, 
                                                    h_null, 
                                                    t_span[:1], 
                                                    guidance_scale)
        dt = init_step(self.combined_vector_field, 
                       k1, initial_state, h, h_null, t_span[:1], guidance_scale, 
                       self.solver.order, self.atol, self.rtol)
        dt = dt.to(device)

        if dt == 0:
            raise ValueError("initial step size is zero")

        t_eval, state_sol = self.adaptive_odeint(self.combined_vector_field, 
                                                 k1, initial_state, h, h_null, dt, t_span, guidance_scale, 
                                                 self.solver, self.atol, self.rtol)
        
        return t_eval.cpu().detach(), state_sol.cpu().detach()
    
    @staticmethod
    def adaptive_odeint(f, k1, x, h, h_null, dt, t_span, guidance_scale, 
                                    solver, atol=1e-4, rtol=1e-4, args=None, 
                                    interpolator=None, return_all_eval=False, seminorm=(False, None)):
            
        t_eval, t, T = t_span[1:], t_span[:1], t_span[-1]
        ckpt_counter, ckpt_flag = 0, False
        eval_times, sol = [t], [x]

        while t < T:
            if t + dt > T:
                dt = T - t

            # Handle checkpoint interpolation (for t_eval grid)
            if t_eval is not None and ckpt_counter < len(t_eval) and t + dt > t_eval[ckpt_counter]:
                if interpolator is None:
                    dt_old, ckpt_flag = dt, True
                    dt = t_eval[ckpt_counter] - t

            f_new, x_new, x_err, stages = solver.step(f, x, h, h_null, t, guidance_scale, dt, 
                                                      k1=k1, args=args)

            # Error control
            if seminorm[0]:
                state_dim = seminorm[1]
                error = x_err[:state_dim]
                error_scaled = error / (atol + rtol * torch.max(x[:state_dim].abs(), x_new[:state_dim].abs()))
            else:
                error = x_err
                error_scaled = error / (atol + rtol * torch.max(x.abs(), x_new.abs()))
            error_ratio = hairer_norm(error_scaled)
            accept_step = error_ratio <= 1

            if accept_step:
                # Handle interpolation to hit exact t_eval checkpoints
                if t_eval is not None and interpolator is not None:
                    coefs = None
                    while ckpt_counter < len(t_eval) and t + dt > t_eval[ckpt_counter]:
                        t0, t1 = t, t + dt
                        x_mid = x + dt * sum(interpolator.bmid[i] * stages[i] for i in range(len(stages)))
                        f0, f1, x0, x1 = k1, f_new, x, x_new
                        if coefs is None:
                            coefs = interpolator.fit(dt, f0, f1, x0, x1, x_mid)
                        x_in = interpolator.evaluate(coefs, t0, t1, t_eval[ckpt_counter])
                        sol.append(x_in)
                        eval_times.append(t_eval[ckpt_counter][None])
                        ckpt_counter += 1

                if return_all_eval or (ckpt_counter < len(t_eval) and t + dt == t_eval[ckpt_counter]):
                    sol.append(x_new)
                    eval_times.append(t + dt)
                    if t + dt == t_eval[ckpt_counter]:
                        ckpt_counter += 1

                # Update for next step
                t, x = t + dt, x_new
                k1 = f_new

            # Reset step size after non-interpolated checkpoint
            if ckpt_flag:
                dt = dt_old - dt
                ckpt_flag = False

            # Update dt using adaptive rule
            dt = adapt_step(dt, error_ratio,
                            solver.safety,
                            solver.min_factor,
                            solver.max_factor,
                            solver.order)

        return torch.cat(eval_times), torch.stack(sol)
    
    
class CombinedVectorField(torch.nn.Module):
    def __init__(self, cfg_vector_field):
        super().__init__()
        self.cfg_vector_field = cfg_vector_field

    def forward(self, state, h, h_null, t, guidance_scale):
        """
        args
        ----
            state: (batch_size, dim_x+1)
            h:
            t: 

        returns
        -------
            
        """
        x_t = state[:,:-1] # the last element is log det J

        # compute vector field
        v_h = self.cfg_vector_field(x_t, h, t)  # (batch_size, dim_x)
        v_h_null = self.cfg_vector_field(x_t, h_null, t)  # (batch_size, dim_x)
        v = (1-guidance_scale)*v_h_null + guidance_scale*v_h
        
        # compute divergence
        divergence_h = compute_divergence(self.cfg_vector_field, x_t, h, t)  # (batch_size, 1)
        divergence_h_null = compute_divergence(self.cfg_vector_field, x_t, h_null, t)  # (batch_size, 1)
        divergence = (1-guidance_scale)*divergence_h_null + guidance_scale*divergence_h

        return torch.cat([v, divergence], dim=1).to(state.device)


def init_step(f, f0, x0, h, h_null, t0, 
              guidance_scale, order, atol, rtol):
    """
    Estimate initial step size for ODE solver with guided flow.

    args
    ----
        f: Callable vector field function f(x, h, h_null, t, guidance_scale)
        f0: Initial derivative f(x0, ...)
        x0: Initial state
        t0: Initial time (scalar)
        h: Conditional vector (e.g., encoder output)
        h_null: Null conditional vector for classifier-free guidance
        guidance_scale: Guidance scale parameter (float or tensor)
        order: Order of ODE solver (int)
        atol: Absolute tolerance
        rtol: Relative tolerance

    returns
    -------
        dt: initial step size
    """

    scale = atol + torch.abs(x0) * rtol
    d0, d1 = hairer_norm(x0 / scale), hairer_norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        init_step_size = torch.tensor(1e-6, dtype=t0.dtype, device=t0.device)
    else:
        init_step_size = 0.01 * d0 / d1

    x_new = x0 + init_step_size * f0
    f_new = f(x_new, h, h_null, t0+init_step_size, guidance_scale)

    d2 = hairer_norm((f_new - f0) / scale) / init_step_size
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=t0.dtype, device=t0.device), init_step_size * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1. / float(order + 1))
    dt = torch.min(100 * init_step_size, h1).to(t0)

    return dt


def init_state(x_0):
    """
    initialize the state vector containing x_0 and logdetJ_0
    """
    logdetJ_0 = torch.zeros((x_0.size(0),1)).to(x_0.device) # logdetJ_0 is zero
    state_0 = torch.cat([x_0, logdetJ_0], -1).to(x_0.device)

    return state_0


def compute_divergence(vector_field, phi_th_x, h, t):
    """
    compute the divergence of the given vector field in terms of the argument
    """

    phi_th_x = phi_th_x.clone().detach().requires_grad_(True)
    #print("phi_th_x : {}".format(phi_th_x.size()))
    #print("h : {}".format(h.size()))
    #print("t : {}".format(t.size()))
    v = vector_field(phi_th_x, h, t) # (batch_size, dim_x)
    divergence = torch.zeros(phi_th_x.size(0), device=phi_th_x.device)

    for i in range(v.size(1)):

        vi = v[:, i] # (batch_size, 1)
        grad_vi = torch.autograd.grad(outputs=vi, inputs=phi_th_x,
                                        grad_outputs=torch.ones_like(vi),
                                        retain_graph=True, create_graph=True)[0][:, i] 
        divergence += grad_vi

    return divergence.view((phi_th_x.size(0), 1))


def estimate_region_size(combined_ode: CombinedODE, h, target_coverage, sampling_num, scale_cov, device):
    """
    Monte Carlo estimate the determinant of the Jacobian of the flow by solving Jacobian ODE
    y0 corresponding each h_i is sampled 
    we assume that sampling_dist is isotropic gaussian, thats why we sample from sphere
    """
    
    det_jacobian_mean_list = []
    est_region_size_list = []
    y_dim = combined_ode.combined_dt.vector_field.model[-1].out_features # the dimension of the last layer
    target_coverage_radii = gaussian_sphere_radius_scale_cov(target_coverage, y_dim, scale_cov)
    base_region_size = n_dimensional_sphere_volume(y_dim, target_coverage_radii)
    
    for i in tqdm(range(h.size(0))):
        
        x0_batch = uniform_sample_from_sphere_between_radii(y_dim, 0, target_coverage_radii, sampling_num)
        x0_batch = torch.tensor(x0_batch, dtype=torch.float32) # (sample_num, x_dim)
        h_batch = h[i,:].unsqueeze(0).repeat(sampling_num,1) # (sample_num, h_dim)
        t_span = torch.linspace(0, 1, 3) 
        
        _, state_sol = combined_ode(x0_batch, h_batch, t_span, device=device)
        
        logdet_jacobian_i = state_sol[-1,:,-1].numpy()
        det_jacobian_mean = np.mean(np.exp(logdet_jacobian_i))
        est_region_size = base_region_size * det_jacobian_mean
        det_jacobian_mean_list.append(det_jacobian_mean)
        est_region_size_list.append(est_region_size)
        
        if i % 50 == 0:
            print("count : {}".format(i+1))
            print("accumulated avg det: {}".format(np.mean(np.array(det_jacobian_mean_list))))
            print("accumulated avg estimated region size: {}".format(np.mean(np.array(est_region_size_list))))
        
    return est_region_size_list, det_jacobian_mean_list, base_region_size