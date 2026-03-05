import torch

"""
backbone of the code is adapted from torchdiffeq: https://github.com/rtqichen/torchdiffeq
"""


class ConditionalDiffEqSolver(torch.nn.Module):
    def __init__(
            self, 
            order, 
            stepping_class:str="fixed", 
            min_factor:float=0.2, 
            max_factor:float=10, 
            safety:float=0.9
        ):

        super(ConditionalDiffEqSolver, self).__init__()
        self.order = order
        self.min_factor = torch.tensor([min_factor])
        self.max_factor = torch.tensor([max_factor])
        self.safety = torch.tensor([safety])
        self.tableau = None
        self.stepping_class = stepping_class

    def sync_device_dtype(self, x, h, t_span, device="cpu"):
        "Ensures `x`, `h`, `t_span`, `tableau` and other solver tensors are on the same device with compatible dtypes"
        x = x.to(device)
        h = h.to(device)
        t_span = t_span.to(device)
        
        if self.tableau is not None:
            c, a, bsol, berr = self.tableau
            self.tableau = c.to(x), [a.to(x) for a in a], bsol.to(x), berr.to(x)
        t_span = t_span.to(device)
        self.safety = self.safety.to(device)
        self.min_factor = self.min_factor.to(device)
        self.max_factor = self.max_factor.to(device)

        return x, h, t_span
    
    def step(self, f, x, h, t, dt, k1=None, args=None):
        raise NotImplementedError("Stepping rule not implemented for the solver")


class ConditionalDormandPrince45(ConditionalDiffEqSolver):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=5)
        self.dtype = dtype
        self.stepping_class = 'adaptive'
        self.tableau = construct_dopri5(self.dtype)

    def step(self, f, x, h, h_null, t, guidance_scale, dt, k1=None, args=None):
        """
        args
        ----
            f
            x
            h
            t
            dt
            k1
        """
        if len(t.size()) == 1:
            # if t is scalar tensor, expand its dimension to (batch_size, 1)
            t = t.repeat(x.size(0)).view(x.size(0),1)

        c, a, bsol, berr = self.tableau

        if k1 == None: 
            k1 = f(x, h, h_null, t, guidance_scale)

        k2 = f(x + dt*a[0]*k1, 
               h, h_null, t + c[0]*dt, guidance_scale)
        k3 = f(x + dt*(a[1][0]*k1 + a[1][1]*k2), 
               h, h_null, t + c[1]*dt, guidance_scale)
        k4 = f(x + dt*a[2][0]*k1 + dt*a[2][1]*k2 + dt*a[2][2]*k3, 
               h, h_null, t + c[2]*dt, guidance_scale)
        k5 = f(x + dt*a[3][0]*k1 + dt*a[3][1]*k2 + dt*a[3][2]*k3 + dt*a[3][3]*k4, 
               h, h_null, t + c[3]*dt, guidance_scale)
        k6 = f(x + dt*a[4][0]*k1 + dt*a[4][1]*k2 + dt*a[4][2]*k3 + dt*a[4][3]*k4 + dt*a[4][4]*k5, 
               h, h_null, t + c[4]*dt, guidance_scale)
        k7 = f(x + dt*a[5][0]*k1 + dt*a[5][1]*k2 + dt*a[5][2]*k3 + dt*a[5][3]*k4 + dt*a[5][4]*k5 + dt*a[5][5]*k6, 
               h, h_null, t + c[5]*dt, guidance_scale)

        x_sol = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4 + bsol[4] * k5 + bsol[5] * k6)
        err = dt * (berr[0] * k1 + berr[1] * k2 + berr[2] * k3 + berr[3] * k4 + berr[4] * k5 + berr[5] * k6 + berr[6] * k7)
        
        return k7, x_sol, err, (k1, k2, k3, k4, k5, k6, k7)


def construct_dopri5(dtype):
    c = torch.tensor([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.], dtype=dtype)
    a = [
        torch.tensor([1 / 5], dtype=dtype),
        torch.tensor([3 / 40, 9 / 40], dtype=dtype),
        torch.tensor([44 / 45, -56 / 15, 32 / 9], dtype=dtype),
        torch.tensor([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], dtype=dtype),
        torch.tensor([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], dtype=dtype),
        torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=dtype),
    ]
    bsol = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=dtype)
    berr_sub = torch.tensor([1951 / 21600, 0, 22642 / 50085, 451 / 720, -12231 / 42400, 649 / 6300, 1 / 60.], dtype=dtype)

    return (c, a, bsol, bsol - berr_sub)

