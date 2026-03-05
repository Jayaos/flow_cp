import torch
import math
import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import chi, qmc


def load_data(data_dir):
    file = open(data_dir,'rb')
    
    return pickle.load(file)


def save_data(save_dir, data_dict):
    with open(save_dir, 'wb') as f:
        pickle.dump(data_dict, f)


def find_config_file(directory):
    for file in os.listdir(directory):
        if file.endswith("config.pkl"):
            return os.path.join(directory, file)
        
    raise ValueError("config file does not exist")


def expand_tensor_like(input_tensor, expand_to):
    """`input_tensor` is a 1d vector of length equal to the batch size of `expand_to`,
    expand `input_tensor` to have the same shape as `expand_to` along all remaining dimensions.

    Args:
        input_tensor (Tensor): (batch_size, past_window).
        expand_to (Tensor): (batch_size, past_window, dim)

    Returns:
        Tensor: expanded tensor of input_tensor, (batch_size, past_window, dim)
    """
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The second (past_window) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    t_expanded = input_tensor.clone().view(input_tensor.size(0), input_tensor.size(1), 1)
    t_expanded = t_expanded.repeat(1,1,expand_to.size(2))

    return t_expanded


def gaussian_sphere_radius(probability: float, dimensions: int):
    """
    Calculate the radius of a sphere that covers a given probability region 
    for an isotropic Gaussian centered at the origin in d-dimensional space.
    
    args
    ----
        probability: float, The desired probability region (0 < p < 1).
        dimensions: int, The number of dimensions (d).
    
    returns
    -------
        radius: float, The radius of the sphere.
    """
    if not (0 < probability < 1):
        raise ValueError("Probability must be in the range (0, 1).")
    if dimensions < 1:
        raise ValueError("Dimensions must be a positive integer.")
    
    # The radius is the quantile of the chi distribution for the given probability
    radius = chi.ppf(probability, df=dimensions)
    
    return radius


def gaussian_sphere_radius_scale_cov(probability: float, dimensions: int, scale: float):
    """
    Calculate the radius of a hypersphere enclosing a given probability for an n-dimensional Gaussian.
    
    Parameters:
    - probability (float): The desired probability (0 < p < 1).
    - dimensions (int): The number of dimensions (n).
    - k (float): The scalar for the covariance matrix (k > 0).
    
    Returns:
    - float: The radius of the hypersphere.
    """
    if not (0 < probability < 1):
        raise ValueError("Probability must be in the range (0, 1).")
    if dimensions < 1:
        raise ValueError("Dimensions must be a positive integer.")
    if scale <= 0:
        raise ValueError("k must be a positive scalar.")
    
    radius = np.sqrt(scale) * chi.ppf(probability, df=dimensions)
    
    return radius


def uniform_sample_from_sphere_between_radii(dim, d1, d2, num_samples):
    """
    Uniformly sample points from a spherical shell with radii between d1 and d2.
    
    Parameters:
    - x_dim: Dimensionality of the space (e.g., 3 for 3D space).
    - d1: The inner radius.
    - d2: The outer radius.
    - num_samples: Number of points to sample.
    
    Returns:
    - A numpy array of shape (num_samples, x_dim) with uniformly distributed points.
    """
    samples = []
    
    for _ in range(num_samples):
        # Generate a random point on the unit sphere (uniformly distributed direction)
        point = np.random.normal(0, 1, size=dim)
        point /= np.linalg.norm(point)  # Normalize to unit length
        
        # Randomly sample the radius between sqrt(d1) and sqrt(d2)
        radius = np.random.uniform(d1, d2)
        
        # Scale the point to the desired radius
        point *= radius
        
        samples.append(point)
    
    return np.array(samples)


def quasi_uniform_sample_from_sphere_between_radii(dim, d1, d2, num_samples):
    """
    Quasi-uniformly sample points from a spherical shell with radii between d1 and d2
    using a Sobol low-discrepancy sequence.
    """
    # Generate Sobol samples in [0,1]^dim
    sobol = qmc.Sobol(d=dim, scramble=True)
    u = sobol.random(num_samples)
    
    # Map to normal space and normalize to unit sphere
    normal_samples = qmc.scale(u, l_bounds=[-1]*dim, u_bounds=[1]*dim)
    norms = np.linalg.norm(normal_samples, axis=1, keepdims=True)
    directions = normal_samples / norms  # Project to unit sphere

    # Volume-corrected radius sampling
    radius_samples = np.random.uniform(d1**dim, d2**dim, size=(num_samples,)) ** (1/dim)
    radius_samples = radius_samples[:, None]
    
    # Scale directions
    samples = directions * radius_samples
    
    return samples


def n_dimensional_sphere_volume(dim, radii):
    """
    Compute the volume of an n-dimensional sphere with radius r.
    
    Parameters:
        n (int): Number of dimensions.
        r (float): Radius of the sphere.
    
    Returns:
        float: Volume of the n-dimensional sphere.
    """
    gamma = math.gamma(dim / 2 + 1)

    return (math.pi ** (dim / 2) * radii ** dim) / gamma


def estimate_region_size(flow, combined_ode, dataloader, guidance_scale, 
                         target_coverage, initial_distribution, sampling_num, batch_processing=None, device="cpu"):
    """
    Monte Carlo estimate the determinant of the Jacobian of the flow by solving Jacobian ODE
    y0 corresponding each h_i is sampled
    """
    
    # encoder history for the test data
    h = []
    y = []

    flow.to(device)
    flow.eval()
    for x_batch, y_batch in tqdm(dataloader):
        # test data should come from in sequential order
        with torch.no_grad():
            
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x_batch.size(1)) # causal mask
            x_batch = x_batch.to(device)
            attn_mask = attn_mask.to(device)
            h_batch = flow.encode(x_batch, src_mask=attn_mask, src_key_padding_mask=None)
            
            h.append(h_batch.cpu().detach()[:,-1,:])
            y.append(y_batch[:,-1,:])
    
    h = torch.from_numpy(np.array(h)).squeeze(1).to(device)
    y = torch.from_numpy(np.array(y)).squeeze(1).to(device)

    det_jacobian_mean_list = []
    est_region_size_list = []
    y_dim = initial_distribution.loc.size(0)

    if isinstance(initial_distribution, torch.distributions.MultivariateNormal):
        radii_d = gaussian_sphere_radius_scale_cov(target_coverage, 
                                                   initial_distribution.loc.size(0), 
                                                   initial_distribution.covariance_matrix[0][0])
        base_region_size = n_dimensional_sphere_volume(y_dim, radii_d)
    
        for i in tqdm(range(h.size(0))):
            
            if batch_processing:
                
                batch_size = sampling_num//batch_processing

                det_jacobian_record = np.zeros(shape=sampling_num)

                for k in range(batch_processing):
                    x0_batch = quasi_uniform_sample_from_sphere_between_radii(y_dim, 0, radii_d, batch_size)
                    x0_batch = torch.tensor(x0_batch, dtype=torch.float32) # (sample_num, x_dim)
                    h_batch = h[i,:].unsqueeze(0).repeat(batch_size,1) # (sample_num, h_dim)
                    t_span = torch.linspace(0, 1, 3) 
                    _, state_sol = combined_ode(x0_batch, h_batch, t_span, guidance_scale, device=device)
                    
                    logdet_jacobian_i = state_sol[-1,:,-1].numpy()

                    det_jacobian_record[k*batch_size:(k+1)*batch_size] = np.exp(logdet_jacobian_i)

                det_jacobian_mean = np.mean(det_jacobian_record)
                est_region_size = base_region_size * det_jacobian_mean
                det_jacobian_mean_list.append(det_jacobian_mean)
                est_region_size_list.append(est_region_size)

            else:
                x0_batch = quasi_uniform_sample_from_sphere_between_radii(y_dim, 0, radii_d, sampling_num)
                x0_batch = torch.tensor(x0_batch, dtype=torch.float32) # (sample_num, x_dim)
                h_batch = h[i,:].unsqueeze(0).repeat(sampling_num,1) # (sample_num, h_dim)
                t_span = torch.linspace(0, 1, 3) 
                _, state_sol = combined_ode(x0_batch, h_batch, t_span, guidance_scale, device=device)
                
                logdet_jacobian_i = state_sol[-1,:,-1].numpy()
                det_jacobian_mean = np.mean(np.exp(logdet_jacobian_i))
                est_region_size = base_region_size * det_jacobian_mean
                det_jacobian_mean_list.append(det_jacobian_mean)
                est_region_size_list.append(est_region_size)
            
            #if i % 50 == 0:
            #    print("count : {}".format(i+1))
             #   print("accumulated avg det: {}".format(np.mean(np.array(det_jacobian_mean_list))))
              #  print("accumulated avg estimated region size: {}".format(np.mean(np.array(est_region_size_list))))
    
    else:
        raise NotImplementedError("wrong initial distribution assigned")
        
    return est_region_size_list, det_jacobian_mean_list, base_region_size


def estimate_region_size_error(flow, combined_ode, dataloader, guidance_scale, 
                         target_coverage, initial_distribution, sampling_num, device):
    """
    Monte Carlo estimate the determinant of the Jacobian of the flow by solving Jacobian ODE
    y0 corresponding each h_i is sampled
    """
    
    # encoder history for the test data
    h = []
    y = []

    flow.to(device)
    flow.eval()
    for x_batch, y_batch in tqdm(dataloader):
        # test data should come from in sequential order
        with torch.no_grad():
            
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x_batch.size(1)) # causal mask
            x_batch = x_batch.to(device)
            attn_mask = attn_mask.to(device)
            h_batch = flow.encode(x_batch, src_mask=attn_mask, src_key_padding_mask=None)
            
            h.append(h_batch.cpu().detach()[:,-1,:])
            y.append(y_batch[:,-1,:])
    
    h = torch.from_numpy(np.array(h)).squeeze(1).to(device)
    y = torch.from_numpy(np.array(y)).squeeze(1).to(device)

    det_jacobian_relative_se_list = []
    y_dim = initial_distribution.loc.size(0)

    if isinstance(initial_distribution, torch.distributions.MultivariateNormal):
        radii_d = gaussian_sphere_radius_scale_cov(target_coverage, 
                                                   initial_distribution.loc.size(0), 
                                                   initial_distribution.covariance_matrix[0][0])
        base_region_size = n_dimensional_sphere_volume(y_dim, radii_d)

        det_jacobian_record = np.zeros(shape=(h.size(0), sampling_num))
    
        for i in tqdm(range(h.size(0))):
            
            x0_batch = quasi_uniform_sample_from_sphere_between_radii(y_dim, 0, radii_d, sampling_num)
            x0_batch = torch.tensor(x0_batch, dtype=torch.float32) # (sample_num, x_dim)
            h_batch = h[i,:].unsqueeze(0).repeat(sampling_num,1) # (sample_num, h_dim)
            t_span = torch.linspace(0, 1, 3) 
            _, state_sol = combined_ode(x0_batch, h_batch, t_span, guidance_scale, device=device)
            
            logdet_jacobian_i = state_sol[-1,:,-1].numpy()
            det_jacobian_i = np.exp(logdet_jacobian_i)

            det_jacobian_record[i, :] = det_jacobian_i

            det_jacobian_mean = np.mean(np.exp(logdet_jacobian_i))
            det_jacobian_se = np.std(np.exp(logdet_jacobian_i)) / np.sqrt(sampling_num) # standard error of det_jacobian
            det_jacobian_relative_se = det_jacobian_se / det_jacobian_mean

            det_jacobian_relative_se_list.append(det_jacobian_relative_se)
    
    else:
        raise NotImplementedError("wrong initial distribution assigned")
        
    return det_jacobian_record, det_jacobian_relative_se_list


def flow_encode(flow, dataloader, device):

    h = []
    y = []

    flow.to(device)
    flow.eval()
    for x_batch, y_batch in tqdm(dataloader):
        with torch.no_grad():
            
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x_batch.size(1)) # causal mask
            x_batch = x_batch.to(device)
            attn_mask = attn_mask.to(device)
            h_batch = flow.encode(x_batch, src_mask=attn_mask, src_key_padding_mask=None)
            
            h.append(h_batch.cpu().detach()[:,-1,:])
            y.append(y_batch[:,-1,:])
    
    h = torch.from_numpy(np.array(h)).squeeze(1).to(device)
    y = torch.from_numpy(np.array(y)).squeeze(1).to(device)

    return h, y


def flow_transform_reverse(flow, flow_ode, dataloader, guidance_scale, device):

    # encoder history for the test data
    h = []
    y = []

    flow.to(device)
    flow.eval()
    for x_batch, y_batch in tqdm(dataloader):
        # test data should come from in sequential order
        with torch.no_grad():
            
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x_batch.size(1)) # causal mask
            x_batch = x_batch.to(device)
            attn_mask = attn_mask.to(device)
            h_batch = flow.encode(x_batch, src_mask=attn_mask, src_key_padding_mask=None)
            
            h.append(h_batch.cpu().detach()[:,-1,:])
            y.append(y_batch[:,-1,:])
    
    h = torch.from_numpy(np.array(h)).squeeze(1).to(device)
    y = torch.from_numpy(np.array(y)).squeeze(1).to(device)

    # the number of spans does not matter if adaptive step size is used
    t_span = torch.linspace(1, 0, 3).to(device)
    _, y_transformed = flow_ode(y, h, t_span, guidance_scale, device=device)

    return y_transformed[-1,:,:].cpu().detach()


def compute_empirical_coverage(flow, flow_ode, test_dataloader, guidance_scale, target_coverage, 
                               initial_distribution, device):

    # data transformed to p_init space
    y_transformed = flow_transform_reverse(flow, flow_ode, test_dataloader, guidance_scale, device=device)

    if not (0 < target_coverage < 1):
        raise ValueError("target coverage must be in the range (0, 1)")

    if isinstance(initial_distribution, torch.distributions.MultivariateNormal):
        radii_d = gaussian_sphere_radius_scale_cov(target_coverage, 
                                                   initial_distribution.loc.size(0), 
                                                   initial_distribution.covariance_matrix[0][0])
    else:
        raise NotImplementedError("wrong initial distribution assigned")

    square_dist = y_transformed.square().sum(-1) # square distance from the origin
    coverage_list = (square_dist <= (radii_d**2)) * 1
    empirical_coverage = (square_dist <= (radii_d**2)).sum() / y_transformed.size(0)

    return empirical_coverage.item(), coverage_list


def evenly_sample_points_2d_circle(radius, n_points):
    """
    Evenly sample n points on the circumference of a circle of radius r.

    Parameters:
        radius (float): Radius of the circle.
        n_points (int): Number of points to sample.

    Returns:
        np.ndarray: An array of shape (n_points, 2) containing the sampled points.
    """
    # Compute evenly spaced angles between 0 and 2π
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    # Compute x and y coordinates
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    # Combine x and y coordinates
    points = np.column_stack((x, y))
    
    return torch.from_numpy(points)

