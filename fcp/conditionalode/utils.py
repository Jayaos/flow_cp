import torch
import math
import numpy as np
from tqdm import tqdm
from scipy.stats import chi


def hairer_norm(tensor):
    return tensor.abs().pow(2).mean().sqrt()


@torch.no_grad()
def adapt_step(dt, error_ratio, safety, min_factor, max_factor, order):

    safety = safety.to(dt.device)
    max_factor = max_factor.to(dt.device)

    if error_ratio == 0: 
        return dt * max_factor
    if error_ratio < 1: 
        min_factor = torch.ones_like(dt).to(dt.device)
    exponent = torch.tensor(order, dtype=dt.dtype, device=dt.device).reciprocal()
    factor = torch.min(max_factor, torch.max(safety / error_ratio ** exponent, min_factor))
    return dt * factor


def list_tensor_device_set(list_tensor, device):
    
    list_tensor_set = []
    
    for tensor in list_tensor:
        list_tensor_set.append(tensor.to(device))
        
    return list_tensor_set


def gaussian_sphere_radius(probability: float, dimensions: int):
    """
    Calculate the radius of a sphere that covers a given probability region 
    for an isotropic Gaussian centered at the origin in d-dimensional space.
    
    Parameters:
    - probability (float): The desired probability region (0 < p < 1).
    - dimensions (int): The number of dimensions (d).
    
    Returns:
    - radius (float): The radius of the sphere.
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


def flow_transform(flow_ode, test_dataloader, reverse=True):

    ...

