import torch
from torch.distributions import MultivariateNormal


def set_initial_gaussian_distribution(cov_scale, dim):

    return MultivariateNormal(torch.zeros(dim), cov_scale*torch.eye(dim))