import torch
from torch import nn, Tensor
from torch.distributions import Distribution
from fcp.path import ProbPath
from .encoder import AutoregressiveTransformer, IdentityEncoder
import random


class CFGFlow(nn.Module):
    """
    Guided conditional normalizing flow trained using flow matching
    """
    
    def __init__(self, 
                 encoder: nn.Module, 
                 vector_field: nn.Module, 
                 initial_dist: Distribution,
                 probability_path: ProbPath):
        super(CFGFlow, self).__init__()

        """
        args
        ----
            encoder: an encoder to encode the historical features
            vector_field: neural network to model vector field of the flow
            initial_dist: initial distribution where x_0 is sampled from
        """
        
        self.encoder = encoder
        self.null_condition_embedding = nn.Parameter(torch.randn(self.encoder.dim_model), requires_grad=True)
        self.vector_field = vector_field
        self.initial_dist = initial_dist
        self.probability_path = probability_path

    def forward(self, 
                x: Tensor, 
                src_mask: Tensor, 
                src_key_padding_mask: Tensor, 
                y: Tensor,
                null_condition_prob: float,
                device) -> dict:

        """
        args
        ----
            x: time series historical features to generate guidance through the encoder
            src_mask: 
            src_key_padding_mask: mask tensor corresponding to the input tensor x
            y: target outcome
            null_condition_prob: probability to assign null condition guidance

        returns
        -------
            output: output tensor

        """

        if random.random() <= null_condition_prob:
            # unguided
            h = self.null_condition_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, dim_condition)
            h = h.expand(x.size(0), x.size(1), -1)  # (batch_size, seq_len, dim_output)
        else:
            # guided
            if isinstance(self.encoder, AutoregressiveTransformer):
                h = self.encoder(x, src_mask, src_key_padding_mask) # h summarizing the historical time series
            elif isinstance(self.encoder, IdentityEncoder): 
                h = self.encoder(x) # h is current feature

        y0 = self.initial_dist.sample((y.size(0),y.size(1),)).to(device) # (batch_size, past_window, dim_output)
        t = torch.rand((y0.size(0),y0.size(1))).type_as(y0).to(device) # (batch_size, past_window)

        # sample probability path
        path_sample = self.probability_path.sample(t=t, x_0=y0, x_1=y)
        vt = self.vector_field(path_sample.x_t, h, path_sample.t.unsqueeze(-1)) 
        loss = torch.mean((vt - path_sample.dx_t) ** 2)
        
        return {"h" : h, "vt" : vt, "path_sample" : path_sample, "loss" : loss}

    def encode(self, x: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor) -> Tensor:

        if isinstance(self.encoder, AutoregressiveTransformer):
            return self.encoder(x, src_mask, src_key_padding_mask) # h summarizing the historical time series
        elif isinstance(self.encoder, IdentityEncoder): 
            return self.encoder(x) # h is current feature

