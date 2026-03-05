import torch
from torch import nn, Tensor
import math
import math
import numpy as np
from fcp.config import TransformerEncoderConfig, IdentityEncoderConfig


class AutoregressiveTransformer(nn.Module):
    """
    simple Transformer encoder with a mask to preserve autoregressiveness

    Arguments
    ---------
    x: input tensor
    mask: mask tensor corresponding to the input tensor x

    Returns
    -------
    output: output tensor
    """
    
    def __init__(self, dim_input: int, dim_output: int, dim_model: int, num_head: int, dim_ff: int, num_layer: int, 
                 dropout: float = 0.1, batch_first: bool=True):
        
        super(AutoregressiveTransformer, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_model = dim_model
        self.positional_encoding = PositionalEncoding(dim_model, dropout=dropout)
        Encoder_Layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_head, dim_feedforward=dim_ff, 
                                                   dropout=dropout, batch_first=batch_first)
        self.Encoder = nn.TransformerEncoder(Encoder_Layer, num_layers=num_layer)
        self.input_linear = nn.Linear(dim_input, dim_model) # this will work as embedding layer for features

    def forward(self, x: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor):

        src_emb = self.input_linear(x)
        src_emb = src_emb * math.sqrt(self.dim_model)
        src_emb = self.positional_encoding(src_emb)

        return self.Encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments
        ---------
            x: Tensor, [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class IdentityEncoder(nn.Module):
    """
    output only the last feature in the feature sequence
    used in ablation experiment

    Arguments
    ---------
    x: input tensor, (batch_size, context_size, feature_dim)
    mask: mask tensor corresponding to the input tensor x

    Returns
    -------
    output: output tensor
    """
    def __init__(self, dim_input, dim_output):
        super(IdentityEncoder, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_model = dim_input

    def forward(self, x: Tensor):

        return x
    

class LinearEncoder(nn.Module):
    def __init__(self, dim_input: int, dim_model: int):
        super(LinearEncoder, self).__init__()
        self.dim_input = dim_input
        self.dim_model = dim_model

        self.linear = nn.Linear(dim_input, dim_model)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor):

        return self.relu(self.linear(x)) # (batch_size, past_window, dim_model)
    

class AveragePoolingEncoder(nn.Module):
    def __init__(self, dim_input: int):
        super(AveragePoolingEncoder, self).__init__()
        self.dim_input = dim_input
        self.dim_model = dim_input

    @staticmethod
    def average_pool(x):

        ones = torch.ones_like(x)
        x_cumsum = torch.cumsum(x, dim=1)
        divisor = torch.cumsum(ones, dim=1)

        return x_cumsum / divisor

    def forward(self, x: Tensor):
        """
        args
        ----
            x_avg_pooled: (batch_size, past_window, input_dim)
        """

        return self.average_pool(x) # (batch_size, past_window, input_dim)
    

class LinearAveragePoolingEncoder(nn.Module):
    def __init__(self, dim_input: int, dim_model: int):
        super(LinearAveragePoolingEncoder, self).__init__()
        self.dim_input = dim_input
        self.dim_model = dim_model

        self.linear = nn.Linear(dim_input, dim_model)
        self.relu = nn.ReLU()

    @staticmethod
    def average_pool(x):

        ones = torch.ones_like(x)
        x_cumsum = torch.cumsum(x, dim=1)
        divisor = torch.cumsum(ones, dim=1)

        return x_cumsum / divisor

    def forward(self, x: Tensor):
        """
        args
        ----
            x_avg_pooled: (batch_size, past_window, input_dim)
        """

        return self.relu(self.linear(self.average_pool(x))) # (batch_size, past_window, dim_model)
    

def initialize_encoder(encoder_config, dim_input, dim_output):

    if isinstance(encoder_config, TransformerEncoderConfig):
        print("use Transformer as an encoder")
        return AutoregressiveTransformer(dim_input, dim_output, encoder_config.model_dim,
                                         encoder_config.num_head, encoder_config.dim_ff,
                                         encoder_config.num_layer, encoder_config.dropout)
    
    elif isinstance(encoder_config, IdentityEncoderConfig):
        print("use current feature as the context")
        return IdentityEncoder(dim_input, dim_output)
    
    # TODO: add other types of time series encoders (LSTM, ...)
    """
    elif isinstance(encoder_config) == EncoderConfig:
        print("use the last feature without time series encoder")
        return IdentityEncoder(dim_input)
           
    elif isinstance(encoder_config) == LinearEncoderConfig:
        print("use a single hidden layer on the last feature as time series encoder")
        return LinearEncoder(dim_input, ts_encoder_config.dim_model)
    
    elif isinstance(encoder_config) == AvgPoolingEncoderConfig:
        print("use the avg pooling as time series encoder")
        return AveragePoolingEncoder(dim_input)
    
    elif isinstance(encoder_config) == AvgPoolingEncoderConfig:
        print("use a single hidden layer on the avg pooling as time series encoder")
        return LinearAveragePoolingEncoder(dim_input, ts_encoder_config.dim_model)
    """