import torch
from torch import nn, Tensor
from torch.nn.utils import spectral_norm


class ConcatSquashLinear(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, time_dim: int, output_dim: int, activation=None):
        super(ConcatSquashLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        combined_dim = time_dim + condition_dim
        self.hypernet_bias = nn.Linear(combined_dim, output_dim, bias=False)
        self.hypernet_gate = nn.Linear(combined_dim, output_dim)

        self.activation = activation


    def forward(self, x: Tensor, h: Tensor, t: Tensor) -> Tensor:
        # concatenate time and conditional vectors
        ht = torch.cat([h, t], dim=-1)  

        # Time- and condition-modulated gate and bias
        gate = torch.sigmoid(self.hypernet_gate(ht))
        bias = self.hypernet_bias(ht)

        out = self.linear(x) * gate + bias

        if self.activation is not None:
            out = self.activation(out)

        return out


class ConcatLinear(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, time_dim: int, output_dim: int, activation=None):
        super(ConcatLinear, self).__init__()
        self.linear = nn.Linear(input_dim + time_dim + condition_dim, output_dim)
        self.activation = activation

    def forward(self, x: Tensor, h: Tensor, t: Tensor) -> Tensor:
        xht = torch.cat([x, h, t], dim=-1) 
        out = self.linear(xht)

        if self.activation is not None:
            out = self.activation(out)

        return out
        

class Linear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:

        if self.activation is None:
            return self.linear(x)
        else:
            return self.activation(self.linear(x))
        

class iResNetBlock(nn.Module):
    def __init__(self, x_dim, h_dim, t_dim, hidden_dim, activation=nn.Softplus(), scale=0.9):
        super().__init__()
        self.activation = activation
        self.scale = scale  # Ensures Lip(g) < 1

        # Spectrally normalized MLP with conditional input
        self.fc1 = spectral_norm(nn.Linear(x_dim + h_dim + t_dim, hidden_dim))
        self.fc2 = spectral_norm(nn.Linear(hidden_dim, x_dim))

    def forward(self, x, h, t):

        inp = torch.cat([x, h, t], dim=-1)
        out = self.activation(self.fc1(inp))
        out = self.fc2(out)
        return x + self.scale * out  # Invertible residual map
        

class CFGVectorField(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 condition_dim: int, 
                 time_dim: int, 
                 hidden_dims: list, 
                 layer_type: str, 
                 activation=None):
        """
        MLP with ConcatLinear or ConcatSquashLinear for the first layer 
        to model guided vector field of guided flow

        args
        ----
        :param input_dim: Input feature dimension
        :param time_dim: Time input dimension
        :param hidden_dims: List of hidden layer dimensions
        :param layer_type: Type of layers to use ('linear' or 'squash')
        :param activation: Activation function (default: None)

        returns
        -------
        """
        super(CFGVectorField, self).__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        self.hidden_dims = hidden_dims
        self.layer_type = layer_type
        self.activation = activation

        # Validate layer type
        layer_cls = {"concatlinear": ConcatLinear, 
                     "concatsquash": ConcatSquashLinear}.get(layer_type)
        if layer_cls is None:
            raise ValueError(f"Invalid layer type: {layer_type}. Choose 'linear' or 'squash'.")


        layers = nn.ModuleList()

        # first layer
        self.first_layer = layer_cls(input_dim, condition_dim, time_dim, hidden_dims[0], activation)

        # hidden layers
        if len(hidden_dims) > 1:
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
                layers.append(Linear(in_dim, out_dim, activation))
        else:
            pass

        # last layer
        layers.append(Linear(hidden_dims[-1], input_dim, activation=None))
        self.layers = nn.Sequential(*layers)
                
    def forward(self, x, h, t):
        """
        Args
        ----
            x: input tensor, (batch_size, context_window, input_dim)
            h: condition tensor, (batch_size, context_window, condition_dim)
            t: time tensor, (batch_size, context_window, time_dim)

        Returns
        -------
            output tensor, (batch_size, input_dim)
        """
        if len(t.size()) == 1:
            # if t is scalar tensor, expand its dimension to (batch_size, 1)
            t = t.repeat(x.size(0)).view(x.size(0),1)

        return self.layers(self.first_layer(x, h, t))
    
    def cfg_forward(self, x, h, h_null, t, guidance_scale):
        
        if len(t.size()) == 1:
            # if t is scalar tensor, expand its dimension to (batch_size, 1)
            t = t.repeat(x.size(0)).view(x.size(0),1)

        o_guided = self.layers(self.first_layer(x, h, t))
        o_unguided = self.layers(self.first_layer(x, h_null, t))

        return (1-guidance_scale)*o_unguided + guidance_scale*o_guided 
    

class CFGiVectorField(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 condition_dim: int, 
                 time_dim: int, 
                 hidden_dims: list, 
                 activation=None):
        """
        MLP with ConcatLinear or ConcatSquashLinear for the first layer 
        to model guided vector field of guided flow

        args
        ----
        :param input_dim: Input feature dimension
        :param time_dim: Time input dimension
        :param hidden_dims: List of hidden layer dimensions
        :param layer_type: Type of layers to use ('linear' or 'squash')
        :param activation: Activation function (default: None)

        returns
        -------
        """
        super(CFGiVectorField, self).__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        self.layers = nn.ModuleList()
            
        for hidden_dim in self.hidden_dims:
            self.layers.append(iResNetBlock(self.input_dim, self.condition_dim, self.time_dim, hidden_dim,
                                       activation=self.activation))
                
    def forward(self, x, h, t):
        """
        Args
        ----
            x: input tensor, (batch_size, context_window, input_dim)
            h: condition tensor, (batch_size, context_window, condition_dim)
            t: time tensor, (batch_size, context_window, time_dim)

        Returns
        -------
            output tensor, (batch_size, input_dim)
        """
        if len(t.size()) == 1:
            # if t is scalar tensor, expand its dimension to (batch_size, 1)
            t = t.repeat(x.size(0)).view(x.size(0),1)

        for layer in self.layers:
            x = layer(x, h, t)

        return x
    
    def cfg_forward(self, x, h, h_null, t, guidance_scale):
        
        if len(t.size()) == 1:
            # if t is scalar tensor, expand its dimension to (batch_size, 1)
            t = t.repeat(x.size(0)).view(x.size(0),1)

        for i, layer in enumerate(self.layers):
            if i == 0:
                o_guided = layer(x, h, t)
            else:
                o_guided = layer(o_guided, h, t)

        for i, layer in enumerate(self.layers):
            if i == 0:
                o_unguided = layer(x, h_null, t)
            else:
                o_unguided = layer(o_unguided, h_null, t)

        return (1-guidance_scale)*o_unguided + guidance_scale*o_guided 
    

def initialize_vector_field(input_dim, condition_dim, time_dim, hidden_dims, layer_type, activation):

    if layer_type in ["concatlinear", "concatsquash"]:
        return CFGVectorField(input_dim, condition_dim, time_dim, hidden_dims, layer_type, activation)
    elif layer_type in ["iresnet"]:
        return CFGiVectorField(input_dim, condition_dim, time_dim, hidden_dims, activation)

