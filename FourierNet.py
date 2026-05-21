import torch
import torch.nn as nn
from FNN import LinearFNN
import math
from utils import *
from typing import List, Optional, Callable

# Wrapper for bare functions
class FuncLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

"""ON THE EIGENVECTOR BIAS OF FOURIER FEATURE NETWORKS:
FROM REGRESSION TO SOLVING MULTI-SCALE PDES WITH
PHYSICS-INFORMED NEURAL NETWORKS"""
class FourierFeatureMapping(nn.Module):
    """
    Encodes inputs using random Fourier feature embeddings:
      r(x) = [sin(2π B x), cos(2π B x)]
    where B_ij ~ N(0, scale^2).
    """
    def __init__(self, input_dim: int, mapping_size: int, scale: float):
        super().__init__()
        B = torch.randn(mapping_size, input_dim) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, input_dim]
        x_proj = 2.0 * math.pi * (x @ self.B.t())  # [batch_size, mapping_size]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierFeatureMapping_Leanable(nn.Module):
    def __init__(self, input_dim, mapping_size, init_scale: float):
        super().__init__()
        # B is now a parameter, initialized with the desired variance
        B = nn.Parameter(torch.randn(mapping_size, input_dim) * init_scale)
        self.B = B

    def forward(self, x):
        x_proj = 2*math.pi*(x @ self.B.t())
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierFeatureMapping_MultiScale(nn.Module):
    def __init__(self, spatial_dim=3, freq_dim=1, mapping_size=256, spatial_scale=1.0, freq_scale=10.0):
        super().__init__()
        input_dim = spatial_dim + freq_dim
        
        # Initialize B with different scales for different columns
        # Shape: [mapping_size, input_dim]
        B_init = torch.randn(mapping_size, input_dim)
        
        # Apply spatial_scale to the first 3 columns (x, y, z)
        B_init[:, :spatial_dim] *= spatial_scale
        
        # Apply freq_scale to the last column (f)
        B_init[:, spatial_dim:] *= freq_scale
        
        self.B = nn.Parameter(B_init)

    def forward(self, x):
        # x shape: [batch, 4] -> (x, y, z, f)
        x_proj = 2 * math.pi * (x @ self.B.t())
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
class SoftFourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size, init_scale: float=0.01):
        super().__init__()
        self.B = nn.Parameter(torch.randn(mapping_size, input_dim))  # Learnable frequencies
        self.init_scale = init_scale

    def forward(self, x):
        proj = 2 * torch.pi * x @ self.B.T
        decay = torch.exp(-self.init_scale * torch.norm(self.B, dim=1)**2)  # Soft amplitude
        sin_part = torch.sin(proj) * decay
        cos_part = torch.cos(proj) * decay
        return torch.cat([sin_part, cos_part], dim=-1)

# class FourierFNN(nn.Module):
#     """
#     Fourier feature mapping followed by a feed-forward network (FNN).

#     Args:
#         input_dim (int): Dimensionality of each input sample.
#         mapping_size (int): Number of Fourier features per scale.
#         scales (list of float): If learnable=False, list of scales for each Fourier mapping.
#         hidden_dims (list of int): Sizes of hidden layers for the FNN.
#         output_dim (int): Dimensionality of the network output.
#         activation (str or list): Activation(s) after each hidden layer.
#         kernel_initializer (str): Name of weight initializer.
#         init_scale (float): Initial scale for learnable mapping (stddev of B).
#         learnable (bool): Whether to use a learnable Fourier mapping.
#         include_input (bool): If True, append raw input to the Fourier embeddings.
#         regularization: Optional regularizer applied to weights.
#     """
#     def __init__(
#         self,
#         input_dim: int,
#         mapping_size: int,
#         hidden_dims: list,
#         activation,
#         scales: list=[1,1,1],
#         init_scale: float = 1.0,
#         learnable: bool = False,
#         include_input: bool = False,
#         normalization: str = "ln",   # "bn", "ln", or None
#         regularization=None,
#     ):
#         super().__init__()
#         self.learnable = learnable
#         self.include_input = include_input
#         self.regularizer = regularization

#         # --- Fourier mappings ---
#         if not learnable:
#             self.ff_layers = nn.ModuleList([
#                 FourierFeatureMapping(input_dim, mapping_size, scale)
#                 for scale in scales
#             ])
#             embed_dim = len(scales) * 2 * mapping_size
#         else:
#             self.ff_layers = FourierFeatureMapping_Leanable(
#                 input_dim=input_dim,
#                 mapping_size=mapping_size,
#                 init_scale=init_scale,
#             )
#             embed_dim = 2 * mapping_size

#         if include_input:
#             embed_dim += input_dim

#         # --- FNN backbone ---

#         self.layers = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         prev_dim = embed_dim
#         # Hidden layers
#         for h in hidden_dims:
#             lin = nn.Linear(prev_dim, h, dtype=torch.float32)

#             self.layers.append(lin)

#             if normalization == "bn":
#                 self.norms.append(nn.BatchNorm1d(h))
#             elif normalization == "ln":
#                 self.norms.append(nn.LayerNorm(h))
#             else:
#                 self.norms.append(nn.Identity())
            
#             prev_dim = h
#         # Output layer
#         # lin = nn.Linear(prev_dim, output_dim, dtype=config.real(torch))
#         # init_w(lin.weight)
#         # init_b(lin.bias)
#         # self.layers.append(lin)

#         # Activation functions
        
#         self.activations = activation
        

#     def forward(self, inputs):
#         x = inputs
#         # Fourier feature embedding
#         if not self.learnable:
#             embeds = [ff(x) for ff in self.ff_layers]
#             h = torch.cat(embeds, dim=-1)
#         else:
#             h = self.ff_layers(x)
#         if self.include_input:
#             h = torch.cat([h, x], dim=-1)

#         # Feedforward through FNN layers
#         out = h
#         for i, lin in enumerate(self.layers):
#             out = lin(out)
#             out = self.norms[i](out)
#             # Apply activation after every layer except last
#             if i < len(self.layers) - 1:
#                 if isinstance(self.activations, list):
#                     act_fn = self.activations[i]
#                 else:
#                     act_fn = self.activations
#                 out = act_fn(out)

#         return out
    
class FourierFNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mapping_size: int,
        hidden_dims: list,
        activation: Optional[Callable[[], nn.Module]] = nn.SiLU,
        scales: list = [1, 1, 1],
        init_scale: float = 1.0,
        learnable: bool = False,
        include_input: bool = False,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        self.learnable = learnable
        self.include_input = include_input

        # --- Fourier mapping ---
        if not learnable:
            self.ff_layers = nn.ModuleList([
                FourierFeatureMapping(input_dim, mapping_size, scale)
                for scale in scales
            ])
            embed_dim = len(scales) * 2 * mapping_size
        else:
            self.ff_layers = FourierFeatureMapping_Leanable(
                input_dim=input_dim,
                mapping_size=mapping_size,
                init_scale=init_scale,
            )
            embed_dim = 2 * mapping_size

        if include_input:
            embed_dim += input_dim

        # --- Build FNN backbone ---
        layer_sizes = [embed_dim] + hidden_dims

        self.backbone = LinearFNN(
            layer_sizes=layer_sizes,
            activation=activation,
            last_activation=None,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, inputs):
        x = inputs

        # Fourier embedding
        if not self.learnable:
            embeds = [ff(x) for ff in self.ff_layers]
            h = torch.cat(embeds, dim=-1)
        else:
            h = self.ff_layers(x)

        if self.include_input:
            h = torch.cat([h, x], dim=-1)

        out = self.backbone(h)
        return out


class FourierUnetFnn(nn.Module):
    """
    U-Net with Fourier feature mapped input.

    This wrapper applies random Fourier feature embeddings to the original
    input channels at each sequence position, then feeds the resulting
    higher-dimensional representation into a standard UNet1D.
    learnable: If True, the Fourier feature mapping scales are learnable parameters.
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 features: list,
                 activation: Optional[Callable[[], nn.Module]],
                 fourier_mapping_size: int,
                 fourier_scales: list,
                 init_scale: float = 1.0,
                 learnable: bool = False,
                 normalization = "batch"):
        super().__init__()
        self.learnable = learnable
        # Initialize Fourier feature mappings for each scale
        if self.learnable == False:
            self.ff_layers = nn.ModuleList([
                FourierFeatureMapping(input_dim=in_channels,
                                    mapping_size=fourier_mapping_size,
                                    scale=s)
                for s in fourier_scales
            ])
            mapping_dim = len(fourier_scales) * 2 * fourier_mapping_size
        
        if self.learnable == True:
            self.ff_layers = FourierFeatureMapping_Leanable(
                input_dim=in_channels,
                mapping_size=fourier_mapping_size,
                init_scale=init_scale
            )
            mapping_dim = 2 * fourier_mapping_size

            # self.ff_layers = FourierFeatureMapping_MultiScale(
            #     spatial_dim=3,
            #     freq_dim = 1,
            #     mapping_size=fourier_mapping_size,
            #     spatial_scale=init_scale,
            #     freq_scale = 0.5*init_scale
            # )
            # mapping_dim = 2 * fourier_mapping_size

            # self.ff_layers = FourierFeatureMapping_LogLinear(
            #     spatial_dim=3,
            #     mapping_size=fourier_mapping_size,
            #     spatial_scale=init_scale,
            #     f_min=20.0,
            #     f_max=50.0
            # )
            # mapping_dim = 2 * fourier_mapping_size
            
        # Initialize base U-Net with updated in_channels
        self.unet = UNetFNN(
            in_channels=mapping_dim,
            out_channels=out_channels,
            features=features,
            activation=activation,
            normalization=normalization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, orig_in_channels, length)
        Returns:
            Tensor of shape (batch, out_channels)
        """
        # Handle single-sample input without batch dimension
        single = False
        if x.dim() == 2:
            if x.shape[0]>x.shape[1]:
                x = x.permute(1,0)
            # x: (C, L) -> (1, C, L)
            x = x.unsqueeze(0)

            single = True

        B, C, L = x.shape
        # Flatten batch and length dims to apply Fourier mapping per position
        x_flat = x.permute(0, 2, 1).reshape(B * L,C)
        # Apply Fourier feature mapping for each scale
        if self.learnable == False:
            embeds = [ff(x_flat) for ff in self.ff_layers]
            # Concatenate embeddings along feature dimension
            h = torch.cat(embeds, dim=-1)
            
        if self.learnable == True:
            h = self.ff_layers(x_flat)
        
        # Reshape for U-Net input: restore batch and length dims
        # h = torch.reshape(h,(B,h.shape[1],C))
        # h = h.reshape(B, L, -1).permute(0, 2, 1)
        # Forward through the underlying U-Net
        out = self.unet(h)
        # out = torch.reshape(out,(-1, out.shape[2],out.shape[1]))
        # Remove batch dimension if input was single-sample
        if single:
            out = out.squeeze(0)

        return out
    


class UNetFNN(nn.Module):
    """
    UNet-style architecture using fully-connected (FNN) blocks.

    Each encoder/decoder block consists of two Linear layers,
    with optional BatchNorm and activation, and skip-connections.

    Args:
        in_features (int): Dimensionality of input vectors.
        out_features (int): Dimensionality of output vectors.
        features (list of int): Hidden sizes for each level.
        activation (str): Activation name (e.g. "relu").
        kernel_initializer (str): Initializer for weights (e.g. "Glorot uniform").
        normalization (str): "batch" or "none" for BatchNorm1d or identity.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        features,
        activation:Optional[Callable[[], nn.Module]],
        normalization="batch",
        num_groups = 8
    ):
        super().__init__()
        # Activation & init
        # if isinstance(activation, str):
        #     # Call the returned class/lambda to create an instance
        #     act_class = get_activation(activation)
        #     act = act_class() 
        # elif isinstance(activation, nn.Module):
        #     act = activation
        # else:
        #     # Fallback for custom objects or FuncLayers
        #     act = nn.Sequential(nn.Identity(), FuncLayer(activation))
        self.act = activation
        self.num_groups = num_groups
        self.norm_type = normalization.lower()

        def make_norm(dim):
            if self.norm_type == "batch":
                return nn.BatchNorm1d(dim)
            elif self.norm_type == "group":
                return nn.GroupNorm(self.num_groups, dim)
            elif self.norm_type == "layer":
                return nn.LayerNorm(dim)
            else:
                return nn.Identity()

        # Encoder blocks
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_dim = in_channels
        for dim in features:
            block = nn.Sequential(
                nn.Linear(prev_dim, dim, dtype=torch.float32),
                make_norm(dim),
                self.act(),
                nn.Linear(dim, dim, dtype=torch.float32),
                make_norm(dim),
                self.act(),
            )
            self.encoders.append(block)
            # For FNN, pooling is identity
            self.pools.append(nn.Identity())
            prev_dim = dim

        # Bottleneck
        bott_dim = features[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Linear(features[-1], bott_dim, dtype=torch.float32),
            make_norm(bott_dim), self.act(),
            nn.Linear(bott_dim, bott_dim, dtype=torch.float32),
            make_norm(bott_dim), self.act(),
        )

        # Decoder blocks
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_features = list(reversed(features))
        prev_dim = bott_dim
        for dim in rev_features:
            # Linear upconv reduces dim to match skip
            self.upconvs.append(
                nn.Linear(prev_dim, dim, dtype=torch.float32)
            )
            decoder = nn.Sequential(
                nn.Linear(dim * 2, dim, dtype=torch.float32),
                make_norm(dim), self.act(),
                nn.Linear(dim, dim, dtype=torch.float32),
                make_norm(dim), self.act(),
            )
            self.decoders.append(decoder)
            prev_dim = dim

        # Final projection
        self.final = nn.Linear(prev_dim, out_channels, dtype=torch.float32)

    def forward(self, x):  # x: (batch, in_features)
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        x = self.final(x)
        return x

