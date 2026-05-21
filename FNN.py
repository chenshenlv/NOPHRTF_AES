import torch
import torch.nn as nn
from typing import List, Optional, Callable

class LinearFNN(nn.Module):
    """
        net = LinearFNN(
            layer_sizes=[128, 64, 32, 10],
            activation=nn.ReLU,
            last_activation=None
        )
    """
    def __init__(
        self,
        layer_sizes: List[int],
        activation: Optional[Callable[[], nn.Module]] = nn.SiLU,
        last_activation: Optional[Callable[[], nn.Module]] = None,
        dropout: float = 0.0,
        use_batchnorm: bool = False,

    ):
       
        super().__init__()

        assert len(layer_sizes) >= 2, "Need at least input and output size."


        layers = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            # Linear layer
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = (i == len(layer_sizes) - 2)

            # Optional BatchNorm for hidden layers only
            if use_batchnorm and not is_last:
                layers.append(nn.BatchNorm1d(out_dim))

            # Activation
            if not is_last:
                if activation is not None:
                    layers.append(activation())
                # Optional dropout
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            else:
                # Last layer activation (optional)
                if last_activation is not None:
                    layers.append(last_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)