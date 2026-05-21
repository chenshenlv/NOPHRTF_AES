import torch
import torch.nn.functional as F
from torch import nn
class ResidualBlock(nn.Module):
    """Residual block containing N dense layers with a skip connection spanning the block."""
    def __init__(self, input_dim, hidden_dims, activation, kernel_initializer, regularization=None):
        """
        Args:
            input_dim (int): Dimension of the block input.
            hidden_dims (list): A list of 3 integers, defining the output dimension of each linear layer in the block.
            activation (str or callable): Activation function applied after each layer.
            kernel_initializer (str): Name of the initializer for the weights.
            regularization: Regularization setting (if any).
        """
        super().__init__()
        if len(hidden_dims) < 2:
            raise ValueError("Each residual block must have more than 2 layers (2 hidden_dims).")
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")
        self.regularizer = regularization

        # Create three dense layers for the block.
        self.linears = torch.nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(1, len(dims)):
            layer = torch.nn.Linear(dims[i-1], dims[i], dtype=config.real(torch))
            self.linears.append(layer)
            initializer(layer.weight)
            initializer_zero(layer.bias)

        # Create a projection if input and final output dimensions differ.
        if input_dim != dims[-1]:
            self.projection = torch.nn.Linear(
                input_dim, dims[-1], bias=False, dtype=config.real(torch)
            )
            initializer(self.projection.weight)
        else:
            self.projection = None

    def forward(self, x):
        identity = x
        for layer in self.linears:
            x = self.activation(layer(x))
        if self.projection is not None:
            identity = self.projection(identity)
        return x + identity

class ResNetDenseArchitecture(NN):
    """ResNet architecture with N residual blocks, each block containing m dense layers."""
    def __init__(self, input_dim, block_hidden_dims, activation, kernel_initializer, regularization=None, final_output_dim=100):
        """
        Args:
            input_dim (int): Input feature dimension.
            block_hidden_dims (list): A list of N lists. Each sublist must contain m integers specifying the hidden dimensions for that block.
            activation (str or callable): Activation function applied in each block.
            kernel_initializer (str): Name of the initializer for the weights.
            regularization: Regularization setting (if any).
            final_output_dim (int): Final output dimension (should be 100 for your case).
        """
        super().__init__()
        
        if len(block_hidden_dims) < 2:
            raise ValueError("Expected more than 2 residual blocks; got {}.".format(len(block_hidden_dims)))
        self.blocks = torch.nn.ModuleList()
        current_dim = input_dim
        for block_dims in block_hidden_dims:
            block = ResidualBlock(current_dim, block_dims, activation, kernel_initializer, regularization)
            self.blocks.append(block)
            current_dim = block_dims[-1]  # Update dimension to output of current block.

        # Final dense layer to map to the desired output dimension.
        # self.activation = activations.get(activation)
        self.final_linear = torch.nn.Linear(current_dim, final_output_dim, dtype=config.real(torch))
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")
        initializer(self.final_linear.weight)
        initializer_zero(self.final_linear.bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_linear(x)
        # x = self.activation(x) # The final linear layer just project to a desired dimension, no activation function.
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
    

if __name__ == "__main__":
    print("test")
    activation = "relu"
    kernel_initializer = "Glorot normal"
    Resnet_config = {
    "input_dim": 3,  # e.g., each coordinate input has 3 features
    "block_hidden_dims": [
         [16, 16, 16],
         [16, 16, 16],
         [16, 32, 32],
         [32, 32, 32],
         [32, 32, 32]
    ],
    "final_output_dim": 64,
    "fnn": [64, 200, 200]
    }

    model = ResNetDenseArchitecture(
        input_dim=Resnet_config["input_dim"],
        block_hidden_dims=Resnet_config["block_hidden_dims"],
        activation=activation,
        kernel_initializer=kernel_initializer,
        regularization=None,
        final_output_dim=Resnet_config["final_output_dim"],
    )

    batch_size = 8
    x = torch.randn(batch_size, Resnet_config["input_dim"], dtype=config.real(torch))

    y = model(x)

    # 4. Inspect shapes
    print(f"Input shape : {x.shape}")   # (batch_size, input_dim)
    print(f"Output shape: {y.shape}")   # (batch_size, final_output_dim)
