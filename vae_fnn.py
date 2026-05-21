import torch
from torch import nn


def _get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(in_dim, hidden_dims, out_dim, activation, dropout):
    layers = []
    last_dim = in_dim
    act = _get_activation(activation)
    for dim in hidden_dims:
        layers.append(nn.Linear(last_dim, dim))
        layers.append(act)
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        last_dim = dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class FNNVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_hidden,
        decoder_hidden,
        activation="silu",
        dropout=0.0,
    ):
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive.")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = _build_mlp(
            input_dim, encoder_hidden, 2 * latent_dim, activation, dropout
        )
        self.decoder = _build_mlp(
            latent_dim, decoder_hidden, input_dim, activation, dropout
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
