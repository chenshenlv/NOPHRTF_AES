import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vae_fnn import FNNVAE


class NumpyDataset(Dataset):
    def __init__(self, array):
        if array.ndim != 2:
            raise ValueError("Input array must be 2D: [num_samples, num_features].")
        self.array = array.astype(np.float32)

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, idx):
        return self.array[idx]


def load_array(path, key=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        data = np.load(path)
        if key is None:
            if len(data.files) != 1:
                raise ValueError("NPZ has multiple arrays; provide --data-key.")
            key = data.files[0]
        return data[key]
    raise ValueError("Unsupported file type. Use .npy or .npz.")


def split_train_val(array, val_ratio, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(array.shape[0])
    split = int(array.shape[0] * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    return array[train_idx], array[val_idx]


def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


def get_beta(epoch, total_epochs, beta, warmup_epochs):
    if warmup_epochs <= 0:
        return beta
    progress = min(1.0, epoch / warmup_epochs)
    return beta * progress


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_array = load_array(args.train_data, args.data_key)
    if args.val_data:
        val_array = load_array(args.val_data, args.val_key)
    else:
        train_array, val_array = split_train_val(
            train_array, args.val_ratio, args.seed
        )

    if args.standardize:
        mean = train_array.mean(axis=0, keepdims=True)
        std = train_array.std(axis=0, keepdims=True) + 1e-8
        train_array = (train_array - mean) / std
        val_array = (val_array - mean) / std
    else:
        mean, std = None, None

    train_loader = DataLoader(
        NumpyDataset(train_array),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        NumpyDataset(val_array),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device(args.device)
    model = FNNVAE(
        input_dim=train_array.shape[1],
        latent_dim=args.latent_dim,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = get_beta(epoch, args.epochs, args.beta, args.warmup_epochs)
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss, recon_l, kl_l = vae_loss(recon, batch, mu, logvar, beta)
                val_loss += loss.item() * batch.size(0)
                recon_loss += recon_l.item() * batch.size(0)
                kl_loss += kl_l.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)
        recon_loss /= len(val_loader.dataset)
        kl_loss /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt = {
                "model_state": model.state_dict(),
                "input_dim": train_array.shape[1],
                "latent_dim": args.latent_dim,
                "encoder_hidden": args.encoder_hidden,
                "decoder_hidden": args.decoder_hidden,
                "activation": args.activation,
                "dropout": args.dropout,
                "mean": mean,
                "std": std,
            }
            torch.save(ckpt, os.path.join(args.output_dir, "vae_fnn.pt"))

        print(
            f"Epoch {epoch:03d} | beta={beta:.4f} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"recon={recon_loss:.6f} | kl={kl_loss:.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train FNN VAE on tabular data.")
    parser.add_argument("--train-data", required=True, help="Path to .npy/.npz data.")
    parser.add_argument("--data-key", default=None, help="NPZ key for train data.")
    parser.add_argument("--val-data", default=None, help="Optional .npy/.npz val data.")
    parser.add_argument("--val-key", default=None, help="NPZ key for val data.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument(
        "--encoder-hidden",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Encoder hidden layer sizes.",
    )
    parser.add_argument(
        "--decoder-hidden",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Decoder hidden layer sizes.",
    )
    parser.add_argument("--activation", default="silu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="VAE/checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
