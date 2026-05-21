
import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from joblib import load,dump
from ae_fnn import FNNAE
from scaler import Scaler_single,Scaler_single_subject
from loss import LSD_l_mag_db
from DataLoader_operator import Preprocessor
import torch, traceback
import loss as metric


def load_array(path:str, key=None)->np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.endswith(".sav"):
        return load(path)
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


def split_train_val(array:np.ndarray, val_ratio, seed) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(array.shape[0])
    split = int(array.shape[0] * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    return array[train_idx], array[val_idx]


def prepare_array(array:np.ndarray) -> np.ndarray:
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array
    if array.ndim == 3:
        array = array.reshape(-1, array.shape[-1])
    if array.ndim <= 2:
        raise ValueError(
            "Expected shape [n_sub, n_loc, n_fre, 1] or [n_sub, n_loc, n_fre]."
        )
    return array

def preprocess_HRTF(HRTF_file_path:str, inputs_file_path:str, input_augment:bool=True) -> np.ndarray:
    """Flip the hrtf"""
    H = load_array(HRTF_file_path)
    x_inputs = load(inputs_file_path)
    if len(x_inputs) == 6:
        _,_,x_loc,_,_,_ = x_inputs
    if len(x_inputs) == 5:
        _,_,x_loc,_,_, = x_inputs
    preprocessor = Preprocessor()
    H = preprocessor.preprocess_outputs(x_loc,H,input_augment)
    return H

class NumpyDataset(Dataset):
    def __init__(self, array):
        if array.ndim != 2:
            raise ValueError("Input array must be 2D: [num_samples, num_features].")
        self.array = array.astype(np.float32)

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, idx):
        return self.array[idx]

class VAE():
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = self.args.device
    
    def build_data(self, train: bool = True):
        train_loader = None
        val_loader = None
        test_loader = None
        n_train = 0
        n_val = 0
        n_test = 0
        n_loc = 0
        n_freq = 0
        input_dim = 0
        if train:
            train_array = prepare_array(load_array(self.args.train_data, self.args.data_key))
            if self.args.val_data:
                val_array = prepare_array(load_array(self.args.val_data, self.args.val_key))
            else:
                train_array, val_array = split_train_val(
                    train_array, self.args.val_ratio, self.args.seed
                )
            # train_array = train_array[0:190,:,:,:]
            # val_array = val_array[0:10,:,:,:]
            n_train, n_loc, n_freq = train_array.shape[0], train_array.shape[1], train_array.shape[2]
            n_val = val_array.shape[0]

            scaler = None
            if self.args.standardize:
                train_array_d = torch.from_numpy(train_array).to(device=self.device, dtype=torch.float32, non_blocking=True)
                val_array_d = torch.from_numpy(val_array).to(device=self.device, dtype=torch.float32, non_blocking=True)
                scaler = Scaler_single(
                    y=train_array_d,
                    itd=None,
                    hrtf_scaler_path="./Scaler/hrtf_scaler_temp",
                    force_recompute=True,
                )
                scaler.init_scalers(True)
                train_array_d = scaler.normalize_hrtf(train_array_d).reshape(n_train, n_loc, n_freq, -1)
                val_array_d = scaler.normalize_hrtf(val_array_d).reshape(n_val, n_loc, n_freq, -1)
                train_array_h = train_array_d.cpu().numpy()
                val_array_h = val_array_d.cpu().numpy()
                train_array_h = train_array_h.reshape(n_train*n_loc, n_freq)
                val_array_h = val_array_h.reshape(n_val*n_loc, n_freq)
            else:
                train_array_h = train_array.reshape(n_train*n_loc, n_freq)
                val_array_h = val_array.reshape(n_train*n_loc, n_freq)

            train_loader = DataLoader(
                NumpyDataset(train_array_h),
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=False,
            )
            val_loader = DataLoader(
                NumpyDataset(val_array_h),
                batch_size=self.args.batch_size,
                shuffle=False,
                drop_last=False,
            )

            input_dim = n_freq
        
        if not train: #test
            if self.args.test_data:
                test_array = prepare_array(load_array(self.args.test_data, self.args.test_key))
            else:
                raise ValueError(
                    "test_data is not defined."
                )
            # test_array = test_array[0:10,:,:,:]
            n_test, n_loc, n_freq = test_array.shape[0], test_array.shape[1], test_array.shape[2]

            scaler = None
            if self.args.standardize:
                test_array_d = torch.from_numpy(test_array).to(device=self.device, dtype=torch.float32, non_blocking=True)
                scaler = Scaler_single(
                    y=None,
                    itd=None,
                    hrtf_scaler_path="./Scaler/hrtf_scaler_temp",
                    force_recompute=False,
                )
                scaler.init_scalers(False)
                test_array_d = scaler.normalize_hrtf(test_array_d).reshape(n_test, n_loc, n_freq, -1)
                test_array_h = test_array_d.cpu().numpy()
                test_array_h = test_array_h.reshape(n_test*n_loc, n_freq)
            else:
                test_array_h = test_array.reshape(n_test*n_loc, n_freq)
            test_loader = DataLoader(
                NumpyDataset(test_array_h),
                batch_size=self.args.batch_size,
                shuffle=False,
                drop_last=False,
            )

            input_dim = n_freq

        return (
            train_loader,
            val_loader,
            test_loader,
            scaler,
            n_train,
            n_val,
            n_test,
            n_loc,
            n_freq,
            input_dim,
        )

    def build_net(self, input_dim:int):
        model = FNNAE(
            input_dim=input_dim,
            latent_dim=self.args.latent_dim,
            encoder_hidden=self.args.encoder_hidden,
            decoder_hidden=self.args.decoder_hidden,
            activation=self.args.activation,
            dropout=self.args.dropout,
            ).to(self.device)
        return model
    
    def train(self,):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

        (
            train_loader,
            val_loader,
            _,
            scaler,
            n_train,
            n_val,
            _,
            n_loc,
            n_freq,
            input_dim,
        ) = self.build_data(train=True)
        model = self.build_net(input_dim=input_dim)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,weight_decay=self.args.l2_lambda,)
        best_val = float("inf")
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.5
        )

        for epoch in range(1, self.args.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                recon, _ = model(batch)
                # loss = nn.functional.mse_loss(recon, batch, reduction="mean")
                loss = metric.calculate_rmse(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    recon, _ = model(batch)
                    # loss = nn.functional.mse_loss(recon, batch, reduction="mean")
                    loss = metric.calculate_rmse(recon, batch)
                    val_loss += loss.item() * batch.size(0)
            val_loss /= len(val_loader.dataset)
            scheduler.step()
            if val_loss < best_val:
                best_val = val_loss
                os.makedirs(self.args.output_dir, exist_ok=True)
                ckpt = {
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "latent_dim": self.args.latent_dim,
                    "encoder_hidden": self.args.encoder_hidden,
                    "decoder_hidden": self.args.decoder_hidden,
                    "activation": self.args.activation,
                    "dropout": self.args.dropout,
                    # "mean": mean,
                    # "std": std,
                }
                torch.save(ckpt, os.path.join(self.args.output_dir, "ae_fnn_aug.pt"))
                print(f'val improved, model saved')
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}, LR: {current_lr}"
            )

    def test(self,):
        (
            _,
            _,
            test_loader,
            scaler,
            _,
            _,
            n_test,
            n_loc,
            n_freq,
            input_dim,
        ) = self.build_data(train=False)

        ckpt_path = os.path.join(self.args.output_dir, "ae_fnn_aug.pt")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model = self.build_net(input_dim=input_dim)
            model.load_state_dict(ckpt["model_state"])
        if model is None:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                recon, _ = model(batch)
                loss = nn.functional.mse_loss(recon, batch, reduction="mean") 
                test_loss += loss.item() * batch.size(0)
        test_loss /= len(test_loader.dataset)
        print(f"Test (val reuse) | mse={test_loss:.6f}")

        if self.args.standardize and scaler is not None:
            preds = []
            truths = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.device)
                    recon, _ = model(batch)
                    preds.append(recon.detach().cpu())
                    truths.append(batch.detach().cpu())
            y_pred = torch.cat(preds, dim=0).reshape(n_test, n_loc, n_freq)
            y_true = torch.cat(truths, dim=0).reshape(n_test, n_loc, n_freq)
            y_pred_inv = scaler.inverse_hrtf(
                y_pred, n_sub=n_test, n_loc=n_loc, n_freq=n_freq, n_dim=1
            )
            y_true_inv = scaler.inverse_hrtf(
                y_true, n_sub=n_test, n_loc=n_loc, n_freq=n_freq, n_dim=1
            )
            lse_loss = LSD_l_mag_db(
                y_true_inv[..., 0].cpu().numpy(), y_pred_inv[..., 0].cpu().numpy()
            )
            print(f"Test (val reuse, inverse) | lsd={lse_loss:.6f}")

        

def parse_args():
    parser = argparse.ArgumentParser(description="Train FNN AE on HRTF magnitude data.")
    parser.add_argument(
        "--train-data",
        default="VAE/augment_data/y_train.sav",
        help="Path to .npy/.npz data.",
    )
    parser.add_argument("--data-key", default=None, help="NPZ key for train data.")
    parser.add_argument("--val-data", default="VAE/augment_data/y_val.sav", help="Optional .npy/.npz/.sav val data.")
    parser.add_argument("--val-key", default=None, help="NPZ key for val data.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-data", default="VAE/augment_data/y_val.sav", help="Optional .npy/.npz/.sav val data.")
    parser.add_argument("--test-key", default=None, help="NPZ key for val data.")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument(
        "--encoder-hidden",
        type=int,
        nargs="+",
        default=[512,128],
        help="Encoder hidden layer sizes.",
    )
    parser.add_argument(
        "--decoder-hidden",
        type=int,
        nargs="+",
        default=[128,512],
        help="Decoder hidden layer sizes.",
    )
    parser.add_argument("--activation", default="silu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_lambda", type=float, default=1e-5)
    parser.add_argument("--standardize",default=True, action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="VAE/saved_model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    preprocess = False
    train = False
    test = True
    input_augment = False
    # base_folder = "./model_point/data/AXD/AE/mag/raw/msr_44_ff_ref_/5k/"
    # base_folder = "./model_point/data/AXD/dis_weighted/mag/raw/msr_44_ff/5k/"
    base_folder = "VAE/data/AES/5sets/"
    if preprocess:
        HRTF_file_path = base_folder + "y_train.sav"
        inputs_file_path = base_folder +  "X_train.sav"
        H_processed = preprocess_HRTF(HRTF_file_path, inputs_file_path,input_augment)
        save_path = "VAE/augment_data/y_train.sav"
        dump(H_processed, save_path)
        HRTF_file_path = base_folder + "y_val.sav"
        inputs_file_path = base_folder + "X_val.sav"
        H_processed = preprocess_HRTF(HRTF_file_path, inputs_file_path,input_augment)
        save_path = "VAE/augment_data/y_val.sav"
        dump(H_processed, save_path)
        print(f"Preprocessed HRTF saved to {save_path}")
    Model = VAE(args=args)
    try:
        if train:
            Model.train()
        if test:
            Model.test()
    except Exception as e:
        print("An error occurred during training/testing:")
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
