import os
import numpy as np
from joblib import load as joblib_load
from joblib import dump as joblib_dump
import torch
from sklearn.preprocessing import StandardScaler
# hrtf_dim = 4
"""sclaer using isotropic scaler for points cloud and frequency wise scaler for hrtf and including scaler for itd"""

class HRTFScaler_single:
    """
    Per-frequency standardization (global over sub,loc) with optional per-sample
    (sub,loc) de-meaning across frequency to emphasize spectral shape.

    Expected input H:
      - [n_sub, n_loc, n_fre, 1]  or
      - [*, n_fre, 1] (any leading dims)

    Channels: in log magnitude / dB domain.
    """
    def __init__(
        self,
        eps: float = 1e-9,
        device=None,
        dtype=torch.float32,
        std_min: float = 1e-6,
        remove_sample_mean: bool = False,
    ):
        self.eps = float(eps)
        self.device = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self.std_min = float(std_min)

        # Options
        self.remove_sample_mean = bool(remove_sample_mean)
        # If True, transform() returns (H_norm, mean_L, mean_R) and inverse_transform
        # can reconstruct exactly using those means.

        # Learned stats: [n_fre]
        # self.sample_mu = None
        self.mu = None
        self.s = None

    @torch.no_grad()
    def fit(self, H: torch.Tensor, method: str):
        """
        Fit global per-frequency mean/std on TRAIN split only.

        H: [n_sub, n_loc, n_fre, 1]

        method: sub: over sub
                fre: over frequency
                all: global
        """
        H = H.to(device=self.device, dtype=self.dtype, non_blocking=True)
        sample_mu = H.mean(dim=(1), keepdim=True) #center the hrtf manifold for each subject
        # sample_mu = H.mean(dim=(1,2), keepdim=True)
        # # Optional: remove per-sample mean across the defined dimension BEFORE computing global stats
        if self.remove_sample_mean:
            H = H - sample_mu

        # Global mean
        if method == 'fre':
            self.mu = H.mean(dim=(0, 1),keepdim=True)  # [n_fre]
            var_H = (H - self.mu).pow(2).mean(dim=(0, 1),keepdim=True)
        elif method == 'sub':
            self.mu = H.mean(dim=(0), keepdim=True)  # [n_loc,n_fre,1]
            var_H = (H - self.mu).pow(2).mean(dim=(0),keepdim=True)
        elif method == 'loc':
            self.mu = H.mean(dim=(1), keepdim=True)  # [n_sub,n_fre,1]
            var_H = (H - self.mu).pow(2).mean(dim=(1),keepdim=True)
        elif method == 'all':
            self.mu = H.mean()  # scalar
            var_H = (H - self.mu).pow(2).mean() # scalar
        
        self.s = torch.sqrt(var_H + self.eps).clamp_min(self.std_min)

        return self

    def _check_fitted(self):
        if any(x is None for x in (self.mu, self.s)):
            raise RuntimeError("Call fit() before transform()/inverse_transform().")

    def transform(self, H: torch.Tensor):
        """
        Normalize H.

        If store_sample_mean=False:
            returns H_norm with same shape as H
        If store_sample_mean=True and remove_sample_mean=True:
            returns (H_norm, m_L, m_R) where m_* has shape H[...,0].mean(dim=-1,keepdim=True)
            so inverse_transform can exactly reconstruct.
        """
        self._check_fitted()
        H = H.to(device=self.device, dtype=self.dtype, non_blocking=True)
        m = None
        if self.remove_sample_mean:
            # per-sample mean across frequency (broadcastable)
            m = H.mean(dim=(1), keepdim=True)
            H = H - m
        # H_norm = (H - self.mu.view(1, 1, -1, 1)) / self.s.view(1, 1, -1, 1) # over freq
        H_norm = (H - self.mu) / self.s # over sub

        return H_norm, m

    def inverse_transform(self, H_norm: torch.Tensor, m: torch.Tensor = None):
        """
        Invert normalization.

        If remove_sample_mean=True:
          - If you want exact reconstruction, pass m returned by transform() when store_sample_mean=True.
          - If you don't pass means, this returns the de-meaned reconstruction (shape correct, but level/offset lost).
        """
        self._check_fitted()
        H_norm = H_norm.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # H = H_norm * self.s.view(1, 1, -1, 1) + self.mu.view(1, 1, -1, 1) # over frequency
        H = H_norm * self.s + self.mu # over sub

        if self.remove_sample_mean:
            if (m is not None):
                m = m.to(device=self.device, dtype=self.dtype, non_blocking=True)
                H = H + m
            # else: intentionally keep de-meaned output

        return H

    # def state_dict(self):
    #     return {
    #         "eps": self.eps,
    #         "std_min": self.std_min,
    #         "remove_sample_mean": self.remove_sample_mean,
    #         "store_sample_mean": self.store_sample_mean,
    #         "s": None if self.s is None else self.s.detach().cpu(),
    #         "mu": None if self.mu is None else self.mu.detach().cpu(),
    #     }

    # def load_state_dict(self, state):
    #     self.eps = float(state["eps"])
    #     self.std_min = float(state.get("std_min", self.std_min))
    #     self.remove_sample_mean = bool(state.get("remove_sample_mean", False))
    #     self.store_sample_mean = bool(state.get("store_sample_mean", False))

    #     self.s = state["s"].to(self.device, dtype=self.dtype)
    #     self.mu = state["mu"].to(self.device, dtype=self.dtype)
   
class HRTFScaler:
    """
    Per-frequency standardization (global over sub,loc) with optional per-sample
    (sub,loc) de-meaning across frequency to emphasize spectral shape.

    Expected input H:
      - [n_sub, n_loc, n_fre, 2]  or
      - [*, n_fre, 2] (any leading dims)

    Channels: [L_mag, R_mag] in log magnitude / dB domain.
    """
    def __init__(
        self,
        eps: float = 1e-9,
        device=None,
        dtype=torch.float32,
        std_min: float = 1e-6,
        remove_sample_mean: bool = False,
        store_sample_mean: bool = False,
    ):
        self.eps = float(eps)
        self.device = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self.std_min = float(std_min)

        # Options
        self.remove_sample_mean = bool(remove_sample_mean)
        # If True, transform() returns (H_norm, mean_L, mean_R) and inverse_transform
        # can reconstruct exactly using those means.
        self.store_sample_mean = bool(store_sample_mean)

        # Learned stats: [n_fre]
        self.mu_L = None; self.mu_R = None
        self.s_L = None;  self.s_R  = None

    @torch.no_grad()
    def fit(self, H: torch.Tensor):
        """
        Fit global per-frequency mean/std on TRAIN split only.

        H: [n_sub, n_loc, n_fre, 2]
        """
        H = H.to(device=self.device, dtype=self.dtype, non_blocking=True)

        L = H[..., 0]  # [n_sub, n_loc, n_fre]
        R = H[..., 1]

        # Optional: remove per-sample mean across frequency BEFORE computing global stats
        if self.remove_sample_mean:
            L = L - L.mean(dim=-1, keepdim=True)
            R = R - R.mean(dim=-1, keepdim=True)

        # Global per-frequency mean
        self.mu_L = L.mean(dim=(0, 1))  # [n_fre]
        self.mu_R = R.mean(dim=(0, 1))

        # Global per-frequency std
        var_L = (L - self.mu_L).pow(2).mean(dim=(0, 1))
        var_R = (R - self.mu_R).pow(2).mean(dim=(0, 1))
        self.s_L = torch.sqrt(var_L + self.eps).clamp_min(self.std_min)
        self.s_R = torch.sqrt(var_R + self.eps).clamp_min(self.std_min)
        return self

    def _check_fitted(self):
        if any(x is None for x in (self.mu_L, self.mu_R, self.s_L, self.s_R)):
            raise RuntimeError("Call fit() before transform()/inverse_transform().")

    def transform(self, H: torch.Tensor):
        """
        Normalize H.

        If store_sample_mean=False:
            returns H_norm with same shape as H
        If store_sample_mean=True and remove_sample_mean=True:
            returns (H_norm, m_L, m_R) where m_* has shape H[...,0].mean(dim=-1,keepdim=True)
            so inverse_transform can exactly reconstruct.
        """
        self._check_fitted()
        H = H.to(device=self.device, dtype=self.dtype, non_blocking=True)

        L = H[..., 0]
        R = H[..., 1]

        m_L = None
        m_R = None
        if self.remove_sample_mean:
            # per-sample mean across frequency (broadcastable)
            m_L = L.mean(dim=-1, keepdim=True)
            m_R = R.mean(dim=-1, keepdim=True)
            L = L - m_L
            R = R - m_R

        L_norm = (L - self.mu_L) / self.s_L
        R_norm = (R - self.mu_R) / self.s_R
        H_norm = torch.stack((L_norm, R_norm), dim=-1)

        if self.store_sample_mean and self.remove_sample_mean:
            return H_norm, m_L, m_R
        return H_norm

    def inverse_transform(self, H_norm: torch.Tensor, m_L: torch.Tensor = None, m_R: torch.Tensor = None):
        """
        Invert normalization.

        If remove_sample_mean=True:
          - If you want exact reconstruction, pass (m_L, m_R) returned by transform() when store_sample_mean=True.
          - If you don't pass means, this returns the de-meaned reconstruction (shape correct, but level/offset lost).
        """
        self._check_fitted()
        H_norm = H_norm.to(device=self.device, dtype=self.dtype, non_blocking=True)

        L_norm = H_norm[..., 0]
        R_norm = H_norm[..., 1]

        L = L_norm * self.s_L + self.mu_L
        R = R_norm * self.s_R + self.mu_R

        if self.remove_sample_mean:
            if (m_L is not None) and (m_R is not None):
                m_L = m_L.to(device=self.device, dtype=self.dtype, non_blocking=True)
                m_R = m_R.to(device=self.device, dtype=self.dtype, non_blocking=True)
                L = L + m_L
                R = R + m_R
            # else: intentionally keep de-meaned output

        return torch.stack((L, R), dim=-1)

    def state_dict(self):
        return {
            "eps": self.eps,
            "std_min": self.std_min,
            "remove_sample_mean": self.remove_sample_mean,
            "store_sample_mean": self.store_sample_mean,
            "s_L": None if self.s_L is None else self.s_L.detach().cpu(),
            "s_R": None if self.s_R is None else self.s_R.detach().cpu(),
            "mu_L": None if self.mu_L is None else self.mu_L.detach().cpu(),
            "mu_R": None if self.mu_R is None else self.mu_R.detach().cpu(),
        }

    def load_state_dict(self, state):
        self.eps = float(state["eps"])
        self.std_min = float(state.get("std_min", self.std_min))
        self.remove_sample_mean = bool(state.get("remove_sample_mean", False))
        self.store_sample_mean = bool(state.get("store_sample_mean", False))

        self.s_L = state["s_L"].to(self.device, dtype=self.dtype)
        self.s_R = state["s_R"].to(self.device, dtype=self.dtype)
        self.mu_L = state["mu_L"].to(self.device, dtype=self.dtype)
        self.mu_R = state["mu_R"].to(self.device, dtype=self.dtype)

class Scaler:
    """scaler class for inital data x and y
    compute new StandardScalers from the provided (x, y) data, or
    load existing scalers from disk,
    use IsotropicPointScaler
    """
    def __init__(self, itd:np.ndarray|None,
                 y:torch.Tensor,
                 hrtf_scaler_path: str | None = None,
                 itd_scaler_path: str | None = None,
                 force_recompute: bool = False,):

        self.y = y
        self.itd = itd
        if y is not None:
            _, self.n_eval_loc, self.n_freq,_ = y.shape


        # where to save / load
        self.hrtf_scaler_path   = hrtf_scaler_path
        self.itd_scaler_path   = itd_scaler_path

        # placeholders for the actual scalers
        self.hrtf_scaler   = None
        self.itd_scaler = None

        # attempt to load existing scalers if they exist and we're not forcing recompute
        if not force_recompute:
            if hrtf_scaler_path and os.path.isfile(hrtf_scaler_path):
                self.hrtf_scaler = joblib_load(hrtf_scaler_path)
            if itd_scaler_path and os.path.isfile(itd_scaler_path):
                self.itd_scaler = joblib_load(itd_scaler_path)
        # L/R spectral mean
        self.mL = None
        self.mR = None

    def init_scalers(self, force_recompute:bool = False):
        """
        Ensure both scalers are available.  If missing (or force_recompute=True)
        we compute them from self.x / self.y, then save to disk if paths given.
        """

        if self.hrtf_scaler is None or force_recompute:
            self.compute_hrtf_scaler()
            if self.hrtf_scaler_path:
                joblib_dump(self.hrtf_scaler, self.hrtf_scaler_path)

        if self.itd_scaler is None or force_recompute:
            if self.itd is not None:
                self.compute_itd_scaler()
                if self.itd_scaler_path:
                    joblib_dump(self.itd_scaler, self.itd_scaler_path)
        

    def compute_hrtf_scaler(self):
      
        hrtf_scaler = HRTFScaler(remove_sample_mean=True, store_sample_mean=True)
        self.hrtf_scaler = hrtf_scaler.fit(self.y)
    
    def compute_itd_scaler(self):
        itd_scaler = StandardScaler()
        itd = self.itd.reshape((-1,1))
        self.itd_scaler = itd_scaler.fit(itd)
    
    def normalize_itd(self, itd):
        assert self.itd_scaler is not None, "Call init_scalers() first."
        itd = np.reshape(itd,(-1,1))
        itd_norm = self.itd_scaler.transform(itd)
        return itd_norm.reshape((-1,self.n_eval_loc))
    
    def normalize_hrtf(self, Y:torch.Tensor):
        assert self.hrtf_scaler is not None, "Call init_scalers() first."
        n_sub = Y.shape[0]
        n_eval_loc = Y.shape[1]
        n_freq = Y.shape[2]
        n_dim = Y.shape[3]
        Y,self.mL, self.mR = self.hrtf_scaler.transform(Y)
        Y = torch.reshape(Y,(n_sub,n_eval_loc,n_freq,n_dim))
       
        return Y
    
    
    def inverse_hrtf(self,Y:torch.Tensor,n_sub:int, n_loc:int=793, n_freq:int=220,n_dim:int=4 ):
       
        assert self.hrtf_scaler is not None, "Call init_scalers() first."

        Y = torch.reshape(Y,(-1,n_loc,n_freq,n_dim))
        # Y = self.hrtf_scaler.inverse_transform(Y)
        Y = self.hrtf_scaler.inverse_transform(Y, self.mL, self.mR)
        Y = torch.reshape(Y,(n_sub,n_loc,n_freq,n_dim))
        return Y
    
    def inverse_itd(self, itd:np.ndarray,n_sub:int,n_loc:int=793):
        assert self.itd_scaler is not None, "Call init_scalers() first."
        itd = np.reshape(itd,(-1,1))
        itd = self.itd_scaler.inverse_transform(itd)
        itd = np.reshape(itd,(n_sub,n_loc))
        return itd

class Scaler_single(Scaler):
    def __init__(self, itd:np.ndarray|None,
                y:torch.Tensor,
                hrtf_scaler_path: str | None = None,
                itd_scaler_path: str | None = None,
                force_recompute: bool = False,):
        super().__init__(itd, y)
        self.y = y
        self.itd = itd
        if y is not None:
            _, self.n_eval_loc, self.n_freq,_ = y.shape


        # where to save / load
        self.hrtf_scaler_path   = hrtf_scaler_path
        self.itd_scaler_path   = itd_scaler_path

        # placeholders for the actual scalers
        self.hrtf_scaler   = None
        self.itd_scaler = None

        # attempt to load existing scalers if they exist and we're not forcing recompute
        if not force_recompute:
            if hrtf_scaler_path and os.path.isfile(hrtf_scaler_path):
                self.hrtf_scaler = joblib_load(hrtf_scaler_path)
            if itd_scaler_path and os.path.isfile(itd_scaler_path):
                self.itd_scaler = joblib_load(itd_scaler_path)
        # L/R spectral mean
        self.m = None
    def compute_hrtf_scaler(self):
    
        hrtf_scaler = HRTFScaler_single(remove_sample_mean=True)
        self.hrtf_scaler = hrtf_scaler.fit(self.y, method='fre')
    
    def normalize_hrtf(self, Y:torch.Tensor):
        assert self.hrtf_scaler is not None, "Call init_scalers() first."
        n_sub = Y.shape[0]
        n_eval_loc = Y.shape[1]
        n_freq = Y.shape[2]
        n_dim = Y.shape[3]
        Y, self.m = self.hrtf_scaler.transform(Y)
        # if len(out) == 2:
        #     Y,self.m = out
        # elif len(out) < 2: 
        #     Y= self.hrtf_scaler.transform(Y)

        Y = torch.reshape(Y,(n_sub,n_eval_loc,n_freq,n_dim))
       
        return Y

    def inverse_hrtf(self,Y:torch.Tensor,n_sub:int, n_loc:int=793, n_freq:int=220,n_dim:int=4, ):
       
        assert self.hrtf_scaler is not None, "Call init_scalers() first."
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y)
        Y = torch.reshape(Y,(-1,n_loc,n_freq,n_dim))
        # Y = self.hrtf_scaler.inverse_transform(Y)
        if self.m == None:
            Y = self.hrtf_scaler.inverse_transform(Y)
        elif self.m != None: 
            Y = self.hrtf_scaler.inverse_transform(Y, self.m)
        Y = torch.reshape(Y,(n_sub,n_loc,n_freq,n_dim))
        return Y

class HRTFScaler_single_subject:
    """
    subject wise standardization (only over sub) .

    Expected input H:
      - [n_sub, n_loc, n_fre, 1]  or
      - [*, n_fre, 1] (any leading dims)

    Channels: in log magnitude / dB domain.
    """
    def __init__(
        self,
        eps: float = 1e-9,
        device=None,
        dtype=torch.float32,
        std_min: float = 1e-6,
    ):
        self.eps = float(eps)
        self.device = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self.std_min = float(std_min)

        # Learned stats: [n_loc,n_fre,1]
        self.mu = None
        self.s = None

    @torch.no_grad()
    def fit(self, H: torch.Tensor):
        """
        Fit global per-frequency mean/std on TRAIN split only.

        H: [n_sub, n_loc, n_fre, 1]
        """
        H = H.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # Global per-frequency mean
        self.mu = H.mean(dim=0)  # [n_loc,n_fre]
       
        # Global per-frequency std
        var_H = (H - self.mu).pow(2).mean(dim=(0))
        self.s = torch.sqrt(var_H + self.eps).clamp_min(self.std_min)
        return self

    def _check_fitted(self):
        if any(x is None for x in (self.mu, self.s)):
            raise RuntimeError("Call fit() before transform()/inverse_transform().")

    def transform(self, H: torch.Tensor):
        """
        Normalize H.

        If store_sample_mean=False:
            returns H_norm with same shape as H
        If store_sample_mean=True and remove_sample_mean=True:
            returns (H_norm, m_L, m_R) where m_* has shape H[...,0].mean(dim=-1,keepdim=True)
            so inverse_transform can exactly reconstruct.
        """
        self._check_fitted()
        H = H.to(device=self.device, dtype=self.dtype, non_blocking=True)
        # a = self.mu[None,:,:,:]
        H_norm = (H - self.mu[None,:,:,:]) / self.s[None,:,:,:]

        return H_norm

    def inverse_transform(self, H_norm: torch.Tensor, m: torch.Tensor = None):
        """
        Invert normalization.

        If remove_sample_mean=True:
          - If you want exact reconstruction, pass m returned by transform() when store_sample_mean=True.
          - If you don't pass means, this returns the de-meaned reconstruction (shape correct, but level/offset lost).
        """
        self._check_fitted()
        H_norm = H_norm.to(device=self.device, dtype=self.dtype, non_blocking=True)

        H = H_norm * self.s[None,:,:,:] + self.mu[None,:,:,:]

        return H

    def state_dict(self):
        return {
            "eps": self.eps,
            "std_min": self.std_min,
            # "remove_sample_mean": self.remove_sample_mean,
            # "store_sample_mean": self.store_sample_mean,
            "s": None if self.s is None else self.s.detach().cpu(),
            "mu": None if self.mu is None else self.mu.detach().cpu(),
        }

    def load_state_dict(self, state):
        self.eps = float(state["eps"])
        self.std_min = float(state.get("std_min", self.std_min))
        # self.remove_sample_mean = bool(state.get("remove_sample_mean", False))
        # self.store_sample_mean = bool(state.get("store_sample_mean", False))

        self.s = state["s"].to(self.device, dtype=self.dtype)
        self.mu = state["mu"].to(self.device, dtype=self.dtype)

class Scaler_single_subject(Scaler):
    def __init__(self, itd:np.ndarray|None,
                y:torch.Tensor,
                hrtf_scaler_path: str | None = None,
                itd_scaler_path: str | None = None,
                force_recompute: bool = False,):
        super().__init__(itd, y)
        self.y = y
        self.itd = itd
        if y is not None:
            _, self.n_eval_loc, self.n_freq,_ = y.shape


        # where to save / load
        self.hrtf_scaler_path   = hrtf_scaler_path
        self.itd_scaler_path   = itd_scaler_path

        # placeholders for the actual scalers
        self.hrtf_scaler   = None
        self.itd_scaler = None

        # attempt to load existing scalers if they exist and we're not forcing recompute
        if not force_recompute:
            if hrtf_scaler_path and os.path.isfile(hrtf_scaler_path):
                self.hrtf_scaler = joblib_load(hrtf_scaler_path)
            if itd_scaler_path and os.path.isfile(itd_scaler_path):
                self.itd_scaler = joblib_load(itd_scaler_path)
        
    def compute_hrtf_scaler(self):
    
        hrtf_scaler = HRTFScaler_single_subject()
        self.hrtf_scaler = hrtf_scaler.fit(self.y)
    
    def normalize_hrtf(self, Y:torch.Tensor):
        assert self.hrtf_scaler is not None, "Call init_scalers() first."
        n_sub = Y.shape[0]
        n_eval_loc = Y.shape[1]
        n_freq = Y.shape[2]
        n_dim = Y.shape[3]
        Y= self.hrtf_scaler.transform(Y)
        Y = torch.reshape(Y,(n_sub,n_eval_loc,n_freq,n_dim))
       
        return Y

    def inverse_hrtf(self,Y:torch.Tensor,n_sub:int, n_loc:int=793, n_freq:int=220,n_dim:int=4, ):
       
        assert self.hrtf_scaler is not None, "Call init_scalers() first."
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y)
        Y = torch.reshape(Y,(-1,n_loc,n_freq,n_dim))
        # Y = self.hrtf_scaler.inverse_transform(Y)
        Y = self.hrtf_scaler.inverse_transform(Y)
        Y = torch.reshape(Y,(n_sub,n_loc,n_freq,n_dim))
        return Y
