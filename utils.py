import torch
import numpy as np
from torch import nn



def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "gelu":
        return nn.GELU
    if name == "silu":
        return nn.SiLU
    if name == "wavelet":
        return WaveletActivation
    if name == "leaky_relu":
         return lambda: nn.LeakyReLU(0.2)
    raise ValueError(f"Unsupported activation: {name}")

class WaveletActivation(nn.Module):
    """

    f(x) = w1 * sin(x) + w2 * cos(x)

    PINNSFORMER: A TRANSFORMER-BASED FRAME-WORK FOR PHYSICS-INFORMED NEURAL NETWORKS.
    
    Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding
    """
    def __init__(self, w1: float = 1.0, w2: float = 1.0):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(w1, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor(w2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


def grading_resample(points, centers, M, sigma=0.7, mode="max", side="both", eps=1e-8):
    """
    points: (N,6) array of original cloud with first 3 is position and last 3 is normals
    centers: list or array of shape (K,3) for K focus points
    M: desired number of samples
    sigma: Gaussian kernel bandwidth
    mode: "sum" or "max" to combine per-center weights
    side: "both", "left", or "right" to indicate which side to sample from
    """
    centers = np.atleast_2d(centers)
    pos = points[:,:3]
    norm = points[:,3:]
    centroid = np.mean(centers,axis=0)
    R_max = abs(np.max(np.linalg.norm(centers - centroid, axis=1)))
    pos = pos/R_max
    centers = centers/R_max

    # Compute (N,K) array of distances
    pts = pos[:, None, :]           # (N,1,3)
    if side == "both":
        ctr = np.array(centers)[None, :, :] # (1,K,3)
    elif side == "left":
        ctr = np.array([c for c in centers if c[1]>=0])[None, :, :] # (1,K,3)
    elif side == "right":
        ctr = np.array([c for c in centers if c[1]<=0])[None, :, :] # (1,K,3)
    else:
        raise ValueError("side must be 'both', 'left', or 'right'")
    


    # pts = pts[:,:,2]
    # pts = pts[:,:,None]
    # ctr = ctr[:,:,2].reshape(-1,2,1)
    dists = np.linalg.norm(pts - ctr, axis=2)  # (N,K)
    
    # Gaussian weights around each center
    w_centers = np.exp(-0.5 * (dists / sigma)**2)  # (N,K)
    
    # Combine across centers
    if mode == "sum":
        w = w_centers.sum(axis=1)      # (N,)
    elif mode == "max":
        w = np.max(w_centers, axis=1)  # (N,)
    else:
        raise ValueError("mode must be 'sum' or 'max'")
    
    # Avoid zeros and normalize
    w += eps
    p = w / np.sum(w)
    
    # Subsample M points without replacement
    idx = np.random.choice(len(pos), size=M, replace=False, p=p)
    downsample = np.concatenate((points[idx,:3],norm[idx,:]),axis=1)
    return downsample


def check_branch_latent(z):
    """[B,C,N]/ [B,D]
    offdiag mean in the range of 0.95-1.0 and tiny std is bad
    feature var small is bad """
    z = z.reshape(z.shape[0], -1)               # [B,D]
    z_norm = torch.nn.functional.normalize(z, dim=1) 

    S = z_norm @ z_norm.t()   
    
    B = S.shape[0]
    off = S[~torch.eye(B, dtype=torch.bool, device=S.device)]

    print("cos sim offdiag: mean=", off.mean().item(),
        "std=", off.std(unbiased=False).item(),
        "min=", off.min().item(),
        "max=", off.max().item())
    
    z0 = z - z.mean(dim=0, keepdim=True)      
    var = z0.var(dim=0, unbiased=False)

    print("feature var: mean=", var.mean().item(),
        "median=", var.median().item(),
        "min=", var.min().item())

    C = (z0 @ z0.t()) / (z0.shape[1] + 1e-12)   
    eig = torch.linalg.eigvalsh(C).clamp_min(1e-12)
    p = eig / eig.sum()
    eff_rank = torch.exp(-(p * torch.log(p)).sum())
    print("effective rank (on BxB gram):", eff_rank.item())

def check_trunk_latent(t_feat, eps=1e-12):  # t_feat: [B,L,C]
    B, L, C = t_feat.shape

    # (A) token cosine stats
    t = torch.nn.functional.normalize(t_feat, dim=-1)  # normalize per token
    eye = torch.eye(L, dtype=torch.bool, device=t.device)

    cos_stats = []
    eranks = []
    for b in range(B):
        S = t[b] @ t[b].t()          # [L,L]
        off = S[~eye]
        cos_stats.append(torch.stack([
            off.mean(), off.std(unbiased=False), off.min(), off.max()
        ]))

        # (B) token effective rank on centered features (do NOT use normalized t here)
        Tb = t_feat[b]                          # [L,C]
        Tb0 = Tb - Tb.mean(dim=0, keepdim=True) # center over tokens
        G = (Tb0 @ Tb0.t()) / (C + eps)         # [L,L]
        eig = torch.linalg.eigvalsh(G).clamp_min(eps)
        p = eig / eig.sum()
        erank = torch.exp(-(p * torch.log(p)).sum())
        eranks.append(erank)

    cos_stats = torch.stack(cos_stats, dim=0)  # [B,4]
    eranks = torch.stack(eranks, dim=0)        # [B]

    print("per-sample token cos-sim offdiag mean/std/min/max (avg over B) =",
          cos_stats.mean(dim=0).tolist())
    print("per-sample token effective rank (avg/min/max over B) =",
          eranks.mean().item(), eranks.min().item(), eranks.max().item())
    
def lsd_loss_db(y_true:torch.Tensor, y_pred:torch.Tensor):
    n_sample = y_true.shape[0]
    n_freq = y_true.shape[1]
    error = torch.pow((y_pred - y_true),2)
    Lsd = torch.sqrt(error.mean(dim=1)).mean()
    
    return Lsd

def relative_l2_loss(y_true:torch.Tensor, y_pred:torch.Tensor, eps=1e-8):
    num = torch.sum((y_pred - y_true) ** 2, dim=1)
    den = torch.sum(y_true ** 2, dim=1) + eps
    return torch.mean(num / den)

def relative_l1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, eps=1e-8):
    num = torch.sum(abs(y_pred - y_true), dim=1)
    den = torch.sum(abs(y_true), dim=1) + eps
    return torch.mean(num / den)