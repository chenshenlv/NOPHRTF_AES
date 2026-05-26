import torch
import torch.nn as nn
from utils import check_branch_latent,check_trunk_latent
import torch.nn.functional as F
import math

class PointTokenCrossAttn(nn.Module):
    def __init__(self, C_point: int, C_trunk: int, C_attn: int,
                 D_out: int, n_heads: int = 8, M: int = 1024, dropout: float = 0.0,
                cond_dim: int = 1, film_hidden: int = 128, film_on=("",""),drop_path_rate: float = 0.1):
        super().__init__()
        assert C_attn % n_heads == 0

        self.M = M
        # self.p_proj = nn.Sequential(
        #     nn.Linear(C_point, C_attn,bias=False),
        #     nn.LayerNorm(C_attn)
        # )
        # # self.t_proj = nn.Sequential(
        # #     nn.Linear(C_trunk, C_attn,bias=False),
        # #     nn.LayerNorm(C_attn)
        # # )

        self.p_proj = nn.Linear(C_point, C_attn,bias=False)
          
        self.t_proj = nn.Linear(C_trunk, C_attn,bias=False)

        self.q_mlp = nn.Sequential(
            nn.Linear(C_attn, 2 * C_attn),
            nn.GELU(),
            nn.Linear(2 * C_attn, C_attn),
        )
            
        self.attn = nn.MultiheadAttention(embed_dim=C_attn, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        # self.attn = kernel_base_MultiAtten(d_emb=C_attn, num_heads=n_heads,dropout=dropout)
        
        self.film_on = set(film_on)
        self.film_trunk = FiLM(cond_dim, C_trunk, hidden=film_hidden) if "trunk" in self.film_on else None
        self.film_branch = FiLM(cond_dim, C_point, hidden=film_hidden) if "branch" in self.film_on else None
        # self.gate = nn.Sequential(nn.Linear(C_attn, C_attn, bias=False), nn.Sigmoid())
        self.gate = nn.Sequential(
            nn.Linear(C_attn, C_attn // 2, bias=True),
            nn.GELU(),
            nn.Linear(C_attn // 2, 1, bias=True),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.constant_(self.gate[-2].bias, -4.0)

        self.alpha = nn.Parameter(torch.tensor([1.0])) 
        self.drop_path_rate = drop_path_rate

        self.norm1 = nn.LayerNorm(C_attn)
        self.ffn = nn.Sequential(
            nn.Linear(C_attn, 4 * C_attn),
            nn.GELU(),
            nn.Linear(4 * C_attn, C_attn),
        )
        self.norm2 = nn.LayerNorm(C_attn)
        self.out = nn.Linear(C_attn, D_out)
        nn.init.zeros_(self.out.weight)

    def sample_points(self, Pn: torch.Tensor) -> torch.Tensor:
        # Pn: [B, N, C]
        B, N, C = Pn.shape
        if self.M >= N:
            return Pn
        idx = torch.randint(0, N, (B, self.M), device=Pn.device)
        idx = idx.unsqueeze(-1).expand(-1, -1, C)  # [B,M,C]
        return torch.gather(Pn, dim=1, index=idx)
    
    def apply_rope(self, Q: torch.Tensor, loc: torch.Tensor):
        """
        Q: [B, L, C]  where C must be divisible by 6 (2 dims per axis * 3 axes)
        loc: [B, L, 3] (x, y, z coordinates)
        """
        B, L, C = Q.shape
        assert C % 6 == 0, "Embedding dim must be divisible by 6 for 3D RoPE"

        half_dim = C // 2  # Total pairs
        chunk_size = C // 3 # Space allocated for each axis (X, Y, Z)
        
        # 1. Generate frequencies (omega) for a single axis chunk
        # We use chunk_size // 2 because each pair (sin/cos) takes 2 channels
        k = torch.arange(chunk_size // 2, device=Q.device)
        omega = 10000 ** (-2 * k / chunk_size) # [chunk_size // 2]

        # 2. Calculate the rotation angles (theta) for each axis
        # theta_x, y, z shape: [B, L, chunk_size // 2]
        theta_x = loc[..., 0].unsqueeze(-1) * omega
        theta_y = loc[..., 1].unsqueeze(-1) * omega
        theta_z = loc[..., 2].unsqueeze(-1) * omega
        
        # Concatenate them to cover the full embedding dimension
        # Full theta shape: [B, L, C // 2]
        theta = torch.cat([theta_x, theta_y, theta_z], dim=-1)

        cos = torch.cos(theta)
        sin = torch.sin(theta)

        # 3. Apply the rotation to pairs
        Q = Q.view(B, L, C // 2, 2)
        q1 = Q[..., 0]
        q2 = Q[..., 1]

        Q_rot_1 = q1 * cos - q2 * sin
        Q_rot_2 = q1 * sin + q2 * cos

        return torch.stack([Q_rot_1, Q_rot_2], dim=-1).view(B, L, C)
    
    def get_sinusoidal_encoding(self, loc: torch.Tensor, C: int):
        """
        loc: [B, L, 3] (x, y, z)
        C: output embedding dimension (must be divisible by 6)
        """
        B, L, _ = loc.shape
        device = loc.device
        
        # Divide dimension by 3 (for X, Y, Z)
        # Each axis gets half sine and half cosine
        dims_per_axis = C // 3 
        half_dim = dims_per_axis // 2
        
        # Frequency scaling: 10000^(2i/d)
        # We use a base of 100 or 1000 for spatial data usually, 
        # but 10000 is the Transformer standard.
        emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000.0) / (half_dim - 1)))
        
        # Calculate for each axis [B, L, half_dim]
        args_x = loc[..., 0:1] * emb
        args_y = loc[..., 1:2] * emb
        args_z = loc[..., 2:3] * emb
        
        # Create [sin, cos] pairs for each axis
        def axis_enc(args):
            return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        enc_x = axis_enc(args_x)
        enc_y = axis_enc(args_y)
        enc_z = axis_enc(args_z)
        
        # Result: [B, L, C]
        return torch.cat([enc_x, enc_y, enc_z], dim=-1)
    
    def knn_neighbor_augment(self, Q:torch.Tensor, loc:torch.Tensor, k=8, gamma=0.1):
        # Q:   [B,L,C]   loc: [B,L,3] assumed ~unit sphere
        B, L, C = Q.shape
        sim = loc @ loc.transpose(1, 2)                 # [B,L,L], cosine similarity
        sim = sim - torch.eye(L, device=loc.device)[None] * 1e9  # mask self
        idx = sim.topk(k, dim=-1).indices               # [B,L,k]
        # gather neighbors correctly
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, C)  # [B,L,k,C]
        nbr = torch.take_along_dim(
            Q.unsqueeze(2).expand(-1, -1, k, -1),         # [B,L,k,C]
            idx_exp,
            dim=1
        )
        ctx = nbr.mean(dim=2)                           # [B,L,C]
        return Q + gamma * ctx

    def forward(self, P: torch.Tensor, T: torch.Tensor, x_pts:torch.Tensor,  cond=None) -> torch.Tensor:
        """
        P: point_clouds features
        T: location features + locations, 
        x_pts: origional point_clouds
        """
        # P: [B,Cp,N] -> [B,N,Cp]
        Pn = P.transpose(1, 2).contiguous()
      
        loc = T[:,:,256:]
        loc_feats = T[:,:,:256]
       

        # Project into attention space
        KVm = self.p_proj(Pn)            # [B,N,C_attn]
        K_ = KVm + self.get_sinusoidal_encoding(x_pts.transpose(1, 2).contiguous(), KVm.shape[-1])
        # K_ = self.apply_rope(KVm,x_pts.transpose(1,2))
        Q = self.t_proj(loc_feats)               # [B,L,C_attn]
        Q_ = Q + self.get_sinusoidal_encoding(loc, Q.shape[-1])
        # Q_ = self.apply_rope(Q,loc)
        # Q = Q + self.q_mlp(Q)
        # Q = self.knn_neighbor_augment(Q, loc, k=8, gamma=0.1)
        
        #-----------Check-------------#
        # print('Check Q:')
        # check_trunk_latent(Q)

        # Reduce tokens: [B,M,C_attn]
        # KVm = self.sample_points(KVm)

        # Cross-attention: query=Q, key/value=KVm
        attn_out, _ = self.attn(Q_, K_, KVm, need_weights=False)
        # attn_out = self.attn(Q, KVm, KVm, R)
        #-----------Check-------------#
        # print('Check A:')
        # check_trunk_latent(attn_out)
        gated_attn = self.alpha * attn_out
        # res_connection = drop_path(gated_attn, self.drop_path_rate, self.training)
        x = self.norm1(Q + gated_attn)
        # Transformer block
        # x = self.norm1(Q + attn_out)
        x = self.norm2(x + self.ffn(x))

        # g = self.gate(Q)
        # x = self.norm1(Q + g*attn_out)
        # x = self.norm2(x + self.ffn(x))
   
        # g = self.gate(Q)
        # x = token_scaling(Q + g*attn_out)
        # x = token_scaling(x + self.ffn(x))

        # Output latent per location
        Z = self.out(x)                  # [B,L,D_out]
        #-----------Check-------------#
        # print('Check Z:')
        # check_trunk_latent(Z)
        return Z


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth (DropPath) per sample.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class FiLM(nn.Module):
    """
    Produces (gamma, beta) from a conditioning vector and applies:
      h <- (1 + gamma) * h + beta
    Using (1+gamma) helps keep identity at init if gamma,beta start near 0.
    """
    def __init__(self, cond_dim: int, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * feat_dim)
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor):
        """
        h: [B, *, C]
        cond: [B, cond_dim]
        """
        gb = self.mlp(cond)                     # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1)       # [B,C], [B,C]
        # broadcast to h's shape
        while gamma.dim() < h.dim():
            gamma = gamma.unsqueeze(1)
            beta  = beta.unsqueeze(1)
        return (1.0 + gamma) * h + beta
    


