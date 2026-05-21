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

        self.alpha = nn.Parameter(torch.tensor([0.5])) 
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
        # # Pn = torch.zeros_like(Pn)
        # # Pn = P[:,None,:]
        loc = T[:,:,256:]
        loc_feats = T[:,:,:256]
        # loc_feats = torch.zeros_like(loc_feats)
        # Pn = T.transpose(1, 2).contiguous()
        # loc = P[:,:,256:]
        # loc_feats = P[:,:,:256]
        # loc_feats = torch.zeros_like(loc_feats)

        # print('Check Q before projection:')
        # check_trunk_latent(T)
        # if self.film_branch is not None:
        #     Pn = self.film_branch(Pn, cond)  # broadcast over N_pts
        # if self.film_trunk is not None:
        #     loc_feats = self.film_trunk(loc_feats, cond)     # broadcast over N_eval

        # Project into attention space
        KVm = self.p_proj(Pn)            # [B,N,C_attn]
        # K = KVm + self.get_sinusoidal_encoding(x_pts.transpose(1, 2).contiguous(), KVm.shape[-1])
        K_ = self.apply_rope(KVm,x_pts.transpose(1,2))
        Q = self.t_proj(loc_feats)               # [B,L,C_attn]
        # Q = Q + self.get_sinusoidal_encoding(loc, Q.shape[-1])
        Q_ = self.apply_rope(Q,loc)
        # Q = Q + self.q_mlp(Q)
        # Q = self.knn_neighbor_augment(Q, loc, k=8, gamma=0.1)
        

        #----- Swap Q and K --------
        # Q_, K_ = K_, Q_


        # --- TEMPERATURE TRICK ---
        # temp = 0.5  # T < 1.0 sharpens the attention
        # inv_sqrt_temp = 1.0 / math.sqrt(temp)
        
        # # Pre-scale Q and K before passing to MultiheadAttention
        # Q = Q * inv_sqrt_temp
        # K = K * inv_sqrt_temp
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
        res_connection = drop_path(gated_attn, self.drop_path_rate, self.training)
        x = self.norm1(Q + res_connection)
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


# class PointTokenCrossAttn(nn.Module):
#     def __init__(self, C_point: int, C_trunk: int, C_attn: int,
#                  D_out: int, n_heads: int = 8, M: int = 1024, dropout: float = 0.0,
#                 cond_dim: int = 1, film_hidden: int = 128, film_on=("",""),n_experts=4,):
#         super().__init__()
#         self.n_experts = n_experts
#         self.C_attn = C_attn

#         self.p_proj = nn.Linear(C_point, C_attn, bias=False)
#         self.t_proj = nn.Linear(C_trunk, C_attn, bias=False)

#         # Router: token-wise expert weights
#         self.router = nn.Sequential(
#             nn.Linear(C_attn, C_attn),
#             nn.GELU(),
#             nn.Linear(C_attn, n_experts)
#         )

#         # Expert cross-attention blocks
#         self.experts = nn.ModuleList([
#             nn.MultiheadAttention(
#                 embed_dim=C_attn,
#                 num_heads=n_heads,
#                 dropout=dropout,
#                 batch_first=True
#             )
#             for _ in range(n_experts)
#         ])

#         self.norm1 = nn.LayerNorm(C_attn)
#         self.ffn = nn.Sequential(
#             nn.Linear(C_attn, 4 * C_attn),
#             nn.GELU(),
#             nn.Linear(4 * C_attn, C_attn),
#         )
#         self.norm2 = nn.LayerNorm(C_attn)
#         self.out = nn.Linear(C_attn, D_out)

#     def forward(self, P: torch.Tensor, T: torch.Tensor, x_pts:torch.Tensor,  cond=None):
#         # P: [B,Cp,N] -> [B,N,Cp]
#         Pn = P.transpose(1, 2).contiguous()

#         # Example split
#         loc_feats = T[:, :, :256]

#         KVm = self.p_proj(Pn)        # [B,N,C]
#         Q = self.t_proj(loc_feats)   # [B,L,C]

#         # Router weights
#         logits = self.router(Q)                  # [B,L,E]
#         gates = F.softmax(logits, dim=-1)        # [B,L,E]

#         # Expert outputs
#         expert_outs = []
#         for expert in self.experts:
#             A_e, _ = expert(Q, KVm, KVm, need_weights=False)   # [B,L,C]
#             expert_outs.append(A_e)

#         # Stack: [B,L,E,C]
#         A = torch.stack(expert_outs, dim=2)

#         # Weighted sum over experts
#         A_mix = torch.sum(A * gates.unsqueeze(-1), dim=2)      # [B,L,C]

#         x = self.norm1(Q + A_mix)
#         x = self.norm2(x + self.ffn(x))
#         Z = self.out(x)
#         return Z

# def token_scaling(X:torch.Tensor,eps=1e-6):
#     # X: [B,L,C]
#     scale = torch.sqrt(X.pow(2).mean(dim=-1, keepdim=True) + eps)
#     return X / scale

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
    
class kernel_base_MultiAtten(nn.Module):
    def __init__(self, d_emb:int, num_heads:int, hidden:int=64, dropout=0.0):
        super().__init__()
        assert d_emb % num_heads == 0, "d_emb must be divisible by num_heads"
        self.d_emb = d_emb
        self.num_heads = num_heads
        self.d_head = d_emb // num_heads
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        # one projection per Q, K, V
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x:torch.Tensor):
        """
        x: [B, N, d_emb] -> [B, num_heads, N, d_head]
        """
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.d_head)
        x = x.permute(0, 2, 1, 3)   # [B, H, N, d_head]
        return x

    def _combine_heads(self, x:torch.Tensor):
        """
        x: [B, H, N, d_head] -> [B, N, d_emb]
        """
        B, H, L, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, N, H, d_head]
        x = x.view(B, L, H * Dh)                # [B, N, d_emb]
        return x

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, R:torch.Tensor,  mask=None):
        """
        R shape: B,L,N
        """
        B, L, D = Q.shape
        _, N, _ = K.shape
        assert D == self.d_emb

        Q = self._split_heads(Q)  # [B, H, L, d_head]
        K = self._split_heads(K)  # [B, H, N, d_head]
        V = self._split_heads(V)  # [B, H, N, d_head]


        # scaled dot-product attention
        # scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5) # scores: [B, H, L, N]
        psi_r = self.mlp(R.unsqueeze(-1)).squeeze(-1)  # [B, L, N]
        psi_r = psi_r.unsqueeze(1)  
        psi_r = psi_r.expand(-1, self.num_heads, -1, -1)  # [B, H, L, N]                 
        # scores = scores + psi_r

        # attn = torch.softmax(scores, dim=-1)         # [B, H, L, N]
        # attn = self.dropout(attn)
        # out = torch.matmul(attn, V)                  # [B, H, N, d_head]
        out = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=psi_r,
        dropout_p=self.dropout.p if self.training else 0.0,
        is_causal=False
        )

        out = self._combine_heads(out)               # [B, N, d_emb]
        return out


    


