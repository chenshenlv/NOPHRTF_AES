import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (B, C, N)
    return idx: (B, N, k) neighbor indices (excluding self not guaranteed)
    Uses squared Euclidean distance in feature space.
    """
    # (B, N, N) pairwise distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a^T b
    B, C, N = x.shape
    xx = (x ** 2).sum(dim=1, keepdim=True)          # (B, 1, N)
    # inner: (B, N, N)
    inner = torch.matmul(x.transpose(2, 1), x)      # (B, N, N)
    dist = xx.transpose(2, 1) + xx - 2.0 * inner    # (B, N, N)
    # take k smallest distances
    idx = dist.topk(k=k, dim=-1, largest=False)[1]  # (B, N, k)
    return idx

def get_graph_feature(x: torch.Tensor, k: int, idx: torch.Tensor = None) -> torch.Tensor:
    """
    x: (B, C, N)
    idx: (B, N, k)
    returns edge features: (B, 2C, N, k) = [x_i, x_j - x_i]
    """
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k)

    # flatten for gather
    device = x.device
    idx_base = torch.arange(B, device=device).view(-1, 1, 1) * N  # (B,1,1)
    idx = (idx + idx_base).view(-1)                               # (B*N*k,)

    x_t = x.transpose(2, 1).contiguous()                          # (B, N, C)
    feat = x_t.view(B * N, C)[idx, :]                             # (B*N*k, C)
    feat = feat.view(B, N, k, C).permute(0, 3, 1, 2).contiguous()  # (B, C, N, k)

    x_i = x.unsqueeze(-1).expand(-1, -1, -1, k)                    # (B, C, N, k)
    edge = torch.cat((x_i, feat - x_i), dim=1)                     # (B, 2C, N, k)
    return edge

# --------- EdgeConv block ---------
class EdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        super().__init__()
        self.k = int(k)
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        edge = get_graph_feature(x, k=self.k)      # (B, 2C, N, k)
        h = self.mlp(edge)                         # (B, Cout, N, k)
        h = h.max(dim=-1)[0]                       # max over neighbors -> (B, Cout, N)
        return h

# --------- Point cloud graph encoder ---------
class GraphPointCloudEncoder(nn.Module):
    """
    Input:  X (B, 6, N)
    Output: per-point code H (B, d, N) and/or global code z (B, d_g)

    Switch behavior via mode:
      - mode="point": return H only
      - mode="global": return z only
      - mode="both": return (H, z)
    """
    def __init__(
        self,
        k: int = 20,
        d_point: int = 256,
        d_global: int = 256,
        use_dynamic_graph: bool = True,
        mode: str = "point"
    ):
        super().__init__()
        self.k = int(k)
        self.use_dynamic_graph = bool(use_dynamic_graph)
        self.mode = mode
        # EdgeConv stack (typical DGCNN widths)
        self.ec1 = EdgeConv(in_channels=6,   out_channels=64,  k=k)
        self.ec2 = EdgeConv(in_channels=64,  out_channels=64,  k=k)
        self.ec3 = EdgeConv(in_channels=64,  out_channels=128, k=k)
        self.ec4 = EdgeConv(in_channels=128, out_channels=256, k=k)

        # fuse multi-scale point features
        fuse_in = 64 + 64 + 128 + 256
        self.fuse = nn.Sequential(
            nn.Conv1d(fuse_in, d_point, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_point),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # global head: pool + MLP
        self.global_head = nn.Sequential(
            nn.Linear(d_point * 2, d_global, bias=False),  # max+mean concat
            nn.BatchNorm1d(d_global),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor,):
        """
        x: (B, 6, N)
        mode: "point" | "global" | "both"
        """
        if x.ndim != 3: #or x.shape[1] != 6:
            raise ValueError(f"Expected x of shape (B,6,N), got {tuple(x.shape)}")

        # If not using dynamic graph, you could compute kNN on xyz once and reuse.
        # Here we implement the common dynamic variant implicitly because get_graph_feature
        # uses the current x passed into each EdgeConv.
        h1 = self.ec1(x)          # (B, 64, N)
        h2 = self.ec2(h1)         # (B, 64, N)
        h3 = self.ec3(h2)         # (B, 128, N)
        h4 = self.ec4(h3)         # (B, 256, N)

        h_cat = torch.cat([h1, h2, h3, h4], dim=1)  # (B, 512, N)
        H = self.fuse(h_cat)                        # (B, d_point, N)

        if self.mode == "point":
            return H

        # global pooling: max + mean (often better than max alone)
        H_max = H.max(dim=-1)[0]          # (B, d_point)
        H_mean = H.mean(dim=-1)           # (B, d_point)
        z = self.global_head(torch.cat([H_max, H_mean], dim=1))  # (B, d_global)

        if self.mode == "global":
            return z
        if self.mode == "both":
            return H, z
        raise ValueError(f"Unknown mode={self.mode}. Use 'point', 'global', or 'both'.")