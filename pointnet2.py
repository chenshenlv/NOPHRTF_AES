import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import (
    PointNetSetAbstractionMsg,
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    PointNetEncoder,

)

class PointNet(nn.Module):
    def __init__(self, latent_size=512, global_feat=False, normal_channel=False):
        super().__init__()
        self.normal_channel = normal_channel
        channel = 6 if normal_channel else 3
        # encoder that returns only the global feature
        self.encoder = PointNetEncoder(global_feat=global_feat,
                                       feature_transform=True,
                                       channel=channel)
        input_dim = 1024 if global_feat else 1088
        # self.fc_latent = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     # nn.SiLU(),
        #     nn.Linear(512, latent_size)
        # )
        self.fc_latent = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, latent_size),
        )

    def forward(self, x):
        # x: [B, C, N]
        if not self.normal_channel:
            x = x[:,:3,:]
        global_feat, trans, trans_feat = self.encoder(x)   # [B, 1024]
        if global_feat.dim() == 3:
            global_feat = global_feat.transpose(1, 2).contiguous()
        z = self.fc_latent(global_feat)             # [B, latent_dim]
        if global_feat.dim() == 3:
            z = z.transpose(1, 2).contiguous()
        return z
    
class PointNetHybrid(nn.Module):
    def __init__(self, latent_size=512, with_point_feats=False, normal_channel=True):
        super().__init__()
        channel = 6 if normal_channel else 3
        # exactly as in part-seg: returns per-point features [B,1088,N] and trans_feat
        self.encoder = PointNetEncoder(global_feat=False,
                                       feature_transform=True,
                                       channel=channel)
        self.with_point_feats = with_point_feats

        # projection from the per-point feature map to a compact code
        # first reduce 1088 → 1024 with a 1×1 conv, then pool, then FC→latent_dim
        self.conv_reduce = nn.Conv1d(1088, 1024, 1)
        self.bn_reduce   = nn.BatchNorm1d(1024)
        self.fc_latent = nn.Sequential(
            nn.Linear(1024, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(),
        )

        if with_point_feats:
            # if you still want to use per-point features later
            self.point_proj = nn.Conv1d(1088, 256, 1)

    def forward(self, x):
        # x: [B, C, N]
        pointfeat,trans, trans_feat = self.encoder(x)
        # pointfeat: [B,1088,N]

        # 1) Reduce & pool to get global code
        x = F.relu(self.bn_reduce(self.conv_reduce(pointfeat)))  # → [B,1024,N]
        x = torch.max(x, 2)[0]                                   # → [B,1024]
        z = self.fc_latent(x)                                    # → [B,latent_dim]

        if self.with_point_feats:
            # 2) also return a compact per-point embedding if desired
            p = self.point_proj(pointfeat)                      # → [B,256,N]
            return z#, p, trans_feat

        return z#, trans_feat

class PointNetmsg(nn.Module):
    def __init__(self, latent_size=256, normal_channel=False):
        super().__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstractionMsg(
            512,
            [0.03, 0.06, 0.12],#[0.1, 0.2, 0.4],# [0.03, 0.06, 0.12],
            [16, 32, 64],# [16, 32, 128],
            in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128,
            [0.12, 0.24, 0.48],#[0.2, 0.4, 0.8], # [0.12, 0.24, 0.48],
            [32, 64, 128],
            320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(
            None, None, None,
            640 + 3,
            [256, 512, 1024],
            True
        )

        self.fc_latent = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_size)
        )


        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.fc2 = nn.Linear(512, latent_size)
        # self.bn2 = nn.BatchNorm1d(latent_size)

        

    def forward(self, xyz):
        """
        Returns:
            latent: [B, latent_dim]
            global_feat: [B, 1024] (pre-MLP global feature, if you want it)
        """
        B, _, _ = xyz.shape

        # if self.normal_channel:
        #     norm = xyz[:, 3:, :]
        #     xyz = xyz[:, :3, :]
        # else:
        #     norm = None
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            xyz = xyz[:, :3, :]
            norm = xyz
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        global_feat = l3_points.view(B, 1024)
        # latent = F.relu(self.bn1(self.fc1(global_feat)))
        # latent = F.relu(self.bn2(self.fc2(latent)))
        # latent = self.fc_latent(global_feat)
        return global_feat


class PointNetssg(nn.Module):
    def __init__(self, latent_size: int = 256):
        super(PointNetssg, self).__init__()
        # --- encoder: exactly your SA layers ---
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32, in_channel=6 + 3, mlp=[32, 32, 64], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=0.8, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False
        )

        # --- decoder / upsampling: exactly your FP layers ---
        self.fp4 = PointNetFeaturePropagation(in_channel=768, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=320, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128])

        # --- bottleneck conv before pooling ---
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

        # --- final projection to latent ---
        self.fc_latent = nn.Linear(128, latent_size)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [B, 3 + F, N]  (first 3 dims are XYZ; extra per‐point features OK)
        Returns:
            latent: [B, latent_size]
        """
        # split coords vs. features
        l0_xyz = xyz[:, :3, :]       # [B, 3, N]
        l0_points = xyz             # [B, 3+F, N]

        # set abstraction (down-sampling)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # feature propagation (up-sampling)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        # now l0_points is [B, 128, N]

        # bottleneck conv + activation + dropout
        x = F.relu(self.bn1(self.conv1(l0_points)))  # [B,128,N]
        x = self.drop1(x)

        # global max‐pool over N → [B,128]
        x, _ = torch.max(x, dim=2)

        # project to latent_size → [B, latent_size]
        latent = self.fc_latent(x)
        return latent


class PointNetmsg_sem(nn.Module):
    def __init__(self, latent_size: int = 256, normal_channel: bool = False):
        super(PointNetmsg_sem, self).__init__()

        self.normal_channel = normal_channel
        in_channel = 3 if normal_channel else 0   # feature dimension only

        # -------- Set Abstraction (Encoder) --------
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=in_channel,   # must be C-3
            mlp_list=[[16, 16, 32], [32, 32, 64]],
        )

        self.sa2 = PointNetSetAbstractionMsg(
            npoint=256,
            radius_list=[0.1, 0.2],
            nsample_list=[16, 32],
            in_channel=32 + 64,
            mlp_list=[[64, 64, 128], [64, 96, 128]],
        )

        self.sa3 = PointNetSetAbstractionMsg(
            npoint=64,
            radius_list=[0.2, 0.4],
            nsample_list=[16, 32],
            in_channel=128 + 128,
            mlp_list=[[128, 196, 256], [128, 196, 256]],
        )

        self.sa4 = PointNetSetAbstractionMsg(
            npoint=16,
            radius_list=[0.4, 0.8],
            nsample_list=[16, 32],
            in_channel=256 + 256,
            mlp_list=[[256, 256, 512], [256, 384, 512]],
        )

        # -------- Feature Propagation (Decoder) --------
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # -------- Per-point latent head --------
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

        self.conv_latent = nn.Conv1d(128, latent_size, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Input:
            xyz: [B, C, N]
        Output:
            per-point latent: [B, latent_size, N]
        """

        B, C, N = xyz.shape

        # ---------- Split xyz and features ----------
        l0_xyz = xyz[:, :3, :]  # always first 3 dims
        if self.normal_channel and C > 3:
            l0_points = xyz[:, 3:, :]
        else:
            l0_points = None

        # ---------- Set Abstraction ----------
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # ---------- Feature Propagation ----------
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # l0_points: [B,128,N]

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(x)

        z = self.conv_latent(x)  # [B, latent_size, N]

        return z