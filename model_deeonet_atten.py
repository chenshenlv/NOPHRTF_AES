import torch
from torch import nn
from FourierNet import *
from pointnet2 import *
from DGCNN import *
from FNN import LinearFNN as FNN
from Cross_atten import PointTokenCrossAttn


"""This mode using one trunk for the trunk input (x,y,z,f)
output is the mag of left and right"""



class DeepONet(nn.Module):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch:  a dict with keys "cnn" and "fnn".

        layer_sizes_trunk: A dict containing the configuration for the trunk net.

        activation: If a string, the same activation is used in both trunk and branch nets.
            If a dict, the branch net uses activation["branch"] and the trunk net uses activation["trunk"].
        num_outputs (integer): Number of outputs. For multiple outputs, set multi_output_strategy accordingly.
        multi_output_strategy (str or None): None.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        layer_sizes_decoder,
        activation,
        num_outputs=1,
        decoder_enable = True,
        film_on=("trunk","branch"), 
        cond_use=("scale",)
    ):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_trunk = get_activation(activation["trunk"])
            self.activation_branch = get_activation(activation["branch"])
            self.activation_decoder = get_activation(activation["decoder"])


        self.cond_use = set(cond_use)
        cond_dim = 0
        cond_dim += 1 if "scale" in self.cond_use else 0
        cond_dim += 3 if "src_l" in self.cond_use else 0
        self.decoder_enable = decoder_enable
        self.num_outputs = num_outputs
        self.branch_cnn = self.build_branch_cnn(layer_sizes_branch, "point")
        self.branch_merge = self.build_branch_merge(layer_sizes_branch)
        self.trunk = self.build_trunk(layer_sizes_trunk, "Fourier_FNN")
        self.cross_atten = self.build_cross_atten(film_on,cond_dim)
        self.decoder = self.build_decoder(layer_sizes_decoder, "fnn")
       

    def build_branch_cnn(self, branch_config, cnn_type):
        """Build the branch CNN.

        Args:
            branch_config (dict): config file for the network.
            cnn_type (str): simple cnn or unet

        """

        if cnn_type == "point":
            cnn_config = branch_config["point"]
            cnn = PointNet(latent_size=cnn_config["latent_size"])
        
        if cnn_type == "pointmsg":
            cnn_config = branch_config["point"]
            cnn = PointNetmsg_sem(latent_size=cnn_config["latent_size"])
        
        if cnn_type == "dgcnn":
            cnn_config = branch_config["point"]
            cnn = GraphPointCloudEncoder(
                k=20,
                d_point=512,
                d_global=1024,
                use_dynamic_graph=True,
                mode="point"
            )
        
        if cnn_type == "Fourier_UFNN":
            cnn_config = branch_config["point"]
            cnn = FourierUnetFnn(
                in_channels=3,
                out_channels=cnn_config["latent_size"],
                features=[256,128,64,32,16],
                activation=self.activation_trunk,
                fourier_mapping_size=32,
                fourier_scales=[0.3, 1, 1.2, 1.4],
                init_scale=1,
                learnable=True,
                normalization="None"
            )

        return cnn
    
    def build_branch_merge(self,branch_config):
        # input_size = branch_config["point"]["latent_size"]+1
        input_size = branch_config["point"]["latent_size"]+4
        output_size = branch_config["point"]["latent_size"]
        # fnn = FNN(layer_sizes=[input_size,input_size], activation=self.activation_branch,dropout=0.2,use_batchnorm=False)
        fnn = FNN(layer_sizes=[input_size,256,output_size], activation=self.activation_trunk)
        return fnn

    def build_trunk(self, trunk_config, trunk_type):
        
        if trunk_type == "Fourier_FNN":
            trunk_config = trunk_config["Fourier_FNN"]
            trunk_net = FourierFNN(
                input_dim=trunk_config["in_dim"],
                hidden_dims=trunk_config["hidden_dims"],
                activation=self.activation_trunk,
                mapping_size=trunk_config["mapping_size"],
                init_scale=trunk_config["init_scale"],
                learnable=trunk_config["learnable"],
                dropout=0.2
            )
        
        if trunk_type == "Fourier_UFNN":
            trunk_config = trunk_config["Fourier_UFNN"]
            trunk_net = FourierUnetFnn(
                in_channels=trunk_config["in_channels"],
                out_channels=trunk_config["out_channels"],
                features=trunk_config["features"],
                activation=self.activation_trunk,
                fourier_mapping_size=trunk_config["mapping_size"],
                fourier_scales=trunk_config["scales"],
                init_scale=trunk_config["init_scale"],
                learnable=trunk_config["learnable"],
                normalization="None"
            )
        return trunk_net

    def build_cross_atten(self,film_on,cond_dim):
        cross_atten = PointTokenCrossAttn(
            C_point=1024,
            C_trunk=256,   
            C_attn=144, # for rope
            D_out=64,
            n_heads=8,
            cond_dim=cond_dim,
            film_on=film_on,
            M=3000,
            dropout=0
        )
        return cross_atten

    def build_decoder(self, layer_sizes_decoder, decoder_type):
        if decoder_type == "fnn":
            decoder_config = layer_sizes_decoder["fnn"]
            decoder = FNN(
                layer_sizes=decoder_config, 
                activation=self.activation_decoder,
                last_activation=None,
                dropout= 0.2,
                use_batchnorm=False
                )
        return decoder

    @staticmethod
    def concatenate_outputs(ys):
        return torch.stack(ys, dim=3)

    def forward(self, inputs):
        """input: x_pt, x_loc, src_l, src_r, mu, scale
        x_pt: (B,6,N)
        x_loc: (B,L,3)
        src_l: (B, 3)
        src_r: (B, 3)
        mu: (B,1,3)
        scale: (B,1)"""
        x_pts = inputs[0]
        x_loc = inputs[1]
        src_l = inputs[2]
        src_r = inputs[3]
        mu = inputs[4]
        scale = inputs[5]
        
        n_pts = x_pts.shape[2]
        n_batch = x_pts.shape[0]
        n_eval = x_loc.shape[1]

        # noise_scale = 0.01
        # noise = torch.randn_like(x_loc) * noise_scale
        # x_loc = x_loc + noise

        # ------Conditioning vector (scale and/or src_l)------
        cond_parts = []
        if "scale" in self.cond_use:
            cond_parts.append(scale)         # [B,1]
        if "src_l" in self.cond_use:
            cond_parts.append(src_l)         # [B,3]
        cond = torch.cat(cond_parts, dim=-1) # [B, cond_dim]

        x_func = self.branch_cnn(x_pts)
        # print("Check branch lantent Var:")
        # check_latent(x_func)
        x_func_lantent = x_func
        scale_ = scale.unsqueeze(1).repeat(1,1,n_pts)  # (B,1) -> (B,1,N)
        # src_l_ = src_l.unsqueeze(-1).repeat(1,1,n_pts)
        # x_func = torch.concatenate([x_func.reshape(n_batch,-1,n_pts),scale_,src_l_],dim=1)
        # x_func = self.branch_merge(x_func.reshape(n_batch*n_pts,-1))
        # x_func = x_func.reshape(n_batch,-1,n_pts)

        x_loc_ = torch.reshape(x_loc,(-1,x_loc.shape[2]))
        x_trunk = self.trunk(x_loc_)
        # x_trunk = x_loc
        # print("\nCheck trunk lantent Var:")
        # check_latent(x_trunk)
        x_trunk_latent = x_trunk.reshape(n_batch, n_eval, -1)
        scale_ = scale.unsqueeze(1).repeat(1,n_eval,1)  # (B,1) -> (B,L,1)
        src_l_ = src_l.unsqueeze(1).repeat(1,n_eval,1)
        # x_trunk = torch.cat([x_trunk_latent,x_loc.reshape(n_batch, n_eval, -1),src_l_], dim=-1)
        x_trunk = torch.cat([x_trunk_latent,x_loc], dim=-1)
        # x_trunk = x_trunk.reshape(n_batch, n_eval, -1)+x_loc.reshape(n_batch, n_eval, -1)
        diff = x_loc.unsqueeze(2) - x_pts[:,:3,:].transpose(1,2).unsqueeze(1)
        R = torch.norm(diff, dim=-1)   # [B, L, N]
        x_latent = self.cross_atten(x_func, x_trunk.reshape(n_batch, n_eval, -1), x_pts[:,:3,:], cond)
        # x_latent = self.cross_atten(x_trunk.reshape(n_batch, n_eval, -1),x_func.reshape(n_batch,-1,n_pts), x_pts[:,:3,:], cond)
        # print(torch.norm(x_pts[0,:3,0]))
        # print(torch.norm(x_pts[0,3:,0] ))

        # using decoder
        if self.decoder_enable: 
            x = torch.reshape(x_latent, (x_latent.shape[0]* x_latent.shape[1], -1))
            x = self.decoder(x)
        if not self.decoder_enable:
            x = x_latent
          

        x = torch.reshape(x, (n_batch*n_eval, -1))
        return x,x_func_lantent,x_trunk_latent




    def num_trainable_parameters(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)