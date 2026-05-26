import os,gc
# Hide all GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DataLoader_operator import Dataloader_augment,Dataloader_augment_loc_split,Preprocessor
from model_deeonet_atten import DeepONet
from utils import check_branch_latent,check_trunk_latent,lsd_loss_db, relative_l2_loss, relative_l1_loss
from ae_fnn import FNNAE
from scaler import *
from joblib import load as joblib_load
from joblib import dump as joblib_dump
from utils import grading_resample

from loss import LSD_l_mag_db
from contextlib import contextmanager

# import gc, os, torch

class ModelLowFTrainer():
    """
    Trainer for model residue all frequency architecture.
    """
    def __init__(self, arg_config, device=None):
        self.arg_config = arg_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.data_arg = self.arg_config.get("data","")
        self.model_save_path =self.arg_config.get("model save path","")
        self.model_name = self.arg_config.get("model name","")
        self.data_save_folder = self.arg_config.get("data save folder","")
        self.ae_path = self.arg_config.get("ae_path","")
        self.latent_dim = self.arg_config.get("latent_dim",32)
        self.encoder_hidden = self.arg_config.get("encoder_hidden",[128,64])
        self.decoder_hidden = self.arg_config.get("decoder_hidden",[64,1128])
        self.activation_ae = self.arg_config.get("activation_ae","silu")
        self.dropout = self.arg_config.get("dropout",0)
        self.l2_lambda = self.arg_config.get("l2_lambda",1e-5)
        self.hrtf_scaler_path = self.arg_config.get("hrtf scaler path","")
        self.n_resample = self.arg_config.get("head resampling num",5000)
        self.sigma = self.arg_config.get("sigma",0.3)
        self.temp_model_path = self.arg_config.get("temp model path","")
        self.result_path = self.arg_config.get("results path","")
        self.iterations = self.arg_config.get("iterations","")
        self.epochs = self.arg_config.get("epochs",100)
        self.batch_size = self.arg_config.get("batch size",32)
        self.learning_rate = self.arg_config.get("learning rate",1e-3)
        self.decay_step = self.arg_config.get("decay step",1)
        self.decay_rate = self.arg_config.get("decay_rate",self.learning_rate/5)
        self.decoder_enable = self.arg_config.get("decoder_enable", True)
        self.point_size = self.arg_config.get("point_size", "")
        self.op_net = None
        self.ae_net = None
        self.scaler = None
        self.resume = False

    def _get_data(self,save=True):
        X_train = joblib_load(self.data_save_folder+"X_train.sav")
        X_val = joblib_load(self.data_save_folder+"X_val.sav")
        # X_val = joblib_load(self.data_save_folder+"X_val.sav")
        y_train = joblib_load(self.data_save_folder+"y_train.sav")
        y_val = joblib_load(self.data_save_folder+"y_val.sav")
        # y_val = joblib_load(self.data_save_folder+"y_val.sav")
        return X_train, y_train, X_val, y_val


    def _build_data(self, X_train, y_train, X_val, y_val):
        return Dataloader_augment(X_train, y_train, X_val, y_val,n_resample=self.n_resample,sigma=self.sigma)
       
        
    def _build_test_data(self, x_name, y_name):
        x_test = joblib_load(self.data_save_folder+x_name)
        y_test = joblib_load(self.data_save_folder+y_name)
        return x_test, y_test

    def _build_predict_data(self,):
        # x_test = joblib_load(self.data_save_folder+"X_test.sav")
        # y_test = joblib_load(self.data_save_folder+"y_test.sav")
        x_test = joblib_load(self.data_save_folder+"X_train.sav")
        y_test = joblib_load(self.data_save_folder+"y_train.sav")
        # np.save(self.model_save_path+"/result/gt.npy", y_test)
        return x_test, y_test

    def _load_AE(self,input_dim):
        ckpt_path = os.path.join(self.ae_path, "ae_fnn_aug.pt")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.ae_net = FNNAE(
                input_dim=input_dim,
                latent_dim=self.latent_dim,
                encoder_hidden=self.encoder_hidden,
                decoder_hidden=self.decoder_hidden,
                activation=self.activation_ae,
                dropout=self.dropout,
                ).to(self.device)
            self.ae_net.load_state_dict(ckpt["model_state"])
        if self.ae_net is None:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    def _build_net(self):
        p = 256
        # if not self.decoder_enable:
        #     trunk_p = int(p*4)
        # else: 
        #     trunk_p = p
        trunk_p = p

        branch_config = {
            "point": {
                "latent_size":1024,
                }
            }
        trunk_config = {
            "Fourier_UFNN":{
                "in_channels": 3,
                "out_channels": trunk_p,
                "features": [256,128,64,32],
                "mapping_size": 64,
                "scales": [0.3, 1, 1.2, 1.4], 
                "init_scale": 1.0,
                "learnable": True

            },
            "Fourier_FNN":{
                "in_dim": 3,
                "mapping_size": 64,
                "hidden_dims": [64,128,trunk_p],
                "init_scale": 1.0,
                "learnable":False,
            },
        }
        decoder_config = {
            "fnn":[p,64,32],
            }
        activation = {
            "branch": "gelu", 
            "trunk": "wavelet",
            "decoder": "relu"
            }
        # load low_fidality model
        self.op_net = DeepONet(
            layer_sizes_branch=branch_config,
            layer_sizes_trunk=trunk_config,
            layer_sizes_decoder=decoder_config,
            activation = activation,
            num_outputs=1,
            decoder_enable= self.decoder_enable,       
        ).to(device=self.device)
        print(f'trainable paras: {self.op_net.num_trainable_parameters()}')

    def resolve(self):
        checkpoint_path = os.path.join(self.model_save_path, self.temp_model_path)
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            return checkpoint   
        else:
            print(f'checkpoint_path: {checkpoint_path} does not exist')
            # # Load the states
            # self.op_net.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
            # # Start from the next epoch
            # start_epoch = checkpoint['epoch'] + 1
            # print(f"Resuming from epoch {start_epoch}")


    def train(self):
        """
        Train models
        """        
        # preprocess data 
        if self.data_save_folder is not None:
            X_train, y_train, X_val, y_val = self._get_data()
        n_ex = 900
        X_train = (X_train[0],X_train[1][0:n_ex],X_train[2],X_train[3][0:n_ex],X_train[4][0:n_ex])
        y_train = y_train[0:n_ex,:,:,:]
        n_train = y_train.shape[0]
        n_test = y_val.shape[0]
        n_freq = X_train[0].shape[0]
        n_loc = X_train[2].shape[0]
        freq_min = 100.0
        freq_max = 22000.0
        x_fre = ((X_train[0]+1)*100 - freq_min) / (freq_max - freq_min)
        
        
        assert np.isfinite(y_train).all(), f"hrtf array contains {np.count_nonzero(~np.isfinite(y_train[0]))} invalid entries"
        data = self._build_data(X_train, y_train, X_val, y_val)
        self._build_net()
        self._load_AE(n_freq)
        self.ae_net.eval()
        for p in self.ae_net.parameters():
            p.requires_grad_(False)

        
        # optimizer = torch.optim.AdamW(self.op_net.parameters(), lr=self.learning_rate,weight_decay=self.l2_lambda,)
        optimizer = torch.optim.RAdam(self.op_net.parameters(), lr=self.learning_rate,weight_decay=self.l2_lambda,)
        
        
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=50,
        #     gamma=0.5
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        if self.resume:
            checkpoint = self.resolve()
            self.op_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        best_val = float("inf")
        total_iterations = (n_train*n_loc*2)//(self.batch_size[0]*self.batch_size[1])
        alpha_latent = 1.0
        beta_recon   = 1.0
        for epoch in range(1, self.epochs + 1):
            self.op_net.train()
            train_loss = 0.0
            train_lat = 0.0
            train_rec = 0.0
            for _ in range (total_iterations):
                x_train_b, y_train_b = data.train_next_batch(self.batch_size)
                inputs = tuple(
                        map(lambda x: torch.as_tensor(x,device=self.device), x_train_b)
                    )
                y_train_b = torch.as_tensor(y_train_b,device=self.device, dtype=torch.float32)
                # target latent (fixed); no grad needed
                with torch.no_grad():
                    y_rd = self.ae_net.encoder(y_train_b)

                # predicted latent
                y_rd_hat,branch_latent,trunk_latent = self.op_net(inputs)

                # optional: decoded reconstruction from predicted latent (keep grad to op_net!)
                y_rec_hat = self.ae_net.decoder(y_rd_hat)

                # losses
                # loss_latent = F.mse_loss(y_rd_hat, y_rd, reduction="sum")
                loss_latent = relative_l1_loss(y_rd, y_rd_hat)
                # loss_recon  = F.mse_loss(y_rec_hat, y_train_b, reduction="mean")
                loss_recon = lsd_loss_db(y_rec_hat, y_train_b)
                loss = alpha_latent * loss_latent + beta_recon * loss_recon
                # loss =  loss_latent


                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.op_net.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_lat  += loss_latent.item()
                train_rec  += loss_recon.item()

            train_loss /= total_iterations
            train_lat  /= total_iterations
            train_rec  /= total_iterations
            #     train_loss += loss.item() * (self.batch_size[0]*self.batch_size[1])
            # train_loss /= (n_train*n_loc)
            # scheduler.step()

            self.op_net.eval()
            val_loss = 0.0
            val_latent = 0.0
            branch_latent = None
            trunk_latent = None
            with torch.no_grad():
                x_test, y_test = data.test()
                inputs = tuple(
                        map(lambda x: torch.as_tensor(x,device=self.device), x_test)
                    )
                y_test = torch.as_tensor(y_test,device=self.device) #gt HRTF
                y_rd_hat,branch_latent,trunk_latent = self.op_net(inputs) # predicted latent
                y_rec = self.ae_net.decoder(y_rd_hat) # predicted HRTF
                # reconstruction loss
                # loss = F.mse_loss(y_rec, y_test, reduction="mean")
                loss = lsd_loss_db(y_rec, y_test) 
                val_loss += loss.item()
                # latent loss
                y_rd = self.ae_net.encoder(y_test) # gt latent
                # loss_latent = F.mse_loss(y_rd_hat, y_rd, reduction="mean")
                loss_latent = relative_l1_loss(y_rd, y_rd_hat)
                val_latent += loss_latent.item()
                scheduler.step(val_loss)
                # val_loss /= total_iterations
                # val_latent /=total_iterations

            if val_loss < best_val:
                print(f'Validation loss improve from {best_val:.6f} to {val_loss:.6f}')
                best_val = val_loss
                os.makedirs(self.model_save_path, exist_ok=True)
                ckpt = {
                    "model_state_dict": self.op_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
               
                torch.save(ckpt, os.path.join(self.model_save_path, self.model_name))

            print(
                f"Epoch {epoch:03d} | train={train_loss:.6f},lat={train_lat:.6f}, rec={train_rec:.6f}  \
                | val_latent={val_latent:.6f}, val_recon={val_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.6f} "
            )
            if epoch % 10 ==0:
                print('Check Branch Latent:')
                check_branch_latent(branch_latent)
                print('\nCheck Trunk Latent:')
                check_trunk_latent(trunk_latent)

    
    def test(self, model_name, result_folder, x_name, y_name ):
        x_test, y_test = self._build_test_data(x_name, y_name)
        preprocessor = Preprocessor()
        x_fre,x_pt,x_cent, x_eval,x_src_l,x_src_r = preprocessor.preprocess_inputs(x_test,input_augment=False)
        n_sub = x_pt.shape[0]
        n_eval_loc = x_eval.shape[0] ## x[3] has 2*n_loc
        n_freq = x_fre.shape[0]
        y_test = preprocessor.preprocess_outputs(x_eval, y_test, input_augment= False)
       

        pt_resample = np.zeros((n_sub,6,self.n_resample),dtype=np.float32)
        for i in range(n_sub):
            pt_i = x_pt[i]
            center_i = x_cent[i]
            down = grading_resample(points=pt_i, centers=center_i, M=self.n_resample, sigma=self.sigma,side="both",)
            # down = uniform_resample_torch(pt_i,5000)
            pt_resample[i] = down.T

        xyz = pt_resample[:,:3,:]
        mu = xyz.mean(axis=2, keepdims=True)
        xyz_c = xyz - mu
        scale = np.linalg.norm(xyz_c, axis=1).max(axis=1, keepdims=True)
        scale = scale[:, :, None].astype(np.float32)   # (B,1,1)
        xyz_n = xyz_c / scale
        # xyz_n = xyz_c / 1.5
        resample = np.concatenate([xyz_n, pt_resample[:, 3:, :]], axis=1)
        
        x_eval = np.broadcast_to(x_eval[None, :, :], (n_sub, n_eval_loc, 3)).copy() #(B,n_loc,3)
        x_eval = (x_eval - mu.transpose(0, 2, 1)) / scale
        # x_eval = (x_eval - mu.transpose(0, 2, 1)) / 1.5
        mu_vec = mu[:, :, 0]                                       # (B,3)
        scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
        src_l = (x_src_l - mu_vec) / scale_vec
        src_r = (x_src_r - mu_vec) / scale_vec


        ## test scalar
        scaler = preprocessor.get_scaler(None, None, init_hrtf_scaler=False)
        y_test_norm = preprocessor.scale_outputs(scaler, y_test)
        y_test_rever = scaler.inverse_hrtf(y_test_norm,n_sub=n_sub,n_loc=n_eval_loc,n_freq=220,n_dim=1)
        y_test_rever_h = y_test_rever.cpu().numpy()
        error = np.mean(np.abs(y_test-y_test_rever_h))
        print(f'The error from scalar is: {error}')
        
        assert n_freq>1, "number of frequencies smaller than 1"
        self._build_net()
        ckpt_path = os.path.join(self.model_save_path, model_name)
        print(f'Load model parameters from: {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location=torch.device(self.device), weights_only=True)
        self.op_net.load_state_dict(checkpoint["model_state_dict"])
        for param in self.op_net.parameters():
            param.requires_grad = False
        self.op_net.eval()
        self._load_AE(n_freq)
        for param in self.ae_net.parameters():
            param.requires_grad = False
        self.ae_net.eval()


        pt = torch.as_tensor(resample,device=self.device)         # keep on CPU for slicing → move per batch
        x_eval = torch.as_tensor(x_eval,device=self.device)
        x_source_l = torch.as_tensor(src_l,device=self.device)
        x_source_r = torch.as_tensor(src_r,device=self.device)
        mu = torch.as_tensor(mu.transpose(0, 2, 1),device=self.device)
        scale = torch.as_tensor(np.reshape(scale,(n_sub,1)),device=self.device)
        N = n_sub
        # N = 1

        outputs = []
        branch_latent = []
        trunk_latent = []
        start = time.time()
        batch_size = 10
        ctx = torch.inference_mode()
        with ctx:
            for s in range(0, N, batch_size):
                e = min(s + batch_size, N)

                # move the big slice only
                pt_b = pt[s:e]
                x_eval_b = x_eval[s:e]
                x_source_l_b = x_source_l[s:e]
                x_source_r_b = x_source_r[s:e]
                mu_b = mu[s:e]
                scale_b= scale[s:e]

                inputs_b = (pt_b,x_eval_b,x_source_l_b,x_source_r_b,mu_b,scale_b)

                try:
                    y_rd_hat_b,branch_latent_b,trunk_latent_b = self.op_net(*inputs_b)
                    y_b = self.ae_net.decode(y_rd_hat_b)

                except TypeError:
                    y_rd_hat_b,branch_latent_b,trunk_latent_b = self.op_net(inputs_b)
                    y_b = self.ae_net.decode(y_rd_hat_b)

                outputs.append(y_b.detach().to("cpu"))
                branch_latent.append(branch_latent_b.detach().to("cpu"))
                trunk_latent.append(trunk_latent_b.detach().to("cpu"))


            del y_b, pt_b
        total_time = time.time() - start
        # Concatenate along the batch dimension (assumes batch is dim 0)
        y_pred_t = torch.cat(outputs, dim=0)
        branch_latent = torch.cat(branch_latent,dim=0)
        trunk_latent = torch.cat(trunk_latent,dim=0)
        y_pred_t = y_pred_t.reshape(N,n_eval_loc,n_freq,1)
        print(f"Prediction time: {total_time:.2f}s  |  y_pred shape={tuple(y_pred_t.shape)}")
        # if n_sub > 100:
        #     n_sub_ = 100
        #     y_pred_t = y_pred_t[:n_sub_, :]
        #     y_test = y_test[:n_sub_,:]

        # reshape by every feature scaler
        y_pred_t = scaler.inverse_hrtf(y_pred_t,n_sub=n_sub, n_loc=n_eval_loc,n_freq=n_freq,n_dim=1)
        # y_test = scaler.normalize_hrtf(y_test).reshape(n_sub,-1)
        y_pred = y_pred_t.cpu().numpy()
        y_pred = np.reshape(y_pred,(-1,n_eval_loc,n_freq,1))
        y_test = self.ae_net(torch.as_tensor(y_test_norm,dtype=torch.float32,device='cuda').reshape(-1,220))
        y_test = scaler.inverse_hrtf(y_test_norm,n_sub=n_sub,n_loc=n_eval_loc,n_freq=220,n_dim=1)
        y_test = y_test.cpu().numpy()
        y_test = np.reshape(y_test,(-1,n_eval_loc,n_freq,1))

        # y_pred = y_pred_t.cpu().numpy()
        # y_pred = np.reshape(y_pred,(-1,n_eval_loc,n_freq,1))
        # y_test,_ = self.ae_net(torch.as_tensor(y_test_norm,dtype=torch.float32,device='cuda').reshape(-1,220))
        # y_test = y_test.cpu().numpy()
        # y_test = np.reshape(y_test,(-1,n_eval_loc,n_freq,1))
        

   
        # calculate the lsd
        LSD_loss = LSD_l_mag_db(y_pred[...,0],y_test[...,0])
        print(f'The LSD deviation from Test is: {LSD_loss}')
        # Analyze the latent
        # print('Check Branch Latent:')
        # check_branch_latent(branch_latent)
        # print('\nCheck Trunk Latent:')
        # check_trunk_latent(trunk_latent)
        
        os.makedirs(self.result_path, exist_ok=True)
        np.save(os.path.join(self.result_path, result_folder, "gt.npy"), y_test)
        np.save(os.path.join(self.result_path, result_folder, "predicts.npy"), y_pred)
        torch.cuda.synchronize()

    

@contextmanager
def trainer_ctx(cfg):
    t = ModelLowFTrainer(cfg)
    try:
        yield t
    finally:
        t.close()
        del t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_train(cfg, use_saved_scalar, resume):
    trainer = None
    try:
        trainer = ModelLowFTrainer(cfg)
        trainer.train()
    finally:
        del trainer
        gc.collect()        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def run_test(cfg, model_name, result_folder, x_name, y_name):
    trainer = None
    try:
        trainer = ModelLowFTrainer(cfg)
        trainer.test(model_name,result_folder,x_name,y_name)
    finally:
        del trainer
        gc.collect()        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def run_prediction(cfg, model_name):
    trainer = None
    try:
        trainer = ModelLowFTrainer(cfg)
        trainer.prediction(model_name)
    finally:
        del trainer
        gc.collect()        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

if __name__ == "__main__":
    # warm-up
    # torch.cuda.init()
    # torch.empty(1, device="cuda").sin_()
    lr = 1e-3
    # pt_size=["15k","10k","5k","2k","1k"]
    # pt_size=["2k","1k"]
    # pt_size=["10k"]
    # pt_size=["1k"]
    # pt_size=["1k"]
    pt_size=["5k"]


    train = False
    test = True
    prediction = False
    resume_train = False
    use_saved_scalar = False
    decoder_enable = False
    database = "AXD"
    datatype = "msr_44_ff"
    for pt in pt_size:
        if pt == "5k":
            sampling_num = 5000
        elif pt == "2k":
            sampling_num = 2000
        elif pt == "1k":
            sampling_num = 1000
        if pt == "05k":
            sampling_num = 2000
        arg_config ={
            "data":{
                # "database": "UHM_interp_train",   # "UHM_dense_noitd_train" "AXD"
                "database": database,   #database name does not matter if load data directly
                "sub": "", #small_ or ""
                "type": datatype  # simu / msr /msr-ffmp/"msr_44_noitd"/"msr_44_ff"
            },
        "model save path": "saved_model",
        "model name": "model_point_single"+ database +"_"+ datatype+ "_" + pt+ "_pe.pt",
        # "data save folder": "./model_point/data/"+ database +"/dis_weighted/mag/raw/"+datatype+"/" + pt +"/",
        "data save folder": "data/AES/5sets/",
        "ae_path":"saved_model",
        "latent_dim":64,
        "encoder_hidden":[512,128],
        "decoder_hidden":[128,512],
        "activation_ae":"silu",
        "dropout":0,
        "l2_lambda":1e-5,
    
        "hrtf scaler path": "./Scaler/hrtf_scaler_temp", 
        "head resampling num": sampling_num,
        "sigma":0.6,
        "temp model path": "model_point_single"+ database +"_"+ datatype+ "_" + pt +"_1.pt",
        "results path": "result/"+ database +"/AES/"+pt+"_pe"+"/",
        "iterations": 0,
        "epochs": 30,
        "batch size": [16,128],#default is [32,330] for UHM
        "learning rate": lr,
        "decay step": 1,
        "decay_rate": lr/5,
        "point_size": pt,
        "decoder_enable": decoder_enable,
        }
        
        if train:
            run_train(arg_config,use_saved_scalar,resume_train)

        if test:
            # model_name = "model_point_dense_" + pt + ".pt"
            model_name = arg_config["model name"]
            run_test(arg_config,model_name,"Test_1","X_test_1.sav","y_test_1.sav")
            run_test(arg_config,model_name,"Test_2","X_test_2.sav","y_test_2.sav")
            run_test(arg_config,model_name,"Test_3","X_test_3.sav","y_test_3.sav")
            run_test(arg_config,model_name,"Test_4","X_test_4.sav","y_test_4.sav")





