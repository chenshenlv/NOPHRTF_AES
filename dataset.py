import numpy as np
import os
from scipy.io import loadmat
from random import randint
from torch import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import load as joblib_load
from joblib import dump as joblib_dump
import argparse

class HRTFDataset(Dataset):
    def __init__(self, arg, val=False):
        super(HRTFDataset,self).__init__()
        # Load HRTF Data
        self.arg = arg
        database = self.arg["database"]
        database_type = self.arg["type"]
        self.val = val
        HRTF = loadmat("/app/Dataset/"+database+"/Dataset"+database+"_"+database_type+"HRTF.mat")["Dataset"] 
        self.freq = HRTF['frequency'][0,0].flatten()            # 1*220
        self.source_l = HRTF['source_left'][0,0]                # n_sub * 3
        self.source_r = HRTF['source_right'][0,0]               # n_sub * 3
        self.evaluation_loc = HRTF['evaluation_loc'][0,0]       # n_eval_loc * 3
        self.HRTF_l = HRTF['HRTF_left'][0,0]                    # (n_sub * n_eval_loc) * n_freq
        self.HRTF_r = HRTF['HRTF_right'][0,0]                  # (n_sub * n_eval_loc) * n_freq
        self.source_mag_left = HRTF['source_mag_left'][0,0]                # n_sub * n_eval_loc* n_freq
        self.source_mag_right = HRTF['source_mag_right'][0,0]              # n_sub * n_eval_loc* n_freq
        if hasattr(HRTF,'HRIR_left'):
            self.HRIR_l = HRTF['HRIR_left'][0,0]                    # (n_sub * n_eval_loc) * n_time
            self.HRIR_r = HRTF['HRIR_right'][0,0]                  # (n_sub * n_eval_loc) * n_time
            self.timesteps = np.transpose(HRTF['timesteps'][0,0])                      # 256*1
            self.num_time = self.timesteps.shape[0]

        # Load coefficients
        self.SH_coef = np.load("/app/Dataset/"+database+"/SH_coef.npy")      # n_sub * n_coef * 3
        self.SCH_coef = np.load("/app/Dataset/"+database+"/SCH_coef.npy")    # (n_sub * 2) * n_coef * 3 [left, right]
        if self.SH_coef.ndim > 3:
            self.SH_coef = np.squeeze(self.SH_coef)

        self.num_sub = len(self.source_l)
        self.num_eval_loc = len(self.evaluation_loc)
        self.num_freq = len(self.freq)
        self.num_side = 2
        

        # Normalize HRTF
        self.HRTF_l = self.min_max_normalize(self.HRTF_l)
        self.HRTF_r = self.min_max_normalize(self.HRTF_r)
        self.source_mag_left = self.min_max_normalize(self.source_mag_left)
        self.source_mag_right = self.min_max_normalize(self.source_mag_right)
        self.SCH_coef = self.normalize_sch(self.SCH_coef)
        self.SH_coef = self.normalize_sch(self.SH_coef)

        # split train and val
        if not self.val:
            self.split_idx = np.random.choice(range(self.num_sub), 10, replace=False)
            print(f'split indices: {self.split_idx}')
        if self.val: 
            self.split_idx = np.array([0,1,2])
            print(f'split indices: {self.split_idx}')
        self.source_l_train = np.delete(self.source_l,self.split_idx, axis=0)
        self.source_r_train = np.delete(self.source_r,self.split_idx, axis=0)
        self.SH_coef_train = np.delete(self.SH_coef, self.split_idx, axis=0)
        # rows_to_remove_sch = np.ravel([[idx * 2, idx * 2 + 1] for idx in self.split_idx])
        rows_to_remove_sch = self.split_idx
        self.SCH_coef_train = np.delete(self.SCH_coef,rows_to_remove_sch, axis=0)
        row_to_remove_hrtf = np.ravel([list(range(idx * self.num_eval_loc, (idx + 1) * self.num_eval_loc)) for idx in self.split_idx])
        self.HRTF_l_train = np.delete(self.HRTF_l,row_to_remove_hrtf, axis=0)
        self.HRTF_r_train = np.delete(self.HRTF_r,row_to_remove_hrtf, axis=0)
        if hasattr(HRTF,'HRIR_left'):
            self.HRIR_l_train = np.delete(self.HRIR_l,row_to_remove_hrtf, axis=0)
            self.HRIR_r_train = np.delete(self.HRIR_r,row_to_remove_hrtf, axis=0)
        self.source_mag_left_train = np.delete(self.source_mag_left,row_to_remove_hrtf, axis=0)
        self.source_mag_right_train = np.delete(self.source_mag_right,row_to_remove_hrtf, axis=0)

        self.source_l_val = self.source_l[self.split_idx,:]
        self.source_r_val = self.source_r[self.split_idx,:]
        self.SH_coef_val = self.SH_coef[self.split_idx,:,:]
        
        # self.SCH_coef_val = self.SCH_coef[rows_to_remove_sch,:,:]
        self.SCH_coef_val = self.SCH_coef[rows_to_remove_sch,:,:,:]
        self.HRTF_l_val = self.HRTF_l[row_to_remove_hrtf,:]
        self.HRTF_r_val = self.HRTF_r[row_to_remove_hrtf,:] 
        if hasattr(HRTF,'HRIR_left'):
            self.HRIR_l_val = self.HRIR_l[row_to_remove_hrtf,:]
            self.HRIR_r_val = self.HRIR_r[row_to_remove_hrtf,:] 
        self.source_mag_left_val = self.source_mag_left[row_to_remove_hrtf,:]
        self.source_mag_right_val = self.source_mag_right[row_to_remove_hrtf,:]

    
    def __len__(self):
        if (self.val):
            return len(self.split_idx) * self.num_eval_loc * self.num_freq * self.num_side
        else:
            return (self.num_sub-len(self.split_idx)) * self.num_eval_loc * self.num_freq * self.num_side
        
    def __getitem__(self, index):
        if (self.val):
            lr_index = index % 2 # 0: left, 1: right
            xyz_index = (index // 2) % self.num_eval_loc
            freq_index = (index // (self.num_eval_loc * 2)) % self.num_freq
            sub_index = index // (self.num_freq * self.num_eval_loc * 2)
            freq_item = self.freq[freq_index]
            eval_loc_item = self.evaluation_loc[xyz_index]
            SH_item = self.SH_coef_val[sub_index]
            if lr_index == 0:
                source_loc_item = self.source_l_val[sub_index]
                # SCH_item = self.SCH_coef_val[sub_index * 2]
                SCH_item = self.SCH_coef_val[sub_index, 0]
                HRTF_idx = sub_index*self.num_eval_loc+xyz_index
                HRTF_item = self.HRTF_l_val[sub_index*self.num_eval_loc+xyz_index, freq_index]
                source_mag_item = self.source_mag_left_val[sub_index*self.num_eval_loc+xyz_index, freq_index]

            if lr_index == 1:
                source_loc_item = self.source_r_val[sub_index]
                # SCH_item = self.SCH_coef_val[sub_index * 2 + 1]
                SCH_item = self.SCH_coef_val[sub_index, 1]
                HRTF_idx = sub_index*self.num_eval_loc+xyz_index
                HRTF_item = self.HRTF_r_val[sub_index*self.num_eval_loc+xyz_index, freq_index]
                source_mag_item = self.source_mag_right_val[sub_index*self.num_eval_loc+xyz_index, freq_index]

        else:
            lr_index = index % 2 # 0: left, 1: right
            xyz_index = (index // 2) % self.num_eval_loc
            freq_index = (index // (self.num_eval_loc * 2)) % self.num_freq
            sub_index = index // (self.num_freq * self.num_eval_loc * 2)
            freq_item = self.freq[freq_index]
            eval_loc_item = self.evaluation_loc[xyz_index]
            SH_item = self.SH_coef_train[sub_index]
            if lr_index == 0:
                source_loc_item = self.source_l_train[sub_index]
                # SCH_item = self.SCH_coef_val[sub_index * 2]
                SCH_item = self.SCH_coef_val[sub_index, 0]
                HRTF_idx = sub_index*self.num_eval_loc+xyz_index
                HRTF_item = self.HRTF_l_train[sub_index*self.num_eval_loc+xyz_index, freq_index]
                source_mag_item = self.source_mag_left_train[sub_index*self.num_eval_loc+xyz_index, freq_index]
            if lr_index == 1:
                source_loc_item = self.source_r_train[sub_index]
                # SCH_item = self.SCH_coef_val[sub_index * 2 + 1]
                SCH_item = self.SCH_coef_val[sub_index, 1]
                HRTF_idx = sub_index*self.num_eval_loc+xyz_index
                HRTF_item = self.HRTF_r_train[sub_index*self.num_eval_loc+xyz_index, freq_index]
                source_mag_item = self.source_mag_right_train[sub_index*self.num_eval_loc+xyz_index, freq_index]
                
        return sub_index,HRTF_idx,freq_item, source_loc_item, eval_loc_item, SH_item, SCH_item, HRTF_item, source_mag_item
    
    def min_max_normalize(self, HRTF):
        r"""normalize the real and image part of HRTF at each frequency for all subjects across all positions, such that for a given frequency, the HRTF of all subjects are normalized between 0 and 1.
        Args:
            HRTF (np.array): HRTF data of shape (n_sub * n_eval_loc) * n_freq
            Returns: normalized HRTF data: HRTF data of shape (n_sub * n_eval_loc) * n_freq *2 (real:0 and image:1 part)"""
        
        HRTF_real = HRTF.real
        HRTF_imag = HRTF.imag
        # scaler = MinMaxScaler(feature_range=(-1,1))
        # scaler_real = scaler.fit(HRTF_real)
        # scaler_imag = scaler.fit(HRTF_imag)
        # HRTF_real = scaler_real.transform(HRTF_real)
        # HRTF_imag = scaler_imag.transform(HRTF_imag)
        # for i in range(self.num_freq):
        #     scaler = MinMaxScaler(feature_range=(-1,1))
        #     HRTF_real[:,i] = scaler.fit_transform(HRTF_real[:,i].reshape(-1,1)).flatten()
        #     HRTF_imag[:,i] = scaler.fit_transform(HRTF_imag[:,i].reshape(-1,1)).flatten()
        HRTF_reshaped = np.stack((HRTF_real, HRTF_imag), axis=-1)
        # for i in range(0, HRTF_real.shape[0]):
        #     for j in range(0,HRTF_real.shape[1]):
        #         HRTF_reshaped[i,j,0] = HRTF_real[i,j]
        #         HRTF_reshaped[i,j,1] = HRTF_imag[i,j]

        # HRTF_reshaped = np.concatenate((HRTF_real,HRTF_imag),axis=1)
        return HRTF_reshaped
    def normalize_sch(self, sch: np.array):
        
        scaler = MinMaxScaler(feature_range=(-1,1))
        if sch.ndim > 3:
            temp = np.reshape(sch,(sch.shape[0]*sch.shape[1]*sch.shape[2],sch.shape[3]))
            scaler.fit(temp)
            sch = scaler.transform(temp).reshape([sch.shape[0],sch.shape[1],sch.shape[2],sch.shape[3]])
        if sch.ndim ==3:
            temp = np.reshape(sch,(sch.shape[0]*sch.shape[1],sch.shape[2]))
            scaler.fit(temp)
            sch = scaler.transform(temp).reshape([sch.shape[0],sch.shape[1],sch.shape[2]])
        return sch


class HRTFDataset2(Dataset):
    """ 1st, extract each data from mat.\\
        2nd, pack each data into dataset(X,Y) for all frequenice"""
    
    def __init__(self, arg,):
        import h5py
        super(HRTFDataset2,self).__init__()
        # Load HRTF Data
        self.arg = arg
        database = self.arg["database"]
        database_type = self.arg["type"]
        database_sub = self.arg["sub"]
        point_size = self.arg.get("points", "1k")
        point_format = self.arg.get("point_format", "unique")  # "unique" or "repeat"
      
        filename = "/app/Dataset/"+database+"/Dataset"+database+"_"+database_sub+database_type+"HRTF.mat"
        print(f"Loading Raw HRTF data from {filename}")
        if h5py.is_hdf5(filename):
            with h5py.File(filename, 'r') as f:
                # print("Keys:", list(f.keys()))
                # f.visititems(print_structure)
                HRTF = f['Dataset']
                self.freq = np.array(HRTF['frequency']).flatten()

                self.source_l = np.array(HRTF['source_left']).T  # shape: n_sub x 3
                self.source_r = np.array(HRTF['source_right']).T
                self.evaluation_loc = np.array(HRTF['evaluation_loc']).T
                self.evaluation_loc_angle = np.array(HRTF['evaluation_loc_angle']).T

                self.HRTF_l = np.array(HRTF['HRTF_left']).T
                self.HRTF_r = np.array(HRTF['HRTF_right']).T
                try:
                    self.delay_left = np.array(HRTF['delay_left']).T
                    self.delay_right = np.array(HRTF['delay_right']).T
                    self.itd = self.delay_right - self.delay_left
                except KeyError:
                    self.delay_left = None
                    self.delay_right = None
                    self.itd = None

                try:
                    self.source_mag_left = np.array(HRTF['source_mag_left']).T
                    self.source_mag_right = np.array(HRTF['source_mag_right']).T
                except KeyError:
                    self.source_mag_left = None
                    self.source_mag_right = None


               
                
        else:
            HRTF = loadmat(filename)["Dataset"] 
            self.freq = HRTF['frequency'][0,0].flatten()     # 1*220
            # self.freq = self.freq[0:50]
            self.source_l = HRTF['source_left'][0,0]                # n_sub * 3
            self.source_r = HRTF['source_right'][0,0]               # n_sub * 3
            self.evaluation_loc = HRTF['evaluation_loc'][0,0]       # n_eval_loc * 3
            self.evaluation_loc_angle = HRTF['evaluation_loc_angle'][0,0]       # n_eval_loc * 3
            self.HRTF_l = HRTF['HRTF_left'][0,0]                    # (n_sub * n_eval_loc) * n_freq
            self.HRTF_r = HRTF['HRTF_right'][0,0]                  # (n_sub * n_eval_loc) * n_freq
            if 'delay_left' in HRTF.dtype.names:
                    self.delay_left = HRTF['delay_left'][0,0].astype(np.int8)
                    self.delay_right = HRTF['delay_right'][0,0].astype(np.int8)
                    self.itd = self.delay_right - self.delay_left
            if 'source_mag_left' in HRTF.dtype.names:
                    self.source_mag_left = HRTF['source_mag_left'][0,0]    # n_sub * n_eval_loc* n_freq
                    self.source_mag_right = HRTF['source_mag_right'][0,0]  # n_sub * n_eval_loc* n_freq
        
        # self.freq = self.freq[120:200]
        # self.HRTF_l = self.HRTF_l[:,120:200]
        # self.HRTF_r = self.HRTF_r[:,120:200]
        # # Load coefficients
        # self.SH_coef = np.load("../Dataset/"+database+"/SH_coef.npy")      # n_sub * n_coef * 3
        # self.SCH_coef = np.load("../Dataset/"+database+"/SCH_coef.npy")    # n_sub * 2 * n_coef * 3 [left, right]
        # if self.SH_coef.ndim > 3:
        #     self.SH_coef = np.squeeze(self.SH_coef)
        if point_format == "unique":
            point_file = "/app/Dataset/"+database+"/points_"+point_size+".npy"
            if os.path.isfile(point_file):
                print(f"Loading points from {point_file}")
                self.points = np.load(point_file)
            else: 
                self.points = np.array([0])
                print(f"points_{point_size}.npy not found, please generate it")
                raise ValueError
        elif point_format == "repeat":
            point_file = "/app/Dataset/"+database+"/points_"+point_size+"_repeat.npy"
            if os.path.isfile(point_file):
                print(f"Loading points from {point_file}")
                self.points = np.load(point_file)  
            else: 
                self.points = np.array([0])
                print(f"points_{point_size}_repeat.npy not found, please generate it")
                raise ValueError
        elif point_format == "raw":
            point_file = "/app/Dataset/"+database+"/points.job"
            if os.path.isfile(point_file):
                print(f"Loading points from {point_file}")
                self.points = joblib_load(point_file)  
            else: 
                self.points = np.array([0])
                print(f"points_{point_size}_repeat.npy not found, please generate it")
                raise ValueError


        self.num_eval_loc = len(self.evaluation_loc)
        self.num_freq = len(self.freq)
        self.num_sub = self.HRTF_l.shape[0] // self.num_eval_loc
        if isinstance(self.points, np.ndarray):
            self.repeat_times = self.points.shape[0] // self.num_sub
        if isinstance(self.points, list):
            self.repeat_times = len(self.points) // self.num_sub
        self.num_side = 2

        if database =="UHM_dense_train" or database =="UHM_dense_test" or database =="UHM_dense_noitd_train" or database =="UHM_dense_noitd_test":
            self.points = self.points[:500,:]
        if point_format == "repeat":
            if database =="UHM_dense_train" or database =="UHM_dense_test" or database =="UHM_dense_noitd_train" or database =="UHM_dense_noitd_test":
                self.points = self.points[:500*self.repeat_times,:]
        print(f"Number of subjects: {self.num_sub}, Number of evaluation locations: {self.num_eval_loc}, Number of frequencies: {self.num_freq},  Number of repeat of points: {self.repeat_times}")

        # reshape HRTF
        self.HRTF_l = self.extract_real_image(self.HRTF_l)
        self.HRTF_r = self.extract_real_image(self.HRTF_r)
        if np.iscomplexobj(self.source_mag_left):
            self.source_mag_left = self.extract_real_image(self.source_mag_left)
            self.source_mag_right = self.extract_real_image(self.source_mag_right)
        # normalize sh/sch

       
        # self.SH_coef = self.SH_coef[0:1000]
        # self.SCH_coef = self.SCH_coef[0:1000]

    
    def __len__(self):
        return  self.num_sub*self.num_eval_loc * self.num_freq * self.num_side
             
    def getitem(self, sub_index, eval_loc_index, freq_index, lr_index):
        """ sub_index: subject index, \\
            eval_loc_index: evaluation location index, \\
            freq_index: frequency index, \\
            lr_index: left[0] or right ear[1]"""
        assert sub_index < self.num_sub, f"sub_index {sub_index} out of range"
        assert eval_loc_index < self.num_eval_loc, f"eval_loc_index {eval_loc_index} out of range"
        assert freq_index < self.num_freq, f"freq_index {freq_index} out of range"
        assert lr_index < self.num_side, f"lr_index {lr_index} out of range"
        if lr_index == 0:
            source_loc_item = self.source_l[sub_index]
            HRTF_item = self.HRTF_l[sub_index*self.num_eval_loc+eval_loc_index, freq_index,:]
        elif lr_index == 1:
            source_loc_item = self.source_r[sub_index]
            HRTF_item = self.HRTF_r[sub_index*self.num_eval_loc+eval_loc_index, freq_index,:]
        
        freq_item = self.freq[freq_index]
        eval_loc_item = self.evaluation_loc[eval_loc_index]
        SH_item = self.SH_coef[sub_index]   
        SCH_item = self.SCH_coef[sub_index, lr_index]
                
        return sub_index, freq_item, source_loc_item, eval_loc_item, SH_item, SCH_item, HRTF_item,
    
    def extract_real_image(self, HRTF):
        r"""extract real and image part"""
        if np.iscomplexobj(HRTF):
            HRTF_real = HRTF.real
            HRTF_imag = HRTF.imag
        else:
            HRTF_real = HRTF['real']
            HRTF_imag = HRTF['imag']
        
        HRTF_reshaped = np.stack((HRTF_real, HRTF_imag), axis=-1)
        return HRTF_reshaped
  
    
def print_structure(name, obj):
    print(name, type(obj))    


class StandardizerBase:
    """
    Base class for per-sample standardization along the batch (0th) dimension.
    Subclasses must assert any shape constraints before calling super().__init__.
    """
    def __init__(self, x: torch.Tensor, eps: float = 1e-6):
        """
        x: Tensor of shape [N, ...] where statistics are computed over the N axis.
        eps: small constant to avoid division by zero.
        """
        self.eps = eps
        # compute mean/std over the batch dimension (dim=0)
        # resulting shape is x.shape[1:]
        self.means = x.mean(dim=0)
        self.stds  = x.std(dim=0)
        # zero-std → ∞ so (x–mean)/∞ → 0
        self.stds[self.stds == 0] = float('inf')

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize: (x - mean) / (std + eps)
        x: Tensor of shape [batch_size, ...] matching the shape of self.means
        """
        return (x - self.means.unsqueeze(0)) / (self.stds.unsqueeze(0) + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Invert normalization: x * (std + eps) + mean,
        then mask any inf/NaN back to zero.
        """
        out = x * (self.stds.unsqueeze(0) + self.eps) + self.means.unsqueeze(0)
        mask = ~torch.isfinite(out)
        out[mask] = 0.
        return out

    def cpu(self):
        """Move stored statistics to CPU."""
        self.means = self.means.cpu()
        self.stds  = self.stds.cpu()


class StandardizerFreq(StandardizerBase):
    def __init__(self, x: torch.Tensor, eps: float = 1e-6):
        """
        x: [N, n_freqs, 2] tensor (real & imag)
        """
        assert x.dim() == 3 and x.size(2) == 2, "Expected shape [N, n_freqs, 2]"
        super().__init__(x, eps)


class StandardizerAnthrop(StandardizerBase):
    def __init__(self, x: torch.Tensor, eps: float = 1e-6):
        """
        x: [N, 3] tensor of anthropometric features
        """
        assert x.dim() == 2 and x.size(1) == 3, "Expected shape [N, 3]"
        super().__init__(x, eps)


if __name__ == "__main__":

    arg_config = {
        "database": "UHM",
        "sub": "", #small_ or ""
        "type": "simu",
    }
    dataset = HRTFDataset2(arg=arg_config)
    sub_index = 0
    eval_loc_index = 30
    freq_index = 2
    lr_index = 0
    # sub_index, freq_item, source_loc_item, eval_loc_item, SH_item, SCH_item, HRTF_item = dataset.getitem(sub_index, eval_loc_index, freq_index, lr_index)
    print("OK ")
    
