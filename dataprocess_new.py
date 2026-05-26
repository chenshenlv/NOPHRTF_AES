import numpy as np
import torch
import os, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
if root not in sys.path:
    sys.path.insert(0, root)
from dataset import HRTFDataset2, StandardizerFreq, StandardizerAnthrop
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler
from sklearn.model_selection import train_test_split
from joblib import load as joblib_load
from joblib import dump as joblib_dump
from deepxde import config
from deepxde.utils import run_if_all_none
# config.set_default_float("float64")
config.set_default_float("float32")

class GenerateDataBase:
    def __init__(self, HRTFDataset: HRTFDataset2,):
        self.source_l = HRTFDataset.source_l.astype(config.real(np))
        self.source_r = HRTFDataset.source_r.astype(config.real(np))
        # self.SH_coef = HRTFDataset.SH_coef.astype(config.real(np))
        # self.SCH_coef = HRTFDataset.SCH_coef.astype(config.real(np))
        if isinstance(HRTFDataset.points,np.ndarray):
            self.points = HRTFDataset.points.astype(config.real(np))
        if isinstance(HRTFDataset.points,list):
            self.points = HRTFDataset.points.copy()
        self.HRTF_l = HRTFDataset.HRTF_l.astype(config.real(np))
        self.HRTF_r = HRTFDataset.HRTF_r.astype(config.real(np))
        if hasattr(HRTFDataset,'delay_left') and HRTFDataset.itd is not None:
            self.delay_left = HRTFDataset.delay_left.astype(config.real(np))
            self.delay_right = HRTFDataset.delay_right.astype(config.real(np))
        if hasattr(HRTFDataset, 'itd') and HRTFDataset.itd is not None:
            self.itd = HRTFDataset.itd.astype(config.real(np))
        self.source_mag_left = HRTFDataset.source_mag_left
        self.source_mag_right = HRTFDataset.source_mag_right
        self.frequency = HRTFDataset.freq.astype(config.real(np))  # shape: (n_freq,)
        self.n_freq = self.frequency.shape[0]
        self.eval_loc = HRTFDataset.evaluation_loc.astype(config.real(np))  # shape: (n_eval, 3)
        self.eval_loc_angle = HRTFDataset.evaluation_loc_angle.astype(config.real(np))  # shape: (n_eval, 3)
        self.n_eval = self.eval_loc.shape[0]
        self.n_sub = HRTFDataset.num_sub
        self.repeat_times = HRTFDataset.repeat_times
    def generate_data(self):
        raise NotImplementedError
    def generate_train_val(self):
        raise NotImplementedError

class GenerateDataPoint(GenerateDataBase):
    """ for use with model residue """
    def __init__(self, HRTFDataset: HRTFDataset2,):
        super().__init__(HRTFDataset)

    def generate_data(self):
        frequency = self.frequency  # shape: (n_freq,)
        freq_min = frequency[0]
        freq_max = frequency[-1]
        freq_step = (freq_max-freq_min)/(frequency.shape[0]-1)
        frequency_idx = ((frequency - freq_min) / freq_step).astype(config.real(np))
        eval_loc = self.eval_loc # shape: (n_eval, 3)
        n_eval = eval_loc.shape[0]

        branch_list = []  # for the CNN branch input (3x n_coeffs array)
     
        y_list = []           # corresponding targets: each is an array of shape (n_batch, 2)
        # Number of training subjects:
        n_sub = self.n_sub
        for i in range(n_sub):
           
            target =np.concat((self.HRTF_l[i * n_eval:(i + 1) * n_eval, :, :],self.HRTF_r[i * n_eval:(i + 1) * n_eval, :, :]),axis=2) 
            y_list.append(target)  # each target is (n_eval, n_freq, 4)
            
            
            # Build CNN branch input:
            POINTS_T = self.points[i].T   
            branch_list.append(POINTS_T)



        # Convert branch lists to arrays:
        branch_array = np.stack(branch_list, axis=0)  # shape: (N_branch, 3, 970)
        y_array = np.stack(y_list, axis=0)
        # X = (frequency_idx,branch_cnn_array_1,branch_cnn_array_2,np.concatenate((eval_loc,eval_loc),axis=0))
        # X = (frequency_idx,branch_cnn_array_1,branch_cnn_array_2,np.repeat(eval_loc,repeats=2,axis=0))
        X = (frequency_idx,branch_array,eval_loc)
        return X, y_array
    
    def generate_train_val(self,X, Y, test_size=0.13):
        """generate train and val pairs for all frequencies"""
        if test_size == 0: # no validation
            return X, None, Y, None
        
        X_fre, X_points,X_eval = X
        if test_size < 1:
            test_size = float(test_size)
        if test_size >= 1:
            test_size = int(test_size)
        X_points_tr, X_points_val, \
        y_train,  y_val  = train_test_split(
            X_points, Y,
            test_size=test_size,
            random_state=42
        )
        X_train = (X_fre, X_points_tr, X_eval)
        X_val   = (X_fre, X_points_val, X_eval)

        return X_train, X_val, y_train, y_val
    
class GenerateDataPoint_disweighted(GenerateDataBase):
    """ for use with model point and eval_loc distance weighted """
    def __init__(self, HRTFDataset: HRTFDataset2,):
        super().__init__(HRTFDataset)

    def generate_data(self):
        frequency = self.frequency  # shape: (n_freq,)
        freq_min = frequency[0]
        freq_max = frequency[-1]
        freq_step = (freq_max-freq_min)/(frequency.shape[0]-1)
        frequency_idx = ((frequency - freq_min) / freq_step).astype(config.real(np))
        eval_loc = self.eval_loc # shape: (n_eval, 3)
        n_eval = eval_loc.shape[0]
        source_l = self.source_l # shape: (n_sub, 3)
        source_r = self.source_r # shape: (n_sub, 3)
        branch_list = []  # for the CNN branch input (3x n_coeffs array)
     
        y_list = []           # corresponding targets: each is an array of shape (n_batch, 2)
        # Number of training subjects:
        n_sub = self.n_sub
        for i in range(n_sub):
           
            target =np.concat((self.HRTF_l[i * n_eval:(i + 1) * n_eval, :, :],self.HRTF_r[i * n_eval:(i + 1) * n_eval, :, :]),axis=2) 
            y_list.append(target)  # each target is (n_eval, n_freq, 4)
            
            
            # Build CNN branch input:
            POINTS_T = self.points[i].T   # shape: (3, 529)
            branch_list.append(POINTS_T)



        # Convert branch lists to arrays:
        branch_array = np.stack(branch_list, axis=0)  # shape: (N_branch, 3, 970)
        y_array = np.stack(y_list, axis=0)
        # X = (frequency_idx,branch_cnn_array_1,branch_cnn_array_2,np.concatenate((eval_loc,eval_loc),axis=0))
        # X = (frequency_idx,branch_cnn_array_1,branch_cnn_array_2,np.repeat(eval_loc,repeats=2,axis=0))
        X = (frequency_idx,branch_array,eval_loc,source_l,source_r)
        return X, y_array
    
    def generate_train_val(self,X, Y, test_size=0.13):
        """generate train and val pairs for all frequencies"""
        if test_size == 0: # no validation
            return X, None, Y, None, 
        
        X_fre, X_points,X_eval,X_source_l, X_source_r = X
        if test_size < 1:
            test_size = float(test_size)
        if test_size >= 1:
            test_size = int(test_size)
       
        (X_points_tr, X_points_val,
         X_source_l_tr, X_source_l_val,
         X_source_r_tr, X_source_r_val,
         y_train, y_val,
         ) = train_test_split(
            X_points, X_source_l, X_source_r, Y,
            test_size=test_size,
            random_state=42
        )

        # Group training and validation sets
        X_train = (X_fre, X_points_tr, X_eval, X_source_l_tr, X_source_r_tr)
        X_val   = (X_fre, X_points_val, X_eval, X_source_l_val, X_source_r_val)

        return X_train, X_val, y_train, y_val
    
class GenerateDataPoint_disweighted_joint(GenerateDataBase):
    """ for use with model point and eval_loc distance weighted and joint train itd """
    def __init__(self, HRTFDataset: HRTFDataset2,):
        super().__init__(HRTFDataset)

    def generate_data(self):
        frequency = self.frequency  # shape: (n_freq,)
        freq_min = frequency[0]
        freq_max = frequency[-1]
        freq_step = (freq_max-freq_min)/(frequency.shape[0]-1)
        frequency_idx = ((frequency - freq_min) / freq_step).astype(config.real(np))
        eval_loc = self.eval_loc # shape: (n_eval, 3)
        n_eval = eval_loc.shape[0]
        source_l = self.source_l # shape: (n_sub, 3)
        source_r = self.source_r # shape: (n_sub, 3)
        branch_list = []  # for the CNN branch input (3x n_coeffs array)
     
        y_list = []           # corresponding targets: each is an array of shape (n_batch, 4)
        itd_list = []         # corresponding targets: each is an array of shape (n_batch, 1)
        # Number of training subjects:
        n_sub = self.n_sub
        for i in range(n_sub):
           
            target =np.concat(
                (self.HRTF_l[i * n_eval:(i + 1) * n_eval, :, :],
                 self.HRTF_r[i * n_eval:(i + 1) * n_eval, :, :]),axis=2) 
            itd = self.itd[i, :]  # shape: (n_eval, 1)
            y_list.append(target)  # each target is (n_eval, n_freq, 4)
            itd_list.append(itd)
            
            # Build CNN branch input:
            POINTS_T = self.points[i].T   # shape: (3, 529)
            branch_list.append(POINTS_T)



        # Convert branch lists to arrays:
        branch_array = np.stack(branch_list, axis=0)  # shape: (N_branch, 3, 970)
        y_array = np.stack(y_list, axis=0)
        itd_array = np.stack(itd_list, axis=0)
        # X = (frequency_idx,branch_cnn_array_1,branch_cnn_array_2,np.concatenate((eval_loc,eval_loc),axis=0))
        # X = (frequency_idx,branch_cnn_array_1,branch_cnn_array_2,np.repeat(eval_loc,repeats=2,axis=0))
        X = (frequency_idx,branch_array,eval_loc,source_l,source_r)
        return X, y_array, itd_array 
    
    def generate_train_val(self,X, Y, ITD, test_size=0.13):
        """generate train and val pairs for all frequencies"""
        if test_size == 0: # no validation
            return X, None, Y, None, ITD, None
        
        X_fre, X_points,X_eval,X_source_l, X_source_r = X
        if test_size < 1:
            test_size = float(test_size)
        if test_size >= 1:
            test_size = int(test_size)
       
        (X_points_tr, X_points_val,
         X_source_l_tr, X_source_l_val,
         X_source_r_tr, X_source_r_val,
         y_train, y_val,
         itd_train, itd_val) = train_test_split(
            X_points, X_source_l, X_source_r, Y, ITD,
            test_size=test_size,
            random_state=42
        )

        # Group training and validation sets
        X_train = (X_fre, X_points_tr, X_eval, X_source_l_tr, X_source_r_tr)
        X_val   = (X_fre, X_points_val, X_eval, X_source_l_val, X_source_r_val)

        return X_train, X_val, y_train, y_val, itd_train, itd_val
    

class GenerateDataPoint_disweighted_mag_noitd(GenerateDataBase):
    """ for use with model point and eval_loc distance weighted and joint train itd, generate mag only """
    def __init__(self, HRTFDataset: HRTFDataset2,):
        super().__init__(HRTFDataset)

    def generate_data(self):
        frequency = self.frequency  # shape: (n_freq,)
        freq_min = frequency[0]
        freq_max = frequency[-1]
        freq_step = (freq_max-freq_min)/(frequency.shape[0]-1)
        frequency_idx = ((frequency - freq_min) / freq_step).astype(config.real(np))
        eval_loc = self.eval_loc # shape: (n_eval, 3)
        n_eval = eval_loc.shape[0]
        source_l = self.source_l # shape: (n_sub, 3)
        source_r = self.source_r # shape: (n_sub, 3)
        repeat_times = self.repeat_times
        # Number of training subjects:
        # n_sub = np.floor(self.n_sub*0.6).astype(int)
        n_sub = self.n_sub
        N = n_sub * repeat_times

        if isinstance(self.points,np.ndarray):
            branch_array = np.empty((N, self.points.shape[2], self.points.shape[1]), dtype=np.float32) # (N, 6, n_pts)
        if isinstance(self.points,list):
            branch_array = self.points
        y_array   = np.empty((N, n_eval, self.n_freq, 2), dtype=np.float32)

        k = 0
        for i in range(n_sub):
            HRTF_l_mag = np.linalg.norm(self.HRTF_l[i * n_eval:(i + 1) * n_eval, :, :], axis=-1, keepdims=True)
            HRTF_r_mag = np.linalg.norm(self.HRTF_r[i * n_eval:(i + 1) * n_eval, :, :], axis=-1, keepdims=True)
            HRTF_l_mag_db = 20 * np.log10(HRTF_l_mag)
            HRTF_r_mag_db = 20 * np.log10(HRTF_r_mag)
            target =np.concatenate(
                (HRTF_l_mag_db,
                HRTF_r_mag_db),axis=2) 
            

            
            for _ in range(repeat_times):
                if isinstance(self.points,np.ndarray):
                    branch_array[k] = self.points[k].T.astype(np.float32, copy=False)
                y_array[k] = target
                k += 1
        if repeat_times > 1:        
            source_l_rep = np.repeat(source_l, repeats=repeat_times, axis=0)  # (N,3)
            source_r_rep = np.repeat(source_r, repeats=repeat_times, axis=0)  # (N,3)
        else:
            source_l_rep = source_l
            source_r_rep = source_r
        X = (frequency_idx,branch_array,eval_loc,source_l_rep,source_r_rep)
        return X, y_array, 
    

    def generate_train_val(self, X, Y, val_size=0.13, test_size=10, random_state=42):
        """
        Requirements implemented (repeat_times > 1):
        1) Test set: pick `test_size` random subjects from n_sub.
           For each selected subject, take ONLY ONE sample from its repeats.
        2) Remove ALL samples (all repeats) of those test subjects from the remaining pool.
        3) Train/val split: split by subjects on the remaining subjects, but the resulting
           train/val sets should include ALL repeats for the selected train/val subjects.
        """

        X_fre, X_points, X_eval, X_source_l, X_source_r = X
        n_sub = self.n_sub
        r = int(self.repeat_times)
        if isinstance(X_points, list):
            X_points = np.array(X_points,dtype=object)

        # If no test requested
        if test_size == 0:
            # Optional: still do subject-aware val split when r>1
            return self._split_train_val_only(X, Y, val_size=val_size, random_state=random_state)

        # Total number of samples must match repeats structure
        N = X_points.shape[0]
        expected_N = n_sub * r
        if N != expected_N:
            raise ValueError(f"Expected N=n_sub*repeat_times={expected_N}, but got N={N}.")

        # Convert test_size to subject-count
        if test_size < 1:
            # fraction of subjects
            n_test_sub = int(np.round(n_sub * float(test_size)))
        else:
            n_test_sub = int(test_size)

        n_test_sub = max(0, min(n_test_sub, n_sub))

        rng = np.random.default_rng(random_state)

        # subject_id for each sample: [0,0,...,0, 1,1,...,1, ..., n_sub-1,...]
        subject_id = np.repeat(np.arange(n_sub), r)  # shape (N,)

        # ---------- Choose test subjects ----------
        test_subjects = rng.choice(n_sub, size=n_test_sub, replace=False)

        # ---------- For each test subject, choose ONE repeat to keep ----------
        # Collect exactly one sample index per test subject
        test_sample_indices = []
        for s in test_subjects:
            idx_s = np.where(subject_id == s)[0]          # all repeats indices for subject s
            pick = rng.choice(idx_s, size=1, replace=False)[0]
            test_sample_indices.append(pick)
        test_sample_indices = np.array(test_sample_indices, dtype=np.int64)

        # ----------  Remove ALL repeats of test subjects from remaining pool ----------
        is_test_subject_sample = np.isin(subject_id, test_subjects)   # all repeats of test subjects
        remaining_indices = np.where(~is_test_subject_sample)[0]

        remaining_subjects = np.setdiff1d(np.arange(n_sub), test_subjects)

        # ----------  Train/val split by SUBJECTS on remaining subjects ----------
        if val_size <= 0:
            train_subjects = remaining_subjects
            val_subjects = np.array([], dtype=int)
        else:
            train_subjects, val_subjects = train_test_split(
                remaining_subjects,
                test_size=val_size,
                random_state=random_state,
                shuffle=True
            )

        # Expand subjects -> sample indices (include ALL repeats)
        train_indices = np.where(np.isin(subject_id, train_subjects))[0]
        val_indices   = np.where(np.isin(subject_id, val_subjects))[0]

        # ---------- Build datasets ----------
        # Note: X_eval and X_fre are shared (not sample-dependent) in current design.
        X_train = (
            X_fre,
            X_points[train_indices],
            X_eval,
            X_source_l[train_indices],
            X_source_r[train_indices],
        )
        y_train = Y[train_indices]
       

        X_val = (
            X_fre,
            X_points[val_indices],
            X_eval,
            X_source_l[val_indices],
            X_source_r[val_indices],
        )
        y_val = Y[val_indices]
       
        # Test set: one sample per test subject
        X_test = (
            X_fre,
            X_points[test_sample_indices],
            X_eval,
            X_source_l[test_sample_indices],
            X_source_r[test_sample_indices],
        )
        y_test = Y[test_sample_indices]
        

        return (X_train, X_val, X_test,
                y_train, y_val, y_test)

    def _split_train_val_only(self, X, Y, val_size=0.13, random_state=42):
        """Subject-aware train/val split only (no test), still includes ALL repeats."""
        X_fre, X_points, X_eval, X_source_l, X_source_r = X
        n_sub = self.n_sub
        r = int(self.repeat_times)

        if isinstance(X_points, list):
            X_points = np.array(X_points,dtype=object)
        N = X_points.shape[0]
        expected_N = n_sub * r
        if N != expected_N:
            raise ValueError(f"Expected N=n_sub*repeat_times={expected_N}, but got N={N}.")

        subject_id = np.repeat(np.arange(n_sub), r)

        subjects = np.arange(n_sub)
        if val_size <= 0:
            train_subjects = subjects
            val_subjects = np.array([], dtype=int)
        else:
            train_subjects, val_subjects = train_test_split(
                subjects, test_size=val_size, random_state=random_state, shuffle=True
            )

        train_indices = np.where(np.isin(subject_id, train_subjects))[0]
        val_indices   = np.where(np.isin(subject_id, val_subjects))[0]

        X_train = (X_fre, X_points[train_indices], X_eval, X_source_l[train_indices], X_source_r[train_indices])
        X_val   = (X_fre, X_points[val_indices],   X_eval, X_source_l[val_indices],   X_source_r[val_indices])
        return X_train, X_val, Y[train_indices], Y[val_indices]
    

class GenerateDataPoint_mag_noitd_ref(GenerateDataBase):
    """ generate mag only, and also genereate source reference mag, no itd data generated """
    def __init__(self, HRTFDataset: HRTFDataset2,):
        super().__init__(HRTFDataset)

    def generate_data(self):
        frequency = self.frequency  # shape: (n_freq,)
        freq_min = frequency[0]
        freq_max = frequency[-1]
        freq_step = (freq_max-freq_min)/(frequency.shape[0]-1)
        frequency_idx = ((frequency - freq_min) / freq_step).astype(config.real(np))
        eval_loc = self.eval_loc # shape: (n_eval, 3)
        n_eval = eval_loc.shape[0]
        source_l = self.source_l # shape: (n_sub, 3)
        source_r = self.source_r # shape: (n_sub, 3)
        ref_l = self.source_mag_left
        ref_r = self.source_mag_right
        repeat_times = self.repeat_times
        # Number of training subjects:
        # n_sub = np.floor(self.n_sub*0.6).astype(int)
        n_sub = self.n_sub
        N = n_sub * repeat_times

        if isinstance(self.points,np.ndarray):
            branch_array = np.empty((N, self.points.shape[2], self.points.shape[1]), dtype=np.float32) # (N, 6, n_pts)
        if isinstance(self.points,list):
            branch_array = self.points
        y_array = np.empty((N, n_eval, self.n_freq, 2), dtype=np.float32)
        ref_array = np.empty((N, n_eval, self.n_freq, 2), dtype=np.float32)

        k = 0
        for i in range(n_sub):
            HRTF_l_mag = np.linalg.norm(self.HRTF_l[i * n_eval:(i + 1) * n_eval, :, :], axis=-1, keepdims=True)
            HRTF_r_mag = np.linalg.norm(self.HRTF_r[i * n_eval:(i + 1) * n_eval, :, :], axis=-1, keepdims=True)
            HRTF_l_mag_db = 20 * np.log10(HRTF_l_mag)
            HRTF_r_mag_db = 20 * np.log10(HRTF_r_mag)
            target =np.concatenate(
                (HRTF_l_mag_db,
                HRTF_r_mag_db),axis=2) 
            ref_l_mag = np.linalg.norm(ref_l[i * n_eval:(i + 1) * n_eval, :, :], axis=-1, keepdims=True)
            ref_r_mag = np.linalg.norm(ref_r[i * n_eval:(i + 1) * n_eval, :, :], axis=-1, keepdims=True)
            ref_l_mag_db = 20 * np.log10(ref_l_mag)
            ref_r_mag_db = 20 * np.log10(ref_r_mag)
            ref = np.concatenate(
                (ref_l_mag_db,
                ref_r_mag_db),axis=2) 


            
            for _ in range(repeat_times):
                if isinstance(self.points,np.ndarray):
                    branch_array[k] = self.points[k].T.astype(np.float32, copy=False)
                y_array[k] = target
                ref_array[k] = ref
                k += 1
        if repeat_times > 1:        
            source_l_rep = np.repeat(source_l, repeats=repeat_times, axis=0)  # (N,3)
            source_r_rep = np.repeat(source_r, repeats=repeat_times, axis=0)  # (N,3)
        else:
            source_l_rep = source_l
            source_r_rep = source_r
        X = (frequency_idx,branch_array,eval_loc,source_l_rep,source_r_rep, ref_array)
        return X, y_array, 

    def generate_train_val(self, X, Y, val_size=0.13, test_size=10, random_state=42):
        """
        Requirements implemented (repeat_times > 1):
        1) Test set: pick `test_size` random subjects from n_sub.
           For each selected subject, take ONLY ONE sample from its repeats.
        2) Remove ALL samples (all repeats) of those test subjects from the remaining pool.
        3) Train/val split: split by subjects on the remaining subjects, but the resulting
           train/val sets should include ALL repeats for the selected train/val subjects.
        """

        X_fre, X_points, X_eval, X_source_l, X_source_r, ref_array = X
        n_sub = self.n_sub
        r = int(self.repeat_times)
        if isinstance(X_points, list):
            X_points = np.array(X_points,dtype=object)

        # If no test requested
        if test_size == 0:
            # Optional: still do subject-aware val split when r>1
            return self._split_train_val_only(X, Y, val_size=val_size, random_state=random_state)

        # Total number of samples must match repeats structure
        N = X_points.shape[0]
        expected_N = n_sub * r
        if N != expected_N:
            raise ValueError(f"Expected N=n_sub*repeat_times={expected_N}, but got N={N}.")

        # Convert test_size to subject-count
        if test_size < 1:
            # fraction of subjects
            n_test_sub = int(np.round(n_sub * float(test_size)))
        else:
            n_test_sub = int(test_size)

        n_test_sub = max(0, min(n_test_sub, n_sub))

        rng = np.random.default_rng(random_state)

        # subject_id for each sample: [0,0,...,0, 1,1,...,1, ..., n_sub-1,...]
        subject_id = np.repeat(np.arange(n_sub), r)  # shape (N,)

        # ----------  Choose test subjects ----------
        test_subjects = rng.choice(n_sub, size=n_test_sub, replace=False)

        # ----------  For each test subject, choose ONE repeat to keep ----------
        # Collect exactly one sample index per test subject
        test_sample_indices = []
        for s in test_subjects:
            idx_s = np.where(subject_id == s)[0]          # all repeats indices for subject s
            pick = rng.choice(idx_s, size=1, replace=False)[0]
            test_sample_indices.append(pick)
        test_sample_indices = np.array(test_sample_indices, dtype=np.int64)

        # ----------  Remove ALL repeats of test subjects from remaining pool ----------
        is_test_subject_sample = np.isin(subject_id, test_subjects)   # all repeats of test subjects
        remaining_indices = np.where(~is_test_subject_sample)[0]

        remaining_subjects = np.setdiff1d(np.arange(n_sub), test_subjects)

        # ----------  Train/val split by SUBJECTS on remaining subjects ----------
        if val_size <= 0:
            train_subjects = remaining_subjects
            val_subjects = np.array([], dtype=int)
        else:
            train_subjects, val_subjects = train_test_split(
                remaining_subjects,
                test_size=val_size,
                random_state=random_state,
                shuffle=True
            )

        # Expand subjects -> sample indices (include ALL repeats)
        train_indices = np.where(np.isin(subject_id, train_subjects))[0]
        val_indices   = np.where(np.isin(subject_id, val_subjects))[0]

        # ---------- Build datasets ----------
        # Note: X_eval and X_fre are shared (not sample-dependent) in  current design.
        X_train = (
            X_fre,
            X_points[train_indices],
            X_eval,
            X_source_l[train_indices],
            X_source_r[train_indices],
            ref_array[train_indices]
        )
        y_train = Y[train_indices]
       

        X_val = (
            X_fre,
            X_points[val_indices],
            X_eval,
            X_source_l[val_indices],
            X_source_r[val_indices],
            ref_array[val_indices]
        )
        y_val = Y[val_indices]
       
        # Test set: one sample per test subject
        X_test = (
            X_fre,
            X_points[test_sample_indices],
            X_eval,
            X_source_l[test_sample_indices],
            X_source_r[test_sample_indices],
            ref_array[test_sample_indices]
        )
        y_test = Y[test_sample_indices]
        

        return (X_train, X_val, X_test,
                y_train, y_val, y_test)

    def _split_train_val_only(self, X, Y, val_size=0.13, random_state=42):
        """Subject-aware train/val split only (no test), still includes ALL repeats."""
        X_fre, X_points, X_eval, X_source_l, X_source_r, ref_array = X
        n_sub = self.n_sub
        r = int(self.repeat_times)

        if isinstance(X_points, list):
            X_points = np.array(X_points,dtype=object)
        N = X_points.shape[0]
        expected_N = n_sub * r
        if N != expected_N:
            raise ValueError(f"Expected N=n_sub*repeat_times={expected_N}, but got N={N}.")

        subject_id = np.repeat(np.arange(n_sub), r)

        subjects = np.arange(n_sub)
        if val_size <= 0:
            train_subjects = subjects
            val_subjects = np.array([], dtype=int)
        else:
            train_subjects, val_subjects = train_test_split(
                subjects, test_size=val_size, random_state=random_state, shuffle=True
            )

        train_indices = np.where(np.isin(subject_id, train_subjects))[0]
        val_indices   = np.where(np.isin(subject_id, val_subjects))[0]

        X_train = (X_fre, X_points[train_indices], X_eval, X_source_l[train_indices], X_source_r[train_indices], ref_array[train_indices])
        X_val   = (X_fre, X_points[val_indices],   X_eval, X_source_l[val_indices],   X_source_r[val_indices], ref_array[val_indices])
        return X_train, X_val, Y[train_indices], Y[val_indices]

class GenerateDataPoint_disweighted_joint_mag(GenerateDataBase):
    """ for use with model point and eval_loc distance weighted and joint train itd, generate mag only """
    def __init__(self, HRTFDataset: HRTFDataset2,):
        super().__init__(HRTFDataset)

    def generate_data(self):
        frequency = self.frequency  # shape: (n_freq,)
        freq_min = frequency[0]
        freq_max = frequency[-1]
        freq_step = (freq_max-freq_min)/(frequency.shape[0]-1)
        frequency_idx = ((frequency - freq_min) / freq_step).astype(config.real(np))
        eval_loc = self.eval_loc # shape: (n_eval, 3)
        n_eval = eval_loc.shape[0]
        source_l = self.source_l # shape: (n_sub, 3)
        source_r = self.source_r # shape: (n_sub, 3)
        repeat_times = self.repeat_times
        # Number of training subjects:
        n_sub = self.n_sub
        N = n_sub * repeat_times

        if isinstance(self.points,np.ndarray):
            branch_array = np.empty((N, self.points.shape[2], self.points.shape[1]), dtype=np.float32) # (N, 6, n_pts)
        if isinstance(self.points,list):
            branch_array = self.points
        y_array   = np.empty((N, n_eval, self.n_freq, 2), dtype=np.float32)
        itd_array = np.empty((N, n_eval), dtype=np.float32)

        k = 0
        for i in range(n_sub):
            HRTF_l_mag = np.linalg.norm(self.HRTF_l[i * n_eval:(i + 1) * n_eval, :, :], axis=-1, keepdims=True)
            HRTF_r_mag = np.linalg.norm(self.HRTF_r[i * n_eval:(i + 1) * n_eval, :, :], axis=-1, keepdims=True)
            HRTF_l_mag_db = 20 * np.log10(HRTF_l_mag)
            HRTF_r_mag_db = 20 * np.log10(HRTF_r_mag)
            target =np.concatenate(
                (HRTF_l_mag_db,
                HRTF_r_mag_db),axis=2) 
            
            itd = self.itd[i, :]  # shape: (n_eval, 1)

            
            for _ in range(repeat_times):
                if isinstance(self.points,np.ndarray):
                    branch_array[k] = self.points[k].T.astype(np.float32, copy=False)
                y_array[k] = target
                itd_array[k] = itd
                k += 1
        if repeat_times > 1:        
            source_l_rep = np.repeat(source_l, repeats=repeat_times, axis=0)  # (N,3)
            source_r_rep = np.repeat(source_r, repeats=repeat_times, axis=0)  # (N,3)
        else:
            source_l_rep = source_l
            source_r_rep = source_r
        X = (frequency_idx,branch_array,eval_loc,source_l_rep,source_r_rep)
        return X, y_array, itd_array 
    

    def generate_train_val(self, X, Y, ITD, val_size=0.13, test_size=10, random_state=42):
        """
        Requirements implemented (repeat_times > 1):
        1) Test set: pick `test_size` random subjects from n_sub.
           For each selected subject, take ONLY ONE sample from its repeats.
        2) Remove ALL samples (all repeats) of those test subjects from the remaining pool.
        3) Train/val split: split by subjects on the remaining subjects, but the resulting
           train/val sets should include ALL repeats for the selected train/val subjects.
        """

        X_fre, X_points, X_eval, X_source_l, X_source_r = X
        n_sub = self.n_sub
        r = int(self.repeat_times)
        if isinstance(X_points, list):
            X_points = np.array(X_points,dtype=object)

        # If no test requested
        if test_size == 0:
            # Optional: still do subject-aware val split when r>1
            return self._split_train_val_only(X, Y, ITD, val_size=val_size, random_state=random_state)

        # Total number of samples must match repeats structure
        N = X_points.shape[0]
        expected_N = n_sub * r
        if N != expected_N:
            raise ValueError(f"Expected N=n_sub*repeat_times={expected_N}, but got N={N}.")

        # Convert test_size to subject-count
        if test_size < 1:
            # fraction of subjects
            n_test_sub = int(np.round(n_sub * float(test_size)))
        else:
            n_test_sub = int(test_size)

        n_test_sub = max(0, min(n_test_sub, n_sub))

        rng = np.random.default_rng(random_state)

        # subject_id for each sample: [0,0,...,0, 1,1,...,1, ..., n_sub-1,...]
        subject_id = np.repeat(np.arange(n_sub), r)  # shape (N,)

        # ----------  Choose test subjects ----------
        test_subjects = rng.choice(n_sub, size=n_test_sub, replace=False)

        # ----------  For each test subject, choose ONE repeat to keep ----------
        # Collect exactly one sample index per test subject
        test_sample_indices = []
        for s in test_subjects:
            idx_s = np.where(subject_id == s)[0]          # all repeats indices for subject s
            pick = rng.choice(idx_s, size=1, replace=False)[0]
            test_sample_indices.append(pick)
        test_sample_indices = np.array(test_sample_indices, dtype=np.int64)

        # ----------  Remove ALL repeats of test subjects from remaining pool ----------
        is_test_subject_sample = np.isin(subject_id, test_subjects)   # all repeats of test subjects
        remaining_indices = np.where(~is_test_subject_sample)[0]

        remaining_subjects = np.setdiff1d(np.arange(n_sub), test_subjects)

        # ----------  Train/val split by SUBJECTS on remaining subjects ----------
        if val_size <= 0:
            train_subjects = remaining_subjects
            val_subjects = np.array([], dtype=int)
        else:
            train_subjects, val_subjects = train_test_split(
                remaining_subjects,
                test_size=val_size,
                random_state=random_state,
                shuffle=True
            )

        # Expand subjects -> sample indices (include ALL repeats)
        train_indices = np.where(np.isin(subject_id, train_subjects))[0]
        val_indices   = np.where(np.isin(subject_id, val_subjects))[0]

        # ---------- Build datasets ----------
        # Note: X_eval and X_fre are shared (not sample-dependent) in current design.
        X_train = (
            X_fre,
            X_points[train_indices],
            X_eval,
            X_source_l[train_indices],
            X_source_r[train_indices],
        )
        y_train = Y[train_indices]
        itd_train = ITD[train_indices]

        X_val = (
            X_fre,
            X_points[val_indices],
            X_eval,
            X_source_l[val_indices],
            X_source_r[val_indices],
        )
        y_val = Y[val_indices]
        itd_val = ITD[val_indices]

        # Test set: one sample per test subject
        X_test = (
            X_fre,
            X_points[test_sample_indices],
            X_eval,
            X_source_l[test_sample_indices],
            X_source_r[test_sample_indices],
        )
        y_test = Y[test_sample_indices]
        itd_test = ITD[test_sample_indices]

        return (X_train, X_val, X_test,
                y_train, y_val, y_test,
                itd_train, itd_val, itd_test)

    def _split_train_val_only(self, X, Y, ITD, val_size=0.13, random_state=42):
        """Subject-aware train/val split only (no test), still includes ALL repeats."""
        X_fre, X_points, X_eval, X_source_l, X_source_r = X
        n_sub = self.n_sub
        r = int(self.repeat_times)

        if isinstance(X_points, list):
            X_points = np.array(X_points,dtype=object)
        N = X_points.shape[0]
        expected_N = n_sub * r
        if N != expected_N:
            raise ValueError(f"Expected N=n_sub*repeat_times={expected_N}, but got N={N}.")

        subject_id = np.repeat(np.arange(n_sub), r)

        subjects = np.arange(n_sub)
        if val_size <= 0:
            train_subjects = subjects
            val_subjects = np.array([], dtype=int)
        else:
            train_subjects, val_subjects = train_test_split(
                subjects, test_size=val_size, random_state=random_state, shuffle=True
            )

        train_indices = np.where(np.isin(subject_id, train_subjects))[0]
        val_indices   = np.where(np.isin(subject_id, val_subjects))[0]

        X_train = (X_fre, X_points[train_indices], X_eval, X_source_l[train_indices], X_source_r[train_indices])
        X_val   = (X_fre, X_points[val_indices],   X_eval, X_source_l[val_indices],   X_source_r[val_indices])
        return X_train, X_val, Y[train_indices], Y[val_indices], ITD[train_indices], ITD[val_indices]

                
    
if __name__ == "__main__":
    # import os, sys
    # root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # if root not in sys.path:
    #     sys.path.insert(0, root)
    # pt_size=["15k","10k","5k","2k","1k"]
    pt_size = ["5k"]
    test_size = 0      # number of subjects in the test set for training data
    itd = False        # whether to generate ITD data
    magnitude = True  # whether to generate magnitude only data
    train_data = True  # generate training/validation (and optionally test) data
    prediction_data = False  # generate only prediction data
    ref = True  # whether to include ref magnitude data
    point_format = "raw"  # "unique" or "repeat" or "raw"

    for pt in pt_size:
        arg_config = {
            "database": "AXD",  # "UHM_dense_train", "UHM_dense_test", "UHM", "AXD", "HUTUBUS","UHM_dense_noitd_train","UHM_noitd"
            "sub": "",                            # "small_" or ""
            "type": "msr_44_ff_ref_",                       # "msr" or "simu","msr_44_ff"
            "points": pt,
            "point_format": point_format,
        }   

        data_save_folder = (
            f"/app/Network/model_point/data/"
            f"{arg_config['database']}/AE/{arg_config['type']}/{pt}/"
        )
        if magnitude and point_format != "raw":
            data_save_folder = (
            f"/app/Network/model_point/data/"
            f"{arg_config['database']}/AE/mag/{arg_config['type']}/{pt}/"
        )
        if point_format == "raw":
            data_save_folder = (
            f"/app/Network/model_point/data/"
            f"{arg_config['database']}/AE/mag/raw/{arg_config['type']}/{pt}/"
        )

        if not os.path.exists(data_save_folder):
            os.makedirs(data_save_folder)
            print(f"Created folder: {data_save_folder}")
        else:
            print(f"Folder already exists: {data_save_folder}")

        # Choose generator based on ITD flag
        dataset = HRTFDataset2(arg_config)
        if itd and magnitude:
            GeneratorCls = GenerateDataPoint_disweighted_joint_mag
        elif not itd and magnitude and not ref:
            GeneratorCls = GenerateDataPoint_disweighted_mag_noitd
        elif itd and not magnitude:
            GeneratorCls = GenerateDataPoint_disweighted_joint
        elif not itd and not magnitude:
            GeneratorCls = GenerateDataPoint_disweighted
        elif not itd and (magnitude and ref):
            GeneratorCls = GenerateDataPoint_mag_noitd_ref
        generator = GeneratorCls(dataset)

        # Generate raw data
        if itd:
            X, Y, ITD = generator.generate_data()
        else:
            X, Y = generator.generate_data()
            ITD = None  # for uniform handling below

        # ---------------------------
        # Training / validation / test
        # ---------------------------
        if train_data:
            #  Split out a test set only if test_size > 0
            if test_size > 0 and itd:
                X_train, X_val, X_test,y_train, y_val, y_test,itd_train, itd_val, itd_test = \
                        generator.generate_train_val(X, Y, ITD, val_size=0.05, test_size=test_size)
            elif test_size > 0 and not itd:
                X_train, X_val, X_test,y_train, y_val, y_test = \
                        generator.generate_train_val(X, Y, val_size=0.05, test_size=test_size)
                itd_train_val = itd_test = None
            elif test_size == 0 and itd:
                 X_train, X_val, y_train, y_val, itd_train, itd_val = \
                        generator.generate_train_val(X=X, Y=Y, ITD=ITD, val_size=0.05, test_size=0)
            elif test_size == 0 and not itd:
                X_train, X_val, y_train, y_val = \
                    generator.generate_train_val(X=X, Y=Y, val_size=0.05, test_size=0)
                itd_train = itd_val = None

            # 3) Save train/val (and optionally ITD)
            joblib_dump(X_train, data_save_folder + "X_train.sav")
            joblib_dump(X_val,   data_save_folder + "X_val.sav")
            joblib_dump(y_train, data_save_folder + "y_train.sav")
            joblib_dump(y_val,   data_save_folder + "y_val.sav")

            if itd:
                joblib_dump(itd_train, data_save_folder + "itd_train.sav")
                joblib_dump(itd_val,   data_save_folder + "itd_val.sav")

            # 4) Save test only if test_size > 0
            if test_size > 0:
                joblib_dump(X_test, data_save_folder + "X_test.sav")
                joblib_dump(y_test, data_save_folder + "y_test.sav")
                if itd:
                    joblib_dump(itd_test, data_save_folder + "itd_test.sav")

            print("Process data done!")

        # ---------------------------
        # Prediction-only data dump
        # ---------------------------
        if prediction_data:
            # here X,Y (and ITD) are the full dataset, treated as "test"
            joblib_dump(X, data_save_folder + "X_test.sav")
            joblib_dump(Y, data_save_folder + "y_test.sav")
            if itd:
                joblib_dump(ITD, data_save_folder + "itd_test.sav")
            print("Process data done!")
    