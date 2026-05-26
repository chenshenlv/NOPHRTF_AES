import numpy as np
from scipy.io import savemat
from sklearn.model_selection import train_test_split
from dataprocess_new import GenerateDataBase
from dataset import HRTFDataset2



class GenerateDataPoint_multi_split_1(GenerateDataBase):
    '''Not generate seperate val set, but directly split the data into 4 sets based on subject and location splits.'''
    def __init__(self, HRTFDataset: HRTFDataset2):
        super().__init__(HRTFDataset)

    def generate_train_val(self):
        return

    def generate_data(self, val_size=0.13, loc_test_size=0.2, random_state=42):
        """
        Generates 4 datasets based on Subject and Location splits.
        """
        rng = np.random.default_rng(random_state)
        
        # 1. Split LOCATIONS (L_seen, L_unseen)
        n_loc = self.eval_loc.shape[0]
        loc_indices = np.arange(n_loc)
        L_seen_idx, L_unseen_idx = train_test_split(
            loc_indices, test_size=loc_test_size, random_state=random_state
        )
        
        # 2. Split SUBJECTS (sub_tr, sub_ts)
        n_sub = self.n_sub
        sub_indices = np.arange(n_sub)
        sub_tr_idx, sub_ts_idx = train_test_split(
            sub_indices, test_size=val_size, random_state=random_state
        )

        # Helper to extract and format data for specific sub/loc groups
        def bundle_data(s_indices, l_indices):
            # s_indices: list of subject IDs
            # l_indices: list of location IDs
            
            # Filter eval_loc
            curr_eval_loc = self.eval_loc[l_indices]
            
            # sub_points = [self.points[idx] for idx in s_indices]
            sub_points = np.array([self.points[idx] for idx in s_indices], dtype=object)
            sub_source_l = self.source_l[s_indices]
            sub_source_r = self.source_r[s_indices]

            # Prepare Y (Magnitude in dB)
            # Conceptually: [len(s_indices), len(l_indices), n_freq, 2]
            # We need to slice self.HRTF_l which is [n_sub * n_loc, n_freq, 1]
            y_subset = np.empty((len(s_indices), len(l_indices), self.n_freq, 2), dtype=np.float32)
            
            for i, s_id in enumerate(s_indices):
                # Calculate start/end for this subject in the original flat HRTF array
                start = s_id * n_loc
                h_l = self.HRTF_l[start : start + n_loc][l_indices] # Slice locations
                h_r = self.HRTF_r[start : start + n_loc][l_indices]
                
                mag_l = 20 * np.log10(np.linalg.norm(h_l, axis=-1, keepdims=True) + 1e-8)
                mag_r = 20 * np.log10(np.linalg.norm(h_r, axis=-1, keepdims=True) + 1e-8)
                
                y_subset[i] = np.concatenate((mag_l, mag_r), axis=-1)
            
            X = (self.frequency, sub_points, curr_eval_loc, sub_source_l, sub_source_r)
            return X, y_subset

        # Generate the 4 sets
        # Train: sub_tr, L_seen
        ds_train = bundle_data(sub_tr_idx, L_seen_idx)
        
        # Test 1: sub_ts, L_seen (Unseen Subjects, Seen Locations)
        ds_test_1 = bundle_data(sub_ts_idx, L_seen_idx)
        
        # Test 2: sub_tr, L_unseen (Seen Subjects, Unseen Locations)
        ds_test_2 = bundle_data(sub_tr_idx, L_unseen_idx)
        
        # Test 3: sub_ts, L_unseen (Unseen Both)
        ds_test_3 = bundle_data(sub_ts_idx, L_unseen_idx)

        # Test 4: Unseen Subjects, ALL Locations
        ds_test_4 = bundle_data(sub_ts_idx, loc_indices)

        

        return ds_train, ds_test_1, ds_test_2, ds_test_3, ds_test_4,L_seen_idx, L_unseen_idx
    

class GenerateDataPoint_multi_split_2(GenerateDataBase):
    '''Generate seperate val set, and split the data into 4 sets based on subject and location splits.'''
    def __init__(self, HRTFDataset: HRTFDataset2):
        super().__init__(HRTFDataset)

    def generate_train_val(self):
        return

    def generate_data(self, test_size=0.1, val_size=0.1, loc_test_size=0.2, random_state=42):
        """
        Generates 5 datasets + 1 extra test set based on Subject and Location splits.
        Target: 180 Train subjects, 10 Val subjects, 10 Test subjects (assuming 200 total).
        """
        rng = np.random.default_rng(random_state)
        
        # 1. Split LOCATIONS (L_seen, L_unseen)
        n_loc = self.eval_loc.shape[0]
        loc_indices = np.arange(n_loc)
        L_seen_idx, L_unseen_idx = train_test_split(
            loc_indices, test_size=loc_test_size, random_state=random_state
        )
        
        # 2. Split SUBJECTS
        n_sub = self.n_sub
        sub_indices = np.arange(n_sub)
        
        # First: Separate out the Test Subjects (e.g., 10 subjects)
        # Remaining will be (Train + Val)
        sub_tv_idx, sub_ts_idx = train_test_split(
            sub_indices, test_size=test_size, random_state=random_state
        )
        
        # Second: Split the remaining (tv) into Train and Val
        # If you want exactly 10 subjects in Val, adjust val_size accordingly 
        # or use an absolute integer if your library supports it.
        sub_tr_idx, sub_val_idx = train_test_split(
            sub_tv_idx, test_size=val_size, random_state=random_state
        )

        # Helper to extract and format data (kept your original logic)
        def bundle_data(s_indices, l_indices):
            curr_eval_loc = self.eval_loc[l_indices]
            sub_points = np.array([self.points[idx] for idx in s_indices], dtype=object)
            sub_source_l = self.source_l[s_indices]
            sub_source_r = self.source_r[s_indices]

            y_subset = np.empty((len(s_indices), len(l_indices), self.n_freq, 2), dtype=np.float32)
            
            for i, s_id in enumerate(s_indices):
                start = s_id * n_loc
                # Slice locations from the flattened HRTF array
                h_l = self.HRTF_l[start : start + n_loc][l_indices] 
                h_r = self.HRTF_r[start : start + n_loc][l_indices]
                
                mag_l = 20 * np.log10(np.linalg.norm(h_l, axis=-1, keepdims=True) + 1e-8)
                mag_r = 20 * np.log10(np.linalg.norm(h_r, axis=-1, keepdims=True) + 1e-8)
                
                y_subset[i] = np.concatenate((mag_l, mag_r), axis=-1)
            
            X = (self.frequency, sub_points, curr_eval_loc, sub_source_l, sub_source_r)
            return X, y_subset

        # --- Generate the Sets ---
        
        # Train: sub_tr, L_seen (180 subs)
        ds_train = bundle_data(sub_tr_idx, L_seen_idx)
        
        # Validation: sub_val, L_seen (10 subs)
        ds_val = bundle_data(sub_val_idx, L_seen_idx)
        
        # Test 1: sub_ts, L_seen (Unseen Subjects, Seen Locations)
        ds_test_1 = bundle_data(sub_ts_idx, L_seen_idx)
        
        # Test 2: sub_tr, L_unseen (Seen Subjects, Unseen Locations)
        ds_test_2 = bundle_data(sub_tr_idx, L_unseen_idx)
        
        # Test 3: sub_ts, L_unseen (Unseen Both)
        ds_test_3 = bundle_data(sub_ts_idx, L_unseen_idx)

        # Test 4: Unseen Subjects, ALL Locations
        ds_test_4 = bundle_data(sub_ts_idx, loc_indices)

        return ds_train, ds_val, ds_test_1, ds_test_2, ds_test_3, ds_test_4, L_seen_idx, L_unseen_idx
    

if __name__ == "__main__":
    import os, sys
    from joblib import dump as joblib_dump
    # root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # if root not in sys.path:
    #     sys.path.insert(0, root)
    # pt_size=["15k","10k","5k","2k","1k"]
    pt_size = ["5k"]
    test_size = 0      # number of subjects in the test set for training data
    magnitude = True  # whether to generate magnitude only data
    train_data = True  # generate training/validation (and optionally test) data
    prediction_data = False  # generate only prediction data
    ref = True  # whether to include ref magnitude data
    point_format = "raw"  # "unique" or "repeat" or "raw"

    for pt in pt_size:
        arg_config = {
            "database": "AXD",  # "UHM_dense_train", "UHM_dense_test", "UHM", "AXD", "HUTUBUS","UHM_dense_noitd_train","UHM_noitd"
            "sub": "",                            # "small_" or ""
            "type": "msr_44_ff",                       # "msr" or "simu","msr_44_ff"
            "points": pt,
            "point_format": point_format,
        }   

        # if point_format == "raw":
        #     data_save_folder = (
        #     f"/app/Network/model_point/data/"
        #     f"{arg_config['database']}/AE/mag/raw/{arg_config['type']}/AES/"
        # )
        if point_format == "raw":
            data_save_folder = (
            f"/app/Network/VAE/data/AES/5sets/"
        )

        if not os.path.exists(data_save_folder):
            os.makedirs(data_save_folder)
            print(f"Created folder: {data_save_folder}")
        else:
            print(f"Folder already exists: {data_save_folder}")

        # Choose generator based on ITD flag
        dataset = HRTFDataset2(arg_config)
       
        # GeneratorCls = GenerateDataPoint_multi_split_2
        # generator = GeneratorCls(dataset)

        # # Generate raw data
        
        # ds_train, ds_test_1, ds_test_2, ds_test_3, ds_test_4,L_seen_idx, L_unseen_idx = generator.generate_data( val_size=0.05, loc_test_size=0.10)
        # X_train,y_train = ds_train
        # X_test_1, y_test_1 = ds_test_1
        # X_test_2, y_test_2 = ds_test_2
        # X_test_3, y_test_3 = ds_test_3
        # X_test_4, y_test_4 = ds_test_4

        # joblib_dump(X_train, data_save_folder + "X_train.sav")
        # joblib_dump(y_train, data_save_folder + "y_train.sav")

        # joblib_dump(X_test_1, data_save_folder + "X_test_1.sav")
        # joblib_dump(y_test_1, data_save_folder + "y_test_1.sav")

        # joblib_dump(X_test_2, data_save_folder + "X_test_2.sav")
        # joblib_dump(y_test_2, data_save_folder + "y_test_2.sav")

        # joblib_dump(X_test_3, data_save_folder + "X_test_3.sav")
        # joblib_dump(y_test_3, data_save_folder + "y_test_3.sav")

        # joblib_dump(X_test_4, data_save_folder + "X_test_4.sav")
        # joblib_dump(y_test_4, data_save_folder + "y_test_4.sav")

        # savemat(os.path.join(data_save_folder,'location_splits.mat'), {
        #     'L_seen_idx': L_seen_idx + 1, 
        #     'L_unseen_idx': L_unseen_idx + 1
        # })


        GeneratorCls = GenerateDataPoint_multi_split_2
        generator = GeneratorCls(dataset)

        # Generate raw data
        
        ds_train, ds_val, ds_test_1, ds_test_2, ds_test_3, ds_test_4, L_seen_idx, L_unseen_idx = generator.generate_data( test_size=10, val_size=10, loc_test_size=0.10)
        X_train,y_train = ds_train
        X_val, y_val = ds_val
        X_test_1, y_test_1 = ds_test_1
        X_test_2, y_test_2 = ds_test_2
        X_test_3, y_test_3 = ds_test_3
        X_test_4, y_test_4 = ds_test_4

        joblib_dump(X_train, data_save_folder + "X_train.sav")
        joblib_dump(y_train, data_save_folder + "y_train.sav")

        joblib_dump(X_val, data_save_folder + "X_val.sav")
        joblib_dump(y_val, data_save_folder + "y_val.sav")

        joblib_dump(X_test_1, data_save_folder + "X_test_1.sav")
        joblib_dump(y_test_1, data_save_folder + "y_test_1.sav")

        joblib_dump(X_test_2, data_save_folder + "X_test_2.sav")
        joblib_dump(y_test_2, data_save_folder + "y_test_2.sav")

        joblib_dump(X_test_3, data_save_folder + "X_test_3.sav")
        joblib_dump(y_test_3, data_save_folder + "y_test_3.sav")

        joblib_dump(X_test_4, data_save_folder + "X_test_4.sav")
        joblib_dump(y_test_4, data_save_folder + "y_test_4.sav")

        savemat(os.path.join(data_save_folder,'location_splits.mat'), {
            'L_seen_idx': L_seen_idx + 1, 
            'L_unseen_idx': L_unseen_idx + 1
        })

        
        print('Check dimension')