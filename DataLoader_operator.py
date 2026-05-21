import numpy as np
from utils import grading_resample
from scaler import Scaler_single, Scaler_single_subject
from scipy.spatial.distance import cdist
import torch
import trimesh
from trimesh.exchange.export import export_mesh
from sklearn.model_selection import train_test_split

def mirror_points(points:np.ndarray, norm=True)-> np.ndarray:
    """Mirror the point cloud along the sagittal plane (XZ plane).

    Args:
        points (np.ndarray): The input point cloud of shape (N, 6).

    Returns:
        np.ndarray: The mirrored point cloud of shape (N, 6).
    """
    mirrored_points = np.copy(points)
    mirrored_points[:, 1] = -mirrored_points[:, 1]  # Invert the y-coordinate
    # downsampledMesh = trimesh.Trimesh(mirrored_points[:,:3],)
    # export_mesh(downsampledMesh, "./Scaler/mirrored_Mesh_5k.ply")
    # downsampledMesh = trimesh.Trimesh(points[:,:3],)
    # export_mesh(downsampledMesh, "./Scaler/origin_Mesh_5k.ply")
    if norm:
        mirrored_points[:, 3:6] = mirrored_points[:, 3:6][::-1]  # Swap normals
    return mirrored_points

def mirror_hrtf_sagittal(X:np.ndarray, H:np.ndarray)-> np.ndarray:
    """
    Assume symmetry along the sagittal plane (XZ plane).
    X: [n_loc, 3]
    H: [n_sub, n_loc, n_fre, 2]  (L,R) in db
    Returns:
      X_m: [n_loc, 3] mirrored locations
      H_m: [n_sub, n_loc, n_fre, 2] mirrored HRTF, aligned with X_m
    """
    assert H.ndim == 4, "H must be of shape [n_sub, n_loc, n_fre, 2]"
    X = np.asarray(X)
    H = np.asarray(H)

    # mirror locations: (x,y,z)->(-x,y,z)
    X_m = X.copy()
    X_m[:, 1] *= -1

    # permutation: for each mirrored point, find nearest original point
    dist = cdist(X_m, X)          # [n_loc, n_loc]
    idx = dist.argmin(axis=1)     # [n_loc]
    
    err = np.linalg.norm(X_m - X[idx], axis=1)
    if err.max() > 1e-6:   # choose tolerance based on your X precision
        print("Warning: mirror grid mismatch, max error =", err.max())

    H_re = np.take(H, idx, axis=1)
    H_m = H_re[..., [1, 0]] 
    # swap ears + reorder locations
    # H_m = np.empty_like(H)
    # H_m[..., 0] = H[:, idx, :, 1]  # L <- R
    # H_m[..., 1] = H[:, idx, :, 0]  # R <- L
    return H_m

def test_mirror_hrtf_sagittal():
    """Assume symmetry along the sagittal plane (XZ plane).
    That is H_left(-y) = H_right(y) and H_right(-y) = H_left(y)"""
    # Create dummy data
    X = np.array([[1, -3, 0], [1, 3, 0], [0, 1, 3], [0, -1, 3]])
    H = np.random.rand(1, 4, 1, 2)  # 2 subjects, 4 locations, 10 frequencies, 2 ears
    
    H_m = mirror_hrtf_sagittal(X, H)
    print("Original H_left:\n", H[:, :, :, 0])
    print("Original H_right:\n", H[:, :, :, 1])
    print("Mirrored H_left:\n", H_m[:, :, :, 0])
    print("Mirrored H_right:\n", H_m[:, :, :, 1])

    # Check if the mirrored HRTF is correct
    for i in range(X.shape[0]):
        mirrored_loc = X[i].copy()
        mirrored_loc[1] *= -1
        dist = np.linalg.norm(X - mirrored_loc, axis=1)
        nearest_idx = np.argmin(dist)
        assert np.allclose(H_m[:, i, :, 0], H[:, nearest_idx, :, 1]), "Left ear mismatch"
        assert np.allclose(H_m[:, i, :, 1], H[:, nearest_idx, :, 0]), "Right ear mismatch"

    print("All tests passed!")

# def mirror_hrtf_sagittal(X:np.ndarray, H:np.ndarray)-> np.ndarray:
#     """
#     X: [n_loc, 3]
#     H: [n_sub, n_loc, n_fre, 2]  (L,R) in db
#     Returns:
#       X_m: [n_loc, 3] mirrored locations
#       H_m: [n_sub, n_loc, n_fre, 2] mirrored HRTF, aligned with X_m
#     """
#     X = np.asarray(X)
#     H = np.asarray(H)

#     # mirror locations: (x,y,z)->(-x,y,z)
#     X_m = X.copy()
#     X_m[:, 0] *= -1

#     # permutation: for each mirrored point, find nearest original point
#     H_m = np.empty_like(H)
#     for i in range(H.shape[0]):
#         for j in range(H.shape[1]):
#             for k in range(H.shape[1]):
#                 if X_m[j,0] == X[k,0] and X_m[j,1] == X[k,1] and X_m[j,2] == X[k,2]:
#                     H_m[i,j,:,0] = H[i,k,:,1]
#                     H_m[i,j,:,1] = H[i,k,:,0]

#     return H_m

class Preprocessor:
    def __init__(self,device=None):
        self.device = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    def preprocess_inputs(self, X, input_augment = True):
        x_fre, x_pt, x_eval, x_src_l, x_src_r = X[:5]
        x_ref = X[5] if len(X) == 6 else None

        n_train = len(x_pt)
        n_eval_locs = len(x_eval)
        n_freqs = len(x_fre)

        # x_pt_tr[i,0] : (N,6)  point cloud
        # x_pt_tr[i,1] : (2,3)  ear entrance locations (L,R)
        if isinstance(x_pt,np.ndarray):
            train_pts_cpu = [
                np.asarray(x_pt[i, 0], dtype=np.float32) for i in range(n_train)
            ]
            train_cent_cpu = [
                np.asarray(x_pt[i, 1], dtype=np.float32) for i in range(n_train)
            ]
            train_pts_cpu = np.asarray(train_pts_cpu)
            train_cent_cpu = np.asarray(train_cent_cpu)
        if isinstance(x_pt, list):
            train_pts_cpu = [
                np.asarray(x_pt[i][0], dtype=np.float32) for i in range(n_train)
            ]
            train_cent_cpu = [
                np.asarray(x_pt[i][1], dtype=np.float32) for i in range(n_train)
            ]
            train_pts_cpu = np.asarray(train_pts_cpu)
            train_cent_cpu = np.asarray(train_cent_cpu)


        train_src_l = x_src_l
        train_src_r = x_src_r
        if input_augment:
            train_pts_cpu_mir = [
                np.asarray(mirror_points(train_pts_cpu[i]), dtype=np.float32) for i in range(n_train)
            ]
            train_cent_cpu_mir = [
                np.asarray([pair[::-1] for pair in x_pt[i, 1]], dtype=np.float32) for i in range(n_train)
            ]
            train_cent_cpu = np.concatenate([train_cent_cpu, train_cent_cpu_mir], axis=0)
            train_pts_cpu = np.concatenate([train_pts_cpu, train_pts_cpu_mir], axis=0)
            train_src_l = np.concatenate([x_src_l,mirror_points(x_src_r,norm=False)] ,axis=0)
            train_src_r = np.concatenate([x_src_r,mirror_points(x_src_l,norm=False)] ,axis=0)
            if len(X) == 6:
                train_ref = self.preprocess_outputs(x_eval,x_ref)
        n_train = len(train_pts_cpu)
        train_fre = x_fre
        train_eval = x_eval
        if len(X) == 6:
            return (train_fre, train_pts_cpu, train_cent_cpu, train_eval, train_src_l, train_src_r, train_ref)
        if len(X) == 5:
            return (train_fre, train_pts_cpu, train_cent_cpu, train_eval, train_src_l, train_src_r)
        
    
    def preprocess_outputs(self, x_eval, y, input_augment = True):
        if input_augment:
            y_mir = mirror_hrtf_sagittal(x_eval, y)
            y = np.concatenate([y[...,0:1], y_mir[...,0:1]], axis=0)
        else:
            y = y[...,0:1]
        return y
    
    def get_scaler(self, y:np.ndarray, itd:np.ndarray|None, init_hrtf_scaler:bool=True, path:str="./Scaler/hrtf_scaler_temp"):
        """ y: np.ndarray :[n_sample, n_eval_loc, n_freqs, 1] """
        if init_hrtf_scaler:
            y_d = torch.from_numpy(y).to(device=self.device, dtype=torch.float32, non_blocking=True)
            scaler = Scaler_single(itd=itd, y=y_d,hrtf_scaler_path=path)
            scaler.init_scalers(True)
        else:
            scaler = Scaler_single(itd=None, y=None,hrtf_scaler_path=path)
            scaler.init_scalers(False)
        return scaler
    
    def scale_outputs(self, scaler:Scaler_single, y:np.ndarray):
        """ y: np.ndarray :[n_sample, n_eval_loc, n_freqs, 1] """
        y_d = torch.from_numpy(y).to(device=self.device, dtype=torch.float32, non_blocking=True)
        y_scaled_d = scaler.normalize_hrtf(y_d)
        y_scaled = y_scaled_d.cpu().numpy()
        return y_scaled

    

class BatchSampler:
    """Samples a mini-batch of indices.

    The indices are repeated indefinitely. Has the same effect as:

    .. code-block:: python

        indices = tf.data.Dataset.range(num_samples)
        indices = indices.repeat().shuffle(num_samples).batch(batch_size)
        iterator = iter(indices)
        batch_indices = iterator.get_next()

    However, ``tf.data.Dataset.__iter__()`` is only supported inside of ``tf.function`` or when eager execution is
    enabled. ``tf.data.Dataset.make_one_shot_iterator()`` supports graph mode, but is too slow.

    This class is not implemented as a Python Iterator, so that it can support dynamic batch size.

    Args:
        num_samples (int): The number of samples.
        shuffle (bool): Set to ``True`` to have the indices reshuffled at every epoch.
    """

    def __init__(self, num_samples, shuffle=True):
        self.num_samples = num_samples
        self.shuffle = shuffle

        self._indices = np.arange(self.num_samples)
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # Shuffle for the first epoch
        if shuffle:
            np.random.shuffle(self._indices)

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def get_next(self, batch_size):
        """Returns the indices of the next batch.

        Args:
            batch_size (int): The number of elements to combine in a single batch.
        """
        if batch_size > self.num_samples:
            raise ValueError(
                "batch_size={} is larger than num_samples={}.".format(
                    batch_size, self.num_samples
                )
            )

        start = self._index_in_epoch
        if start + batch_size <= self.num_samples:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._indices[start:end]
        else:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_samples = self.num_samples - start
            indices_rest_part = np.copy(
                self._indices[start : self.num_samples]
            )  # self._indices will be shuffled below.
            # Shuffle the indices
            if self.shuffle:
                np.random.shuffle(self._indices)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_samples
            end = self._index_in_epoch
            indices_new_part = self._indices[start:end]
            return np.hstack((indices_rest_part, indices_new_part))

class Dataloader():
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_resample=5000,
        sigma=0.6,
        test_batch_size=8,
    ):
        # ---------- TRAIN ----------
        x_fre_tr, x_pt_tr, x_eval_tr, x_src_l_tr, x_src_r_tr = X_train

        n_train = len(x_pt_tr)
        self.n_eval_locs = len(x_eval_tr)
        self.n_freqs = len(x_fre_tr)

        # x_pt_tr[i,0] : (N,6)  point cloud
        # x_pt_tr[i,1] : (2,3)  ear entrance locations (L,R)
        self.train_pts_cpu = [
            np.asarray(x_pt_tr[i, 0], dtype=np.float32) for i in range(n_train)
        ]
        self.train_cent_cpu = [
            np.asarray(x_pt_tr[i, 1], dtype=np.float32) for i in range(n_train)
        ]

        self.train_fre = x_fre_tr
        self.train_eval = x_eval_tr
        self.train_src_l = x_src_l_tr
        self.train_src_r = x_src_r_tr
        self.train_y = y_train

        # ---------- TEST ----------
        x_fre_te, x_pt_te, x_eval_te, x_src_l_te, x_src_r_te = X_test
        n_test = len(x_pt_te)

        self.test_pts_cpu = [
            np.asarray(x_pt_te[i, 0], dtype=np.float32) for i in range(n_test)
        ]
        self.test_cent_cpu = [
            np.asarray(x_pt_te[i, 1], dtype=np.float32) for i in range(n_test)
        ]

        self.test_fre = x_fre_te
        self.test_eval = x_eval_te
        self.test_src_l = x_src_l_te
        self.test_src_r = x_src_r_te
        self.test_y = y_test

        # ---------- SAMPLERS ----------
        self.branch_sampler = BatchSampler(n_train, shuffle=True)
        self.trunk_sampler = BatchSampler(self.n_eval_locs, shuffle=True)
        self.branch_sampler_test = BatchSampler(n_test, shuffle=False)

        self.dim = 1
        self.n_resample = int(n_resample)
        self.sigma = float(sigma)
        self.test_batch_size = int(test_batch_size)
        self.batch_size = None
        self.full_load_test = True

    def _stack_branch_batch(self, indices, train=True):
        pts_list = self.train_pts_cpu if train else self.test_pts_cpu
        cent_list = self.train_cent_cpu if train else self.test_cent_cpu

        pts = (np.stack([pts_list[i] for i in indices], axis=0))
        cent = (np.stack([cent_list[i] for i in indices], axis=0))
        # pts  : (B, N, 6)
        # cent : (B, 2, 3)
        return pts, cent

    # ------------------------------------------------------------------
    # RESAMPLE + NORMALIZE
    # ------------------------------------------------------------------
    def _resample_and_normalize(self, points, centers):
        """
        points  : (B, N, 6)
        centers : (B, 2, 3)
        """
        B = points.shape[0]
        out = np.empty((B, 6, self.n_resample), dtype=np.float32)

        for i in range(B):
            down = grading_resample(points[i], centers[i], self.n_resample)
            out[i] = down.T  # (6,N)

        # coordinate normalization
        xyz = out[:, :3, :]                      # (B,3,N)
        mu = xyz.mean(axis=2, keepdims=True)       # (B,3,1)
        xyz_c = xyz - mu
        scale = np.linalg.norm(xyz_c, axis=1).max(axis=1, keepdims=True)
        scale = scale[:, :, None].astype(np.float32)   # (B,1,1)

        xyz_n = xyz_c / scale
        resample = np.concatenate([xyz_n, out[:, 3:, :]], axis=1)

        return resample, mu, scale

    # ------------------------------------------------------------------
    # TRAIN BATCH
    # ------------------------------------------------------------------
    def train_next_batch(self, batch_size):
        self.batch_size = batch_size
        B_branch, B_trunk = batch_size

        idx_b = self.branch_sampler.get_next(B_branch)
        idx_t = self.trunk_sampler.get_next(B_trunk)


        # n_sample = self.train_y.shape[0]
        # n_freq = self.train_fre.shape[0]

        y = self.train_y.reshape(len(self.train_pts_cpu), self.n_eval_locs,self.n_freqs, self.dim)
        y = y[idx_b]
        y = y[:,idx_t,:]
        # y = y[idx_b][:, loc_indices, fre_indices] # Result: (B_branch, B_trunk, Dim)
        # y = y[idx_b][:,loc_indices,:]
        # y = y[:,:,fre_indices,:]
        y = y.reshape(B_branch*B_trunk, -1)

        pts, cent = self._stack_branch_batch(idx_b, train=True)
        resample, mu, scale = self._resample_and_normalize(pts, cent)

        eval_loc = self.train_eval[idx_t] # (B_trunk, 3)
        eval_loc = np.broadcast_to(eval_loc[None, :, :], (B_branch, B_trunk, 3)).copy()
        eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale
        mu_vec = mu[:, :, 0]                                       # (B,3)
        scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
        src_l = (self.train_src_l[idx_b] - mu_vec) / scale_vec
        src_r = (self.train_src_r[idx_b] - mu_vec) / scale_vec
       
        x_batch = (
            resample,
            eval_loc,
            src_l,
            src_r,
            mu.transpose(0, 2, 1),         # (B,1,3)
            np.reshape(scale,(B_branch,1)),  # (B,)
        )

        return x_batch, y

    # ------------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------------
    def test(self):

        if not self.full_load_test:
            B_branch = self.test_batch_size
            B_trunk = self.batch_size[1]

            idx_b = self.branch_sampler_test.get_next(B_branch)
            idx_t = self.trunk_sampler.get_next(B_trunk)

            y = self.test_y.reshape(len(self.test_pts_cpu), self.n_eval_locs,self.n_freqs, self.dim)
            y = y[idx_b]
            y = y[:,idx_t,:]
            # y = y[idx_b][:, loc_indices, fre_indices] # Result: (B_branch, B_trunk, Dim)
            # y = y[idx_b][:,loc_indices,:]
            # y = y[:,:,fre_indices,:]
            y = y.reshape(B_branch*B_trunk, -1)

            pts, cent = self._stack_branch_batch(idx_b, train=False)
            resample, mu, scale = self._resample_and_normalize(pts, cent)

            eval_loc = self.test_eval[idx_t] # (B_trunk, 3)
            eval_loc = np.broadcast_to(eval_loc[None, :, :], (B_branch, B_trunk, 3)).copy()
            eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale
            mu_vec = mu[:, :, 0]                                       # (B,3)
            scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
            src_l = (self.test_src_l[idx_b] - mu_vec) / scale_vec
            src_r = (self.test_src_r[idx_b] - mu_vec) / scale_vec

            x_batch = (
                resample,
                eval_loc,
                src_l,
                src_r,
                mu.transpose(0, 2, 1),         # (B,1,3)
                np.reshape(scale,(B_branch,1)),  # (B,)
            )
            return x_batch, y


        if self.full_load_test:
            B = self.test_batch_size
            n_sub = len(self.test_pts_cpu)
            if n_sub>50:
                n_sub = int(n_sub*0.5)

            xs, ys = [], []
            eval_loc_all = self.test_eval  # (n_loc,3)
            y = self.test_y.reshape(len(self.test_pts_cpu), self.n_eval_locs,self.n_freqs, self.dim)
            for start in range(0, n_sub, B):
                end = min(start + B, n_sub)
                idx = list(range(start, end))
                bsz = end - start
                pts, cent = self._stack_branch_batch(idx, train=False)
                resample, mu, scale = self._resample_and_normalize(pts, cent)

                eval_loc = np.broadcast_to(eval_loc_all[None, :, :], (bsz, self.n_eval_locs, 3)).copy()
                eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale

                mu_vec = mu[:, :, 0]                                       # (B,3)
                scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
                src_l = (self.test_src_l[idx] - mu_vec) / scale_vec
                src_r = (self.test_src_r[idx] - mu_vec) / scale_vec

                x = (
                    resample,
                    eval_loc,
                    src_l,
                    src_r,
                    mu.transpose(0, 2, 1),
                    np.reshape(scale,(bsz,1)),
                )
                xs.append(x)
                ys.append(y[idx])

            resample_all = np.concatenate([x[0] for x in xs], axis=0)
            eval_loc_all = np.concatenate([x[1] for x in xs], axis=0)
            src_l_all = np.concatenate([x[2] for x in xs], axis=0)
            src_r_all = np.concatenate([x[3] for x in xs], axis=0)
            mu_all = np.concatenate([x[4] for x in xs], axis=0)
            scale_all = np.concatenate([x[5] for x in xs], axis=0)

            x_all = (resample_all, eval_loc_all, src_l_all, src_r_all, mu_all, scale_all)
            y_all = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
            y_all = y_all.reshape(n_sub*self.n_eval_locs,-1)
            return x_all, y_all
        
class Dataloader_augment():
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_resample=5000,
        sigma=0.6,
        test_batch_size=8,
        preprocessor:Preprocessor=Preprocessor(),
        input_augment = False,
        dropout_loc = True,
    ):
        """ Input:
                   y_train: (n_sample, n_eval_loc, n_freqs, 2)
            returns: 
                   y_train: (2*n_sample, n_eval_loc, n_freqs, 1) if input_augement is True, 
                   y_train: (n_sample, n_eval_loc, n_freqs, 1) if input_augement is False,
                   dropout_loc: True, use all locations for training
                                False, randomly dropout locations 
        """
        self.dim = 1
        self.n_resample = int(n_resample)
        self.sigma = float(sigma)
        self.test_batch_size = int(test_batch_size)
        self.batch_size = None
        self.full_load_test = False
        self.dropout_loc = dropout_loc
        self.preprocessor = preprocessor
        
        
        # ---------- TRAIN ----------
        self.train_fre, self.train_pts_cpu, self.train_cent_cpu, self.train_eval, self.train_src_l, self.train_src_r = self.preprocessor.preprocess_inputs(X_train,input_augment=input_augment)
        self.y_train = self.preprocessor.preprocess_outputs(self.train_eval, y_train, input_augment=input_augment)
        n_train = len(self.train_pts_cpu)
        self.n_eval_locs = self.train_eval.shape[0]
        self.n_freqs = self.train_fre.shape[0]
    



        
        # ---------- TEST ----------
        _, self.test_pts_cpu, self.test_cent_cpu, self.test_eval, self.test_src_l, self.test_src_r = self.preprocessor.preprocess_inputs(X_test, input_augment=input_augment)
        n_test = len(self.test_pts_cpu)
        self.y_test = self.preprocessor.preprocess_outputs(self.test_eval, y_test,input_augment=input_augment)

        # ---------- SAMPLERS ----------
        self.branch_sampler = BatchSampler(n_train, shuffle=True)
        self.trunk_sampler = BatchSampler(self.n_eval_locs, shuffle=True)
        self.branch_sampler_test = BatchSampler(n_test, shuffle=True)
    

        # ---------- STANDARDIZER ----------
        self.scaler = self.preprocessor.get_scaler(self.y_train, None, init_hrtf_scaler=True)
        self.y_train = self.preprocessor.scale_outputs(self.scaler, self.y_train)
        self.y_test = self.preprocessor.scale_outputs(self.scaler, self.y_test)
        self.y_train = self.y_train.reshape(n_train,self.n_eval_locs,self.n_freqs, -1)
        self.y_test = self.y_test.reshape(n_test,self.n_eval_locs,self.n_freqs, -1)

        print(f"Number of train subjects: {n_train}")
        print(f"Number of validate subjects: {n_test}")

    def _stack_branch_batch(self, indices, train=True):
        pts_list = self.train_pts_cpu if train else self.test_pts_cpu
        cent_list = self.train_cent_cpu if train else self.test_cent_cpu

        pts = (np.stack([pts_list[i] for i in indices], axis=0))
        cent = (np.stack([cent_list[i] for i in indices], axis=0))
        # pts  : (B, N, 6)
        # cent : (B, 2, 3)
        return pts, cent

    # ------------------------------------------------------------------
    # RESAMPLE + NORMALIZE
    # ------------------------------------------------------------------
    def _resample_and_normalize(self, points, centers):
        """
        points  : (B, N, 6)
        centers : (B, 2, 3)
        """
        B = points.shape[0]
        out = np.empty((B, 6, self.n_resample), dtype=np.float32)

        for i in range(B):
            down = grading_resample(points=points[i], centers=centers[i], M=self.n_resample, side="both",sigma=self.sigma)
            # downsampledMesh = trimesh.Trimesh(down[:,:3],)
            # export_mesh(downsampledMesh, "./Scaler/single_side_downsampledMesh_5k.ply")
            out[i] = down.T  # (6,N)
            


        # coordinate normalization
        xyz = out[:, :3, :]                      # (B,3,N)
        mu = xyz.mean(axis=2, keepdims=True)       # (B,3,1)
        xyz_c = xyz - mu
        scale = np.linalg.norm(xyz_c, axis=1).max(axis=1, keepdims=True)
        scale = scale[:, :, None].astype(np.float32)   # (B,1,1)

        xyz_n = xyz_c / scale
        resample = np.concatenate([xyz_n, out[:, 3:, :]], axis=1)
        # resample = np.concatenate([xyz/1.5, out[:, 3:, :]], axis=1)

        return resample, mu, scale

    # ------------------------------------------------------------------
    # TRAIN BATCH
    # ------------------------------------------------------------------
    # def train_next_batch(self, batch_size):
    #     self.batch_size = batch_size
    #     B_branch, B_trunk = batch_size

    #     idx_b = self.branch_sampler.get_next(B_branch)
    #     idx_t = self.trunk_sampler.get_next(B_trunk)


    #     # n_sample = self.train_y.shape[0]
    #     # n_freq = self.train_fre.shape[0]

    #     y = self.y_train.reshape(len(self.train_pts_cpu), self.n_eval_locs,self.n_freqs, self.dim)
    #     y = y[idx_b]
    #     y = y[:,idx_t,:]
    #     # y = y[idx_b][:, loc_indices, fre_indices] # Result: (B_branch, B_trunk, Dim)
    #     # y = y[idx_b][:,loc_indices,:]
    #     # y = y[:,:,fre_indices,:]
    #     y = y.reshape(B_branch*B_trunk, -1)

    #     pts, cent = self._stack_branch_batch(idx_b, train=True)
    #     resample, mu, scale = self._resample_and_normalize(pts, cent)

    #     eval_loc = self.train_eval[idx_t] # (B_trunk, 3)
    #     eval_loc = np.broadcast_to(eval_loc[None, :, :], (B_branch, B_trunk, 3)).copy()
    #     eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale
    #     mu_vec = mu[:, :, 0]                                       # (B,3)
    #     scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
    #     src_l = (self.train_src_l[idx_b] - mu_vec) / scale_vec
    #     src_r = (self.train_src_r[idx_b] - mu_vec) / scale_vec
       
    #     x_batch = (
    #         resample,
    #         eval_loc,
    #         src_l,
    #         src_r,
    #         mu.transpose(0, 2, 1),         # (B,1,3)
    #         np.reshape(scale,(B_branch,1)),  # (B,)
    #     )

    #     return x_batch, y
    
    def train_next_batch(self, batch_size):
        B_branch, B_trunk = batch_size
        self.batch_size = batch_size
        idx_b = self.branch_sampler.get_next(B_branch)
        if not self.dropout_loc:
            idx_t = self.trunk_sampler.get_next(B_trunk)
            idx_t_mat = np.tile(idx_t[None, :], (B_branch, 1))
            y = self.y_train.reshape(len(self.train_pts_cpu), self.n_eval_locs, self.n_freqs, self.dim)
            y = y[idx_b]  # (B_branch, n_eval_locs, n_freqs, dim)
            y = y[:, idx_t, :, :]  # (B_branch, B_trunk, n_freqs, dim)
            y = y.reshape(B_branch * B_trunk, -1)
            eval_loc = self.train_eval[idx_t_mat]  # (B_branch, B_trunk, 3)
        elif self.dropout_loc: 
            idx_t_mat = np.zeros((B_branch, B_trunk), dtype=int)
            for i in range(B_branch):
                idx_t_mat[i] = np.random.choice(self.n_eval_locs, size=B_trunk, replace=False)
            y = self.y_train.reshape(len(self.train_pts_cpu), self.n_eval_locs, self.n_freqs, self.dim)
            b_idx = np.arange(B_branch)[:, None] 
            y = y[idx_b[:, None], idx_t_mat] # Result: (B_branch, B_trunk, n_freqs, dim)
            y = y.reshape(B_branch * B_trunk, -1)
            eval_loc = self.train_eval[idx_t_mat]  # (B_branch, B_trunk, 3)


        pts, cent = self._stack_branch_batch(idx_b, train=True)
        resample, mu, scale = self._resample_and_normalize(pts, cent)

        

        # Normalize: mu_t has shape (B_branch, 1, 3); scale broadcasts if it's (B_branch,1,1) or (B_branch,1)
        mu_t = mu.transpose(0, 2, 1)           # (B_branch, 1, 3)
        eval_loc = (eval_loc - mu_t) / scale   # broadcasts over trunk dimension
        # eval_loc = (eval_loc - mu_t) / 1.5   # broadcasts over trunk dimension

        mu_vec = mu[:, :, 0]                     # (B_branch, 3)
        scale_vec = scale[:, 0, 0][:, None]      # (B_branch, 1)
        src_l = (self.train_src_l[idx_b] - mu_vec) / scale_vec
        src_r = (self.train_src_r[idx_b] - mu_vec) / scale_vec

        x_batch = (
            resample,
            eval_loc,                 # (B_branch, B_trunk, 3) now unique per branch
            src_l,
            src_r,
            mu_t,                     # (B_branch, 1, 3)
            np.reshape(scale, (B_branch, 1)),  # (B_branch, 1)
        )

        return x_batch, y

    # ------------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------------
    def test(self):

        if not self.full_load_test:
            B_branch = self.test_batch_size
            B_trunk = self.batch_size[1]

            idx_b = self.branch_sampler_test.get_next(B_branch)
            idx_t = self.trunk_sampler.get_next(B_trunk)

            y = self.y_test.reshape(len(self.test_pts_cpu), self.n_eval_locs,self.n_freqs, self.dim)
            y = y[idx_b]
            y = y[:,idx_t,:]
            # y = y[idx_b][:, loc_indices, fre_indices] # Result: (B_branch, B_trunk, Dim)
            # y = y[idx_b][:,loc_indices,:]
            # y = y[:,:,fre_indices,:]
            y = y.reshape(B_branch*B_trunk, -1)

            pts, cent = self._stack_branch_batch(idx_b, train=False)
            resample, mu, scale = self._resample_and_normalize(pts, cent)

            eval_loc = self.test_eval[idx_t] # (B_trunk, 3)
            eval_loc = np.broadcast_to(eval_loc[None, :, :], (B_branch, B_trunk, 3)).copy()
            eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale
            # eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / 1.5
            mu_vec = mu[:, :, 0]                                       # (B,3)
            scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
            src_l = (self.test_src_l[idx_b] - mu_vec) / scale_vec
            src_r = (self.test_src_r[idx_b] - mu_vec) / scale_vec

            x_batch = (
                resample,
                eval_loc,
                src_l,
                src_r,
                mu.transpose(0, 2, 1),         # (B,1,3)
                np.reshape(scale,(B_branch,1)),  # (B,)
            )
            return x_batch, y


        if self.full_load_test:
            B = self.test_batch_size
            n_sub = len(self.test_pts_cpu)
            if n_sub>50:
                n_sub = int(n_sub*0.5)

            xs, ys = [], []
            eval_loc_all = self.test_eval  # (n_loc,3)
            y = self.y_test.reshape(len(self.test_pts_cpu), self.n_eval_locs,self.n_freqs, self.dim)
            for start in range(0, n_sub, B):
                end = min(start + B, n_sub)
                idx = list(range(start, end))
                bsz = end - start
                pts, cent = self._stack_branch_batch(idx, train=False)
                resample, mu, scale = self._resample_and_normalize(pts, cent)

                eval_loc = np.broadcast_to(eval_loc_all[None, :, :], (bsz, self.n_eval_locs, 3)).copy()
                eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale

                mu_vec = mu[:, :, 0]                                       # (B,3)
                scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
                src_l = (self.test_src_l[idx] - mu_vec) / scale_vec
                src_r = (self.test_src_r[idx] - mu_vec) / scale_vec

                x = (
                    resample,
                    eval_loc,
                    src_l,
                    src_r,
                    mu.transpose(0, 2, 1),
                    np.reshape(scale,(bsz,1)),
                )
                xs.append(x)
                ys.append(y[idx])

            resample_all = np.concatenate([x[0] for x in xs], axis=0)
            eval_loc_all = np.concatenate([x[1] for x in xs], axis=0)
            src_l_all = np.concatenate([x[2] for x in xs], axis=0)
            src_r_all = np.concatenate([x[3] for x in xs], axis=0)
            mu_all = np.concatenate([x[4] for x in xs], axis=0)
            scale_all = np.concatenate([x[5] for x in xs], axis=0)

            x_all = (resample_all, eval_loc_all, src_l_all, src_r_all, mu_all, scale_all)
            y_all = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
            y_all = y_all.reshape(n_sub*self.n_eval_locs,-1)
            return x_all, y_all
        
class Dataloader_augment_ref():
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_resample=5000,
        sigma=0.6,
        test_batch_size=8,
        preprocessor:Preprocessor=Preprocessor(),
        input_augment = True
    ):
        """ Input:
                   y_train: (n_sample, n_eval_loc, n_freqs, 2)
            returns: 
                   y_train: (2*n_sample, n_eval_loc, n_freqs, 1)"""
        self.dim = 1
        self.n_resample = int(n_resample)
        self.sigma = float(sigma)
        self.test_batch_size = int(test_batch_size)
        self.batch_size = None
        self.full_load_test = False
        self.preprocessor = preprocessor
        
        
        # ---------- TRAIN ----------
        self.train_fre, self.train_pts_cpu, self.train_cent_cpu, self.train_eval, self.train_src_l, self.train_src_r,self.train_ref = self.preprocessor.preprocess_inputs(X_train,input_augment=input_augment)
        self.y_train = self.preprocessor.preprocess_outputs(self.train_eval, y_train, input_augment=input_augment)
        n_train = len(self.train_pts_cpu)
        self.n_eval_locs = self.train_eval.shape[0]
        self.n_freqs = self.train_fre.shape[0]
    



        
        # ---------- TEST ----------
        _, self.test_pts_cpu, self.test_cent_cpu, self.test_eval, self.test_src_l, self.test_src_r, self.test_ref = self.preprocessor.preprocess_inputs(X_test, input_augment=input_augment)
        n_test = len(self.test_pts_cpu)
        self.y_test = self.preprocessor.preprocess_outputs(self.test_eval, y_test,input_augment=input_augment)

        # ---------- SAMPLERS ----------
        self.branch_sampler = BatchSampler(n_train, shuffle=True)
        self.trunk_sampler = BatchSampler(self.n_eval_locs, shuffle=True)
        self.branch_sampler_test = BatchSampler(n_test, shuffle=True)
    

        # ---------- STANDARDIZER ----------
        self.scaler_HRTF = self.preprocessor.get_scaler(self.y_train, None, init_hrtf_scaler=True, path="./Scaler/hrtf_scaler_temp" )
        self.y_train = self.preprocessor.scale_outputs(self.scaler_HRTF, self.y_train)
        self.y_test = self.preprocessor.scale_outputs(self.scaler_HRTF, self.y_test)
        self.y_train = self.y_train.reshape(n_train,self.n_eval_locs,self.n_freqs, -1)
        self.y_test = self.y_test.reshape(n_test,self.n_eval_locs,self.n_freqs, -1)

        self.scaler_ref = self.preprocessor.get_scaler(self.train_ref, None, init_hrtf_scaler=True, path="./Scaler/ref_scaler_temp")
        self.train_ref = self.preprocessor.scale_outputs(self.scaler_ref, self.train_ref)
        self.test_ref = self.preprocessor.scale_outputs(self.scaler_ref, self.test_ref)
        self.train_fre = self.y_train.reshape(n_train,self.n_eval_locs,self.n_freqs, -1)
        self.test_ref = self.y_test.reshape(n_test,self.n_eval_locs,self.n_freqs, -1)

        print(f"Number of train subjects: {n_train}")
        print(f"Number of validate subjects: {n_test}")

    def _stack_branch_batch(self, indices, train=True):
        pts_list = self.train_pts_cpu if train else self.test_pts_cpu
        cent_list = self.train_cent_cpu if train else self.test_cent_cpu

        pts = (np.stack([pts_list[i] for i in indices], axis=0))
        cent = (np.stack([cent_list[i] for i in indices], axis=0))
        # pts  : (B, N, 6)
        # cent : (B, 2, 3)
        return pts, cent

    # ------------------------------------------------------------------
    # RESAMPLE + NORMALIZE
    # ------------------------------------------------------------------
    def _resample_and_normalize(self, points, centers):
        """
        points  : (B, N, 6)
        centers : (B, 2, 3)
        """
        B = points.shape[0]
        out = np.empty((B, 6, self.n_resample), dtype=np.float32)

        for i in range(B):
            down = grading_resample(points=points[i], centers=centers[i], M=self.n_resample, side="both",sigma=self.sigma)
            # downsampledMesh = trimesh.Trimesh(down[:,:3],)
            # export_mesh(downsampledMesh, "./Scaler/single_side_downsampledMesh_5k.ply")
            out[i] = down.T  # (6,N)
            


        # coordinate normalization
        xyz = out[:, :3, :]                      # (B,3,N)
        mu = xyz.mean(axis=2, keepdims=True)       # (B,3,1)
        xyz_c = xyz - mu
        scale = np.linalg.norm(xyz_c, axis=1).max(axis=1, keepdims=True)
        scale = scale[:, :, None].astype(np.float32)   # (B,1,1)

        xyz_n = xyz_c / scale
        resample = np.concatenate([xyz_n, out[:, 3:, :]], axis=1)

        return resample, mu, scale

    # ------------------------------------------------------------------
    # TRAIN BATCH
    # ------------------------------------------------------------------
    # def train_next_batch(self, batch_size):
    #     self.batch_size = batch_size
    #     B_branch, B_trunk = batch_size

    #     idx_b = self.branch_sampler.get_next(B_branch)
    #     idx_t = self.trunk_sampler.get_next(B_trunk)


    #     # n_sample = self.train_y.shape[0]
    #     # n_freq = self.train_fre.shape[0]

    #     y = self.y_train.reshape(len(self.train_pts_cpu), self.n_eval_locs,self.n_freqs, self.dim)
    #     y = y[idx_b]
    #     y = y[:,idx_t,:]
    #     # y = y[idx_b][:, loc_indices, fre_indices] # Result: (B_branch, B_trunk, Dim)
    #     # y = y[idx_b][:,loc_indices,:]
    #     # y = y[:,:,fre_indices,:]
    #     y = y.reshape(B_branch*B_trunk, -1)

    #     pts, cent = self._stack_branch_batch(idx_b, train=True)
    #     resample, mu, scale = self._resample_and_normalize(pts, cent)

    #     eval_loc = self.train_eval[idx_t] # (B_trunk, 3)
    #     eval_loc = np.broadcast_to(eval_loc[None, :, :], (B_branch, B_trunk, 3)).copy()
    #     eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale
    #     mu_vec = mu[:, :, 0]                                       # (B,3)
    #     scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
    #     src_l = (self.train_src_l[idx_b] - mu_vec) / scale_vec
    #     src_r = (self.train_src_r[idx_b] - mu_vec) / scale_vec
       
    #     x_batch = (
    #         resample,
    #         eval_loc,
    #         src_l,
    #         src_r,
    #         mu.transpose(0, 2, 1),         # (B,1,3)
    #         np.reshape(scale,(B_branch,1)),  # (B,)
    #     )

    #     return x_batch, y
    
    def train_next_batch(self, batch_size):
        B_branch, B_trunk = batch_size
        self.batch_size = batch_size
        idx_b = self.branch_sampler.get_next(B_branch)

        # Sample different trunk indices for each branch element
        # idx_t_mat[b, j] is the j-th trunk location for branch sample b
        # idx_t_mat = np.stack(
        #     [np.random.choice(self.n_eval_locs, size=B_trunk, replace=False)
        #     for _ in range(B_branch)],
        #     axis=0
        # )
        # idx_t_mat = np.random.randint(0, self.n_eval_locs, size=(B_branch, B_trunk))
        idx_t = self.trunk_sampler.get_next(B_trunk)
        idx_t_mat = np.tile(idx_t[None, :], (B_branch, 1))

        y = self.y_train.reshape(len(self.train_pts_cpu), self.n_eval_locs, self.n_freqs, self.dim)
        y = y[idx_b]  # (B_branch, n_eval_locs, n_freqs, dim)
        # if idx_t_mat is identical across branch (your current tile case):
        y = y[:, idx_t, :, :]  # (B_branch, B_trunk, n_freqs, dim)
        y = y.reshape(B_branch * B_trunk, -1)

        ref = self.train_fre.reshape(len(self.train_pts_cpu), self.n_eval_locs, self.n_freqs, self.dim)
        ref = ref[idx_b]  # (B_branch, n_eval_locs, n_freqs, dim)
        # if idx_t_mat is identical across branch (your current tile case):
        ref = ref[:, idx_t, :, :]  # (B_branch, B_trunk, n_freqs, dim)
        ref = ref.reshape(B_branch * B_trunk, -1)

        pts, cent = self._stack_branch_batch(idx_b, train=True)
        resample, mu, scale = self._resample_and_normalize(pts, cent)

        # Now eval_loc is different per branch element, no broadcast needed
        eval_loc = self.train_eval[idx_t_mat]  # (B_branch, B_trunk, 3)

        # Normalize: mu_t has shape (B_branch, 1, 3); scale broadcasts if it's (B_branch,1,1) or (B_branch,1)
        mu_t = mu.transpose(0, 2, 1)           # (B_branch, 1, 3)
        eval_loc = (eval_loc - mu_t) / scale   # broadcasts over trunk dimension

        mu_vec = mu[:, :, 0]                     # (B_branch, 3)
        scale_vec = scale[:, 0, 0][:, None]      # (B_branch, 1)
        # src_l = (self.train_src_l[idx_b] - mu_vec) / scale_vec
        # src_r = (self.train_src_r[idx_b] - mu_vec) / scale_vec
        src_l = (cent[:,0,:] - mu_vec) / scale_vec
        src_r = (cent[:,1,:] - mu_vec) / scale_vec

        x_batch = (
            resample,
            eval_loc,                 # (B_branch, B_trunk, 3) now unique per branch
            src_l,
            src_r,
            mu_t,                     # (B_branch, 1, 3)
            np.reshape(scale, (B_branch, 1)),  # (B_branch, 1)
            ref
        )

        return x_batch, y

    # ------------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------------
    def test(self):

        if not self.full_load_test:
            B_branch = self.test_batch_size
            B_trunk = self.batch_size[1]

            idx_b = self.branch_sampler_test.get_next(B_branch)
            idx_t = self.trunk_sampler.get_next(B_trunk)

            y = self.y_test.reshape(len(self.test_pts_cpu), self.n_eval_locs,self.n_freqs, self.dim)
            y = y[idx_b]
            y = y[:,idx_t,:]
            y = y.reshape(B_branch*B_trunk, -1)


            ref = self.test_ref.reshape(len(self.test_pts_cpu), self.n_eval_locs,self.n_freqs, self.dim)
            ref = ref[idx_b]
            ref = ref[:,idx_t,:]
            ref = ref.reshape(B_branch*B_trunk, -1)

            pts, cent = self._stack_branch_batch(idx_b, train=False)
            resample, mu, scale = self._resample_and_normalize(pts, cent)

            eval_loc = self.test_eval[idx_t] # (B_trunk, 3)
            eval_loc = np.broadcast_to(eval_loc[None, :, :], (B_branch, B_trunk, 3)).copy()
            eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale
            mu_vec = mu[:, :, 0]                                       # (B,3)
            scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
            src_l = (cent[:,0,:] - mu_vec) / scale_vec
            src_r = (cent[:,0,:] - mu_vec) / scale_vec

            x_batch = (
                resample,
                eval_loc,
                src_l,
                src_r,
                mu.transpose(0, 2, 1),         # (B,1,3)
                np.reshape(scale,(B_branch,1)),  # (B,)
                ref
            )
            return x_batch, y


        if self.full_load_test:
            B = self.test_batch_size
            n_sub = len(self.test_pts_cpu)
            if n_sub>50:
                n_sub = int(n_sub*0.5)

            xs, ys = [], []
            eval_loc_all = self.test_eval  # (n_loc,3)
            y = self.y_test.reshape(len(self.test_pts_cpu), self.n_eval_locs,self.n_freqs, self.dim)
            ref = self.test_ref.reshape(len(self.test_pts_cpu), self.n_eval_locs,self.n_freqs, self.dim)
            for start in range(0, n_sub, B):
                end = min(start + B, n_sub)
                idx = list(range(start, end))
                bsz = end - start
                pts, cent = self._stack_branch_batch(idx, train=False)
                resample, mu, scale = self._resample_and_normalize(pts, cent)

                eval_loc = np.broadcast_to(eval_loc_all[None, :, :], (bsz, self.n_eval_locs, 3)).copy()
                eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale

                mu_vec = mu[:, :, 0]                                       # (B,3)
                scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
                src_l = (cent - mu_vec) / scale_vec
                src_r = (cent - mu_vec) / scale_vec

                x = (
                    resample,
                    eval_loc,
                    src_l,
                    src_r,
                    mu.transpose(0, 2, 1),
                    np.reshape(scale,(bsz,1)),
                    ref
                )
                xs.append(x)
                ys.append(y[idx])

            resample_all = np.concatenate([x[0] for x in xs], axis=0)
            eval_loc_all = np.concatenate([x[1] for x in xs], axis=0)
            src_l_all = np.concatenate([x[2] for x in xs], axis=0)
            src_r_all = np.concatenate([x[3] for x in xs], axis=0)
            mu_all = np.concatenate([x[4] for x in xs], axis=0)
            scale_all = np.concatenate([x[5] for x in xs], axis=0)
            ref_all = np.concatenate([x[6] for x in xs], axis=0)

            x_all = (resample_all, eval_loc_all, src_l_all, src_r_all, mu_all, scale_all,ref_all)
            y_all = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
            y_all = y_all.reshape(n_sub*self.n_eval_locs,-1)
            return x_all, y_all
        
class Dataloader_augment_loc_split():
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_resample=5000,
        sigma=0.6,
        test_batch_size=8,
        preprocessor:Preprocessor=Preprocessor(),
        input_augment = True
    ):
        """ Input:
                   y_train: (n_sample, n_eval_loc, n_freqs, 2)
            returns: 
                   y_train: (2*n_sample, n_eval_loc, n_freqs, 1)"""
        self.dim = 1
        self.n_resample = int(n_resample)
        self.sigma = float(sigma)
        self.test_batch_size = int(test_batch_size)
        self.batch_size = None
        self.full_load_test = False
        self.preprocessor = preprocessor


        # ---------- preprocess ----------
        self.train_fre, self.train_pts_cpu, self.train_cent_cpu, self.train_eval, self.train_src_l, self.train_src_r = self.preprocessor.preprocess_inputs(X_train,input_augment=input_augment)
        self.y_train = self.preprocessor.preprocess_outputs(self.train_eval, y_train, input_augment=input_augment)
        # _, self.test_pts_cpu, self.test_cent_cpu, self.test_eval, self.test_src_l, self.test_src_r = self.preprocessor.preprocess_inputs(X_test, input_augment=input_augment)
        # self.y_test = self.preprocessor.preprocess_outputs(self.test_eval, y_test,input_augment=input_augment)
        _, self.test_pts_cpu, self.test_cent_cpu, self.test_eval, self.test_src_l, self.test_src_r = self.preprocessor.preprocess_inputs(X_train,input_augment=input_augment)
        self.y_test = self.y_train.copy()
        n_train = len(self.train_pts_cpu)
        n_test = len(self.test_pts_cpu)
        self.n_freqs = self.train_fre.shape[0]
        self.eval_idx_global = np.arange(self.train_eval.shape[0])

        #---------- split train test location ----------
        self.eval_loc_idx_train, self.eval_loc_idx_test = train_test_split(
            self.eval_idx_global,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        self.eval_loc_train = self.train_eval[self.eval_loc_idx_train]
        self.eval_loc_test  = self.train_eval[self.eval_loc_idx_test]
        self.n_eval_locs_train = self.eval_loc_idx_train.shape[0]
        self.n_eval_locs_test = self.eval_loc_idx_test.shape[0]

        # ---------- SAMPLERS ----------
        self.branch_sampler = BatchSampler(n_train, shuffle=True)
        self.branch_sampler_test = BatchSampler(n_test, shuffle=False)

        self.trunk_sampler = BatchSampler(self.n_eval_locs_train, shuffle=True)
        self.trunk_sampler_test = BatchSampler(self.n_eval_locs_test, shuffle=False)
    

        # ---------- STANDARDIZER ----------
        self.scaler = self.preprocessor.get_scaler(self.y_train, None, init_hrtf_scaler=True)
        self.y_train = self.preprocessor.scale_outputs(self.scaler, self.y_train)
        self.y_test = self.preprocessor.scale_outputs(self.scaler, self.y_test)
        self.y_train = self.y_train.reshape(n_train,self.eval_idx_global.shape[0],self.n_freqs, -1)
        self.y_test = self.y_test.reshape(n_test,self.eval_idx_global.shape[0],self.n_freqs, -1)

        #---------- split train test HRTF ----------
        self.y_train = self.y_train[:,self.eval_loc_idx_train,:,:]
        self.y_test = self.y_test[:,self.eval_loc_idx_test,:,:]

        print(f"Number of train subjects: {n_train}")
        print(f"Number of validate subjects: {n_test}")
        print(f"Number of train locs: {self.n_eval_locs_train}")
        print(f"Number of validate locs: {self.n_eval_locs_test}")

    def _stack_branch_batch(self, indices, train=True):
        pts_list = self.train_pts_cpu if train else self.test_pts_cpu
        cent_list = self.train_cent_cpu if train else self.test_cent_cpu

        pts = (np.stack([pts_list[i] for i in indices], axis=0))
        cent = (np.stack([cent_list[i] for i in indices], axis=0))
        # pts  : (B, N, 6)
        # cent : (B, 2, 3)
        return pts, cent

    # ------------------------------------------------------------------
    # RESAMPLE + NORMALIZE
    # ------------------------------------------------------------------
    def _resample_and_normalize(self, points, centers):
        """
        points  : (B, N, 6)
        centers : (B, 2, 3)
        """
        B = points.shape[0]
        out = np.empty((B, 6, self.n_resample), dtype=np.float32)

        for i in range(B):
            down = grading_resample(points=points[i], centers=centers[i], M=self.n_resample, side="both",sigma=self.sigma)
            # downsampledMesh = trimesh.Trimesh(down[:,:3],)
            # export_mesh(downsampledMesh, "./Scaler/single_side_downsampledMesh_5k.ply")
            out[i] = down.T  # (6,N)
            


        # coordinate normalization
        xyz = out[:, :3, :]                      # (B,3,N)
        mu = xyz.mean(axis=2, keepdims=True)       # (B,3,1)
        xyz_c = xyz - mu
        scale = np.linalg.norm(xyz_c, axis=1).max(axis=1, keepdims=True)
        scale = scale[:, :, None].astype(np.float32)   # (B,1,1)

        xyz_n = xyz_c / scale
        resample = np.concatenate([xyz_n, out[:, 3:, :]], axis=1)

        return resample, mu, scale

    # ------------------------------------------------------------------
    # TRAIN BATCH
    # ------------------------------------------------------------------
    def train_next_batch(self, batch_size):
        B_branch, B_trunk = batch_size
        self.batch_size = batch_size
        idx_b = self.branch_sampler.get_next(B_branch)

        idx_t = self.trunk_sampler.get_next(B_trunk)
        idx_loc_mat = np.tile(idx_t[None, :], (B_branch, 1))

        y = self.y_train[idx_b]  # (B_branch, n_eval_locs, n_freqs, dim)

        # Gather along the location axis (axis=1) using idx_t_mat
        # Build index array with shape (B_branch, B_trunk, 1, 1) for broadcasting
        idx_loc_mat = idx_loc_mat[:, :, None, None]
        y = np.take_along_axis(y, idx_loc_mat, axis=1)  # (B_branch, B_trunk, n_freqs, dim)

        y = y.reshape(B_branch * B_trunk, -1)

        pts, cent = self._stack_branch_batch(idx_b, train=True)
        resample, mu, scale = self._resample_and_normalize(pts, cent)

        # Now eval_loc is different per branch element, no broadcast needed
        eval_loc = self.eval_loc_train[idx_t]  # (B_branch, B_trunk, 3)

        # Normalize: mu_t has shape (B_branch, 1, 3); scale broadcasts if it's (B_branch,1,1) or (B_branch,1)
        mu_t = mu.transpose(0, 2, 1)           # (B_branch, 1, 3)
        eval_loc = (eval_loc - mu_t) / scale   # broadcasts over trunk dimension

        mu_vec = mu[:, :, 0]                     # (B_branch, 3)
        scale_vec = scale[:, 0, 0][:, None]      # (B_branch, 1)
        src_l = (self.train_src_l[idx_b] - mu_vec) / scale_vec
        src_r = (self.train_src_r[idx_b] - mu_vec) / scale_vec

        x_batch = (
            resample,
            eval_loc,                 # (B_branch, B_trunk, 3) now unique per branch
            src_l,
            src_r,
            mu_t,                     # (B_branch, 1, 3)
            np.reshape(scale, (B_branch, 1)),  # (B_branch, 1)
        )

        return x_batch, y

    # ------------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------------
    def test(self):

        if not self.full_load_test:
            B_branch = self.test_batch_size
            B_trunk = self.n_eval_locs_test

            idx_b = self.branch_sampler_test.get_next(B_branch)
            y = self.y_test
            y = y[idx_b,:,:,:]  # (B_branch, n_eval_locs_test, n_freqs, dim)
            y = y.reshape(B_branch*B_trunk, -1)

            pts, cent = self._stack_branch_batch(idx_b, train=False)
            resample, mu, scale = self._resample_and_normalize(pts, cent)

            eval_loc =  self.eval_loc_test # (B_trunk, 3)
            eval_loc = np.broadcast_to(eval_loc[None, :, :], (B_branch, B_trunk, 3)).copy()
            eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale
            mu_vec = mu[:, :, 0]                                       # (B,3)
            scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
            src_l = (self.test_src_l[idx_b] - mu_vec) / scale_vec
            src_r = (self.test_src_r[idx_b] - mu_vec) / scale_vec

            x_batch = (
                resample,
                eval_loc,
                src_l,
                src_r,
                mu.transpose(0, 2, 1),         # (B,1,3)
                np.reshape(scale,(B_branch,1)),  # (B,)
            )
            return x_batch, y


        if self.full_load_test:
            B = self.test_batch_size
            n_sub = len(self.test_pts_cpu)
            if n_sub>50:
                n_sub = int(n_sub*0.5)

            xs, ys = [], []
            eval_loc_all = self.eval_loc_test  # (n_loc,3)
            y = self.y_test
            for start in range(0, n_sub, B):
                end = min(start + B, n_sub)
                idx = list(range(start, end))
                bsz = end - start
                pts, cent = self._stack_branch_batch(idx, train=False)
                resample, mu, scale = self._resample_and_normalize(pts, cent)

                eval_loc = np.broadcast_to(eval_loc_all[None, :, :], (bsz, eval_loc_all.shape[0], 3)).copy()
                eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale

                mu_vec = mu[:, :, 0]                                       # (B,3)
                scale_vec = scale[:, 0, 0][:, None]                        # (B,1)
                src_l = (self.test_src_l[idx] - mu_vec) / scale_vec
                src_r = (self.test_src_r[idx] - mu_vec) / scale_vec

                x = (
                    resample,
                    eval_loc,
                    src_l,
                    src_r,
                    mu.transpose(0, 2, 1),
                    np.reshape(scale,(bsz,1)),
                )
                xs.append(x)
                ys.append(y[idx])

            resample_all = np.concatenate([x[0] for x in xs], axis=0)
            eval_loc_all = np.concatenate([x[1] for x in xs], axis=0)
            src_l_all = np.concatenate([x[2] for x in xs], axis=0)
            src_r_all = np.concatenate([x[3] for x in xs], axis=0)
            mu_all = np.concatenate([x[4] for x in xs], axis=0)
            scale_all = np.concatenate([x[5] for x in xs], axis=0)

            x_all = (resample_all, eval_loc_all, src_l_all, src_r_all, mu_all, scale_all)
            y_all = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
            y_all = y_all.reshape(n_sub*self.eval_loc_idx_test.shape[0],-1)
            return x_all, y_all
        
class Dataloader_augment_loc_split_fixed():
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_resample=5000,
        sigma=0.6,
        test_batch_size=8,
        preprocessor: Preprocessor = Preprocessor(),
        input_augment=True,
    ):
        self.dim = 1
        self.n_resample = int(n_resample)
        self.sigma = float(sigma)
        self.test_batch_size = int(test_batch_size)
        self.batch_size = None
        self.full_load_test = False
        self.preprocessor = preprocessor

        # ---------- TRAIN ----------
        (
            self.train_fre,
            self.train_pts_cpu,
            self.train_cent_cpu,
            self.train_eval,
            self.train_src_l,
            self.train_src_r,
        ) = self.preprocessor.preprocess_inputs(X_train, input_augment=input_augment)
        n_train = len(self.train_pts_cpu)
        self.n_freqs = self.train_fre.shape[0]

        self.eval_idx = np.arange(self.train_eval.shape[0])
        self.eval_loc_idx_train, self.eval_loc_idx_test = train_test_split(
            self.eval_idx,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )
        self.n_eval_locs_train = self.eval_loc_idx_train.shape[0]
        self.n_eval_locs_test = self.eval_loc_idx_test.shape[0]

        self.train_eval_train = self.train_eval[self.eval_loc_idx_train]
        self.train_eval_test = self.train_eval[self.eval_loc_idx_test]

        y_train = self.preprocessor.preprocess_outputs(
            self.train_eval, y_train, input_augment=input_augment
        )
        self.y_train = y_train[:, self.eval_loc_idx_train, :, :]

        # ---------- TEST ----------
        (
            _,
            self.test_pts_cpu,
            self.test_cent_cpu,
            self.test_eval,
            self.test_src_l,
            self.test_src_r,
        ) = self.preprocessor.preprocess_inputs(X_test, input_augment=input_augment)
        n_test = len(self.test_pts_cpu)
        self.test_eval_test = self.test_eval[self.eval_loc_idx_test]

        self.y_test = self.preprocessor.preprocess_outputs(
            self.test_eval, y_test, input_augment=input_augment
        )
        self.y_test = self.y_test[:, self.eval_loc_idx_test, :, :]

        # ---------- SAMPLERS ----------
        self.branch_sampler = BatchSampler(n_train, shuffle=True)
        self.branch_sampler_test = BatchSampler(n_test, shuffle=False)
        self.trunk_sampler = BatchSampler(self.n_eval_locs_train, shuffle=True)
        self.trunk_sampler_test = BatchSampler(self.n_eval_locs_test, shuffle=False)

        # ---------- STANDARDIZER ----------
        self.scaler = self.preprocessor.get_scaler(y_train, None, init_hrtf_scaler=True)
        self.y_train = self.preprocessor.scale_outputs(self.scaler, self.y_train)
        self.y_test = self.preprocessor.scale_outputs(self.scaler, self.y_test)
        self.y_train = self.y_train.reshape(n_train, self.n_eval_locs_train, self.n_freqs, -1)
        self.y_test = self.y_test.reshape(n_test, self.n_eval_locs_test, self.n_freqs, -1)

        print(f"Number of train subjects: {n_train}")
        print(f"Number of validate subjects: {n_test}")

    def _stack_branch_batch(self, indices, train=True):
        pts_list = self.train_pts_cpu if train else self.test_pts_cpu
        cent_list = self.train_cent_cpu if train else self.test_cent_cpu

        pts = np.stack([pts_list[i] for i in indices], axis=0)
        cent = np.stack([cent_list[i] for i in indices], axis=0)
        return pts, cent

    def _resample_and_normalize(self, points, centers):
        B = points.shape[0]
        out = np.empty((B, 6, self.n_resample), dtype=np.float32)

        for i in range(B):
            down = grading_resample(
                points=points[i],
                centers=centers[i],
                M=self.n_resample,
                side="both",
                sigma=self.sigma,
            )
            out[i] = down.T

        xyz = out[:, :3, :]
        mu = xyz.mean(axis=2, keepdims=True)
        xyz_c = xyz - mu
        scale = np.linalg.norm(xyz_c, axis=1).max(axis=1, keepdims=True)
        scale = scale[:, :, None].astype(np.float32)

        xyz_n = xyz_c / scale
        resample = np.concatenate([xyz_n, out[:, 3:, :]], axis=1)
        return resample, mu, scale

    def train_next_batch(self, batch_size):
        B_branch, B_trunk = batch_size
        self.batch_size = batch_size

        idx_b = self.branch_sampler.get_next(B_branch)
        idx_t = self.trunk_sampler.get_next(B_trunk)  # local train-split indices

        y = self.y_train.reshape(
            len(self.train_pts_cpu), self.n_eval_locs_train, self.n_freqs, self.dim
        )
        y = y[idx_b][:, idx_t, :, :]
        y = y.reshape(B_branch * B_trunk, -1)

        pts, cent = self._stack_branch_batch(idx_b, train=True)
        resample, mu, scale = self._resample_and_normalize(pts, cent)

        eval_loc = self.train_eval_train[idx_t]
        eval_loc = np.broadcast_to(eval_loc[None, :, :], (B_branch, B_trunk, 3)).copy()
        mu_t = mu.transpose(0, 2, 1)
        eval_loc = (eval_loc - mu_t) / scale

        mu_vec = mu[:, :, 0]
        scale_vec = scale[:, 0, 0][:, None]
        src_l = (self.train_src_l[idx_b] - mu_vec) / scale_vec
        src_r = (self.train_src_r[idx_b] - mu_vec) / scale_vec

        x_batch = (
            resample,
            eval_loc,
            src_l,
            src_r,
            mu_t,
            np.reshape(scale, (B_branch, 1)),
        )
        return x_batch, y

    def test(self):
        if not self.full_load_test:
            B_branch = self.test_batch_size
            B_trunk = self.n_eval_locs_test

            idx_b = self.branch_sampler_test.get_next(B_branch)
            idx_t = self.trunk_sampler_test.get_next(B_trunk)  # local test-split indices

            y = self.y_test.reshape(
                len(self.test_pts_cpu), self.n_eval_locs_test, self.n_freqs, self.dim
            )
            y = y[idx_b][:, idx_t, :, :]
            y = y.reshape(B_branch * B_trunk, -1)

            pts, cent = self._stack_branch_batch(idx_b, train=False)
            resample, mu, scale = self._resample_and_normalize(pts, cent)

            eval_loc = self.test_eval_test[idx_t]
            eval_loc = np.broadcast_to(eval_loc[None, :, :], (B_branch, B_trunk, 3)).copy()
            eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale

            mu_vec = mu[:, :, 0]
            scale_vec = scale[:, 0, 0][:, None]
            src_l = (self.test_src_l[idx_b] - mu_vec) / scale_vec
            src_r = (self.test_src_r[idx_b] - mu_vec) / scale_vec

            x_batch = (
                resample,
                eval_loc,
                src_l,
                src_r,
                mu.transpose(0, 2, 1),
                np.reshape(scale, (B_branch, 1)),
            )
            return x_batch, y

        if self.full_load_test:
            B = self.test_batch_size
            n_sub = len(self.test_pts_cpu)
            if n_sub > 50:
                n_sub = int(n_sub * 0.5)

            xs, ys = [], []
            eval_loc_all = self.test_eval_test
            y = self.y_test.reshape(
                len(self.test_pts_cpu), self.n_eval_locs_test, self.n_freqs, self.dim
            )
            for start in range(0, n_sub, B):
                end = min(start + B, n_sub)
                idx = list(range(start, end))
                bsz = end - start

                pts, cent = self._stack_branch_batch(idx, train=False)
                resample, mu, scale = self._resample_and_normalize(pts, cent)

                eval_loc = np.broadcast_to(
                    eval_loc_all[None, :, :], (bsz, self.n_eval_locs_test, 3)
                ).copy()
                eval_loc = (eval_loc - mu.transpose(0, 2, 1)) / scale

                mu_vec = mu[:, :, 0]
                scale_vec = scale[:, 0, 0][:, None]
                src_l = (self.test_src_l[idx] - mu_vec) / scale_vec
                src_r = (self.test_src_r[idx] - mu_vec) / scale_vec

                x = (
                    resample,
                    eval_loc,
                    src_l,
                    src_r,
                    mu.transpose(0, 2, 1),
                    np.reshape(scale, (bsz, 1)),
                )
                xs.append(x)
                ys.append(y[idx])

            resample_all = np.concatenate([x[0] for x in xs], axis=0)
            eval_loc_all = np.concatenate([x[1] for x in xs], axis=0)
            src_l_all = np.concatenate([x[2] for x in xs], axis=0)
            src_r_all = np.concatenate([x[3] for x in xs], axis=0)
            mu_all = np.concatenate([x[4] for x in xs], axis=0)
            scale_all = np.concatenate([x[5] for x in xs], axis=0)

            x_all = (resample_all, eval_loc_all, src_l_all, src_r_all, mu_all, scale_all)
            y_all = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
            y_all = y_all.reshape(n_sub * self.n_eval_locs_test, -1)
            return x_all, y_all
        
if __name__ == "__main__":
    test_mirror_hrtf_sagittal()
