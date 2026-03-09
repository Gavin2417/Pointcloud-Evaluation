import torch
import numpy as np
from sklearn.neighbors import KDTree
from network.RandLANet import Network
from utils.config import Config10labels as cfg
from utils.data_process import DataProcessing as DP
import open3d as o3d
class RandlaGroundSegmentor:
    def __init__(self, ckpt_path='log/checkpoint_slope_only.tar', device=None, subsample_grid=0.1):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Network(cfg).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.num_points = cfg.num_points
        self.subsample_grid = subsample_grid
        self.k_n = cfg.k_n
        self.num_layers = cfg.num_layers
        self.sub_ratio = cfg.sub_sampling_ratio

    def segment(self, points_world):
        """
        points_world: (M,3) np.array of live LiDAR points
        returns:
          - if return_probs is False: labels_world (M,) ints
          - if return_probs is True: (labels_world (M,), probs_world (M,C)) with per‑class probabilities
        """
        # Statistical outlier removal 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        clean_pts = np.asarray(pcd.points, dtype= np.float32)

        # Voxel‐grid subsample to reduce density
        pts = DP.grid_sub_sampling(clean_pts, grid_size=self.subsample_grid)

        # Enforce exactly num_points via random trunc/pad
        N = pts.shape[0]
        if N >= self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False)
        else:
            dup = np.random.choice(N, self.num_points - N, replace=True)
            idx = np.concatenate([np.arange(N), dup])
        sampled = pts[idx]

        # Build per-layer xyz lists and features
        xyz_list = [
            torch.from_numpy(sampled).float().to(self.device).unsqueeze(0)
        ]
        feat = xyz_list[0].permute(0, 2, 1).contiguous() 

        neigh_idx, sub_idx, interp_idx = [], [], []
        pts_cur = sampled
        for i in range(self.num_layers):
            # K-NN at this scale
            knn = DP.knn_search(pts_cur[None, ...], pts_cur[None, ...], self.k_n)[0]
            neigh_idx.append(
                torch.from_numpy(knn).long().to(self.device).unsqueeze(0)
            )

            # Subsample
            n_sub = pts_cur.shape[0] // self.sub_ratio[i]
            sel = np.random.permutation(pts_cur.shape[0])[:n_sub]
            sub_idx.append(
                torch.from_numpy(sel[None, ...]).long().to(self.device).unsqueeze(-1)
            )

            # Build next‐level points
            pts_next = pts_cur[sel]
            xyz_list.append(
                torch.from_numpy(pts_next).float().to(self.device).unsqueeze(0)
            )

            # Interpolation indices back
            interp = DP.knn_search(pts_next[None, ...], pts_cur[None, ...], 1)[0]
            interp_idx.append(
                torch.from_numpy(interp).long().to(self.device).unsqueeze(0)
            )

            pts_cur = pts_next

        # Inference
        with torch.no_grad():
            inputs = {
                "xyz":        xyz_list,
                "features":   feat,
                "neigh_idx":  neigh_idx,
                "sub_idx":    sub_idx,
                "interp_idx": interp_idx
            }
            end_pts = self.model(inputs)
            logits = end_pts["logits"]
            preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()  
            probs_sampled = torch.softmax(logits, dim=1).squeeze(0).permute(1, 0).contiguous().cpu().numpy()

        # Project back to original M points
        tree = KDTree(sampled)
        nn = tree.query(points_world, return_distance=False).squeeze(-1)
        labels_world = preds[nn]

        probs_world = probs_sampled[nn]
        return labels_world, probs_world