import os
import torch
import numpy as np
import open3d as o3d
from typing import Iterator, Tuple
from grid_opt.utils.utils_geometry import (
    read_kitti_format_poses,
    transform_poses_from,
    pose_matrix,
    voxel_down_sample_torch,
)
from grid_opt.datasets.submap_dataset import SubmapDataset
import logging
logger = logging.getLogger(__name__)


class LidarSubmapDataset(SubmapDataset):
    """Dataset of LiDAR scans expressed in a common submap frame.

    The dataset reads a sequence of point clouds and their associated poses
    (given in a KITTI style file).  A single frame is chosen as the submap
    anchor and every scan is transformed into this submap using pose chaining.

    The class exposes an iterator yielding ``(points, pose)`` tuples where the
    points are in the submap coordinate system and ``pose`` is the SE(3)
    transform from the LiDAR frame to the submap.  It also implements the
    :class:`~grid_opt.datasets.submap_dataset.SubmapDataset` interface so that
    it can be consumed by :class:`grid_opt.slam.mapper.Mapper`.
    """

    def __init__(
        self,
        lidar_folder: str,
        pose_file: str,
        submap_origin: int = 0,
        frame_batchsize: int = 2 ** 10,
        voxel_size: float = 0.05,
        calib: dict | None = None,
    ) -> None:
        super().__init__()
        self.lidar_folder = lidar_folder
        self.frame_batchsize = frame_batchsize
        self.voxel_size = voxel_size
        self.calib = calib

        # ------------------------------------------------------------------
        # Read poses and determine submap frame
        # ------------------------------------------------------------------
        pose_list = self._read_poses(pose_file)
        self._num_frames = len(pose_list)
        if self._num_frames == 0:
            raise ValueError("No poses found in pose file")

        if not (0 <= submap_origin < self._num_frames):
            raise ValueError("submap_origin must index a valid pose")

        R_w_submap, t_w_submap = pose_list[submap_origin]
        self.R_world_submap = torch.from_numpy(R_w_submap).float()
        self.t_world_submap = torch.from_numpy(t_w_submap).float()

        # Containers for per-frame data
        self.R_world_frame = torch.zeros((self._num_frames, 3, 3))
        self.t_world_frame = torch.zeros((self._num_frames, 3, 1))
        self.frames: list[dict] = []

        pcd_files = sorted(
            [f for f in os.listdir(lidar_folder) if f.endswith(".pcd") or f.endswith(".ply")]
        )
        if len(pcd_files) < self._num_frames:
            logger.warning(
                "Number of point clouds (%d) smaller than poses (%d); truncating", len(pcd_files), self._num_frames
            )
            self._num_frames = len(pcd_files)
            pose_list = pose_list[: self._num_frames]

        for idx in range(self._num_frames):
            R_w_f, t_w_f = pose_list[idx]
            self.R_world_frame[idx] = torch.from_numpy(R_w_f)
            self.t_world_frame[idx] = torch.from_numpy(t_w_f)

            # Pose of frame in submap coordinates via chaining
            R_sub_f, t_sub_f = transform_poses_from(
                torch.from_numpy(R_w_f).unsqueeze(0),
                torch.from_numpy(t_w_f).reshape(1, 3, 1),
                self.R_world_submap,
                self.t_world_submap,
            )
            R_sub_f = R_sub_f[0]
            t_sub_f = t_sub_f[0]

            # Load point cloud
            pcd_path = os.path.join(lidar_folder, pcd_files[idx])
            pcd = o3d.io.read_point_cloud(pcd_path)
            pts = torch.from_numpy(np.asarray(pcd.points)).float()
            if pts.numel() == 0:
                logger.warning("Empty point cloud %s", pcd_path)
                pts_sub = torch.empty((0, 3))
            else:
                if self.voxel_size > 0:
                    keep = voxel_down_sample_torch(pts, self.voxel_size)
                    pts = pts[keep]
                # transform into submap coordinates
                pts_sub = (R_sub_f @ pts.T + t_sub_f).T

            self.frames.append(
                {
                    "points": pts_sub,
                    "R_submap_frame": R_sub_f,
                    "t_submap_frame": t_sub_f,
                }
            )

        self._selected_kfs: list[int] | None = None

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _read_poses(self, pose_file: str) -> list[Tuple[np.ndarray, np.ndarray]]:
        T_list = read_kitti_format_poses(pose_file)
        poses = []
        for T in T_list:
            R = T[:3, :3]
            t = T[:3, 3:]
            poses.append((R, t))
        logger.info("Read %d poses from file %s", len(poses), pose_file)
        return poses

    # ------------------------------------------------------------------
    # SubmapDataset interface
    # ------------------------------------------------------------------
    @property
    def num_kfs(self) -> int:
        return self._num_frames

    def get_odometry_at_pose(self, src_id: int) -> torch.Tensor:
        R_src = self.R_world_frame[src_id]
        t_src = self.t_world_frame[src_id]
        R_dst = self.R_world_frame[src_id + 1]
        t_dst = self.t_world_frame[src_id + 1]
        T_src = pose_matrix(R_src, t_src)
        T_dst = pose_matrix(R_dst, t_dst)
        return torch.linalg.inv(T_src) @ T_dst

    def sampled_points_at_kf(self, kf_id: int) -> torch.Tensor:
        return self.frames[kf_id]["points"]

    def select_keyframes(self, kf_ids):
        self._selected_kfs = list(kf_ids)

    def unselect_keyframes(self):
        self._selected_kfs = None

    def true_kf_pose_in_world(self, kf_id):
        R = self.R_world_frame[kf_id]
        t = self.t_world_frame[kf_id]
        return R, t

    def noisy_kf_pose_in_world(self, kf_id):
        # No noise model provided; use ground truth
        return self.true_kf_pose_in_world(kf_id)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - dataset size small
        return 1

    def __getitem__(self, index):
        pts_list = []
        frame_ids = []
        kf_range = range(self._num_frames) if self._selected_kfs is None else self._selected_kfs
        for kf_id in kf_range:
            pts = self.frames[kf_id]["points"]
            n = pts.shape[0]
            if n == 0:
                continue
            bs = min(self.frame_batchsize, n)
            idxs = torch.randperm(n)[:bs]
            pts_sample = pts[idxs]
            pts_list.append(pts_sample)
            frame_ids.append(torch.full((bs, 1), kf_id, dtype=torch.long))

        if not pts_list:
            raise RuntimeError("No points available for the requested keyframes")

        coords = torch.cat(pts_list, dim=0)
        ids = torch.cat(frame_ids, dim=0)
        N = coords.shape[0]
        weights = torch.ones((N, 1), dtype=torch.float32)
        input_dict = {
            "coords_frame": coords,
            "sample_frame_ids": ids,
            "weights": weights,
        }
        gt_dict = {
            "sdf": torch.zeros((N, 1), dtype=torch.float32),
            "sdf_valid": torch.ones((N, 1), dtype=torch.float32),
            "sdf_signs": torch.zeros((N, 1), dtype=torch.float32),
        }
        return input_dict, gt_dict

    # ------------------------------------------------------------------
    # Convenience iterator
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over frames yielding point cloud and pose.

        Yields:
            Tuple[Tensor, Tensor]: ``(points, pose)`` where ``points`` is a
            ``(N,3)`` tensor in the common submap coordinate system and
            ``pose`` is the ``4x4`` transformation matrix from frame to submap.
        """
        for frame in self.frames:
            R = frame["R_submap_frame"]
            t = frame["t_submap_frame"]
            T = pose_matrix(R, t)
            yield frame["points"], T
