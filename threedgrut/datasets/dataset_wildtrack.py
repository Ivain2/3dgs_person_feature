# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .camera_models import OpenCVPinholeCameraModelParameters, ShutterType
from .protocols import Batch
from .reid_teacher_cache import ReidTeacherCache


class WildtrackDataset(Dataset):
    """Dataset class for WildTrack multi-camera dataset.
    
    Note: This dataset automatically detects image dimensions from the actual images.
    Original WildTrack images are 1920×1080, but the CUDA rendering kernel uses 16×16 tiles,
    so images may be padded to multiples of 16 (e.g., 1920×1088) for optimal performance.
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        downsample_factor: int = 1,
        test_split_interval: int = 8,
        ray_jitter: Optional[float] = None,
        load_teacher_cache: bool = True,
    ) -> None:
        super().__init__()
        
        self.dataset_path = dataset_path
        self.split = split
        self.downsample_factor = downsample_factor
        self.test_split_interval = test_split_interval
        self.ray_jitter = ray_jitter
        
        self.camera_ids = [f"C{i}" for i in range(1, 8)]
        
        # Load image paths first to detect actual dimensions
        self.image_paths = self._load_image_paths()
        
        # Detect actual image dimensions from the dataset
        self.img_width, self.img_height = self._detect_image_dimensions()
        
        # Apply downsample factor
        self.img_height = self.img_height // downsample_factor
        self.img_width = self.img_width // downsample_factor
        
        self.intrinsics, self.extrinsics = self._load_camera_parameters()
        
        # Load ID mapping and annotations (using annotations_remapped)
        self.id_map = self._load_id_map()
        self.annotations = self._load_annotations()
        
        # Load teacher feature cache (optional, for ReID distillation)
        self.teacher_cache = None
        if load_teacher_cache:
            self.teacher_cache = ReidTeacherCache(dataset_path)
        
        self.indices = self._get_split_indices()
        self._scene_bbox = self.compute_spatial_extents()
    
    def _detect_image_dimensions(self) -> Tuple[int, int]:
        """Detect actual image dimensions from the first image in the dataset.
        
        For CUDA compatibility (16×16 tile size), images are padded to multiples of 16.
        WildTrack images are 1920×1080, so we pad height to 1088.
        """
        test_cam = self.camera_ids[0]
        if test_cam in self.image_paths and len(self.image_paths[test_cam]) > 0:
            test_img_path = self.image_paths[test_cam][0]
            test_img = cv2.imread(test_img_path)
            if test_img is not None:
                h, w = test_img.shape[:2]
                print(f"✅ Detected original image dimensions: {w}×{h}")
                
                # Pad to multiples of 16 for CUDA compatibility
                w_padded = ((w + 15) // 16) * 16
                h_padded = ((h + 15) // 16) * 16
                
                if w != w_padded or h != h_padded:
                    print(f"📏 Padding image to CUDA-compatible dimensions: {w_padded}×{h_padded}")
                
                return w_padded, h_padded
        
        # Fallback to default WildTrack dimensions (padded to 1088)
        print(f"⚠️  Could not detect image dimensions, using default: 1920×1088")
        return 1920, 1088
    
    def _load_image_paths(self) -> Dict[str, List[str]]:
        image_paths = {}
        for cam_id in self.camera_ids:
            cam_dir = os.path.join(self.dataset_path, "Image_subsets", cam_id)
            if not os.path.exists(cam_dir):
                raise FileNotFoundError(f"Camera directory not found: {cam_dir}")
            
            paths = sorted([
                os.path.join(cam_dir, f) for f in os.listdir(cam_dir)
                if f.endswith(".png")
            ], key=lambda x: int(re.search(r'([0-9]+)\.png', os.path.basename(x)).group(1)))
            image_paths[cam_id] = paths
        return image_paths
    
    def _load_camera_parameters(self) -> Tuple[Dict[str, OpenCVPinholeCameraModelParameters], Dict[str, np.ndarray]]:
        intrinsics = {}
        extrinsics = {}
        
        # Load intrinsics (keep original calibration parameters unchanged)
        intrinsic_dir = os.path.join(self.dataset_path, "calibrations", "intrinsic_original")
        for cam_id in self.camera_ids:
            xml_path = os.path.join(intrinsic_dir, f"intr_{cam_id}.xml")
            
            if os.path.exists(xml_path):
                fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
                camera_matrix = fs.getNode("camera_matrix").mat()
                dist_coeffs = fs.getNode("distortion_coefficients").mat().flatten()
                image_size = fs.getNode("image_size").mat().flatten()
                fs.release()
                
                # Use original calibration parameters directly
                # Principal point (cx, cy) doesn't need adjustment since padding is at the bottom
                principal_point = np.array([camera_matrix[0, 2], camera_matrix[1, 2]], dtype=np.float32)
                focal_length = np.array([camera_matrix[0, 0], camera_matrix[1, 1]], dtype=np.float32)
                
                # WildTrack XML is OpenCV FileStorage (<opencv_storage>); OpenCV 5-param order is (k1, k2, p1, p2, k3).
                # 3dgrut expects radial = [k1..k6], tangential = [p1,p2]. Map here; do not change 3dgrut/camera_models/CUDA.
                # 5-param: dist[0..4] = k1,k2,p1,p2,k3.  8-param: dist[5..7] = k4,k5,k6.
                radial_coeffs = np.zeros(6, dtype=np.float32)
                tangential_coeffs = np.zeros(2, dtype=np.float32)
                thin_prism_coeffs = np.zeros(4, dtype=np.float32)
                if len(dist_coeffs) >= 5:
                    radial_coeffs[0] = dist_coeffs[0]  # k1
                    radial_coeffs[1] = dist_coeffs[1]  # k2
                    radial_coeffs[2] = dist_coeffs[4]  # k3
                    tangential_coeffs[0] = dist_coeffs[2]  # p1
                    tangential_coeffs[1] = dist_coeffs[3]  # p2
                if len(dist_coeffs) >= 8:
                    radial_coeffs[3] = dist_coeffs[5]  # k4
                    radial_coeffs[4] = dist_coeffs[6]  # k5
                    radial_coeffs[5] = dist_coeffs[7]  # k6
                
                if self.downsample_factor > 1:
                    principal_point /= self.downsample_factor
                    focal_length /= self.downsample_factor
                
                # Use detected image dimensions
                resolution = np.array([self.img_width, self.img_height], dtype=np.int64)
                
                intrinsics[cam_id] = OpenCVPinholeCameraModelParameters(
                    resolution=resolution,
                    shutter_type=ShutterType.GLOBAL,
                    principal_point=principal_point,
                    focal_length=focal_length,
                    radial_coeffs=radial_coeffs,
                    tangential_coeffs=tangential_coeffs,
                    thin_prism_coeffs=thin_prism_coeffs
                )
            else:
                intrinsics[cam_id] = self._get_default_intrinsics(cam_id)
        
        # Load extrinsics (keep original calibration parameters unchanged)
        extrinsic_dir = os.path.join(self.dataset_path, "calibrations", "extrinsic")
        for cam_id in self.camera_ids:
            xml_path = os.path.join(extrinsic_dir, f"extr_{cam_id}.xml")
            if os.path.exists(xml_path):
                fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
                R = fs.getNode("R").mat()
                T = fs.getNode("T").mat().flatten()
                fs.release()
                
                # Use original extrinsics directly
                C2W = np.eye(4, dtype=np.float32)
                C2W[:3, :3] = R
                C2W[:3, 3] = T
                extrinsics[cam_id] = C2W
            else:
                extrinsics[cam_id] = np.eye(4, dtype=np.float32)
        
        return intrinsics, extrinsics
    
    def _get_default_intrinsics(self, cam_id: str) -> OpenCVPinholeCameraModelParameters:
        """Generate default intrinsics using detected image dimensions."""
        return OpenCVPinholeCameraModelParameters(
            resolution=np.array([self.img_width, self.img_height], dtype=np.int64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([self.img_width/2, self.img_height/2], dtype=np.float32),
            focal_length=np.array([1000, 1000], dtype=np.float32) // self.downsample_factor,
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32)
        )
    
    def _load_id_map(self) -> Dict[str, Dict]:
        """Load ID mapping from id_map.json.
        
        Returns:
            Dict with 'raw_to_new' and 'new_to_raw' mappings.
            raw_to_new: {raw_id (int) -> train_id (int)}
            new_to_raw: {train_id (int) -> raw_id (int)}
        """
        id_map_path = os.path.join(self.dataset_path, "id_map.json")
        if not os.path.exists(id_map_path):
            print(f"⚠️  Warning: id_map.json not found at {id_map_path}")
            return {'raw_to_new': {}, 'new_to_raw': {}}
        
        with open(id_map_path, 'r') as f:
            id_map = json.load(f)
        
        # Convert string keys to int keys for raw_to_new
        raw_to_new = {int(k): v for k, v in id_map.get('raw_to_new', {}).items()}
        # Convert string keys to int keys for new_to_raw
        new_to_raw = {int(k): v for k, v in id_map.get('new_to_raw', {}).items()}
        
        print(f"✅ Loaded ID map: {len(raw_to_new)} raw IDs -> {len(new_to_raw)} train IDs")
        return {'raw_to_new': raw_to_new, 'new_to_raw': new_to_raw}
    
    def _load_annotations(self) -> Dict[int, List[Dict]]:
        """Load annotations from annotations_remapped directory.
        
        Each annotation file contains a list of person instances with:
        - image_path, bbox, camera_id, frame_id, raw_id, new_id
        """
        annotations = {}
        # Use annotations_remapped instead of annotations_positions
        annot_dir = os.path.join(self.dataset_path, "annotations_remapped")
        if not os.path.exists(annot_dir):
            print(f"⚠️  Warning: annotations_remapped not found at {annot_dir}")
            print(f"   Falling back to annotations_positions for basic functionality")
            annot_dir = os.path.join(self.dataset_path, "annotations_positions")
            if not os.path.exists(annot_dir):
                return annotations
        
        for annot_file in sorted(os.listdir(annot_dir)):
            if annot_file.endswith(".json"):
                frame_id = int(re.search(r'([0-9]+)\.json', annot_file).group(1))
                with open(os.path.join(annot_dir, annot_file), 'r') as f:
                    annotations[frame_id] = json.load(f)
        
        print(f"✅ Loaded {len(annotations)} annotation files from {annot_dir}")
        return annotations
    
    def _get_split_indices(self) -> List[Tuple[str, int]]:
        indices = []
        num_frames = len(self.image_paths[self.camera_ids[0]])
        for frame_idx in range(num_frames):
            if self.split == "train" and frame_idx % self.test_split_interval != 0:
                for cam_id in self.camera_ids:
                    indices.append((cam_id, frame_idx))
            elif self.split == "val" and frame_idx % self.test_split_interval == 0:
                for cam_id in self.camera_ids:
                    indices.append((cam_id, frame_idx))
        return indices
    
    def compute_spatial_extents(self) -> Tuple[torch.Tensor, torch.Tensor]:
        camera_positions = []
        for cam_id in self.camera_ids:
            C2W = self.extrinsics[cam_id]
            camera_positions.append(C2W[:3, 3])
        
        annotation_points = []
        for frame_annots in self.annotations.values():
            for person in frame_annots:
                if "position_3d" in person:
                    annotation_points.append(person["position_3d"])
        
        all_points = np.array(camera_positions)
        if annotation_points:
            all_points = np.vstack([all_points, np.array(annotation_points)])
        
        if len(all_points) > 0:
            min_bounds = np.min(all_points, axis=0) - 1.0
            max_bounds = np.max(all_points, axis=0) + 1.0
        else:
            min_bounds = np.array([-10.0, -10.0, -2.0], dtype=np.float32)
            max_bounds = np.array([10.0, 10.0, 5.0], dtype=np.float32)
        
        return torch.tensor(min_bounds, dtype=torch.float32), torch.tensor(max_bounds, dtype=torch.float32)
    
    def get_scene_bbox(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._scene_bbox
    
    def get_scene_extent(self) -> float:
        min_bounds, max_bounds = self._scene_bbox
        return float(torch.norm(max_bounds - min_bounds))
    
    def get_observer_points(self) -> np.ndarray:
        camera_positions = []
        for cam_id in self.camera_ids:
            C2W = self.extrinsics[cam_id]
            camera_positions.append(C2W[:3, 3])
        return np.array(camera_positions)
    
    def get_poses(self) -> np.ndarray:
        poses = []
        for cam_id in self.camera_ids:
            poses.append(self.extrinsics[cam_id])
        return np.array(poses)
    
    def __getitem__(self, index: int) -> dict:
        cam_id, frame_idx = self.indices[index]
        
        # Load image
        image_path = self.image_paths[cam_id][frame_idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Pad image to CUDA-compatible dimensions (multiples of 16)
        h, w = image.shape[:2]
        h_padded = ((h + 15) // 16) * 16
        w_padded = ((w + 15) // 16) * 16
        
        if h != h_padded or w != w_padded:
            # Pad with zeros (black border)
            image = cv2.copyMakeBorder(
                image, 
                0, h_padded - h, 
                0, w_padded - w, 
                cv2.BORDER_CONSTANT, 
                value=(0, 0, 0)
            )
        
        if self.downsample_factor > 1:
            h, w = image.shape[:2]
            new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        image = image.astype(np.float32) / 255.0
        
        # Generate rays using actual image dimensions
        C2W = self.extrinsics[cam_id].copy()
        intrinsics = self.intrinsics[cam_id]
        
        h, w = image.shape[:2]
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        
        fx, fy = intrinsics.focal_length
        cx, cy = intrinsics.principal_point
        k1, k2, k3, k4, k5, k6 = intrinsics.radial_coeffs
        p1, p2 = intrinsics.tangential_coeffs
        
        # Back-project to normalized plane
        x = (i - cx) / fx
        y = (j - cy) / fy
        r2 = x*x + y*y
        
        # Distortion correction
        radial_dist = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        tan_dist_x = 2*p1*x*y + p2*(r2 + 2*x*x)
        tan_dist_y = p1*(r2 + 2*y*y) + 2*p2*x*y
        
        x_dist = x * radial_dist + tan_dist_x
        y_dist = y * radial_dist + tan_dist_y
        
        # Generate ray directions (camera coordinate system)
        rays_dir = np.stack([x_dist, y_dist, np.ones_like(x_dist)], axis=-1)
        rays_dir = rays_dir / np.linalg.norm(rays_dir, axis=-1, keepdims=True)
        
        # Transform to world coordinate system
        rays_dir = rays_dir @ C2W[:3, :3].T
        rays_ori = np.broadcast_to(C2W[:3, 3], rays_dir.shape)
        
        if self.ray_jitter is not None and self.split == "train":
            rays_dir += np.random.normal(0, self.ray_jitter, rays_dir.shape)
            rays_dir = rays_dir / np.linalg.norm(rays_dir, axis=-1, keepdims=True)
        
        rays_ori = rays_ori.reshape(h, w, 3).astype(np.float32)
        rays_dir = rays_dir.reshape(h, w, 3).astype(np.float32)
        # ======================================================
        
        intrinsics_dict = {
            "focal_length": intrinsics.focal_length.tolist(),
            "principal_point": intrinsics.principal_point.tolist(),
            "radial_coeffs": intrinsics.radial_coeffs.tolist(),
            "tangential_coeffs": intrinsics.tangential_coeffs.tolist(),
            "thin_prism_coeffs": intrinsics.thin_prism_coeffs.tolist(),
            "resolution": intrinsics.resolution.tolist(),
            "shutter_type": intrinsics.shutter_type.value
        }
        
        # Build instances list for ReID distillation
        instances = self._get_instances_for_frame(frame_idx, cam_id, intrinsics)
        
        return {
            "rays_ori": rays_ori,      # [1088, 1920, 3]
            "rays_dir": rays_dir,      # [1088, 1920, 3]
            "rgb": image,              # [1088, 1920, 3]
            "C2W": C2W.astype(np.float32),
            "intrinsics": intrinsics_dict,
            "image_path": image_path,
            "camera_id": cam_id,
            "frame_idx": frame_idx,
            "instances": instances    # List of instance dicts with bbox, train_id, teacher_embedding
        }
    
    def _get_instances_for_frame(self, frame_idx: int, cam_id: str, intrinsics) -> List[Dict]:
        """Build instances list for a given frame and camera.
        
        Returns:
            List of dicts, each containing:
            - bbox_xyxy: [xmin, ymin, xmax, ymax] in original image coords
            - raw_id: original person ID (for debug/log only)
            - train_id: remapped identity ID (0-based, continuous) - used for ALL training
            - teacher_embedding: ReID feature from cache
            - valid: whether this instance is valid for training
        """
        instances = []
        
        # Check if we have annotations for this frame
        if frame_idx not in self.annotations:
            return instances
        
        # Get annotations for this frame (from annotations_remapped format)
        frame_annots = self.annotations[frame_idx]
        
        for inst_idx, person in enumerate(frame_annots):
            # Get raw_id and train_id
            raw_id = person.get('raw_id') or person.get('personID', 0)
            train_id = person.get('train_id')
            
            if train_id is None:
                # Fallback: use id_map to convert raw_id to train_id
                raw_to_new = self.id_map.get('raw_to_new', {})
                train_id = raw_to_new.get(raw_id, -1)
            
            if raw_id == 0 or train_id == -1:
                # Skip invalid IDs
                continue
            
            # Get camera_id and bbox from views
            views = person.get('views', [])
            bbox_found = False
            
            for view in views:
                view_cam_id = f"C{view.get('viewNum', 0)}"
                if view_cam_id != cam_id:
                    continue
                
                # Get bbox (xmin, ymin, xmax, ymax)
                xmin = view.get('xmin', 0)
                ymin = view.get('ymin', 0)
                xmax = view.get('xmax', 0)
                ymax = view.get('ymax', 0)
                
                # Skip invalid bboxes (xmin == -1 means invisible in this view)
                if xmin == -1 or xmax <= xmin or ymax <= ymin:
                    continue
                
                # Scale bbox if downsampling
                if self.downsample_factor > 1:
                    xmin = int(xmin / self.downsample_factor)
                    ymin = int(ymin / self.downsample_factor)
                    xmax = int(xmax / self.downsample_factor)
                    ymax = int(ymax / self.downsample_factor)
                
                bbox_xyxy = [xmin, ymin, xmax, ymax]
                
                # Get teacher embedding from cache
                teacher_embedding = None
                if self.teacher_cache is not None:
                    cache_key = (frame_idx, cam_id, inst_idx)
                    cache_entry = self.teacher_cache.get(cache_key)
                    if cache_entry is not None:
                        teacher_embedding = cache_entry.get('embedding')
                
                # Build instance dict
                instance = {
                    'bbox_xyxy': bbox_xyxy,
                    'raw_id': raw_id,                    # For debug/log ONLY
                    'train_id': train_id,                # For ALL training (indexing, loss)
                    'teacher_embedding': teacher_embedding,  # May be None if cache not available
                    'valid': teacher_embedding is not None   # Only valid if we have teacher feature
                }
                
                instances.append(instance)
                bbox_found = True
                break  # Only one bbox per (frame, camera, instance)
        
        return instances
    
    def get_gpu_batch_with_intrinsics(self, batch: dict) -> Batch:
        """Convert batch to GPU format with intrinsics.
        
        Ensures all tensors are in [B, H, W, C] format for compatibility with 3DGUT tracer.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Unified processing function: ensure all tensors are in 4D [B, H, W, C] format
        def ensure_4d_bhwc(tensor, name="tensor"):
            """Ensure tensor is in [B, H, W, C] format"""
            if tensor is None:
                return None
            
            # Move to GPU
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.from_numpy(tensor).to(device)
            else:
                tensor = tensor.to(device)
            
            # Ensure floating point type
            if not tensor.is_floating_point():
                tensor = tensor.float()
            
            # Add Batch dimension
            if tensor.ndim == 3:  # [H, W, C] → [1, H, W, C]
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 4:  # Already [B, H, W, C] or [B, C, H, W]
                # Check if it's CHW format (channel count is usually 3 or 4)
                if tensor.shape[1] in [3, 4] and tensor.shape[1] < tensor.shape[2]:
                    # [B, C, H, W] → [B, H, W, C]
                    tensor = tensor.permute(0, 2, 3, 1)
            else:
                raise ValueError(f"{name} has unexpected shape: {tensor.shape}")
            
            return tensor.contiguous()
        
        # Process ray data
        rays_ori = ensure_4d_bhwc(batch["rays_ori"], name="rays_ori")
        rays_dir = ensure_4d_bhwc(batch["rays_dir"], name="rays_dir")
        
        # Process extrinsics
        T_to_world = batch["C2W"]
        if not isinstance(T_to_world, torch.Tensor):
            T_to_world = torch.from_numpy(T_to_world).to(device)
        else:
            T_to_world = T_to_world.to(device)
        
        if T_to_world.ndim == 2:  # [4, 4] → [1, 4, 4]
            T_to_world = T_to_world.unsqueeze(0)
        
        # Process RGB Ground Truth
        rgb_gt = None
        if "rgb" in batch:
            rgb_gt = ensure_4d_bhwc(batch["rgb"], name="rgb")
            
            # If RGB and Rays dimensions don't match, resize RGB
            B, H, W, C = rays_ori.shape
            if rgb_gt.shape[1:3] != (H, W):
                from torchvision.transforms.functional import resize
                rgb_gt_chw = rgb_gt.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
                rgb_gt_resized = resize(rgb_gt_chw, (H, W), antialias=True)
                rgb_gt = rgb_gt_resized.permute(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]
        
        # Process intrinsics
        intrinsics = None
        intrinsics_params = None
        
        if "intrinsics" in batch:
            intrinsic_dict = batch["intrinsics"]
            
            def extract_scalar(val, idx=0):
                if isinstance(val, torch.Tensor):
                    if val.ndim > 0:
                        return val[idx].item() if val.numel() > idx else val.item()
                    return val.item()
                if isinstance(val, (list, np.ndarray)):
                    return float(val[idx]) if len(val) > idx else float(val[0])
                return float(val)
            
            fx = extract_scalar(intrinsic_dict["focal_length"], 0)
            fy = extract_scalar(intrinsic_dict["focal_length"], 1)
            cx = extract_scalar(intrinsic_dict["principal_point"], 0)
            cy = extract_scalar(intrinsic_dict["principal_point"], 1)
            intrinsics = [fx, fy, cx, cy]
            
            intrinsics_params = intrinsic_dict
        
        return Batch(
            rays_ori=rays_ori,
            rays_dir=rays_dir,
            T_to_world=T_to_world,
            rgb_gt=rgb_gt,
            mask=None,
            intrinsics=intrinsics,
            intrinsics_OpenCVPinholeCameraModelParameters=intrinsics_params,
            intrinsics_OpenCVFisheyeCameraModelParameters=None,
            instances=batch.get("instances")  # Pass instances to Batch for ReID distillation
        )
    
    def __len__(self) -> int:
        return len(self.indices)
