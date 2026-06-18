# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .dataset_colmap import ColmapDataset
from .dataset_nerf import NeRFDataset
from .dataset_scannetpp import ScannetppDataset
from .dataset_wildtrack import WildtrackDataset


def make(name: str, config, ray_jitter):
    match name:
        case "nerf":
            train_dataset = NeRFDataset(
                config.path,
                split="train",
                bg_color=config.model.background.color,
                ray_jitter=ray_jitter,
            )
            val_dataset = NeRFDataset(
                config.path,
                split="val",
                bg_color=config.model.background.color,
            )
        case "colmap":
            train_dataset = ColmapDataset(
                config.path,
                split="train",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                ray_jitter=ray_jitter,
            )
            val_dataset = ColmapDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
        case "scannetpp":
            train_dataset = ScannetppDataset(
                config.path,
                split="train",
                ray_jitter=ray_jitter,
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
            val_dataset = ScannetppDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
        case "wildtrack":
            single_frame_id = getattr(config.dataset, 'single_frame_id', None)
            held_out_camera = getattr(config.dataset, 'held_out_camera', None)
            train_dataset = WildtrackDataset(
                config.path,
                split="train",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                ray_jitter=ray_jitter,
                single_frame_id=single_frame_id,
                held_out_camera=held_out_camera,
            )
            val_dataset = WildtrackDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                single_frame_id=single_frame_id,
                held_out_camera=held_out_camera,
            )
        case _:
            raise ValueError(
                f'Unsupported dataset type: {name}. Choose between: ["colmap", "nerf", "scannetpp", "wildtrack"].'
            )

    return train_dataset, val_dataset


def make_test(name: str, config):
    match name:
        case "nerf":
            dataset = NeRFDataset(
                config.path,
                split="test",
                bg_color=config.model.background.color,
            )
        case "colmap":
            dataset = ColmapDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
        case "scannetpp":
            dataset = ScannetppDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
        case "wildtrack":
            # WildTrack has no separate test split; use val split for rendering/evaluation.
            # In LOO mode, val split contains only the held-out camera.
            single_frame_id = getattr(config.dataset, 'single_frame_id', None)
            held_out_camera = getattr(config.dataset, 'held_out_camera', None)
            dataset = WildtrackDataset(
                config.path,
                split="val",  # WildTrack has no test split; val is used for evaluation
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                single_frame_id=single_frame_id,
                held_out_camera=held_out_camera,
            )
        case _:
            raise ValueError(
                f'Unsupported dataset type: {name}. Choose between: ["colmap", "nerf", "scannetpp", "wildtrack"].'
            )
    return dataset
