import concurrent.futures
import glob
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import zarr
import zarr.storage
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.kinematics_helper import KinHelper
from imgaug import augmenters as iaa

from dreamerv3_torch.utils.imagecodecs_numcodecs import Jpeg2k, register_codecs
from dreamerv3_torch.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
    get_twenty_times_normalizer_from_stat,
)
from dreamerv3_torch.utils.pytorch_util import dict_apply
from dreamerv3_torch.utils.replay_buffer import ReplayBuffer
from dreamerv3_torch.utils.sampler import SequenceSampler, get_val_mask

from .base_dataset import BaseImageDataset

register_codecs()


# convert raw hdf5 data to replay buffer, which is used for diffusion policy training
def _convert_real_to_dp_replay(
    store: zarr.storage.Store,
    shape_meta: dict,
    dataset_dir: str,
    n_workers: Optional[int] = None,
    max_inflight_tasks: Optional[int] = None,
    action_mode: str = "bimanual_push",
) -> ReplayBuffer:
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        if type == "depth":
            depth_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    episodes_paths = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split("_")[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)

    episode_ends = list()
    prev_end = 0
    lowdim_data_dict: dict = dict()
    rgb_data_dict: dict = dict()
    mask_data_dict: dict = dict()
    depth_data_dict: dict = dict()
    for epi_idx in tqdm(episodes_idx, desc="Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f"episode_{epi_idx}.hdf5")
        with h5py.File(dataset_path) as file:
            # count total steps
            try:
                episode_length = file["action"].shape[0]
            except Exception:
                print(f"Episode {epi_idx} has no action data")
                continue

            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)

            # save lowdim data to lowedim_data_dict
            for key in [*lowdim_keys, "action"]:
                data_key = "obs/" + key
                if key == "action":
                    data_key = (
                        "action"
                        if "key" not in shape_meta["action"]
                        else shape_meta["action"]["key"]
                    )
                if key not in lowdim_data_dict:
                    lowdim_data_dict[key] = list()
                try:
                    this_data = file[data_key][()]
                except Exception:
                    print(f"Episode {epi_idx} has no {data_key} data")
                    continue
                if key == "action":
                    assert this_data.shape[0] == episode_length
                    assert this_data.shape[1:] == shape_meta["action"]["shape"]
                lowdim_data_dict[key].append(this_data)

            for key in rgb_keys:
                if key not in rgb_data_dict:
                    rgb_data_dict[key] = list()
                try:
                    imgs = file["obs"]["images"][key][()]
                except Exception:
                    print(f"Episode {epi_idx} has no {key} data")
                    continue
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                crop_imgs = [center_crop(img, (h, w)) for img in imgs]
                resize_imgs = [
                    cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    for img in crop_imgs
                ]
                imgs = np.stack(resize_imgs, axis=0)
                assert imgs[0].shape == (h, w, c)
                rgb_data_dict[key].append(imgs)

            for key in depth_keys:
                if key not in depth_data_dict:
                    depth_data_dict[key] = list()
                try:
                    imgs = file["obs"]["images"][key][()]
                except Exception:
                    print(f"Episode {epi_idx} has no {key} data")
                    continue
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                crop_imgs = [center_crop(img, (h, w)) for img in imgs]
                resize_imgs = [
                    cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    for img in crop_imgs
                ]
                imgs = np.stack(resize_imgs, axis=0)[..., None]
                imgs = np.clip(imgs, 0, 1000).astype(np.uint16)
                assert imgs[0].shape == (h, w, c)
                depth_data_dict[key].append(imgs)

            # Load masks for corresponding RGB images
            for key in rgb_keys:
                if key not in mask_data_dict:
                    mask_data_dict[key] = list()
                mask_path = os.path.join(
                    dataset_dir,
                    "kypts",
                    "masks",
                    f"track_masks_episode_{epi_idx}_{key}.npy",
                )
                if os.path.exists(mask_path):
                    masks = np.load(mask_path)
                    # Resize masks to match the RGB image resolution
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    c, h, w = shape
                    resized_masks = []
                    for mask in masks:
                        # Resize mask to match RGB image dimensions
                        mask = center_crop(mask, (h, w))
                        resized_mask = cv2.resize(
                            mask.astype(np.float32),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        resized_masks.append(resized_mask)
                    masks = np.stack(resized_masks, axis=0)
                    assert masks[0].shape == (h, w)
                    mask_data_dict[key].append(masks)
                else:
                    # If no mask file exists, create empty masks
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    c, h, w = shape
                    try:
                        episode_length = file["action"].shape[0]
                    except Exception:
                        print(f"Episode {epi_idx} has no action data")
                        continue
                    empty_masks = np.ones((episode_length, h, w), dtype=np.uint8)
                    mask_data_dict[key].append(empty_masks)

    def img_copy(
        zarr_arr: zarr.Array, zarr_idx: int, hdf5_arr: np.ndarray, hdf5_idx: int
    ) -> bool:
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception:
            return False

    # dump data_dict
    print("Dumping meta data")
    n_steps = episode_ends[-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    print("Dumping lowdim data")
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype,
        )

    print("Dumping rgb data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: set = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint8,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    print("Dumping depth data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint16,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    print("Dumping mask data")
    for key, data in mask_data_dict.items():
        if data:  # Only dump if masks exist
            masks = np.concatenate(data, axis=0).astype(bool)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            mask_arr = data_group.require_dataset(
                name=f"{key}_mask",
                shape=(n_steps, h, w),
                chunks=(1, h, w),
                compressor=None,
                dtype=bool,
            )
            mask_arr[:] = masks

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


class RealAlohaDataset(BaseImageDataset):
    """A dataset for the real-world data collected on Aloha robot."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # assign config
        shape_meta = cfg.shape_meta
        dataset_dir = cfg.dataset_dir
        horizon = cfg.horizon * cfg.skip_frame
        pad_before = cfg.pad_before
        pad_after = cfg.pad_after
        use_cache = cfg.use_cache
        seed = cfg.seed
        val_ratio = cfg.val_ratio
        manual_val_mask = cfg.manual_val_mask if "manual_val_mask" in cfg else False
        manual_val_start = cfg.manual_val_start if "manual_val_start" in cfg else -1
        self.val_horizon = (
            cfg.val_horizon * cfg.skip_frame if "val_horizon" in cfg else horizon
        )
        self.skip_idx = cfg.skip_idx if "skip_idx" in cfg else 1
        self.action_mode = cfg.action_mode
        self.aug_mode = cfg.aug_mode
        if cfg.aug_mode == "img_aug":
            self.aug = iaa.Sequential(
                [
                    iaa.Affine(
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-30, 30),
                        mode="edge",
                    ),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05), per_channel=0.5
                    ),
                    iaa.MultiplyHueAndSaturation(
                        mul_hue=(0.8, 1.2), mul_saturation=(0.8, 1.2)
                    ),
                    iaa.MultiplyBrightness(mul=(0.8, 1.2)),
                ]
            )
        elif cfg.aug_mode == "none":
            self.aug = None
        else:
            raise ValueError(f"Invalid augmentation mode: {cfg.aug_mode}")

        replay_buffer = None
        if cfg.delta_action:
            with h5py.File(f"{dataset_dir}/episode_0.hdf5") as file:
                self.robot_bases = file["obs"]["world_t_robot_base"][0].copy()
        if use_cache:
            cache_info_str = ""
            obs_shape_meta = shape_meta["obs"]
            for _, attr in obs_shape_meta.items():
                type = attr.get("type", "low_dim")
            cache_zarr_path = os.path.join(
                dataset_dir, f"cache{cache_info_str}.zarr.zip"
            )
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("Cache does not exist. Creating!")
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_real_to_dp_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_dir=dataset_dir,
                            action_mode=self.action_mode,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _convert_real_to_dp_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_dir=dataset_dir,
                action_mode=self.action_mode,
            )
        self.replay_buffer = replay_buffer

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "depth":
                depth_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        if not manual_val_mask:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
            )
        else:
            assert manual_val_start >= 0, "manual_val_start must be >= 0"
            assert (
                manual_val_start < replay_buffer.n_episodes
            ), "manual_val_start too large"
            val_mask = np.zeros((replay_buffer.n_episodes,), dtype=np.bool)
            val_mask[manual_val_start:] = True
        train_mask = ~val_mask

        # Add mask keys to the sampler
        if cfg.load_mask:
            mask_keys = [f"{key}_mask" for key in rgb_keys]
        else:
            mask_keys = []
        all_keys = list(self.replay_buffer.keys()) + mask_keys

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.mask_keys = mask_keys
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = cfg.skip_frame
        self.delta_action = cfg.delta_action
        self.goal_sample = cfg.goal_sample
        if self.delta_action:
            self.kin_helper = KinHelper("trossen_vx300s")

        self.normalizer = self.get_normalizer()

    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        """Return a normalizer for the dataset."""
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        if self.delta_action:
            this_normalizer = get_twenty_times_normalizer_from_stat(stat)
        else:
            this_normalizer = get_range_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("vel"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()

        # Add normalizers for mask keys
        for key in self.mask_keys:
            normalizer[key] = get_identity_normalizer_from_stat(
                array_to_stats(self.replay_buffer[key])
            )

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        skip_start = np.random.randint(0, self.skip_frame) + self.skip_frame
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            # obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32)
            obs_dict[key] = sample[key].astype(np.float32)
            obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 1000.0
            obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            del sample[key]

        actions = sample["action"].astype(np.float32)
        action_dim = actions.shape[-1]
        downsample_horizon = actions.shape[0] // self.skip_frame - 1
        action_len = downsample_horizon * self.skip_frame
        action_start = skip_start - self.skip_frame
        actions = actions[action_start : action_start + action_len]
        actions = actions.reshape(downsample_horizon, self.skip_frame, action_dim)
        if self.delta_action:
            joint_pos = obs_dict["joint_pos"].astype(np.float32)
            robot_bases = self.robot_bases
            # compute ee pos in robot_bases[0]
            num_robot = joint_pos.shape[1] // 7
            world_t_ee_pose = np.zeros((joint_pos.shape[0], num_robot, 4, 4))
            for i in range(num_robot):
                for t in range(joint_pos.shape[0]):
                    joint_fk = np.concatenate(
                        [
                            joint_pos[t, i * 7 : (i + 1) * 7],
                            joint_pos[t, i * 7 + 6 : (i + 1) * 7],
                        ]
                    )
                    ee_pose = self.kin_helper.compute_fk_from_link_idx(
                        joint_fk, [self.kin_helper.sapien_eef_idx]
                    )[0]
                    world_t_ee_pose[t, i] = robot_bases[i] @ ee_pose
            if self.action_mode == "bimanual_push":
                d_actions = np.zeros_like(actions)
                d_actions[..., :2] = actions[..., :2] - world_t_ee_pose[:, 0:1, :2, 3]
                d_actions[..., 2:] = actions[..., 2:] - world_t_ee_pose[:, 1:2, :2, 3]
                actions = d_actions
            elif self.action_mode == "single_ee":
                d_actions = np.zeros_like(actions)
                d_actions[..., :3] = actions[..., :3] - world_t_ee_pose[:, 1:2, :3, 3]
                d_actions[..., 3:4] = (
                    actions[..., 3:4] - joint_pos[:, 13:14][:, None]
                ) / 100
            else:
                raise NotImplementedError
        actions = actions.reshape(downsample_horizon, self.skip_frame * action_dim)
        is_first = np.zeros((downsample_horizon,), dtype=np.float32)
        is_first[0] = 1.0
        with torch.no_grad():
            actions = torch.from_numpy(actions).requires_grad_(False)
            norm_actions = self.normalizer["action"].normalize(actions).detach()
        data = {
            "image": torch.from_numpy(obs_dict[self.rgb_keys[0]]),
            "action": norm_actions,
            "is_first": torch.from_numpy(is_first),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data


def dilute_masks(masks: torch.Tensor, n_dilute: int = 10) -> torch.Tensor:
    """Dilute masks using opencv
    Args:
        masks: (B, H, W)
        n_dilute: number of iterations of dilation
    Returns:
        (B, H, W)
    """
    for i in range(masks.shape[0]):
        mask = masks[i]
        mask = mask.detach().cpu().numpy()
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=n_dilute)
        mask = torch.from_numpy(mask).to(masks.device)
        masks[i] = mask
    return masks


def test_real_aloha_dataset() -> None:
    config_path = (
        "/home/yixuan/diffusion-forcing/configurations/dataset/real_aloha_dataset.yaml"
    )
    cfg = OmegaConf.load(config_path)
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/real_aloha/pusht_0412/"  # noqa
    cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/diffusion_wm_rhythm/big_T/2025-08-02_expert_training_data_random_init"  # noqa
    cfg.obs_keys = ["camera_1_color"]
    cfg.delta_action = False
    cfg.horizon = 10
    cfg.action_mode = "bimanual_push"
    cfg.shape_meta.action.shape = (4,)
    cfg.skip_frame = 5
    cfg.skip_idx = 20
    cfg.val_ratio = 0.1
    cfg.val_horizon = 30
    cfg.goal_sample = "aggressive"
    cfg.resolution = 128
    dataset = RealAlohaDataset(cfg)
    batch_size = 64
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=32, shuffle=False
    )

    for j, data in enumerate(tqdm(dataloader)):
        # for data in dataset:
        init_img = data["obs"]["camera_1_color"][:, 0]  # (B, 3, H, W)
        goal_img = data["goal"]["camera_1_color"]  # (B, 3, H, W)
        init_mask = data["obs"]["camera_1_color_mask"][:, 0:1]  # (B, 1, H, W)
        goal_mask = data["goal"]["camera_1_color_mask"][:, None]  # (B, 1, H, W)
        init_img = init_img.permute(0, 2, 3, 1)
        goal_img = goal_img.permute(0, 2, 3, 1)
        init_mask = init_mask.permute(0, 2, 3, 1)
        goal_mask = goal_mask.permute(0, 2, 3, 1)
        init_mask[..., 0] = dilute_masks(init_mask[..., 0], n_dilute=1)
        goal_mask[..., 0] = dilute_masks(goal_mask[..., 0], n_dilute=1)
        if j == 0:
            base_init_img = init_img[0]
            base_goal_img = goal_img[0]
            base_init_count = torch.ones((cfg.resolution, cfg.resolution, 1))
            base_goal_count = torch.ones((cfg.resolution, cfg.resolution, 1))
        for i in range(init_img.shape[0]):
            init_mask_i = init_mask[i, :, :, 0].to(torch.bool)
            init_count = base_init_count[init_mask_i]
            base_init_img[init_mask_i] = base_init_img[init_mask_i] * (
                1 - 1 / (init_count + 1)
            ) + init_img[i][init_mask_i] * 1 / (init_count + 1)
            base_init_count[init_mask_i] += 1
            goal_mask_i = goal_mask[i, :, :, 0].to(torch.bool)
            goal_count = base_goal_count[goal_mask_i]
            base_goal_img[goal_mask_i] = base_goal_img[goal_mask_i] * (
                1 - 1 / (goal_count + 1)
            ) + goal_img[i][goal_mask_i] * 1 / (goal_count + 1)
            base_goal_count[goal_mask_i] += 1

    base_init_img_np = base_init_img.detach().cpu().numpy()
    base_goal_img_np = base_goal_img.detach().cpu().numpy()
    base_init_img_np = (base_init_img_np * 255).astype(np.uint8)
    base_goal_img_np = (base_goal_img_np * 255).astype(np.uint8)
    base_init_img_np = cv2.cvtColor(base_init_img_np, cv2.COLOR_RGB2BGR)
    base_goal_img_np = cv2.cvtColor(base_goal_img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite("base_init_img.png", base_init_img_np)
    cv2.imwrite("base_goal_img.png", base_goal_img_np)


if __name__ == "__main__":
    test_real_aloha_dataset()
