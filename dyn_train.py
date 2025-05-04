import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.dict import Dict
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

import models
from datasets.sim_aloha_dataset import SimAlohaDataset

Path('ckpt').mkdir(parents=True, exist_ok=True)

obs_space = Dict({'image': Box(low=0, high=255, shape=(128,128,3), dtype=np.uint8)})
act_space = Box(low=-1, high=1, shape=(20,), dtype=np.float32)
config_path = 'my_config.yaml'
config = OmegaConf.load(config_path)
config.num_actions = act_space.shape[0]

dataset_cfg_path = 'sim_aloha_dataset.yaml'
dataset_cfg = OmegaConf.load(dataset_cfg_path)
train_dataset = SimAlohaDataset(dataset_cfg)
val_dataset = train_dataset.get_validation_dataset()
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=min(os.cpu_count(), 8),
    persistent_workers=True,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=min(os.cpu_count(), 8),
    persistent_workers=True,
)

wandb.init(
    project="dreamerv3",
    name=f"{config.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    entity="yixuan1999",
    config=OmegaConf.to_container(config, resolve=True),
    mode="online",
)

wm = models.WorldModel(obs_space, act_space, 0, config)
wm.to(config.device)

num_epoch = 1000
step_i = 0
pred_vid_every_steps = 1000
val_every_steps = 10
ckpt_every_steps = 1000

for epoch in range(num_epoch):
    for data in tqdm(train_dataloader, desc=f"Epoch {epoch}", total=len(train_dataloader)):
        post, context, mets = wm._train(data)
        
        # log train metrics
        log_mets = {"train/" + k: v for k, v in mets.items()}
        wandb.log(log_mets)
        if step_i % pred_vid_every_steps == 0:
            vid = wm.video_pred(data)
            for i in range(vid.shape[0]):
                vid_np = vid[i].detach().cpu().numpy()
                vid_np = np.transpose(vid_np, (0, 3, 1, 2))
                vid_np = (vid_np * 255).astype(np.uint8)
                wandb.log({f"train_vis/video_{i}": wandb.Video(vid_np, caption=f"video_{i}")})
        
        # val
        if step_i % val_every_steps == 0:
            val_data = next(iter(val_dataloader))
            post, context, mets = wm._val(val_data)
            log_mets = {"val/" + k: v for k, v in mets.items()}
            wandb.log(log_mets)
            vid = wm.video_pred(val_data)
            for i in range(vid.shape[0]):
                vid_np = vid[i].detach().cpu().numpy()
                vid_np = np.transpose(vid_np, (0, 3, 1, 2))
                vid_np = (vid_np * 255).astype(np.uint8)
                wandb.log({f"val_vis/video_{i}": wandb.Video(vid_np, caption=f"video_{i}")})
        step_i += 1
        
        # save model
        if step_i % ckpt_every_steps == 0:
            ckpt_path = f"ckpt/latest.ckpt"
            torch.save(wm.state_dict(), ckpt_path)
    
