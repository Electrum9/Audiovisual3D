import argparse
import glob
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import AudioVisualModel

torch.autograd.set_detect_anomaly(False)

def loss_sstrim(diff):
    B, M = diff.shape
    diff_abs = torch.abs(diff)
    diff_abs_sorted, _ = torch.sort(diff_abs, dim = 1)
    trimmed = diff_abs_sorted[:, :int(M * .8)]
    return 1 / (2 * M) * torch.sum(trimmed, dim = 1)
    
def loss_reg(diff, grad):
    # TODO: not sure of shape of grad
    return 0

def loss_midas(depth_pred, depth_gt, grad):
    # input: pred and gt of shape (B, H, W), grad of shape ???
    # output: losses of shape (B,)
    B = depth_pred.shape[0]
    depth_pred_1d = depth_pred.view(B, -1)
    depth_gt_1d = depth_gt.view(B, -1)
    diff = depth_pred_1d - depth_gt_1d
    return loss_sstrim(diff) + loss_reg(diff, grad)
    
def loss_log(depth_pred, depth_gt):
    B = depth_pred.shape[0]
    eps = 1e-5
    torch.nn.functional.relu(depth_gt, inplace=True) # clip negative values
    depth_pred_1d = depth_pred.view(B, -1)
    depth_gt_1d = depth_gt.view(B, -1)
    pred_log = torch.log(depth_pred_1d + eps)
    gt_log = torch.log(depth_gt_1d + eps)
    diff = pred_log - gt_log
    B, M = diff.shape
    alpha = -1 / M * torch.sum(diff, dim = 1)
    alpha = alpha.unsqueeze(-1)
    losses = diff + alpha
    losses = losses * losses

    return (1/B) * torch.sum(losses)
    
def depth_loss(depth_pred, depth_gt):
    return loss_log(depth_pred, depth_gt)
    
def scaled_depth_loss(depth_pred, depth_gt):
    B = depth_pred.shape[0]
    eps = 1e-5
    torch.nn.functional.relu(depth_gt, inplace=True) # clip negative values
    depth_pred_1d = depth_pred.view(B, -1)
    depth_gt_1d = depth_gt.view(B, -1)
    pred_log = torch.log(depth_pred_1d + eps)
    gt_log = torch.log(depth_gt_1d + eps)
    losses = pred_log - gt_log
    losses = losses * losses
    return (1 / B) * torch.sum(losses)

class CustomImageDataset(Dataset):
    def __init__(self, args, use_world=False):
        pt_dir = args.dataset_path
        self.pt_files = list(Path(pt_dir).glob('*.pt'))

        if use_world:
            self.micloc_key = 'micloc_world'
            self.speakerloc_key = 'speakerloc_world'
        else:
            self.micloc_key = 'micloc_camera'
            self.speakerloc_key = 'speakerloc_camera'

        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.args = args

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        feed_dict = torch.load(self.pt_files[idx])
        audio = feed_dict['audio'].astype(np.float32)
        rgb = feed_dict['rgb'].astype(np.float32)
        if self.args.use_midas:
            rgb = self.midas_transforms.small_transform(rgb).squeeze()
        else:
            rgb = torch.from_numpy(rgb)
            rgb = rgb.permute(2, 0, 1) # (B, 3, H, W)
        micloc = torch.from_numpy(feed_dict['micloc_camera'])
        speakerloc = torch.from_numpy(feed_dict['speakerloc_camera'].astype(np.float32))
        depth = torch.from_numpy(feed_dict['depth'].astype(np.float32))
        return rgb, audio, speakerloc, micloc, depth

def collate_fn(batch):

    maxlen_audio = max(b[1].size for b in batch)
    stacked_audio = torch.stack([torch.from_numpy(np.pad(b[1], (0, maxlen_audio - b[1].size), 'constant')).unsqueeze(0) for b in batch], dim=0)
    stacked_rgb = torch.stack([b[0] for b in batch], dim=0)
    stacked_speakerloc = torch.stack([b[2] for b in batch], dim=0)
    stacked_micloc = torch.stack([b[3] for b in batch], dim=0)
    stacked_depth = torch.stack([b[4] for b in batch], dim=0)
    
    return stacked_rgb, stacked_audio, stacked_speakerloc, stacked_micloc, stacked_depth

def train_model(args):
    # loader = AVLoader(args)
    dataset = CustomImageDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = AudioVisualModel(args)
    model.to(args.device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_iter = 0
    start_time = time.time()

    scaler = torch.cuda.amp.GradScaler()

    if args.load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")
        
    print("Starting training !")
    train_loader = iter(dataloader)

    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()
        read_start_time = time.time()
        
        if step % len(dataloader) == 0: # completed an epoch
            train_loader = iter(dataloader)

        with torch.cuda.amp.autocast(args.mixed_precision):
            rgb, audio, speaker_pos, mic_pos, depths_gt = next(train_loader)

            rgb = rgb.to(args.device)
            audio = audio.to(args.device)
            speaker_pos = speaker_pos.to(args.device)
            mic_pos = mic_pos.to(args.device)
            depths_gt = depths_gt.to(args.device)

            read_time = time.time() - read_start_time

            depths_pred, transforms = model(rgb, audio, speaker_pos, mic_pos) # depths_pred: (B, 8, 512, 512), transforms: (B, 2, 8)
            scales = transforms[:, 0].unsqueeze(2).unsqueeze(3) # (B, 8, 1, 1)
            translations = transforms[:, 1].unsqueeze(2).unsqueeze(3) # (B, 8, 1, 1)
            depths_pred = depths_pred * scales + translations # (B, 8, 512, 512)
            

        # depths_pred = depths_pred.to(dtype=torch.float32)
        # depths_gt = depths_gt.to(dtype=torch.float32)
            loss = scaled_depth_loss(depths_pred, depths_gt)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        if step % 100 == 0:
            img = depths_pred.detach().cpu().numpy()
            img_gt = depths_gt.detach().cpu().numpy()
            rgb_img_gt = rgb.permute(0,2,3,1).detach().cpu().numpy()
            plt.imsave(f"rgb_gt_images/img_{step}.png", rgb_img_gt[0,...])
            plt.imsave(f"pred_images/img_{step}.png", img[0,...])
            plt.imsave(f"gt_images/gt_img_{step}.png", img_gt[0,...])

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if (step % args.save_freq) == 0:
            print(f"Saving checkpoint at step {step}")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoint_av.pth",
            )
                    

        print(
            "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f"
            % (step, args.max_iter, total_time, read_time, iter_time, loss_vis)
        )
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument("--max_iter", default=100000, type=int)
    parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--fins_config", default='./fins/fins/config.yaml', type=str)
    parser.add_argument("--fins_checkpoint", default='./fins/checkpoints/epoch-20.pt', type=str)
    parser.add_argument("--train_fins", default=False, type=bool)
    parser.add_argument("--backbone", default='resnet50', type=str)
    parser.add_argument("--backbone_freeze", default=False, type=bool)
    parser.add_argument("--backbone_pretrained", default=True, type=bool)
    parser.add_argument("--audio_attn_block", default=False, type=bool)
    parser.add_argument("--dataset_path", default='data/', type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--mixed_precision", default=True, type=bool)
    parser.add_argument("--use_midas", default=False, type=bool)
    parser.add_argument("--midas_checkpoint", default="./midas_v21_384.pt", type=str)
    parser.add_argument("--midas_freeze", default=True, type=bool)

    args = parser.parse_args()
    train_model(args)
