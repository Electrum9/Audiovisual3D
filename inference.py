import argparse
import glob
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pdb
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from model import AudioVisualModel

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
    def __init__(self, args, use_world=True):
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
            rgb = torch.tensor(np.array([self.midas_transforms.small_transform(img).squeeze() for img in rgb]))
            
        else:
            rgb = torch.from_numpy(rgb)
            rgb = rgb.permute(-1, 0, 1, 2) # (B, 3, H, W)
        micloc = torch.from_numpy(feed_dict[self.micloc_key])
        speakerloc = torch.from_numpy(feed_dict[self.speakerloc_key].astype(np.float32))
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


def inference_model(args):
    dataset = CustomImageDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = AudioVisualModel(args)
    model.to(args.device)
    model.eval()
    checkpoint = torch.load("/ix1/tibrahim/jil202/CMU16825/project/Audiovisual3D/checkpoint_av_10000.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    train_loader = iter(dataloader)
    loss_vis = 0
    for idx in range(len(train_loader)):
        rgb, audio, speaker_pos, mic_pos, depths_gt = next(train_loader)
        
        rgb = rgb.to(args.device)
        audio = audio.to(args.device)
        speaker_pos = speaker_pos.to(args.device)
        mic_pos = mic_pos.to(args.device)
        depths_gt = depths_gt.to(args.device)

        with torch.no_grad():
            depths_pred, transforms = model(rgb, audio, speaker_pos, mic_pos)
            pdb.set_trace()
            scales = transforms[:, 0].unsqueeze(2).unsqueeze(3) # (B, 8, 1, 1)
            translations = transforms[:, 1].unsqueeze(2).unsqueeze(3) # (B, 8, 1, 1)
            depths_pred = depths_pred * scales + translations # (B, 8, 512, 512)
            
            loss = scaled_depth_loss(depths_pred, depths_gt)
            loss_vis += loss.cpu().item()

            img = depths_pred.detach().cpu()
            img_gt = depths_gt.detach().cpu()
            rgb_img_gt = rgb.detach().cpu()
            img = plt.get_cmap('viridis')((img[0,0] - img[0,0].min())/(img[0,0].max() - img[0,0].min()))
            showimg = (rgb_img_gt[0,0,...] + abs((rgb_img_gt[0,0,...].min()))).unsqueeze(0)
            showimg = F.interpolate(showimg, [512, 512], mode='bilinear', align_corners=False).squeeze().permute(1,2,0)
            img_gt = plt.get_cmap('viridis')((img_gt[0,0] - img_gt[0,0].min())/(img_gt[0,0].max() - img_gt[0,0].min()))
            output = torch.hstack((showimg/showimg.max(), torch.tensor(img_gt[...,:3]), torch.tensor(img[...,:3])))
            plt.imsave(f"rgb_gt_images/test_{idx}.png", output.numpy())
            
    
        
        
    
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
    parser.add_argument("--dataset_path", default='data-8-livingroom/', type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--mixed_precision", default=True, type=bool)
    parser.add_argument("--use_midas", default=True, type=bool)
    parser.add_argument("--midas_checkpoint", default="./midas_v21_384.pt", type=str)
    parser.add_argument("--midas_freeze", default=True, type=bool)

    args = parser.parse_args()
    inference_model(args)
