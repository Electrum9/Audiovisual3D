import argparse
import glob
from math import isnan
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.transforms import Resize
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import v2

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import AudioVisualModel

torch.autograd.set_detect_anomaly(True)

def loss_sstrim(diff):
    B, M = diff.shape
    diff_abs = torch.abs(diff)
    diff_abs_sorted, _ = torch.sort(diff_abs, dim = 1)
    trimmed = diff_abs_sorted[:, :int(M * .8)]
    return 1 / (2 * M) * torch.sum(trimmed, dim = 1)
    
def loss_reg(diff):
    # TODO: not sure of shape of grad
    return 0

def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
    Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        grad_x = abs(conv_gauss(diff, Gx))
        grad_y = abs(conv_gauss(diff, Gy))
        miniloss = grad_x + grad_y
        pyr.append(miniloss)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=1, device=torch.device('cpu')):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

def loss_midas(disparity_pred, depth_gt):
    # input: pred and gt of shape (B, H, W), grad of shape ???
    # output: losses of shape (B,)
    B = depth_gt.shape[0]
    torch.nn.functional.relu(depth_gt, inplace=True) # clip negative values

    if torch.any(torch.isnan(disparity_pred)).item():
        breakpoint()

    B, H, W = disparity_pred.shape
    disparity_pred_1d = disparity_pred.view(B, -1)
    disparity_gt_1d = 1 / (depth_gt.view(B, -1) + 1e-5) # make this disparity

    translation_pred = torch.median(disparity_pred_1d, dim=-1)[0].unsqueeze(-1)
    scale_pred = torch.norm(disparity_pred_1d - translation_pred, dim=-1, p=1).unsqueeze(-1)
    disparity_pred_1d = disparity_pred_1d.shape[-1] * (((disparity_pred_1d - translation_pred) / (scale_pred + 1e-5))) 

    translation_gt = torch.median(disparity_gt_1d, dim=-1)[0].unsqueeze(-1)
    scale_gt = torch.norm(disparity_gt_1d - translation_gt, dim=-1, p=1).unsqueeze(-1)
    disparity_gt_1d = disparity_gt_1d.shape[-1] * (disparity_gt_1d - translation_gt) / (scale_gt + 1e-5)

    diff = disparity_pred_1d - disparity_gt_1d

    loss = loss_sstrim(diff) 
    reg = LapLoss(max_levels=4, device=torch.device('cuda'))(disparity_pred_1d.reshape(B, 1, H, W), disparity_gt_1d.reshape(B, 1, H, W))

    total = torch.mean(loss + reg)

    if torch.isnan(total):
        breakpoint()

    return total
    
def loss_log(depth_pred, depth_gt):
    B = depth_pred.shape[0]
    eps = 1e-5
    torch.nn.functional.relu(depth_gt, inplace=True) # clip negative values
    depth_pred_1d = depth_pred.view(B, -1)
    depth_gt_1d = depth_gt.view(B, -1)

    breakpoint()
    
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
        self.resize = Resize((256, 256))
        self.aug = v2.RandomChoice([v2.RandomResizedCrop(256, scale=(0.75,1)), v2.RandomErasing(p=1), v2.Lambda(lambda x: x + 0.5*torch.randn_like(x)), torch.nn.Identity()])
        self.args = args

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        feed_dict = torch.load(self.pt_files[idx])
        audio = feed_dict['audio'].astype(np.float32)
        rgb = feed_dict['rgb'].astype(np.float32)
        if self.args.use_midas:
            rgb = torch.stack([self.aug(self.midas_transforms.small_transform(x).squeeze()) for x in rgb], dim=1)
        else:
            rgb = torch.from_numpy(rgb)
            rgb = rgb.permute(0, 3, 1, 2) # (8, H, W, 3)
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

    writer = SummaryWriter()

    scaler = torch.cuda.amp.GradScaler()

    if args.load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")
        
    print("Starting training !")

    train_loader = iter(dataloader)

    resize = Resize((256, 256))

    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()
        read_start_time = time.time()
        
        if step % len(dataloader) == 0:
            train_loader = iter(dataloader)

        with torch.cuda.amp.autocast(args.mixed_precision):
            rgb, audio, speaker_pos, mic_pos, depths_gt = next(train_loader)

            rgb = rgb.permute(0, 2, 1, 3, 4)
            rgb = rgb.reshape(-1, *rgb.shape[2:])
            audio = audio.unsqueeze(2).expand(-1, 8, -1, -1)
            audio = audio.reshape(-1, *audio.shape[2:])
            speaker_pos = speaker_pos.reshape(-1, *speaker_pos.shape[2:])
            mic_pos = mic_pos.reshape(-1, *mic_pos.shape[2:])
            depths_gt = depths_gt.reshape(-1, *depths_gt.shape[2:])

            rgb = resize(rgb.to(args.device))
            audio = audio.to(args.device)
            speaker_pos = speaker_pos.to(args.device)
            mic_pos = mic_pos.to(args.device)
            depths_gt = resize(depths_gt.to(args.device))
            # depths_gt = depths_gt - depths_gt.min()

            read_time = time.time() - read_start_time

            depths_pred = model(rgb, audio, speaker_pos, mic_pos)

            if (step % 50) == 0:
                rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min())
                save_image(rgb_normalized, f"gt_images/gt_img_rgb{step}.png")
                depths_gt_normalized = (depths_gt - depths_gt.min()).unsqueeze(1).expand(-1, 3, -1, -1) / (depths_gt.max() - depths_gt.min())
                save_image(depths_gt_normalized, f"gt_images/gt_img{step}.png")
                actual_depths_pred = 1 / (depths_pred + 1e-5)
                depths_pred_normalized = (actual_depths_pred - actual_depths_pred.min()).unsqueeze(1).expand(-1, 3, -1, -1) / (actual_depths_pred.max() - actual_depths_pred.min())
                save_image(depths_pred_normalized, f"pred_images/img_{step}.png")
                img_pred = torchvision.utils.make_grid(depths_pred_normalized)
                img_gt = torchvision.utils.make_grid(depths_gt_normalized)
                img_rgb = torchvision.utils.make_grid(rgb)
                writer.add_image('RGB', img_rgb)
                writer.add_image('Depth Pred', img_pred)
                writer.add_image('Depth GT', img_gt)

        # depths_pred = depths_pred.to(dtype=torch.float32)
        # depths_gt = depths_gt.to(dtype=torch.float32)
            # loss = depth_loss(depths_pred, depths_gt)
            loss = loss_midas(depths_pred, depths_gt)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # rgb_img = rgb.detach().cpu().numpy()
        # rgb_img /= rgb_img.max()
        # img /= img.max()
        # img_gt = depths_gt.detach().cpu().numpy()
        # img_gt /= img_gt.max()
        # plt.imsave(f"pred_images/img_{step}.png", img[0,...])
        # plt.imsave(f"gt_images/gt_img_{step}.png", img_gt[0,...])
        # plt.imsave(f"gt_images/gt_img_rgb{step}.png", rgb_img)

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

        writer.add_scalar('Loss/train', loss_vis, step) 

    
    writer.close()
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
