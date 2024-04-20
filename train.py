import argparse
import time

import os

import imageio
import numpy as np
import torch
from model import AudioVisualModel

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
    depth_pred_1d = depth_pred.view(B, -1)
    depth_gt_1d = depth_gt.view(B, -1)
    pred_log = torch.log(depth_pred_1d)
    gt_log = torch.log(depth_gt_1d)
    diff = pred_log - gt_log
    B, M = diff.shape
    alpha = -1 / M * torch.sum(diff, dim = 1)
    losses = diff + alpha
    losses = losses * losses
    return torch.sum(losses, dim = 1)
    
def depth_loss(depth_pred, depth_gt):
    return loss_log(depth_pred, depth_gt)

def preprocess(feed_dict, args):
    images = feed_dict["images"]
    audio = feed_dict["audio"]
    depths = feed_dict["depths"]
    return images, depths, audio
    
class AVLoader(object):
    def __init__(self, args):
        points = os.listdir("data/")
        self.n = len(points)
        self.B = args.batch_size
        self.images_shape = (self.B, args.image_size, args.image_size)
        self.audio_shape = (self.B, None) # TODO
        self.depths_shape = (self.B, None, None) # TODO
    def next_datapoint(self):
        res = {}
        images = np.zeros(self.images_shape)
        audio = np.zeros(self.audio_shape)
        depths = np.zeros(self.depths_shape)
        for i in range(self.B):
            images[i] = np.load(f"data/{i}/rgb.npy")
            audio[i] = np.load(f"data/{i}/audio.npy")
            depths[i] = np.load(f"data/{i}/depths.npy")
        res["images"] = torch.from_numpy(images)
        res["audio"] = torch.from_numpy(audio)
        res["depths"] = torch.from_numpy(depths)
        return res

def train_model(args):
    model = AudioVisualModel(args)
    model.to(args.device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_iter = 0
    start_time = time.time()
    
    loader = AVLoader(args)

    if args.load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")
        
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()
        
        feed_dict = loader.next_datapoint()

        read_start_time = time.time()

        images_gt, depths_gt, audio = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        depths_pred = model(images_gt, audio)

        grad = None # TODO: not sure how to retrieve this
        loss = depth_loss(depths_pred, depths_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    parser.add_argument("--fins_checkpoint", default='./fins/checkpoints/epoch19.pt', type=str)
    parser.add_argument("--train_fins", default=False, type=bool)
    parser.add_argument("--backbone", default='resnet50', type=str)
    parser.add_argument("--backbone_freeze", default=False, type=bool)
    parser.add_argument("--backbone_pretrained", default=True, type=bool)
    parser.add_argument("--audio_attn_block", default=False, type=bool)

    args = parser.parse_args()
    train_model(args)
