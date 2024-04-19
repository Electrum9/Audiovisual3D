from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch
from torch import nn
import torch.nn.functional as F

from fins.model import Encoder

# shape defs:
# B: batch size
# H: image height
# W: image width
# M: latent audio 
class AudioVisualModel(nn.Module):
    def __init__(self, args):
        super(AudioVisualModel, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.fins = Encoder()
        # image shape (B, 512, 512, 3)
        # transpose to (B, 3, 512, 512)
        M = 200 # TODO
        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(), # (B, 64, 257, 257)
            nn.Conv2d(3, 64, 64, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (B, 64, 129, 129)
            nn.Conv2d(3, 64, 64, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(), # (B, 64, 66, 66)
            nn.Conv2d(3, 64, 128, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(), # (B, 128, 68, 68)
            nn.Conv2d(3, 128, 128, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)) # (B, 64, 35, 35)
        # reshape to (B, 64 * 35 * 35)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 35 * 35, 1024)
            nn.ReLU())
        # append audio conditioning
        self.fc2 = nn.Sequential(
            nn.Linear(1024 + M, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU())
        # skip layer for audio conditioning
        self.fc3 = nn.Sequential(
            nn.Linear(1024 + M, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU())
        # reshape feature vector for convtranspose, (B, 1024, 1, 1)
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1), # (B, 512, 2, 2)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # (B, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), # (B, 16, 64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), # (B, 8, 128, 128)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, 2, 1), # (B, 4, 256, 256)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 4, 2, 1), # (B, 2, 512, 512)
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 1)) # (B, 1, 512, 512)
        # reshape to (B, 512, 512)
        self.sigmoid = nn.Sigmoid() # (B, 512, 512)

    def forward(self, images, audio):
        # images shape: (B, H, W, 3)
        # audio_cond shape: (B, M)
        audio_cond = self.fins(audio)
        B, M = audio_cond.shape
        print(audio_cond.shape) # to find out M
        # TODO
        res = images.permute(0, 3, 1, 2) # (B, 3, H, W)
        res = self.convs(res) # (B, 64, 35, 35)
        res = res.view(B, -1) # (B, 64 * 35 * 35)
        res = self.fc1(res) # (B, 1024)
        res = torch.cat((res, audio_cond), dim = 1) # (B, 1024 + M)
        res = self.fc2(res) # (B, 1024)
        res = torch.cat((res, audio_cond), dim = 1) # (B, 1024 + M)
        res = self.fc3(res) # (B, 1024)
        res = res.view(B, 1024, 1, 1) # (B, 1024, 1, 1)
        res = self.conv_trans(res) # (B, 1, 512, 512)
        res = res.squeeze() # (B, 512, 512)
        res = self.sigmoid(res) # (B, 512, 512)
        return res
