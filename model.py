
from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch
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
        self.unet = None # TODO

    def forward(self, images, audio):
        # images shape: (B, H, W)
        # audio_cond shape: (B, M)
        audio_cond = self.fins(audio)
        # TODO