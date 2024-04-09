import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline

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
        self.unet = None # TODO

    def forward(self, images, audio_cond):
        # images shape: (B, H, W)
        # audio_cond shape: (B, M)
        # TODO
        pass