import torch
from torch import nn
import torch.nn.functional as F
from unet import Unet
from fins.fins.model import FilteredNoiseShaper, Encoder
from fins.fins.utils.utils import load_config

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

        # self.fins = Encoder()
        fins_config = load_config(args.fins_config)
        self.fins = FilteredNoiseShaper(fins_config.model.params)

        # TODO: Find better way to first get encoder, and only load encoder weights
        fins_checkpoint = torch.load(args.fins_checkpoint)
        self.fins.load_state_dict(fins_checkpoint['model_state_dict'])
        self.fins = self.fins.encoder
        self.fins.requires_grad_(args.train_fins) # optionally freeze encoder

        # image shape (B, 512, 512, 3)
        # transpose to (B, 3, 512, 512)
        # reshape to (B, 512, 512)
        self.unet = Unet(backbone=args.backbone, 
                         encoder_freeze=args.backbone_freeze, 
                         pretrained=args.backbone_pretrained, 
                         preprocessing=True, 
                         in_channels=3,
                         num_classes=1,
                         audio_attn_block=args.audio_attn_block
                         )

        self.audio_cond_net = nn.Sequential(nn.Linear(128+3+3, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, self.unet.encoder_channels[0]),
                                            nn.ReLU(),
                                            )

        self.sigmoid = nn.Sigmoid() # (B, 512, 512)

    def forward(self, images, audio, speaker_pos, mic_pos):
        # images shape: (B, H, W, 3)
        # audio_cond shape: (B, M)
        channel_latent = self.fins(audio) # Bx128

        B, M = channel_latent.shape
        print(channel_latent.shape)

        audio_cond = self.audio_cond_net(torch.cat([channel_latent, # Bx128
                                                    speaker_pos, # Bx3
                                                    mic_pos], 
                                                   dim=-1) # Bx3
                                         )

        res = images.permute(0, 3, 1, 2) # (B, 3, H, W)
        res = self.unet(res, audio_cond)
        res = self.sigmoid(res)

        return res
