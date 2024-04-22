import argparse
import torch
from torch import nn
import torch.nn.functional as F
from unet import Unet
from fins.fins.model import FilteredNoiseShaper, Encoder
from fins.fins.utils.utils import load_config
import torch_liberator

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
        # self.fins.load_state_dict(fins_checkpoint['model_state_dict'])
        # self.fins = self.fins.encoder
        self.fins = self.fins.encoder
        torch_liberator.load_partial_state(self.fins, fins_checkpoint['model_state_dict'], verbose=0)
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
        # audio shape: (B, 1, num_samples)
        # speaker pos shape: (B, 1, 1, 3)
        # mic pos shape: (B, 1, 1, 3)

        breakpoint()
        channel_latent = self.fins(audio) # Bx128

        combined = torch.cat([channel_latent,           # Bx128
                              speaker_pos.squeeze(),    # Bx3
                              mic_pos.squeeze()],       # Bx3
                              dim=-1) 

        audio_cond = self.audio_cond_net(combined)

        res = images.permute(0, 3, 1, 2) # (B, 3, H, W)
        res = self.unet(res, audio_cond)
        res = self.sigmoid(res)

        return res

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

    args = parser.parse_args()

    model = AudioVisualModel(args)
    model.to(args.device)
    breakpoint()

    img = torch.rand(16, 512, 512, 3).to(args.device)
    audio = torch.rand(16, 1, 10000).to(args.device)
    speaker_pos = torch.rand(16, 3).to(args.device)
    mic_pos = torch.rand(16, 3).to(args.device)

    out = model(img, audio, speaker_pos, mic_pos)
