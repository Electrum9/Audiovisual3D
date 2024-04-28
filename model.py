import argparse
import torch
from torch import nn
import torch.nn.functional as F
from unet import Unet
from fins.fins.model import FilteredNoiseShaper, Encoder
from fins.fins.utils.utils import load_config
import torch_liberator

from midas_net import MidasNet

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
        self.args = args
        if args.use_midas:
            midas_features = 256
            self.imageaudio_fusion_net = MidasNet(features=midas_features)
            midas_checkpoint = torch.load(args.midas_checkpoint)
            torch_liberator.load_partial_state(self.imageaudio_fusion_net, model_state_dict=midas_checkpoint)

            if args.midas_freeze:
                self.imageaudio_fusion_net.requires_grad_(False)
                # self.imageaudio_fusion_net.scratch.output_conv.requires_grad_(True)

                # for c in self.imageaudio_fusion_net.scratch.output_conv.children():
                #     if hasattr(c, 'weight'):
                #         torch.nn.init.kaiming_normal_(c.weight)

                if hasattr(self.imageaudio_fusion_net, 'audio_attn_block'):
                    self.imageaudio_fusion_net.audio_attn_block.requires_grad_(True)

            self.encoder_out_channels = midas_features
        else:
            self.imageaudio_fusion_net = Unet(backbone=args.backbone, 
                             encoder_freeze=args.backbone_freeze, 
                             pretrained=args.backbone_pretrained, 
                             preprocessing=True, 
                             in_channels=3,
                             num_classes=1,
                             audio_attn_block=args.audio_attn_block
                             )
            self.encoder_out_channels = self.imageaudio_fusion_net.encoder_channels[0]

        self.audio_cond_net = nn.Sequential(nn.Linear(128+3+3, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, self.encoder_out_channels),
                                            nn.ReLU(),
                                            )

        # self.sigmoid = nn.Sigmoid() # (B, 512, 512)
        self.relu = nn.ReLU()

    def forward(self, images, audio, speaker_pos, mic_pos):
        # images shape: (B, H, W, 3)
        # audio shape: (B, 1, num_samples)
        # speaker pos shape: (B, 1, 1, 3)
        # mic pos shape: (B, 1, 1, 3)

        channel_latent = self.fins(audio) # Bx128

        combined = torch.cat([channel_latent,           # Bx128
                              speaker_pos.squeeze(),    # Bx3
                              mic_pos.squeeze()],       # Bx3
                              dim=-1) 

        audio_cond = self.audio_cond_net(combined)

        res = self.imageaudio_fusion_net(images, audio_cond)
        # res = self.sigmoid(res)

        if self.args.use_midas:
            disparity, scale_translation_factors = res
            depth = 1 / (disparity + 1e-5)
            return depth, scale_translation_factors

        else:
            # TODO: Rewrite code so you could use UNET and decode depth maps, scale and translation factors
            res = self.relu(res)
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
    parser.add_argument("--dataset_path", default='data/', type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--mixed_precision", default=True, type=bool)
    parser.add_argument("--use_midas", default=False, type=bool)
    parser.add_argument("--midas_checkpoint", default="./midas_v21_384.pt", type=str)
    parser.add_argument("--midas_freeze", default=True, type=bool)

    args = parser.parse_args()

    model = AudioVisualModel(args)
    model.to(args.device)
    breakpoint()

    # midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    imgs = torch.rand(args.batch_size, 8, 3, 256, 256).to(args.device) # image is 256x256 after midas transforms
    # imgs = imgs.view(-1, 512, 512, 3)
    # rgb = midas_transforms.small_transform(imgs).squeeze()
    # imgs = imgs.view(args.batch_size, 8, 512, 512, 3)
    audio = torch.rand(args.batch_size, 1, 10000).to(args.device)
    speaker_pos = torch.rand(args.batch_size, 3).to(args.device)
    mic_pos = torch.rand(args.batch_size, 3).to(args.device)

    out = model(imgs, audio, speaker_pos, mic_pos)
