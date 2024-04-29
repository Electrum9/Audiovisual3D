"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from blocks import FeatureFusionBlock, Interpolate, _make_encoder

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


class MidasNet(torch.nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True, audio_attn_block=False):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()

        if audio_attn_block:
            # Spatial Excitation Block: Outputs Bx1xHxW set of scaling factors in interval [0, 1]
            self.audio_attn_block = nn.Sequential(nn.Conv2d(in_channels=features*8,
                                                     out_channels=1, 
                                                     kernel_size=1,
                                                     ), 
                                           nn.Sigmoid(),
                                           )
        else:
            self.audio_attn_block = None

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(backbone="resnext101_wsl", features=features, use_pretrained=use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)

    def forward(self, x, audio_cond):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        if self.audio_attn_block:
            excitation = self.audio_attn_block(layer_4_rn) # Bx1xHxW set of attention scores in [0,1]
            layer_4_rn = excitation * layer_4_rn + (1 - excitation) * audio_cond.view(*audio_cond.shape, 1, 1)
        else:
            layer_4_rn = layer_4_rn + audio_cond.view(*audio_cond.shape, 1, 1) # broadcast along spatial dims

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        out = torch.squeeze(out, dim=1)

        out = torch.nn.functional.interpolate(
            out.unsqueeze(1),
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return out
