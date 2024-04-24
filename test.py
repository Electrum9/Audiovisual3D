import numpy as np
import argparse

import imageio
import pytorch3d
import torch

from model import AudioVisualModel
from train import CustomImageDataset

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--fins_config", default='./fins/fins/config.yaml', type=str)
    parser.add_argument("--fins_checkpoint", default='./fins/checkpoints/epoch-20.pt', type=str)
    parser.add_argument("--train_fins", default=False, type=bool)
    parser.add_argument("--backbone", default='resnet50', type=str)
    parser.add_argument("--backbone_freeze", default=False, type=bool)
    parser.add_argument("--backbone_pretrained", default=True, type=bool)
    parser.add_argument("--audio_attn_block", default=False, type=bool)
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument("--dataset_path", default='data/', type=str)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    num_samples = 5

    # ------ TO DO: Initialize Model for Classification Task ------
    model = AudioVisualModel(args)
    
    # Load Model Checkpoint
    model_path = './checkpoints/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    print ("successfully loaded checkpoint from {}".format(model_path))
    dataset = CustomImageDataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = iter(dataloader)
    os.makedirs('res', exist_ok=True)

    for i in range(num_samples):
        rgb, audio, speaker_pos, mic_pos, depths_gt = next(test_loader)

        rgb = rgb.to(args.device)
        audio = audio.to(args.device)
        speaker_pos = speaker_pos.to(args.device)
        mic_pos = mic_pos.to(args.device)
        depths_gt = depths_gt.to(args.device)
        
        depths_pred = model(rgb, audio, speaker_pos, mic_pos)

        rgb = (rgb.detach().squeeze().cpu().numpy() * 255).astype(np.uint8)
        depths_gt = (depths_gt.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
        depths_pred = (depths_pred.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
        imageio.imsave(f"res/{i}_rgb.png", rgb)
        imageio.imsave(f"res/{i}_gt.png", depths_gt)
        imageio.imsave(f"res/{i}_pred.png", depths_pred)



