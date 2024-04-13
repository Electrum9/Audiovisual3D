import argparse
import time

import imageio
import torch
from model import AudioVisualModel

def depth_loss(depth_pred, depth_gt):
    # TODO
    pass

def fins(audio):
    # TODO: call the (pretrained?) FiNS model on the raw audio data
    # input could be: B audio files, B raw-data audio, etc
    # IDEA: sample a random, ~15s audio sample and pass that into FiNS?
    # idk how this step works help me vik
    pass

def preprocess(feed_dict, args):
    images = feed_dict["images"]
    audio = feed_dict["audio"]
    depths = feed_dict["depths"]
    return images, depths, audio

def train_model(args):
    av_dataset = None # TODO
    
    loader = None # TODO; should look something like the commented code below
    """
    loader = torch.utils.data.DataLoader(
        av_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=???,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    """
    train_loader = iter(loader)

    model = AudioVisualModel(args)
    model.to(args.device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")
        
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, depths_gt, audio = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        depths_pred = model(images_gt, audio)

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
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument("--max_iter", default=100000, type=int)
    parser.add_argument("--save_freq", default=2000, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    args = parser.parse_args()
    train_model(args)
