
import os
from random import random
import shutil

import numpy as np
import pytorch3d

from torchvision.transforms import v2
import numpy as np

N = 1000
image_size = 512
device = "cuda"

def get_random_datapoint():
    # TODO: Lee
    # output: dataset tuple (audio, origin, mic location, spaker location, boundaries, mesh)
    return

def apply_augmentation(image):
    # output: image with random augmentation applied
    augmentations = [v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
                     v2.RandomPosterize(bits=2),
                     v2.RandomAdjustSharpness(sharpness_factor=2),
                     v2.RandomAutocontrast()
                     ]

    aug_idx = np.random.choice(len(augmentations),1)
    aug = augmentations[aug_idx]
    
    return aug(image)

def get_random_camera():
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 3.0,
        elev = 0,
        azim=random() * 360 - 180,
    )

    camera = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    return camera
    
def get_rgb(mesh, camera, renderer, lights):
    rend = renderer(mesh, cameras=camera, lights=lights)
    return (rend.detach().cpu().numpy()[0, ..., :3] * 255).astype(np.uint8)

# TODO: verify that it outputs depths correctly
def get_depth_map(mesh, camera, rasterizer):
    fragments = rasterizer(mesh, cameras=camera)
    depth_map = fragment.zbuf[:,:,:,0] / fragment.zbuf[:,:,:,0].max()
    return depth_map
    
def save_rgb(rgb, i):
    np.save(f"data/{i}/rgb.npy", audio)
    
def save_depth_map(depth_map, i):
    np.save(f"data/{i}/depthmap.npy", audio)
    
def save_audio(audio, i):
    # not really sure what format this is lol
    np.save(f"data/{i}/audio.npy", audio)
    
def save_datapoint(rgb, depth_map, audio, i):
    save_rgb(rgb, i)
    save_depth_map(depth_map, i)
    save_audio(audio, i)

if __name__ == "__main__":
    shutil.rmtree("data")
    os.makedirs("data")
    raster_settings = pytorch3d.renderer.RasterizationSettings(image_size=image_size)
    rasterizer = pytorch3d.renderer.MeshRasterizer(
        raster_settings=raster_settings,
    )
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=rasterizer,
        shader=shader,
    )
    lights = None # TODO? I'm not sure if we can use shader instead
    for i in range(N):
        os.makedirs(f"data/{i}")
        mesh, audio = get_random_datapoint()
        camera = get_random_camera()
        rgb = get_rgb(mesh, camera, renderer, lights)
        depth_map = get_depth_map(mesh, camera, rasterizer)
        save_datapoint(rgb, depth_map, audio, i)
