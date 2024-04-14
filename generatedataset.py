
import os
from random import random
import shutil

import numpy as np
import pytorch3d

N = 1000
image_size = 512
device = "cuda"

num_datapoints = None # TODO
datapoint_locations = [] # TODO; the path to the rooms

def get_datapoint(path):
    # TODO
    # get (mesh, audio) tuple in dataset determined by `path`
    pass

def get_random_datapoint():
    return get_datapoint(datapoint_locations[int(random() * num_datapoints)])

def get_random_camera():
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 3.0, # TODO: maybe change? # np.linspace(6, 0, num_views, endpoint=False),
        elev = 0,
        azim=random() * 360 - 180, # np.linspace(-180, 180, num_views, endpoint=False),
        azim = 0
    )

    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    
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