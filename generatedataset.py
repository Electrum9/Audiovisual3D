
import glob
import math
import os
from random import random
import shutil
import pdb
import numpy as np
import torch
import pytorch3d
from pytorch3d.io import load_objs_as_meshes
from torchvision.transforms import v2

N = 10
image_size = 512
device = "cuda"

def get_datapoint(pt_path):
    pt = torch.load(pt_path)
    room = pt['room']
    if room =='conference':
        room_obj_path = 'path_to_conference.obj'
    elif room == 'treatedroom':
        room_obj_path = 'path_to_treatedroom.obj'
    elif room == 'livingroom':
        room_obj_path = 'path_to_livingroom.obj'

    if pt['audio_mic_pos'] is None:
        pass
    else:
        audio = pt['convovled_audio']
        audio_mic = pt['audio_mic_pos']
        room_mics_pos = pt['mic_pos'] 
        audio_mic_pos = room_mics_pos[int(audio_mic)]
        speaker_pos = pt['speaker_pos']
        mesh = load_objs_as_meshes(room_obj_path)
        x_len = max(abs(pt['x_max']), abs(pt['x_min']))/2
        y_len = max(abs(pt['y_max']), abs(pt['y_min']))/2
        if room =='conference':
            height_len = 1.4906 #centered around zero [-1.4906, 1.4906]
        elif room == 'treatedroom':
            height_len = 1.513 #centered around zero [-1.513, 1.513]
        elif room == 'livingroom':
            height_len = 2.6372 #centered around zero [-2.6372, 2.6372]
            
        boundaries = np.array([[-x_len, -y_len, -height_len],
                                [x_len, y_len, height_len]])
        # TODO: Lee
        # output: dataset tuple (audio, origin, mic location, spaker location, boundaries, mesh)
    return (audio, np.array([[0, 0, 0]]), audio_mic_pos, speaker_pos, boundaries, mesh)

def get_random_datapoint():
    paths = glob.glob("orig_data/conference*.pt") + glob.glob("orig_data/treatedroom*.pt")
    return get_datapoint(paths[int(random() * len(paths))])

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

def get_random_camera(boundaries):
    # currently: chooses a random point in the middle of the air
    camera_loc = [choose_random(boundaries[0][0], boundaries[1][0]), (boundaries[0][1] + boundaries[1][1]) / 2, choose_random(boundaries[0][2], boundaries[1][2])]
    azim = choose_random(-180.0, 180.0)
    elev = choose_random(-45.0, 45.0)
    direc = [math.cos(elev) * math.sin(azim), math.sin(elev), math.cos(elev) * math.cos(azim)]
    R, T = pytorch3d.renderer.look_at_rotation(camera_loc, direc)

    camera = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    return camera
    
def get_rgb(mesh, camera, renderer, lights):
    rend = renderer(mesh, cameras=camera, lights=lights)
    return (rend.detach().cpu().numpy()[0, ..., :3] * 255).astype(np.uint8)

def get_depth_map(mesh, camera, rasterizer):
    fragments = rasterizer(mesh, cameras=camera)
    depth_map = fragments.zbuf[:,:,:,0] / fragments.zbuf[:,:,:,0].max()
    return depth_map
    
def choose_random(a, b):
    return (random() * (b - a)) + a
    
def get_random_lighting(boundaries):
    # currently: chooses a random point on the ceiling
    random_x = choose_random(boundaries[0][0], boundaries[1][0])
    random_z = choose_random(boundaries[0][2], boundaries[1][2])
    return pytorch3d.renderer.lighting.PointLights(location=[[random_x, boundaries[1][1], random_z]], device=device)
    
def to_camera_coords(pos, camera):
    return camera.get_world_to_view_transform().transform_points(pos)
    
def save_datapoint(rgb, depth_map, audio, mic_loc, speaker_loc, i):
    np.save(f"data/{i}/depthmap.npy", depth_map.cpu().detach.numpy())
    np.save(f"data/{i}/rgb.npy", rgb.cpu().detach.numpy())
    np.save(f"data/{i}/audio.npy", audio.cpu().detach.numpy())
    np.save(f"data/{i}/micloc.npy", mic_loc.cpu().detach.numpy())
    np.save(f"data/{i}/speakerloc.npy", speaker_loc.cpu().detach.numpy())

def set_boundaries_buffer(boundaries, margin_ratio):
    size = boundaries[1] - boundaries[0]
    margin = size * margin
    return torch.to_array([boundaries[0] + margin, boundaries[1] - margin], device = device)

def convert_to_torch(audio, origin, mic_loc, spaker_loc, boundaries, mesh):
    return (torch.to_array(audio, device = device), torch.to_array(origin, device = device), torch.to_array(mic_loc, device = device), torch.to_array(spaker_loc, device = device), torch.to_array(boundaries, device = device), torch.to_array(mesh, device = device))

if __name__ == "__main__":
    pdb.set_trace()
    shutil.rmtree("data", ignore_errors = True)
    os.makedirs("data")
    raster_settings = pytorch3d.renderer.RasterizationSettings(image_size=image_size)
    rasterizer = pytorch3d.renderer.MeshRasterizer(
        raster_settings=raster_settings,
    )
    shader = pytorch3d.renderer.HardPhongShader(device=device)
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=rasterizer,
        shader=shader,
    )
    for i in range(N):
        
        os.makedirs(f"data/{i}")
        audio, origin, mic_loc, speaker_loc, boundaries, mesh = get_random_datapoint()
        audio, origin, mic_loc, speaker_loc, boundaries, mesh = convert_to_torch(audio, origin, mic_loc, speaker_loc, boundaries, mesh)
        boundaries = set_boundaries_buffer(boundaries, torch.to_array([0.1, 0.05, 0.1], device = device))
        lights = get_random_lighting(boundaries)
        camera = get_random_camera(boundaries)
        rgb = get_rgb(mesh, camera, renderer, lights)
        rgb_aug = apply_augmentation(rgb)
        depth_map = get_depth_map(mesh, camera, rasterizer)
        mic_loc_std = to_camera_coords(mic_loc)
        speaker_loc_std = to_camera_coords(speaker_loc)
        save_datapoint(rgb_aug, depth_map, audio, mic_loc_std, speaker_loc_std, i)
