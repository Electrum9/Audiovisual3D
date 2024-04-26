
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
# from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 500
image_size = 512
device = "cuda"

def get_datapoint(pt_path):
    pt = torch.load(pt_path)
    room = pt['room']
    if room =='conference':
        room_obj_path = '/mnt/Mercury2/CMU16825/soundcam/scans/ConferenceRoom/poly.obj'
    elif room == 'treatedroom':
        room_obj_path = '/mnt/Mercury2/CMU16825/soundcam/scans/TreatedRoom/poly.obj'
    elif room == 'livingroom':
        room_obj_path = '/mnt/Mercury2/CMU16825/soundcam/scans/LivingRoom/poly.obj'

    if pt['audio_mic_pos'] is None:
        return None
    else:
        audio = pt['convovled_audio']
        audio_mic = pt['audio_mic_pos']
        room_mics_pos = pt['mic_pos'] 
        audio_mic_pos = room_mics_pos[int(audio_mic)]
        speaker_pos = pt['speaker_pos']
        mesh = load_objs_as_meshes([room_obj_path])
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
    return (audio, np.array([[0, 0, 0]]), audio_mic_pos, speaker_pos, boundaries, mesh)

def get_random_datapoint():
    pt_path = '/mnt/Mercury2/CMU16825/soundcam/Audiovisual3D/fins/dataset/convolved_audio_pt'
    paths = glob.glob(f"{pt_path}/conference*.pt") + glob.glob(f"{pt_path}/treatedroom*.pt")
    data_pt = get_datapoint(paths[int(random() * len(paths))])
    
    while data_pt is None:
        data_pt = get_datapoint(paths[int(random() * len(paths))])
        
    return data_pt

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

def get_R(camera_loc, azim, elev):
    direc = [math.cos(elev) * math.sin(azim), math.sin(elev), math.cos(elev) * math.cos(azim)]
    R = pytorch3d.renderer.look_at_rotation(camera_loc, direc)
    return R

def get_random_cameras(boundaries):
    # currently: chooses a random point in the middle of the air
    camera_loc = [choose_random(boundaries[0][0], boundaries[1][0]), (boundaries[0][1] + boundaries[1][1]) / 2, choose_random(boundaries[0][2], boundaries[1][2])]
    camera_loc = torch.tensor([val.item() for val in camera_loc]).unsqueeze(0)
    camera_loc = -camera_loc
    azim_start = choose_random(-180.0, -90.0)
    elev = choose_random(22.5, 45.0)
    Rs = torch.zeros((8, 3, 3), device = device)
    for i in range(4):
        azim = azim_start + 90 * i
        Rs[i * 2] = get_R(camera_loc, azim, elev)
        Rs[i * 2 + 1] = get_R(camera_loc, azim, -elev)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=Rs,
        T=torch.inverse(Rs) @ camera_loc,
        device=device
    )
    return cameras, camera_loc
    
def get_rgb(mesh, cameras, renderer, lights):
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return (rend.detach().cpu().numpy()[..., :3] * 255).astype(np.uint8)

def get_depth_map(mesh, cameras, rasterizer):
    fragments = rasterizer(mesh, cameras=cameras)
    depth_map = fragments.zbuf[:,:,:,0] #/ fragments.zbuf[:,:,:,0].max()
    return depth_map
    
def choose_random(a, b):
    return (random() * (b - a)) + a
    
def get_random_lighting(boundaries):
    # currently: chooses a random point on the ceiling
    random_x = choose_random(boundaries[0][0], boundaries[1][0])
    random_z = choose_random(boundaries[0][2], boundaries[1][2])
    return pytorch3d.renderer.lighting.PointLights(location=[[random_x, boundaries[1][1], random_z]], device=device)
    
def to_camera_coords(pos, camera):
    return camera.get_world_to_view_transform().transform_points(pos.float().unsqueeze(0))
    
def save_datapoint(rgb, depth_map, audio, camera_loc_world, mic_loc_world, mic_loc_camera, speaker_loc_world, speaker_loc_camera, i):
    
    pt = {'depth': depth_map.numpy(),
          'rgb': rgb,
          'audio': audio.detach().cpu().numpy(),
          'cameraloc': camera_loc_world,
          'micloc_world': mic_loc_world.detach().cpu().numpy(),
          'micloc_camera': mic_loc_camera.detach().cpu().numpy(),
          'speakerloc_world': speaker_loc_world.detach().cpu().numpy(),
          'speakerloc_camera': speaker_loc_camera.detach().cpu().numpy()}
    
    torch.save(pt, f"data/{i}.pt")

def set_boundaries_buffer(boundaries, margin):
    size = boundaries[1] - boundaries[0]
    margin = size * margin
    boundaries[0,:] += margin
    boundaries[1,:] -= margin
    return boundaries

def convert2world(pt):
    if len(pt.shape) == 1:
        pt = pt[None]
    pt[:,[0, 1, 2]] = pt[:,[0,2,1]]  
    pt[:, 1] = 0
    return pt

def convert_to_torch(audio, origin, mic_loc, spaker_loc, boundaries, mesh):
    
    return (torch.tensor(audio, device = device), torch.tensor(origin, device = device), torch.tensor(mic_loc, device = device), torch.tensor(spaker_loc, device = device), torch.tensor(boundaries, device = device), mesh.to(device))

if __name__ == "__main__":
    
    shutil.rmtree("data", ignore_errors = True)
    os.makedirs('data', exist_ok=True)

    raster_settings = pytorch3d.renderer.RasterizationSettings(image_size=image_size)
    rasterizer = pytorch3d.renderer.MeshRasterizer(
        raster_settings=raster_settings,
    )
    shader = pytorch3d.renderer.HardPhongShader(device=device)
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=rasterizer,
        shader=shader,
    )
    for i in tqdm(range(N)):
        
        os.makedirs('data', exist_ok=True)
        
        audio, origin, mic_loc, speaker_loc, boundaries, mesh = get_random_datapoint()
        audio, origin, mic_loc, speaker_loc, boundaries, mesh = convert_to_torch(audio, origin, mic_loc, speaker_loc, boundaries, mesh)
        boundaries = set_boundaries_buffer(boundaries, torch.tensor([0.15, 0.05, 0.15], device = device))
        lights = get_random_lighting(boundaries)
        cameras, camera_loc_world = get_random_cameras(boundaries)
        rgb = get_rgb(mesh, cameras, renderer, lights)
        # rgb_aug = apply_augmentation(rgb)
        depth_map = get_depth_map(mesh, cameras, rasterizer).detach().cpu().squeeze()
        
        mic_loc_world = convert2world(mic_loc)
        speaker_loc_world = convert2world(speaker_loc)
        
        mic_loc_camera = to_camera_coords(mic_loc_world, cameras)
        speaker_loc_camera = to_camera_coords(speaker_loc_world, cameras)
        
        save_datapoint(rgb, depth_map, audio, camera_loc_world, mic_loc_world, mic_loc_camera, speaker_loc_world, speaker_loc_camera, i)
