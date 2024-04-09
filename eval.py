import argparse
import imageio
import numpy as np
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
from utils import render_cloud, render_mesh, render_voxel
import utils_vox
import matplotlib.pyplot as plt 

def evaluate_model(args):
    # load checkpoint and call our model to find the average (?) loss
    # TODO
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument("--load_checkpoint", action="store_true")
    args = parser.parse_args()
    evaluate_model(args)
