#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:13:46 2023

@author: willalbert
"""

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import torch
import OSToolBox as ost

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from utils.config import Config
from occupancyGrid import generate_occupancy_grid
from os.path import exists, join

import wandb

# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

# class revRayTracingConfig(Config):
#     """
#     Override the parameters you want to modify for this dataset
#     """

#     ####################
#     # Dataset parameters
#     ####################

#     # Dataset name
#     dataset = 'Kitti-360'

#     # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
#     class_to_extract = None


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    
    #################
    # Initialization
    #################
    
    folder_path = "/home/willalbert/Desktop"
    file_name = "segmentedSteMarthe.ply"
    file_path = join(folder_path, file_name)
    file_name_prob = "segmentedSteMarthePROB.ply"
    file_path_prob = join(folder_path, file_name_prob)

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    
    pntCloud = ost.read_ply(file_path)
    building_inds = np.where(pntCloud["scalar_label"].astype(np.int32) == 6)[0]
    
    x = pntCloud["x"]
    y = pntCloud["y"]
    z = pntCloud["z"]
    lbl = pntCloud["scalar_label"]
    points = np.c_[x, y, z, lbl]
    
    p = points[building_inds]
    
    # Grille d'occupation
    voxel_size = 0.5        # metres
    occupancy_grid, min_coords = generate_occupancy_grid(points, voxel_size)
    
    # Generate voxel grid coordinates
    x, y, z = np.indices(occupancy_grid.shape)
    
    # Convert voxel indices to world coordinates
    x = x * voxel_size + min_coords[0]
    y = y * voxel_size + min_coords[1]
    z = z * voxel_size + min_coords[2]
    
    nb_ligne = occupancy_grid.shape[0]
    nb_col = occupancy_grid.shape[1]
    nb_haut = occupancy_grid.shape[2]
    
    # Reshape the array into a 1D vector
    occupied_voxels = np.reshape(occupancy_grid, (-1,))
    x = np.reshape(x, (-1,))
    y = np.reshape(y, (-1,))
    z = np.reshape(z, (-1,))
    
    
    p = np.c_[x, y, z, occupied_voxels]
    
    ost.write_ply(join(folder_path, "occGrid.ply"), p, ["x","y","z","occ"])
    
    print('Finished')
