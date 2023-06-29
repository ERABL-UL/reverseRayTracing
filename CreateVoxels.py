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
import os
import numpy as np
import OSToolBox as ost

from utils import Config
from OccupancyGrid import generate_occupancy_grid
from os.path import join

import multiprocessing
from functools import partial
    

def createVoxels(name_scalar):
    config = Config
    print("\n###################")
    print("Creating voxels of {} meters from point cloud {} ...".format(config.voxel_size, config.file_name_read))
    
    file_path_read = join(config.folder_path_in, config.file_name_read)
    pntCloud = ost.read_ply(file_path_read)
    
    x = pntCloud["x"]
    y = pntCloud["y"]
    z = pntCloud["z"]
    lbl = pntCloud[name_scalar]
    points = np.c_[x, y, z, lbl]
    
    # Grille d'occupation
    occupancy_grid, min_coords = generate_occupancy_grid(points, config.voxel_size)
    
    # Generate voxel grid coordinates
    x, y, z = np.indices(occupancy_grid.shape)
    
    # Convert voxel indices to world coordinates
    x = x * config.voxel_size + min_coords[0]
    y = y * config.voxel_size + min_coords[1]
    z = z * config.voxel_size + min_coords[2]
    
    nb_ligne = occupancy_grid.shape[0]
    nb_col = occupancy_grid.shape[1]
    nb_haut = occupancy_grid.shape[2]
    
    # Reshape the array into a 1D vector
    occupied_voxels = np.reshape(occupancy_grid, (-1,))
    x = np.reshape(x, (-1,))
    y = np.reshape(y, (-1,))
    z = np.reshape(z, (-1,))
    
    p = np.c_[x, y, z, occupied_voxels]
    ply_path_voxels = join(config.folder_path_out, "occGrid.ply")
    ost.write_ply(ply_path_voxels, p, ["x","y","z","occ"])
    
    print("Voxels created with success!\nA ply file has been created here: {}".format(ply_path_voxels))
    print("###################")
    
    return ply_path_voxels
