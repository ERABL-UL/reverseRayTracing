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
from OccupancyGrid import generate_occupancy_grid, generate_single_occ_grid
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
    
    del x, y, z
    
    # point_cloud=points     # Enlever apres debuggage
    # voxel_size=0.5         # Enlever apres debuggage
    
    # Grille d'occupation
    label_histo_grid, min_coords = generate_occupancy_grid(points, config.voxel_size)
    del points
    label_grid = generate_single_occ_grid(label_histo_grid)
    del label_histo_grid
    
    # Generate voxel grid coordinates
    x, y, z = np.indices(label_grid.shape)
    
    # Reshape the array into a 1D vector
    lbl = np.reshape(label_grid, (-1,))
    
    # Convert voxel indices to world coordinates
    x = x * config.voxel_size + (min_coords[0] + config.voxel_size/2)
    y = y * config.voxel_size + (min_coords[1] + config.voxel_size/2)
    z = z * config.voxel_size + (min_coords[2] + config.voxel_size/2)
    
    x = np.reshape(x, (-1,))
    y = np.reshape(y, (-1,))
    z = np.reshape(z, (-1,))
    
    p = np.c_[x,y,z,lbl]
    
    ply_path_voxels = join(config.folder_path_out, "occGrid.ply")
    ost.write_ply(ply_path_voxels, p, ["x","y","z","lbl"])

    print("Voxels created with success!\nA ply file has been created here: {}".format(ply_path_voxels))
    print("###################")
    
    return ply_path_voxels