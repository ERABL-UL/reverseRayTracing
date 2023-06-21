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


def createVoxels():
    config = Config
    
    folder_path = config.folder_path
    file_name = config.file_name
    file_path = join(folder_path, file_name)
    file_name_prob = config.file_name_prob
    file_path_prob = join(folder_path, file_name_prob)
    
    pntCloud = ost.read_ply(file_path)
    
    x = pntCloud["x"]
    y = pntCloud["y"]
    z = pntCloud["z"]
    lbl = pntCloud["pre"]
    points = np.c_[x, y, z, lbl]
    
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
    
    return p
