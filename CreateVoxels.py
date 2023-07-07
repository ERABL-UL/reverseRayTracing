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

from scipy.spatial import KDTree
    

def createVoxels(name_scalar, ply_path_horiz):
    config = Config
    print("\n###################")
    print("Creating voxels of {} meters from point cloud {} ...".format(config.voxel_size, config.file_name_read))
    
    file_path_read = join(config.folder_path_in, config.file_name_read)
    pntCloud = ost.read_ply(file_path_read)
    
    # Hole point cloud
    x = pntCloud["x"]
    y = pntCloud["y"]
    z = pntCloud["z"]
    lbl = pntCloud[name_scalar]
    zeros_col = np.zeros(x.shape)
    points = np.c_[x, y, z, lbl, zeros_col, zeros_col, zeros_col]
    
    # Point cloud of facades with normals
    pntCloudFacades = ost.read_ply(ply_path_horiz)
    x = pntCloudFacades["x"]
    y = pntCloudFacades["y"]
    z = pntCloudFacades["z"]
    nx = pntCloudFacades["nx"]
    ny = pntCloudFacades["ny"]
    nz = pntCloudFacades["nz"]
    pointsFacades = np.c_[x, y, z, nx, ny, nz]
    
    del x, y, z, nx, ny, nz, lbl                 # Freeing space, useful when debugging
    
    # Give normals from facades to the same facades' points in the hole point cloud
    kdtree = KDTree(points[:, :3])
    _, indices = kdtree.query(pointsFacades[:, :3], k=1, distance_upper_bound=0.00001, workers=24)
    points[:, 4:][indices] = pointsFacades[:, 3:]
    
    
    ost.write_ply(join(config.folder_path_out, "tempPntCloud.ply"), points, ["x","y","z","lbl","nx","ny","nz"])
    
    # point_cloud=points        # Enlever apres debuggage
    # voxel_size=0.5            # Enlever apres debuggage
    
    # Grille d'occupation
    label_histo_grid, normal_histo_grid, min_coords = generate_occupancy_grid(points, config.voxel_size)
    del points                  # Freeing space, useful when debugging
    
    # Check which label is the most common in each voxel
    label_grid = generate_single_occ_grid(label_histo_grid)
    #normal_grid = generate_single_occ_grid(normal_histo_grid)
    del label_histo_grid        # Freeing space, useful when debugging
    
    # Generate voxel grid coordinates
    x, y, z = np.indices(label_grid.shape)
    
    # Reshape the array into a 1D vector
    lbl = np.reshape(label_grid, (-1,))
    
    # Convert voxel indices to "world" coordinates
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



def createBlob(voxels_path):
    voxels_path = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/occGrid.ply"
    voxels = ost.read_ply(voxels_path)
    
    x = voxels["x"]
    y = voxels["y"]
    z = voxels["z"]
    lbl = voxels['lbl']
    voxels2D = np.c_[x, y, z, lbl]
    
    for i in x:
        for j in y:
            for k in z:
                if voxels2D[i,j,k][3] == 6:
                    voxelsBlob[i,j][2]
    
    return 0