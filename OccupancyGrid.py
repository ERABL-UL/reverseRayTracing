#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:22:33 2023

@author: willalbert
"""

import numpy as np
from collections import Counter
from utils import Config
from os.path import join
import OSToolBox as ost


def generate_occupancy_grid(points, voxel_size, grid_type):
    print("  - Generating the occupancy grid ...")
    # Determine the minimum and maximum coordinates of the point cloud
    min_coords = np.min(points, axis=0)[:3]
    max_coords = np.max(points, axis=0)[:3]

    # Calculate the dimensions of the voxel grid based on the voxel size
    grid_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)

    # Create an empty occupancy grid
    #occupancy_grid = np.zeros(grid_dims, dtype=bool)
    histo_grid = np.zeros(grid_dims, dtype=list)
    #label_grid = np.zeros(grid_dims, dtype=int)

    if grid_type == "normal":
        # Iterate through each point in the point cloud
        for point in points:
            # Calculate the voxel indices for the current point
            voxel_indices = np.floor((point[:3] - min_coords) / voxel_size).astype(int)
            
            # Transfert the normals of all the facades points in each voxel with at least one facade point
            if histo_grid[tuple(voxel_indices)] == 0 and (point[4:] != [0., 0., 0.]).all():
                histo_grid[tuple(voxel_indices)] = [point[4:]]
            elif histo_grid[tuple(voxel_indices)] != 0 and (point[4:] != [0., 0., 0.]).all():
                histo_grid[tuple(voxel_indices)].append(point[4:])
                
    elif grid_type == "label":
        # Iterate through each point in the point cloud
        for point in points:
            # Calculate the voxel indices for the current point
            voxel_indices = np.floor((point[:3] - min_coords) / voxel_size).astype(int)
    
            # Mark the corresponding voxel as occupied
            #occupancy_grid[tuple(voxel_indices)] = True
            
            # Transfert the labels of all the points in each voxel
            if histo_grid[tuple(voxel_indices)] == 0:
                histo_grid[tuple(voxel_indices)] = [point[3]]
            else:
                histo_grid[tuple(voxel_indices)].append(point[3])
            
    print("    Done with success")
    
    return histo_grid, min_coords

def generate_single_occ_grid(histo_grid, grid_type, min_coords):
    config=Config
    print("  - Generating occupancy grid with the most popular label in each voxel ...")
    
    nb_ligne = histo_grid.shape[0]
    nb_col = histo_grid.shape[1]
    nb_haut = histo_grid.shape[2]

    
    if grid_type == "normal":
        single_grid = np.zeros((nb_ligne*nb_col*nb_haut, 3), dtype=float)
        # Iterate through each point in the point cloud
        histo_grid = np.reshape(histo_grid, (-1,))
        normal_not_0 = np.where(histo_grid!=0)
        normal_not_0 = np.vstack(normal_not_0).T
        
        # normal_histo_grid = np.where(normal_histo_grid==0, np.array([0,0,0]),normal_histo_grid)
        
        for idx, ijk in enumerate(normal_not_0):
            lst = histo_grid[ijk]
            lst = np.vstack(lst)
            median = np.median(lst, axis=0)
            single_grid[ijk, 0], single_grid[ijk, 1], single_grid[ijk, 2] = median[0], median[1], median[2]
            # normal_grid[tuple((int(normal_not_0[idx, 0]),int(normal_not_0[idx, 1]),int(normal_not_0[idx, 2])))] = median
        
        single_grid = np.round(single_grid, decimals=6)
        del histo_grid
        
    
    elif grid_type == "label":
        single_grid = np.zeros((nb_ligne,nb_col,nb_haut), dtype=int)
        # Iterate through each point in the point cloud
        label_not_0 = np.where(histo_grid!=0)
        label_not_0 = np.vstack(label_not_0).T
        
        for idx, _ in enumerate(label_not_0):
            lst = histo_grid[tuple((int(label_not_0[idx, 0]),int(label_not_0[idx, 1]),int(label_not_0[idx, 2])))]
            single_grid[tuple((int(label_not_0[idx, 0]),int(label_not_0[idx, 1]),int(label_not_0[idx, 2])))] = Counter(lst).most_common(1)[0][0] #https://stackoverflow.com/questions/6987285/find-the-item-with-maximum-occurrences-in-a-list
        
        del histo_grid

        # Reshape the array into a 1D vector
        lbl = np.reshape(single_grid, (-1,))
        
    
    # Generate voxel grid coordinates
    x, y, z = np.indices((nb_ligne,nb_col,nb_haut))
    
    # Convert voxel indices to "world" coordinates
    x = np.reshape(x, (-1,)) * config.voxel_size + (min_coords[0] + config.voxel_size/2)
    y = np.reshape(y, (-1,)) * config.voxel_size + (min_coords[1] + config.voxel_size/2)
    z = np.reshape(z, (-1,)) * config.voxel_size + (min_coords[2] + config.voxel_size/2)
    
    x = np.round(x, decimals=6)
    y = np.round(y, decimals=6)
    z = np.round(z, decimals=6)
    
    if grid_type == "normal":
        p = np.c_[x,y,z,single_grid[:,0],single_grid[:,1],single_grid[:,2]]
        del x,y,z,single_grid
        ply_path_voxels = join(config.folder_path_out, "occGrid_normals.ply")
        ost.write_ply(ply_path_voxels, p, ["x","y","z","nx","ny","nz"])
        
    elif grid_type == "label":
        p = np.c_[x,y,z,lbl]
        del x,y,z,lbl
        ply_path_voxels = join(config.folder_path_out, "occGrid.ply")
        ost.write_ply(ply_path_voxels, p, ["x","y","z","lbl"])
    
    del p
    print("Voxels created with success!\nA ply file has been created here: {}".format(ply_path_voxels))
    
    print("###################")
    
    return ply_path_voxels