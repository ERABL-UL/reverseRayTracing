#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:22:33 2023

@author: willalbert
"""

import numpy as np
from collections import Counter

def generate_occupancy_grid(point_cloud, voxel_size):
    print("  - Generating the occupancy grid with labels of every points and facades' normals in each voxel ...")
    # Determine the minimum and maximum coordinates of the point cloud
    min_coords = np.min(point_cloud, axis=0)[:3]
    max_coords = np.max(point_cloud, axis=0)[:3]

    # Calculate the dimensions of the voxel grid based on the voxel size
    grid_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)

    # Create an empty occupancy grid
    #occupancy_grid = np.zeros(grid_dims, dtype=bool)
    label_histo_grid = np.zeros(grid_dims, dtype=list)
    normal_histo_grid = np.zeros(grid_dims, dtype=list)
    #label_grid = np.zeros(grid_dims, dtype=int)

    # Iterate through each point in the point cloud
    for point in point_cloud:
        # Calculate the voxel indices for the current point
        voxel_indices = np.floor((point[:3] - min_coords) / voxel_size).astype(int)

        # Mark the corresponding voxel as occupied
        #occupancy_grid[tuple(voxel_indices)] = True
        
        # Transfert the labels of all the points in each voxel
        if label_histo_grid[tuple(voxel_indices)] == 0:
            label_histo_grid[tuple(voxel_indices)] = [point[3]]
        else:
            label_histo_grid[tuple(voxel_indices)].append(point[3])
        
        # Transfert the normals of all the facades points in each voxel with at least one facade point
        if normal_histo_grid[tuple(voxel_indices)] == 0 and (point[4:] != [0., 0., 0.]).all():
            normal_histo_grid[tuple(voxel_indices)] = [point[4:]]
        elif normal_histo_grid[tuple(voxel_indices)] != 0 and (point[4:] != [0., 0., 0.]).all():
            normal_histo_grid[tuple(voxel_indices)].append(point[3])
            
    print("    Done with success")
    
    return label_histo_grid, normal_histo_grid, min_coords

def generate_single_occ_grid(label_histo_grid):
    print("  - Generating occupancy grid with the most popular label in each voxel ...")
    
    nb_ligne = label_histo_grid.shape[0]
    nb_col = label_histo_grid.shape[1]
    nb_haut = label_histo_grid.shape[2]
    
    label_grid = np.zeros((nb_ligne,nb_col,nb_haut), dtype=int)
    
    # Iterate through each point in the point cloud
    for i in np.arange(nb_ligne):
        for j in np.arange(nb_col):
            for k in np.arange(nb_haut):
                # Get the most common lbl in the voxel
                lst = label_histo_grid[tuple((int(i),int(j),int(k)))]
                if lst != 0: 
                    label_grid[tuple((i,j,k))] = Counter(lst).most_common(1)[0][0] #https://stackoverflow.com/questions/6987285/find-the-item-with-maximum-occurrences-in-a-list
                else:
                    label_grid[tuple((i,j,k))] = 0
    
    del label_histo_grid        # Freeing space, useful when debugging
    print("    Done with success")

    return label_grid