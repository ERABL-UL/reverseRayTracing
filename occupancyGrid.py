#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:22:33 2023

@author: willalbert
"""

import numpy as np

def generate_occupancy_grid(point_cloud, voxel_size):
    # Determine the minimum and maximum coordinates of the point cloud
    min_coords = np.min(point_cloud, axis=0)[:3]
    max_coords = np.max(point_cloud, axis=0)[:3]

    # Calculate the dimensions of the voxel grid based on the voxel size
    grid_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)

    # Create an empty occupancy grid
    occupancy_grid = np.zeros(grid_dims, dtype=bool)

    # Iterate through each point in the point cloud
    for point in point_cloud:
        # Calculate the voxel indices for the current point
        voxel_indices = np.floor((point[:3] - min_coords) / voxel_size).astype(int)

        # Mark the corresponding voxel as occupied
        occupancy_grid[tuple(voxel_indices)] = True

    return occupancy_grid, min_coords

