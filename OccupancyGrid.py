#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:22:33 2023

@author: willalbert
"""

import numpy as np
from collections import Counter
from utils import Config as config
from os.path import join
import OSToolBox as ost
from tqdm import tqdm


def generate_occupancy_grid(points, grid_type):
    voxel_size = config.voxel_size
    print("  - Generating the occupancy grid ...")
    # Determine the minimum and maximum coordinates of the point cloud
    min_coords = np.min(points, axis=0)[:3]
    max_coords = np.max(points, axis=0)[:3]

    # Calculate the dimensions of the voxel grid based on the voxel size
    grid_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)+1

    # Create an empty occupancy grid
    #occupancy_grid = np.zeros(grid_dims, dtype=bool)
    histo_grid_nor = np.zeros(grid_dims, dtype=list)
    histo_grid_lbl = np.zeros(grid_dims, dtype=list)
    #label_grid = np.zeros(grid_dims, dtype=int)


    if "normal" and "label" in grid_type:
        # Iterate through each point in the point cloud
        for i in tqdm(np.arange(points.shape[0])):
            voxel_indices = np.floor((points[i][:3] - min_coords) / voxel_size).astype(int)
            # Transfert the normals of all the facades points in each voxel with at least one facade point
            if histo_grid_nor[tuple(voxel_indices)] == 0 and (points[i][4:] != [0., 0., 0.]).all():
                histo_grid_nor[tuple(voxel_indices)] = [points[i][4:]]
            elif histo_grid_nor[tuple(voxel_indices)] != 0 and (points[i][4:] != [0., 0., 0.]).all():
                histo_grid_nor[tuple(voxel_indices)].append(points[i][4:])
                
            # Transfert the labels of all the points in each voxel
            if histo_grid_lbl[tuple(voxel_indices)] == 0:
                histo_grid_lbl[tuple(voxel_indices)] = [points[i][3]]
            else:
                histo_grid_lbl[tuple(voxel_indices)].append(points[i][3])
                

    elif "normal" in grid_type:
        histo_grid_lbl = None
        # Iterate through each point in the point cloud
        for i in tqdm(np.arange(points.shape[0])):
            # Calculate the voxel indices for the current point
            voxel_indices = np.floor((points[i][:3] - min_coords) / voxel_size).astype(int)
            
            # Transfert the normals of all the facades points in each voxel with at least one facade point
            if histo_grid_nor[tuple(voxel_indices)] == 0 and (points[i][4:] != [0., 0., 0.]).all():
                histo_grid_nor[tuple(voxel_indices)] = [points[i][4:]]
            elif histo_grid_nor[tuple(voxel_indices)] != 0 and (points[i][4:] != [0., 0., 0.]).all():
                histo_grid_nor[tuple(voxel_indices)].append(points[i][4:])
                
                
    elif "label" in grid_type:
        histo_grid_nor = None
        # Iterate through each point in the point cloud
        for i in tqdm(np.arange(points.shape[0])):
            # Calculate the voxel indices for the current point
            voxel_indices = np.floor((points[i][:3] - min_coords) / voxel_size).astype(int)
            # Transfert the labels of all the points in each voxel
            if histo_grid_lbl[tuple(voxel_indices)] == 0:
                histo_grid_lbl[tuple(voxel_indices)] = [points[i][3]]
            else:
                histo_grid_lbl[tuple(voxel_indices)].append(points[i][3])
            
    print("    Done with success")
    
    return histo_grid_nor, histo_grid_lbl, min_coords


def generate_single_occ_grid(histo_grid_nor, grid_type, min_coords, histo_grid_lbl=None):
    voxel_size = config.voxel_size
    nb_ligne = histo_grid_nor.shape[0]
    nb_col = histo_grid_nor.shape[1]
    nb_haut = histo_grid_nor.shape[2]

    
    if "normal" in grid_type:
        print("  - Generating occupancy grid with the median normal of each voxel ...")
        # Iterate through each point in the point cloud
        normal_not_0 = np.where(histo_grid_nor!=0)
        max_height_norm = np.max(normal_not_0[2]) + 1
        histo_grid_nor = histo_grid_nor[:,:,:max_height_norm]
        single_grid_nor = np.zeros((nb_ligne*nb_col*max_height_norm, 3), dtype=float)
        
        histo_grid_nor = np.reshape(histo_grid_nor, (-1,))
        normal_not_0 = np.where(histo_grid_nor!=0)
        normal_not_0 = np.vstack(normal_not_0).T
        
        # normal_histo_grid = np.where(normal_histo_grid==0, np.array([0,0,0]),normal_histo_grid)
        
        for idx in np.arange(normal_not_0.shape[0]):
            lst = histo_grid_nor[normal_not_0[idx]]
            lst = np.vstack(lst)
            median = np.median(lst, axis=0)
            median = median / np.linalg.norm(median)
            
            single_grid_nor[normal_not_0[idx], 0], single_grid_nor[normal_not_0[idx], 1], single_grid_nor[normal_not_0[idx], 2] = median[0], median[1], median[2]
            
        del histo_grid_nor, normal_not_0, median
        
    
    if "label" in grid_type:
        print("  - Generating occupancy grid with the most popular label in each voxel ...")
        single_grid_lbl = np.zeros((nb_ligne,nb_col,max_height_norm), dtype=int)
        histo_grid_lbl = histo_grid_lbl[:,:,:max_height_norm]
        # Iterate through each point in the point cloud
        label_not_0 = np.where(histo_grid_lbl!=0)
        label_not_0 = np.vstack(label_not_0).T
        
        for idx in np.arange(label_not_0.shape[0]):
            lst = histo_grid_lbl[tuple((int(label_not_0[idx, 0]),int(label_not_0[idx, 1]),int(label_not_0[idx, 2])))]
            single_grid_lbl[tuple((int(label_not_0[idx, 0]),int(label_not_0[idx, 1]),int(label_not_0[idx, 2])))] = Counter(lst).most_common(1)[0][0] #https://stackoverflow.com/questions/6987285/find-the-item-with-maximum-occurrences-in-a-list
        
        del histo_grid_lbl, label_not_0, lst

        # Reshape the array into a 1D vector
        single_grid_lbl = np.reshape(single_grid_lbl, (-1,))
        
    
    # Generate voxel grid coordinates
    x, y, z = np.indices((nb_ligne,nb_col,max_height_norm))
    
    # Convert voxel indices to original coordinate system
    x = np.reshape(x, (-1,)) * voxel_size + (min_coords[0] + voxel_size/2)
    y = np.reshape(y, (-1,)) * voxel_size + (min_coords[1] + voxel_size/2)
    z = np.reshape(z, (-1,)) * voxel_size + (min_coords[2] + voxel_size/2)
    
    if "normal" and "label" in grid_type:
        p = np.c_[x,y,z,single_grid_lbl,single_grid_nor[:,0],single_grid_nor[:,1],single_grid_nor[:,2]]
        del x,y,z,single_grid_nor,single_grid_lbl
        ply_path_voxels = join(config.folder_path_out, "occGrid_norm_lbl.ply")
        ost.write_ply(ply_path_voxels, p, ["x","y","z","lbl","nx","ny","nz"])
        
    elif "normal" in grid_type:
        p = np.c_[x,y,z,single_grid_nor[:,0],single_grid_nor[:,1],single_grid_nor[:,2]]
        del x,y,z,single_grid_nor
        ply_path_voxels = join(config.folder_path_out, "occGrid_normals.ply")
        ost.write_ply(ply_path_voxels, p, ["x","y","z","nx","ny","nz"])
            
    elif "label" in grid_type:
        p = np.c_[x,y,z,single_grid_lbl]
        del x,y,z,single_grid_lbl
        ply_path_voxels = join(config.folder_path_out, "occGrid_lbl.ply")
        ost.write_ply(ply_path_voxels, p, ["x","y","z","lbl"])
    
    del p
    print("Voxels created with success!\nA ply file has been created here: {}\n".format(ply_path_voxels))
    
    print()
    
    return ply_path_voxels