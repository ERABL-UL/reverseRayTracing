#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:58:52 2023

@author: willalbert


#######################

ply_path_voxels : Path of the labaled voxel cloud
lbl_no1: Voxels from which the reverse ray tracing starts (usualy class Building - 6)
lbl_no2: Voxels from which the reverse ray tracing finishes (usualy class Flat - 1)


"""

from utils import Config
import OSToolBox as ost
import numpy as np
from numpy import cos, sin, tan
from scipy.interpolate import griddata


def revRayTracing(ply_path_voxels, ply_path_normals):
    config = Config
    
    voxels = ost.read_ply("/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/TESToccGrid_norm_lbl.ply")
    x = voxels["x"]
    y = voxels["y"]
    z = voxels["z"]
    lbl = voxels["lbl"]
    nx = voxels["nx"]
    ny = voxels["ny"]
    nz = voxels["nz"]
    voxels = np.c_[x, y, z, lbl, nx, ny, nz]
    del x, y, z, nx, ny, nz
    
    dist_at_angle_0 = 13                   # Meters
    angle_step = np.deg2rad(0.5)    # 0.5 degrees
    angle_max = np.deg2rad(60)     # 120 degrees
    
    nb_of_steps = np.ceil((angle_max*2) / angle_step).astype(np.int32)
    
    idx_building_all = np.where(lbl.astype(np.int32) == 6)[0].astype(np.int32)          # Get indices of all points corresponding to Building
    idx_flat_all = np.where(lbl.astype(np.int32) == 1)[0].astype(np.int32)              # Get indices of all points corresponding to Flat
    
    for idx_building in idx_building_all:
        current_point = voxels[idx_building][:2]
        current_normal = voxels[idx_building]
        for idx_current_ray in np.arange(nb_of_steps):
            dist_to_travel = dist_at_angle_0 * cos(angle_max - angle_step*idx_current_ray)
        
    # voxels[idx_building_all[3]] PAS TOUS LES VOXELS BUILDING QUI ONT UNE NORMAL !?!?
        
    num_steps = dist_at_angle_0 * config.voxel_size
    
    
    
    
    
    #for vxl in voxCenter
    
    return 0