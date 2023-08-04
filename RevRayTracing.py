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
    del x, y, z, lbl, nx, ny, nz
    
    dist_at_angle_0 = 13                   # Meters
    angle_step = np.deg2rad(0.5)    # 0.5 degrees
    angle_max = np.deg2rad(60)     # 120 degrees
    
    nb_of_steps = np.ceil((angle_max*2) / angle_step).astype(np.int32)
    
    idx_facade= np.where(voxels[:,4:] != (0.,0.,0.))[0].astype(np.int32)   # Get indices of all facade building points (where normal != 0)
    
    for idx in np.arange(idx_facade.shape[0]):
        if voxels[idx_facade[idx]][3] != 6:
            continue
        else:
            current_point = voxels[idx_facade[idx]][:3]
            current_normal = voxels[idx_facade[idx]][4:]
            for idx_current_ray in np.arange(nb_of_steps):
                dist_to_travel = dist_at_angle_0 * cos(angle_max - angle_step*idx_current_ray)
                # Écrire ensuite un algo qui avance de vxl en vxl tout le long de la trajectoire pour regarder s'il s'y trouve quelque chose, le cas échéant, on enregistre ce voxel avec son lbl
        
    # voxels[idx_building_all[3]] PAS TOUS LES VOXELS BUILDING QUI ONT UNE NORMAL car seulement les facades en ont une
        
    num_steps = dist_at_angle_0 * config.voxel_size
    
    
    
    
    
    #for vxl in voxCenter
    
    return 0