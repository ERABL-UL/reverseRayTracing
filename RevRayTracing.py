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
    
    dist_step = 0.4                        # Distance for only one move
    dist_at_angle_0 = 13                    # Meters
    angle_step = np.deg2rad(0.5)            # 0.5 degrees
    angle_max = np.deg2rad(30)              # 120 degrees 
    
    nb_angle_steps = np.ceil((angle_max*2) / angle_step).astype(np.int32)
    idx_facade = np.where(voxels[:,4:] != (0.,0.,0.))[0].astype(np.int32)   # Get indices of all facade building voxels (where normal != 0)
    
    idx_flat = np.where(voxels[:,3] == 1)[0].astype(np.int32)   # Get indices of all flat voxels to get the min and max
    z_min = np.min(voxels[idx_flat][:,2])
    z_max = np.max(voxels[idx_flat][:,2])
    z_mean = np.mean(voxels[idx_flat][:,2])
    nb_dist_steps_verti = np.ceil((z_max - z_min)/0.5)
    
    lst = []
    for idx in np.arange(idx_facade.shape[0]):
        idx = 1000
        # input("Press Enter to continue...")
        if voxels[idx_facade[idx]][3] != 6:
            continue
        else:
            starting_vxl_center = voxels[idx_facade[idx]][:3]
            starting_vxl_normal = voxels[idx_facade[idx]][4:]
            for idx_current_ray in np.arange(nb_angle_steps):
                dist_tot_to_travel = dist_at_angle_0 / cos(angle_max - angle_step*idx_current_ray)
                nb_dist_steps = np.ceil(dist_tot_to_travel / dist_step)         # Number of time we have to move forward in the current ray
                
                for idx_dist_step in np.arange(1, nb_dist_steps+1):
                    angle_x = np.arccos(np.dot(starting_vxl_normal, np.array([1, 0, 0])))       # Rad
                    angle_y = np.arccos(np.dot(starting_vxl_normal, np.array([0, 1, 0])))       # Rad
                    angle_z = np.arccos(np.dot(starting_vxl_normal, np.array([0, 0, 1])))       # Rad
                    
                    adjustment_x = cos(angle_x - (angle_max - angle_step*idx_current_ray)) * (dist_step * idx_dist_step)
                    adjustment_y = -sin(angle_y - (angle_max - angle_step*idx_current_ray)) * (dist_step * idx_dist_step)
                    
                    point_ray_interm_x = starting_vxl_center[0] + adjustment_x
                    point_ray_interm_y = starting_vxl_center[1] + adjustment_y
                    
                    for idx_dist_step_verti in np.arange(1, nb_dist_steps_verti+1):
                        angle = np.arctan2((abs(starting_vxl_center[2]-z_mean)-idx_dist_step_verti*dist_step), dist_tot_to_travel)
                        corr_z = np.sqrt(adjustment_x**2 + adjustment_y**2) * tan(angle)
                        point_ray_interm_z = starting_vxl_center[2] - corr_z
                    
                        lst.append([point_ray_interm_x, point_ray_interm_y, point_ray_interm_z])
                    
        break
                
        
    # voxels[idx_building_all[3]] PAS TOUS LES VOXELS BUILDING QUI ONT UNE NORMAL car seulement les facades en ont une
        
    num_steps = dist_at_angle_0 * config.voxel_size
    ost.write_ply("grosTEST.ply", np.array(lst), ['x','y','z'])
    
    
    
    
    #for vxl in voxCenter
    
    return 0