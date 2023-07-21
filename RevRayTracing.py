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
from scipy.interpolate import griddata


def uniformNormals():
    
    return 0


def revRayTracing(ply_path_voxels, ply_path_normals):
    config = Config
    
    voxCenter = ost.read_ply(ply_path_voxels)
    x = voxCenter["x"]
    y = voxCenter["y"]
    z = voxCenter["z"]
    lbl = voxCenter["lbl"]
    voxCenter = np.c_[x, y, z, lbl]
    
    pntCloudWithNormals = ost.read_ply(ply_path_normals)
    x = pntCloudWithNormals["x"]
    y = pntCloudWithNormals["y"]
    z = pntCloudWithNormals["z"]
    nx = pntCloudWithNormals["nx"]
    ny = pntCloudWithNormals["ny"]
    nz = pntCloudWithNormals["nz"]
    pntCloudWithNormals = np.c_[x, y, z, nx, ny, nz]
    
    max_dist = 13 # meters
    num_steps = max_dist * config.voxel_size
    
    Idx_start_all = np.where(lbl.astype(np.int32) == 6)[0].astype(np.int32)         # Get indices of all points corresponding to Building
    Idx_finish_all = np.where(lbl.astype(np.int32) == 1)[0].astype(np.int32)        # Get indices of all points corresponding to Flat
    
    p = griddata(pntCloudWithNormals[:, :3], pntCloudWithNormals[:, 3:], method='nearest')
    
    #for vxl in voxCenter
    
    return 0