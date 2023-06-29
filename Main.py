#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:57:20 2023

@author: willalbert
"""


from utils import Config
from CreateVoxels import createVoxels
from ExtractClass import extractClass
from ComputeNormals import compute_surface_normals, compute_surface_normals_RG
from CheckPerpendicular import checkPerpendicular
import OSToolBox as ost
from os.path import join
import open3d as o3d

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



if __name__ == "__main__":
    config = Config
   
    name_scalar = "scalar_pre"
    ply_path_voxels = createVoxels(name_scalar)
    # ply_path_voxels = createVoxels(name_scalar)
    
    idx_class = 1       # 1 == flat
    name_class = "flat"
    name_scalar = "scalar_pre"
    ply_path_extracted = extractClass(idx_class, name_scalar, name_class)
    # flat = ost.read_ply(ply_path_extracted)
    
    file_name_out = name_class + "_normals.ply"
    ply_path_normals = compute_surface_normals(config.r_normals, config.nn, ply_path_extracted, file_name_out)
    
    idx_class = 6       # 6 == building
    name_class = "building"
    name_scalar = "scalar_pre"
    ply_path_extracted = extractClass(idx_class, name_scalar, name_class)
    # building = ost.read_ply(ply_path_extracted)
    
    file_name_out = name_class + "_normals.ply"
    ply_path_normals = compute_surface_normals(config.r_normals, config.nn, ply_path_extracted, file_name_out)
    
    file_name_out = name_class + "_normals_RG.ply"
    ply_path_extracted = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/building_extraction.ply"
    ply_path_normals_RG = compute_surface_normals_RG(config.nn, file_path_read=ply_path_extracted, file_name_out=file_name_out)
    
    file_name_out = name_class + "_horiz_RG.ply"
    ply_path_normals_RG = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/building_normals_RG.ply"
    ply_path_horiz = checkPerpendicular(ply_path_normals, file_name_out)
    
    print('\n\nFinished\n')