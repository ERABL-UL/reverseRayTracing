#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:48:06 2023

@author: willalbert
"""

from utils import Config
import OSToolBox as ost
from os.path import join
import open3d as o3d

import numpy as np

def checkPerpendicular(ply_path_normals, file_name_out, tolerance=0.2):
    config = Config
    print("\n###################")
    print("Extracting horizontal normals with a tolerance of {} ...".format(tolerance))
    
    pntCloud = ost.read_ply(ply_path_normals)
    
    x = pntCloud['x']
    y = pntCloud['y']
    z = pntCloud['z']
    nx = pntCloud['nx']
    ny = pntCloud['ny']
    nz = pntCloud['nz']
    pntCloud = np.c_[x, y, z, nx, ny, nz]
    
    perpendicular_mask = np.abs(nz) < tolerance
    
    inds = np.where(perpendicular_mask.astype(np.int32) != 1)[0].astype(np.int32)
    
    pntCloud_horiz = np.delete(pntCloud, inds, axis=0)
    
    ply_path_horiz = join(config.folder_path_out, file_name_out)
    ost.write_ply(ply_path_horiz, pntCloud_horiz, ["x","y","z","nx","ny","nz"])
    
    print("Extraction completed with success!\nA ply file has been created here: {}".format(ply_path_horiz))
    print("###################")
    
    return ply_path_horiz