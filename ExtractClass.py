#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:21:36 2023

@author: willalbert
"""
import numpy as np
import OSToolBox as ost
from utils import Config
from os.path import join

def extractClass(idx_class, name_scalar, name_class):
    config = Config
    
    folder_path_in = config.folder_path_in
    folder_path_out = config.folder_path_out
    file_name = config.file_name
    
    pntCloud = ost.read_ply(join(folder_path_in, file_name))
    inds = np.where(pntCloud[name_scalar].astype(np.int32) == idx_class)[0].astype(np.int32)
    
    points = []
    for i in pntCloud:
        points.append(list(i))
    points = np.array(points)
    print(points.ndim)
    ply_path = join(folder_path_out, name_class + "_extraction.ply")
    
    ost.write_ply(ply_path, points, ["x","y","z","scalar_label"])
    
    return ply_path