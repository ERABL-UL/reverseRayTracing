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

def extractClass(idx_class):
    config = Config
    
    folder_path = config.folder_path
    file_name = config.file_name
    file_path = join(folder_path, file_name)
    
    pntCloud = ost.read_ply(file_path)
    
    building_inds = np.where(pntCloud["scalar_label"].astype(np.int32) == idx_class)[0]
    
    ply_path = join(folder_path, "OUT/tempPLY_extraction.ply")
    
    ost.write_ply(join(folder_path, "OUT/tempPLY_extraction.ply"), building_inds, ["x","y","z","scalar_label"])
    
    return ply_path