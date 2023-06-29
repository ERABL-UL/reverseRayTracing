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
    print("\n###################")
    print("Extracting the class '{}' with label index {} from point cloud {} ...".format(name_class, idx_class, config.file_name_read))
    
    pntCloud = ost.read_ply(join(config.folder_path_in, config.file_name_read))
    x = pntCloud['x']
    y = pntCloud['y']
    z = pntCloud['z']
    lbl= pntCloud[name_scalar]
    pnt_array = np.c_[x, y, z, lbl]
    
    inds = np.where(lbl.astype(np.int32) != idx_class)[0].astype(np.int32)
    
    pnt_extract = np.delete(pnt_array, inds, axis=0)

    ply_path_extracted = join(config.folder_path_out, name_class + "_extraction.ply")
    
    ost.write_ply(ply_path_extracted, pnt_extract, ["x","y","z","scalar_label"])
    
    print("Points extracted with success!\nA ply file has been created here: {}".format(ply_path_extracted))
    print("###################")
    
    return ply_path_extracted

