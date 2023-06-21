#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:57:20 2023

@author: willalbert
"""


from utils import Config
from CreateVoxels import createVoxels
from ExtractClass import extractClass
import OSToolBox as ost
from os.path import join
import open3d as o3d

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    config = Config
    p = createVoxels()
    ost.write_ply(join(config.folder_path, "OUT/occGrid.ply"), p, ["x","y","z","occ"])
    
    idx_class = 1       # 1 == flat
    name_class = "flat"
    name_scalar = "pre"
    ply_path = extractClass(idx_class, name_scalar, name_class)
    flat = ost.read_ply(ply_path)
    
    idx_class = 6       # 1 == building
    name_class = "building"
    name_scalar = "pre"
    ply_path = extractClass(idx_class, name_scalar, name_class)
    building = ost.read_ply(ply_path)
    
    print('Finished')   