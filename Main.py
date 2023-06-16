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


if __name__ == "__main__":
    config = Config
    p = createVoxels()
    ost.write_ply(join(config.folder_path, "OUT/occGrid.ply"), p, ["x","y","z","occ"])
    
    # Compute normal of road
    idx_class = 1
    ply_path = extractClass(idx_class)
    
    road_pnt_cloud = o3d.io.read_point_cloud(ply_path)
    road_pnt_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.75))
    normals = road_pnt_cloud.normals
    
    
    
    print('Finished')