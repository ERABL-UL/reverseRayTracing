#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:57:20 2023

@author: willalbert
"""


from utils import Config
from CreateVoxels import createVoxels, createBlob, findCenters, centroidsImgToPnt
from ExtractClass import extractClass
from ComputeNormals import compute_surface_normals, compute_surface_normals_RG
from CheckPerpendicular import checkPerpendicular
from RevRayTracing import revRayTracing
import OSToolBox as ost

import numpy as np


if __name__ == "__main__":
    config = Config
   
    if False is True:
        # FLAT extraction
        idx_class = 1       # 1 == flat
        name_class = "flat"
        name_scalar = "scalar_pre"
        ply_path_extracted = extractClass(idx_class, name_scalar, name_class)
        # flat = ost.read_ply(ply_path_extracted)
        
        # FLAT normals computing
        # file_name_out = name_class + "_normals.ply"
        # ply_path_normals = compute_surface_normals(config.r_normals, config.nn, ply_path_extracted, file_name_out)
        
        # BUILDING extraction
        idx_class = 6       # 6 == building
        name_class = "building"
        name_scalar = "scalar_pre"
        ply_path_extracted = extractClass(idx_class, name_scalar, name_class)
        # building = ost.read_ply(ply_path_extracted)
        
        # Computing normals
        file_name_out = name_class + "_normals.ply"
        ply_path_normals = compute_surface_normals(config.r_normals, config.nn, ply_path_extracted, file_name_out)
        
        # Computing normals by region growing
        file_name_out = name_class + "_normals_RG.ply"
        ply_path_extracted = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/building_extraction.ply"
        ply_path_normals_RG = compute_surface_normals_RG(config.nn, file_path_read=ply_path_extracted, file_name_out=file_name_out)
        
        # Extraction of points where normals are horizontals with treshold of 0.2
        file_name_out = name_class + "_horiz.ply"
        ply_path_horiz = checkPerpendicular(ply_path_normals, file_name_out)
        
    # Create Voxel Grid
    name_scalar = "scalar_pre"
    ply_path_voxels = createVoxels(name_scalar, ply_path_horiz = '/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/building_horiz.ply')
        
    if 'ply_path_voxels' not in globals():
        ply_path_voxels = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/occGrid.ply"
        
    blob_path, xMin, yMin = createBlob(ply_path_voxels)
    centroids_img = findCenters(blob_path)          # Centroids in the image. NOT IN THE POINT CLOUD
    centroids_pnt = centroidsImgToPnt(centroids_img, xMin, yMin)
    
    ost.write_ply("/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/centroids.ply", centroids_pnt, ['x', 'y'])
    
    r = revRayTracing(ply_path_voxels, ply_path_normals, 6, 1)        # 6: Building   1: Flat
    print(r)
    
    print('\n\nFinished\n')