#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:57:20 2023

@author: willalbert
"""


from utils import Config
from CreateVoxels import preparationVoxels, createBlob, findCenters, coordsImgToPnt
from ExtractClass import extractClass
from ComputeNormals import compute_surface_normals, uniformNormals
from CheckPerpendicular import checkPerpendicular
from RevRayTracing import revRayTracing
import OSToolBox as ost
from OccupancyGrid import generate_occupancy_grid, generate_single_occ_grid
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
        
        # # Computing normals by region growing
        # file_name_out = name_class + "_normals_RG.ply"
        # ply_path_extracted = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/building_extraction.ply"
        # ply_path_normals_RG = compute_surface_normals_RG(config.nn, file_path_read=ply_path_extracted, file_name_out=file_name_out)
        
        # Extraction of points where normals are horizontals with treshold of 0.2
        file_name_out = name_class + "_horiz.ply"
        ply_path_horiz = checkPerpendicular(ply_path_normals, file_name_out)
        
    
    # Create Voxel Grid
    name_scalar = "scalar_pre"
    points = preparationVoxels(name_scalar, ply_path_horiz = '/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/building_horiz.ply')

    grid_type = "normal"
    histo_grid_nor, histo_grid_lbl, min_coords = generate_occupancy_grid(points, config.voxel_size, grid_type)
    del points
    # Check which label is the most common in each voxel
    ply_path_voxels = generate_single_occ_grid(histo_grid_nor, grid_type, min_coords, histo_grid_lbl)
    
    ply_path_voxels = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/occGrid.ply"
        
    label = 6.
    blob_path, xMin, yMin = createBlob(ply_path_voxels, label, grid_type)
    
    # ===================================
    
    ply_path_voxels = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/occGrid.ply"
    ply_path_horiz = '/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/building_horiz.ply'
    blob_path = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/blob.png"
    
    centroidsCoord_img, cornersCoord_img = findCenters(blob_path)          # Centroids in the image. NOT IN THE POINT CLOUD COORDINATES
    xMin=-287.047791; yMin=-81.554855
    centroidsCoord_pnt, cornersCoord_pnt = coordsImgToPnt(centroidsCoord_img, cornersCoord_img, xMin, yMin)
    
    ost.write_ply("/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/centroids.ply", centroidsCoord_pnt, ['x', 'y', 'lbl'])
    ost.write_ply("/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/centroids.ply", cornersCoord_pnt, ['x', 'y'])
    
    ply_path_normals_rect = uniformNormals(centroidsCoord_pnt, cornersCoord_pnt, ply_path_horiz)
    
    # Recreate Voxel Grid with right facades' normals
    name_scalar = "scalar_pre"
    points = preparationVoxels(name_scalar, ply_path_horiz = '/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/building_horiz_rectif_inCntrs.ply')


    # ===================================
     
    grid_type = ["normal", "label"]
    
    p = ost.read_ply("/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/complete_with_normals.ply")
    x = p["x"]
    y = p["y"]
    z = p["z"]
    lbl = p["lbl"]
    nx = p["nx"]
    ny = p["ny"]
    nz = p["nz"]
    points = np.c_[x, y, z, lbl, nx, ny, nz]
    del x, y, z, lbl, nx, ny, nz, p
    
    histo_grid_nor, histo_grid_lbl, min_coords = generate_occupancy_grid(points, config.voxel_size, grid_type)
    del points
    # Check which label is the most common in each voxel
    ply_path_voxels = generate_single_occ_grid(histo_grid_nor, grid_type, min_coords, histo_grid_lbl)
    
    # r = revRayTracing(ply_path_voxels, ply_path_normals, 6, 1)        # 6: Building   1: Flat
    # print(r)
    
    print('\n\nFinished\n')