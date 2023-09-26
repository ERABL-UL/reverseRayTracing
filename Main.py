#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:57:20 2023

@author: willalbert
"""

from os.path import join
from utils import Config as config
from CreateVoxels import preparationVoxels, createBlob, findCenters, coordsImgToPnt
from ExtractClass import extractClass
from ComputeNormals import compute_surface_normals, uniformNormals
from CheckPerpendicular import checkPerpendicular
from RevRayTracing import createRays, revRayTracingFunction, getFacadeOccl
import OSToolBox as ost
from OccupancyGrid import generate_occupancy_grid, generate_single_occ_grid
import numpy as np
from datetime import datetime


if __name__ == "__main__":
    TSTART = datetime.now()

    # FLAT extraction
    idx_class = 1       # 1 == flat
    name_class = "flat"
    name_scalar = "scalar_lbl"
    ply_path_extracted = extractClass(idx_class, name_scalar, name_class)

    # BUILDING extraction
    idx_class = 6       # 6 == building
    name_class = "building"
    name_scalar = "scalar_lbl"
    ply_path_extracted = extractClass(idx_class, name_scalar, name_class)

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
    name_scalar = "scalar_lbl"
    points = preparationVoxels(name_scalar, ply_path_horiz=join(config.folder_path_out, "building_horiz.ply"))

    grid_type = "normal"
    histo_grid_nor, histo_grid_lbl, min_coords = generate_occupancy_grid(points, grid_type)
    del points
    # Check which normal is the median in each voxel
    ply_path_voxels = generate_single_occ_grid(histo_grid_nor, grid_type, min_coords, histo_grid_lbl)  # Creates vxl grid file "occGrid_norm_lbl.ply", "occGrid_normals.ply" or "occGrid_lbl.ply"


    # ===================================
    # FOR DEBUGGING ONLY
    # ply_path_voxels = join(config.folder_path_out, "occGrid.ply")
    # ===================================


    blob_path, xMin, yMin = createBlob(ply_path_voxels, grid_type)
    centroidsCoord_img, cornersCoord_img = findCenters(blob_path)          # Centroids in the image. NOT IN THE POINT CLOUD COORDINATES
    # xMin=-287.047791; yMin=-81.554855         # To debug Ste-Marthe
    # xMin=-46.25; yMin=-44.002899169921875     # To debug SIMULATION
    centroidsCoord_pnt, cornersCoord_pnt = coordsImgToPnt(centroidsCoord_img, cornersCoord_img, xMin, yMin)
    
    ost.write_ply(join(config.folder_path_out, "centroids.ply"), centroidsCoord_pnt, ['x', 'y'])
    ost.write_ply(join(config.folder_path_out, "corners.ply"), np.vstack(cornersCoord_pnt), ['x', 'y'])
    
    ply_path_normals_rect = uniformNormals(centroidsCoord_pnt, cornersCoord_pnt, ply_path_horiz)  # Creates the file join(config.folder_path_out, "building_horiz_rectif_inCntrs.ply")
    
    # Recreate Voxel Grid with right facades' normals
    name_scalar = "scalar_lbl"
    points = preparationVoxels(name_scalar, ply_path_horiz=ply_path_normals_rect)  # Returns a ready to use point cloud and Creates the file "/complete_with_normals.ply"


    # ===================================
    # FOR DEBUGGING ONLY
    # p = ost.read_ply(join(config.folder_path_out, "complete_with_normals.ply"))
    # x = p["x"]
    # y = p["y"]
    # z = p["z"]
    # lbl = p["lbl"]
    # nx = p["nx"]
    # ny = p["ny"]
    # nz = p["nz"]
    # points = np.c_[x, y, z, lbl, nx, ny, nz]
    # del x, y, z, lbl, nx, ny, nz, p
    # ===================================


    grid_type = ["normal", "label"]
    histo_grid_nor, histo_grid_lbl, min_coords = generate_occupancy_grid(points, grid_type)
    del points

    # Check which label is the most common in each voxel
    ply_path_voxels = generate_single_occ_grid(histo_grid_nor, grid_type, min_coords, histo_grid_lbl)  # Creates vxl grid file "occGrid_norm_lbl.ply", "occGrid_normals.ply" or "occGrid_lbl.ply"

    TEST = False
    minProb = 60
    maxProb = 100
    k = 4
    if TEST:
        ply_path_voxels = join(config.folder_path_out, "TESToccGrid_norm_lbl.ply")
        ply_path_groups_rays = join(config.folder_path_out, "TESTallRays/")
        # createRays(ply_path_groups_rays, ply_path_voxels)

        vxl_path_out = join(config.folder_path_out, "TESTocclusions_vxls/")
        # revRayTracingFunction(ply_path_groups_rays, ply_path_voxels, vxl_path_out)

        occlProb = vxl_path_out + "TESTocclusions_more_0.ply"
        getFacadeOccl(ply_path_voxels, vxl_path_out, occlProb, minProb, maxProb, k, TEST)

    else:
        ply_path_voxels = join(config.folder_path_out, "occGrid_norm_lbl.ply")
        ply_path_groups_rays = join(config.folder_path_out, "allRays/")
        createRays(ply_path_groups_rays, ply_path_voxels)

        vxl_path_out = join(config.folder_path_out, "occlusions_vxls/")
        revRayTracingFunction(ply_path_groups_rays, ply_path_voxels, vxl_path_out)

        occlProb = vxl_path_out + "occlusions_more_0.ply"
        getFacadeOccl(ply_path_voxels, vxl_path_out, occlProb, minProb, maxProb, k, TEST)
    
    print('\n\nFinished\n')

    TEND = datetime.now()
    print(f'Processing time: {TEND - TSTART} [HH:MM:SS]')
