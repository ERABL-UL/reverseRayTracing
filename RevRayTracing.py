#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:58:52 2023

@author: willalbert


#######################

ply_path_voxels : Path of the labaled voxel cloud
lbl_no1: Voxels from which the reverse ray tracing starts (usualy class Building - 6)
lbl_no2: Voxels from which the reverse ray tracing finishes (usualy class Flat - 1)


"""

import os, sys
import OSToolBox as ost
import numpy as np
from numpy import cos, tan, pi
from scipy.spatial import KDTree

from tqdm import tqdm
from datetime import datetime



def createRays(ply_path_out, ply_path_voxels):
    voxels = ost.read_ply(ply_path_voxels)
    x = voxels["x"]
    y = voxels["y"]
    z = voxels["z"]
    try:
        lbl = voxels["lbl"]
    except:
        lbl = voxels["scalar_lbl"]
    nx = voxels["nx"]
    ny = voxels["ny"]
    nz = voxels["nz"]
    voxels = np.c_[x, y, z, lbl, nx, ny, nz]
    del x, y, z, lbl, nx, ny, nz
    
    dist_step = 0.4                        # Distance for only one move
    dist_at_angle_0 = 13                    # Meters
    angle_step = np.deg2rad(0.5)            # 0.5 degrees
    angle_max = np.deg2rad(30)              # 60 degrees 
    
    nb_angle_steps = np.ceil((angle_max*2) / angle_step).astype(np.int32)
    idx_facade = np.where(voxels[:,4:] != (0.,0.,0.))[0].astype(np.int32)   # Get indices of all facade building voxels (where normal != 0)
    
    idx_flat = np.where(voxels[:,3] == 1)[0].astype(np.int32)   # Get indices of all flat voxels to get the min and max
    # z_min = np.min(voxels[idx_flat][:,2])
    # z_max = np.max(voxels[idx_flat][:,2])
    # z_mean = np.mean(voxels[idx_flat][:,2])
    dist_steps_verti = 4
    nb_dist_steps_verti = dist_steps_verti*2
    
    max_file_len = len(idx_facade.shape[0].__str__())
    max_folder_len = len(int(np.ceil(idx_facade.shape[0]/500)).__str__())
    curr_folder_number = -1
    curr_file_number = 0
    
    lst_vxl_facade = []
    for idx in np.arange(idx_facade.shape[0]):        
        if voxels[idx_facade[idx]][3] == 6 and not np.any(np.isin(lst_vxl_facade, set(voxels[idx_facade[idx]]))):       # To avoid processing duplicates
            if curr_file_number%300 == 0: print("{:.2f}% done".format(idx/idx_facade.shape[0]*100))
            lst = []
            lst_vxl_facade.append(set(voxels[idx_facade[idx]]))
            
            starting_vxl_center = voxels[idx_facade[idx]][:3]
            starting_vxl_normal = voxels[idx_facade[idx]][4:]
            
            angle_x = np.arctan2(starting_vxl_normal[1], starting_vxl_normal[0])
            
            go_to_next = False
            again = True
            empty_tour = 0
            while again:
                ################################
                # Check the height of the street
                if angle_x < 0: angle_x += 2*pi
                
                if 0 <= angle_x <= pi:
                    adjustment_x = cos(angle_x) * dist_at_angle_0
                    adjustment_y = np.sqrt(dist_at_angle_0**2 - adjustment_x**2)
                
                else:
                    adjustment_x = cos(angle_x) * dist_at_angle_0
                    adjustment_y = -np.sqrt(dist_at_angle_0**2 - adjustment_x**2)
                    
                point_ray_interm_x = starting_vxl_center[0] + adjustment_x
                point_ray_interm_y = starting_vxl_center[1] + adjustment_y
                
                lstTEST = []
                
                test_x = np.where(abs(voxels[idx_flat][:,0] - point_ray_interm_x) <= 5, voxels[idx_flat][:,0], 0)
                test_x = np.where(test_x != 0)
                test_y = np.where(abs(voxels[idx_flat][:,1] - point_ray_interm_y) <= 5, voxels[idx_flat][:,1], 0)
                test_y = np.where(test_y != 0)
                
                test_x_y = np.append(test_x[0], test_y[0])
                u, c = np.unique(test_x_y, return_counts=True)
                lstTEST = u[c > 1]
                
                # for elemx in test_x[0]:
                #     for elemy in test_y[0]:
                #         if elemx == elemy:
                #             lstTEST.append(elemx)
                            
                if lstTEST.size != 0:
                    z_min = np.min(voxels[idx_flat[lstTEST]][:,2])
                    # z_center = z_min + (dist_steps_verti/2)
                    empty_tour = 0
                    again = False
                    
                else:
                    angle_x = np.arctan2(-starting_vxl_normal[1], -starting_vxl_normal[0])
                    empty_tour += 1
                    
                    if empty_tour > 1:
                        go_to_next = True
                        break
            
            if go_to_next:
                continue
            
                ################################
            
            for idx_current_ray in np.arange(nb_angle_steps):
                dist_tot_to_travel = dist_at_angle_0 / cos(angle_max - angle_step*idx_current_ray)
                nb_dist_steps = np.ceil(dist_tot_to_travel / dist_step)         # Number of time we have to move forward in the current ray
                
                for idx_dist_step in np.arange(1, nb_dist_steps+1):
                    ang = angle_x - (angle_max - angle_step*idx_current_ray)
                    if ang < 0: ang += 2*pi
                    
                    if 0 <= ang <= pi or 2*pi < ang:
                        adjustment_x = cos(angle_x - (angle_max - angle_step*idx_current_ray)) * (dist_step * idx_dist_step)
                        adjustment_y = np.sqrt((dist_step * idx_dist_step)**2 - adjustment_x**2)
                        
                    else:
                        adjustment_x = cos(angle_x - (angle_max - angle_step*idx_current_ray)) * (dist_step * idx_dist_step)
                        adjustment_y = -np.sqrt((dist_step * idx_dist_step)**2 - adjustment_x**2)
                    
                    point_ray_interm_x = starting_vxl_center[0] + adjustment_x
                    point_ray_interm_y = starting_vxl_center[1] + adjustment_y
                    
                    for idx_dist_step_verti in np.arange(1, nb_dist_steps_verti+1):
                        angle = np.arctan2((abs(starting_vxl_center[2]-z_min) - idx_dist_step_verti * dist_step), dist_tot_to_travel)
                        corr_z = np.sqrt(adjustment_x**2 + adjustment_y**2) * tan(angle)
                        point_ray_interm_z = starting_vxl_center[2] - corr_z
                        
                        idx_ray = idx_current_ray*nb_dist_steps_verti + (idx_dist_step_verti - 1)
                        idx_dist = idx_dist_step - 1
                        idx_height = idx_dist_step_verti - 1
                        
                        lst.append([point_ray_interm_x, point_ray_interm_y, point_ray_interm_z, idx_ray, idx_dist, idx_height])
            
            if curr_file_number % 500 == 0:
                curr_folder_number += 1
                
            directory = "{}groupedRays_{:0{}d}".format(ply_path_out, curr_folder_number, max_folder_len)
            
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            name = "{}/rays_{:0{}d}.ply".format(directory, np.int32(curr_file_number), max_file_len)
            curr_file_number += 1
            
            ost.write_ply(name, np.array(lst), ['x','y','z','idx_ray','idx_dist','idx_height'])
            # break
                
        # break

    # ost.write_ply("/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/rays.ply", np.array(lst), ['x','y','z','idx_ray','idx_dist','idx_height'])
    print("{}% done".format(100))
    
    return "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/allRays/"


def doRevRayTracing(ply_path_groups_rays, ply_path_voxels, vxl_path_out):
    TSTART = datetime.now()
    if not os.path.exists(vxl_path_out): os.makedirs(vxl_path_out)
    name = "{}occlusions.ply".format(vxl_path_out)
    
    voxels = ost.read_ply(ply_path_voxels)
    x = voxels["x"]
    y = voxels["y"]
    z = voxels["z"]
    try:
        lbl = voxels["lbl"]
    except:
        lbl = voxels["scalar_lbl"]
    occl = np.zeros(x.shape[0])
    voxels_occl = np.c_[x, y, z, lbl, occl]
    list_occluder_lbl = [np.array(int(0))] * x.shape[0]
    list_occluder_occl = [np.array(int(0))] * x.shape[0]
    del voxels, x, y, z, occl
    
    kdtree = KDTree(voxels_occl[:, :3])
    count_folders = len(ost.getDirBySubstr(ply_path_groups_rays, "groupedRays"))
    for folder_num, folder in enumerate(ost.getDirBySubstr(ply_path_groups_rays, "groupedRays")):
        print("\033[J")
        print("{:.2f}% of folders are done\n".format(folder_num/count_folders*100))
        
        count_files = len(ost.getFileByExt(folder, "ply"))
        for file_num, file in enumerate(ost.getFileByExt(folder, "ply")):
            print("\r{:.2f}% of files are done in the current folder".format(file_num/count_files*100), end="")
            # File represent the rays from one voxel
            rays = ost.read_ply(file)
            x = rays["x"]
            y = rays["y"]
            z = rays["z"]
            idx_ray = rays["idx_ray"].astype(int)
            idx_dist = rays["idx_dist"].astype(int)
            # idx_height = rays["idx_height"].astype(int)
            del rays
            
            points_rays = np.c_[x, y, z, idx_ray, idx_dist]
            unique_idx_ray = np.unique(idx_ray)
            
            for current_idx_ray in unique_idx_ray:
                mask_current_ray = np.where(points_rays[:, 3] == current_idx_ray)[0]
                
                current_ray = points_rays[mask_current_ray]
                order = np.lexsort([current_ray[:, -1]])[::-1]      # Descending sorting
                current_ray = current_ray[order]                    # Current ray's points sorted from the farthest to nearest 
                current_ray = current_ray[-25:,:]
                occluder = 0                                        # 0 == False
                
                for point in current_ray:
                    _, indice = kdtree.query(point[:3], k=1, workers=24)
                    point_lbl = lbl[indice]
                    
                    if point_lbl in [2, 3, 4, 6, 7]:
                        occluder = 1                                            # 1 == True
                        current_occluder_lbl = point_lbl
                        list_occluder_lbl[indice] = np.append(list_occluder_lbl[indice], int(current_occluder_lbl))
                        counts = np.bincount(list_occluder_lbl[indice][1:])
                        voxels_occl[:, 3][indice] = np.argmax(counts)
                        
                    elif point_lbl not in [2, 3, 4, 6, 7] and occluder == 1:                                         # -> point_lbl NOT in [2, 3, 4, 6, 7], occluder == 1
                        # voxels_occl[:, 4][indice] += 1
                        list_occluder_occl[indice] = np.append(list_occluder_occl[indice], int(1))
                        counts = np.bincount(list_occluder_occl[indice][1:])
                        try:
                            voxels_occl[:, 4][indice] = int(counts[1]/np.sum(counts)*100)
                        except: pass
                            
                        list_occluder_lbl[indice] = np.append(list_occluder_lbl[indice], int(current_occluder_lbl))
                        counts = np.bincount(list_occluder_lbl[indice][1:])
                        voxels_occl[:, 3][indice] = np.argmax(counts)
                        
                    else:                                                       # -> point_lbl NOT in [2, 3, 4, 6, 7], occluder == 0
                        # voxels_occl[:, 4][indice] -= 70
                        list_occluder_occl[indice] = np.append(list_occluder_occl[indice], int(0))
                        counts = np.bincount(list_occluder_occl[indice][1:])
                        try:
                            voxels_occl[:, 4][indice] = int(counts[1]/np.sum(counts)*100)
                        except: pass
                        
            
        voxels_occl_copy = voxels_occl.copy()
        voxels_occl_copy = voxels_occl_copy[voxels_occl[:,4] > 0]
        name = "{}occlusions_more_0.ply".format(vxl_path_out)
        ost.write_ply(name, voxels_occl_copy, ['x','y','z','lbl','occl'])
        
        voxels_occl_copy = voxels_occl.copy()
        voxels_occl_copy = voxels_occl_copy[voxels_occl[:,4] > 50]
        name = "{}occlusions_more_50.ply".format(vxl_path_out)
        ost.write_ply(name, voxels_occl_copy, ['x','y','z','lbl','occl'])
            
        print("\r{:.2f}% of files are done in the current folder".format(100), end="")
    
    # voxels_occl[:, 4] = np.where(voxels_occl[:, 4] > 6, 1, 0)
    # voxels_occl = voxels_occl[voxels_occl[:,4] > 0]
    # name = "{}occlusions_last.ply".format(vxl_path_out)
    # ost.write_ply(name, voxels_occl, ['x','y','z','lbl','occl'])
    
    print("\033[J")
    print("\n{:.2f}% of folders are done\n".format(100))
    print("\r{:.2f}% of files are done in the current folder".format(100), end="")
    
    TEND = datetime.now()
    print(f'Processing time: {TEND-TSTART} [HH:MM:SS]')
                
    return 0    


# ply_path_voxels = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/occGrid_norm_lbl.ply"
# ply_path_out = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/allRays/"
# r = createRays(ply_path_out, ply_path_voxels)

ply_path_voxels = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/TESToccGrid_norm_lbl.ply"
ply_path_groups_rays = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/TESTallRays/"
vxl_path_out = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/TESTocclusions_vxls/"
r = doRevRayTracing(ply_path_groups_rays, ply_path_voxels, vxl_path_out)


# ply_path_voxels = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/occGrid_norm_lbl.ply"
# ply_path_groups_rays = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/allRays/"
# vxl_path_out = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/occlusions_vxls/"
# r = doRevRayTracing(ply_path_groups_rays, ply_path_voxels, vxl_path_out)