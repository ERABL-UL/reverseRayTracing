#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:17:01 2023

@author: willalbert
"""

folder_path_in = "/home/willalbert/Documents/GitHub/KPConvPyTorch/test/Log_2023-06-02_15-18-46_FT_BON/predictions"
folder_path_out = "/home/willalbert/Documents/GitHub/reverseRayTracing/OUT"

file_name_read = "89_merged.ply"
#folder_path = "/home/willalbert/Documents/GitHub/KPConvPyTorch/test/Log_2023-06-02_15-18-46_FT_BON/predictions"

file_name_prob = "segmentedSteMarthePROB.ply"

file_name_prob = "89_0000001PROB.ply"

voxel_size = 0.5

r_normals = 1 # Max radius to compute normals
nn = 30