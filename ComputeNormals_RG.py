#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:34:17 2023

@author: willalbert
"""

import numpy as np
import OSToolBox as ost
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from utils import Config
from os.path import join


def compute_surface_normals_RG(radius, nn, file_path_read, file_name_out):
    config = Config
    
    pntCloud = ost.read_ply(file_path_read)
    
    x = pntCloud["x"]
    y = pntCloud["y"]
    z = pntCloud["z"]
    # lbl = pntCloud["scalar_label"]
    points = np.c_[x, y, z]
    
    # Compute nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=nn).fit(points)
    _, indices = neighbors.kneighbors(points)

    normals = []
    for i in range(len(points)):
        # Get the neighbors of the current point
        neighbor_indices = indices[i]
        neighbor_points = points[neighbor_indices]

        # Compute the covariance matrix
        centered_points = neighbor_points - neighbor_points.mean(axis=0)
        covariance_matrix = np.cov(centered_points, rowvar=False)

        # Perform eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        # Select the eigenvector corresponding to the smallest eigenvalue
        normal = eigenvectors[:, -1]

        # Flip the normal to have consistent orientation
        if np.dot(normal, neighbor_points[0] - points[i]) < 0:
            normal = -normal

        normals.append(normal)
        
    np.array(normals)
    
    ply_path_normals_RG = join(config.folder_path_out, file_name_out)
    
    o3d.io.write_point_cloud(ply_path_normals_RG, normals)
    
    print("Normals computed by region growing with success!\nA ply file has been created here: {}".format(ply_path_normals_RG))
    print("###################")
        
    return ply_path_normals_RG