#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:38:06 2023

@author: willalbert
"""

import open3d as o3d
import numpy as np
from utils import Config
import OSToolBox as ost
from os.path import join
from sklearn.neighbors import NearestNeighbors
import cv2
from shapely.geometry import Point, Polygon


def compute_surface_normals(radius, nn, file_path_read, file_name_out):
    config = Config
    print("\n###################")
    print("Computing normals with max nn = {} and max radius = {} ...".format(nn, radius))
    
    pntCloud = ost.read_ply(file_path_read)
    
    x = pntCloud["x"]
    y = pntCloud["y"]
    z = pntCloud["z"]
    # lbl = pntCloud["scalar_label"]
    points = np.c_[x, y, z]
    
    
    # Convert the point cloud to an Open3D format
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(points)

    # Estimate surface normals using Open3D's method
    o3d_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nn)
    )

    # Get the surface normals as a NumPy array
    normals = np.asarray(o3d_cloud.normals)
    
    ply_path_normals = join(config.folder_path_out, file_name_out)
    
    o3d.io.write_point_cloud(ply_path_normals, o3d_cloud)
    
    print("Normals computed with success!\nA ply file has been created here: {}".format(ply_path_normals))
    print("###################")
    
    return ply_path_normals


def compute_surface_normals_RG(nn, file_path_read, file_name_out):
    config = Config
    
    print("\n###################")
    print("Computing normals by region growing with nn = {} ...".format(nn))
    
    pntCloud = ost.read_ply(file_path_read)
    
    x = pntCloud["x"]
    y = pntCloud["y"]
    z = pntCloud["z"]
    # lbl = pntCloud["scalar_label"]
    points = np.c_[x, y, z]
    
    # Compute nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=nn).fit(points)
    _, indices = neighbors.kneighbors(points)

    empty = np.zeros(shape=points.shape)
    normals = np.hstack((points, empty))
    
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
        # if np.dot(normal, neighbor_points[0] - points[i]) < 0:
        if (normal[0] + normal[1] + normal[2]) < 0:
            normal = -normal

        normals[i,3:] = normal
    
    ply_path_normals_RG = join(config.folder_path_out, file_name_out)
    
    ost.write_ply(ply_path_normals_RG, normals, ["x","y","z","nx","ny","nz"])
    
    print("Normals computed by region growing with success!\nA ply file has been created here: {}".format(ply_path_normals_RG))
    print("###################")
        
    return ply_path_normals_RG


def uniformNormals(centroidsCoord_pnt, cornersCoord_pnt, ply_path_horiz):
    config = Config
    
    print("\n###################")
    print("Pointing normals toward the outside of the buildings ...")
    
    cloudFacade = ost.read_ply(ply_path_horiz)
    
    points_normals = np.c_[cloudFacade["x"], cloudFacade["y"], cloudFacade["z"], cloudFacade["nx"], cloudFacade["ny"], cloudFacade["nz"]]          # No need for cloudFacade["z"]
    
    from datetime import datetime
    
    dt = datetime.now()
    print(dt)
    mask = []
    for i in np.arange(points_normals.shape[0]):
        pnt_p = Point(points_normals[i][0], points_normals[i][1])
        pnt_plus_normal = points_normals[i][0:2] + points_normals[i][3:5]
        pnt_xy = points_normals[i][0:2]
        for j in np.arange(cornersCoord_pnt.shape[0]):
            cCoord = cornersCoord_pnt[j]
            poly_p0 = (cCoord[0][0], cCoord[0][1])
            poly_p1 = (cCoord[1][0], cCoord[1][1])
            poly_p2 = (cCoord[2][0], cCoord[2][1])
            poly_p3 = (cCoord[3][0], cCoord[3][1])
            poly = Polygon([poly_p0, poly_p1, poly_p2, poly_p3])
            if poly.contains(pnt_p) is True:
                mask.append(i)
                if np.sqrt(np.sum(np.square(pnt_plus_normal - centroidsCoord_pnt[j]))) < np.sqrt(np.sum(np.square(pnt_xy - centroidsCoord_pnt[j]))):
                    points_normals[i][3:] *= -1
                    break
            
    points_normals_inCntrs = points_normals[mask]
    print(datetime.now() - dt) # 0:06:51.842991 (H:MM:SS)
    print()
    
    ply_path_normals_rect = join(config.folder_path_out, "building_horiz_rectif_inCntrs.ply")
    ost.write_ply(ply_path_normals_rect, points_normals_inCntrs, ["x","y","z","nx","ny","nz"])

    return ply_path_normals_rect