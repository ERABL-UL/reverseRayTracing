#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:13:46 2023

@author: willalbert
"""

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import os
import numpy as np
import OSToolBox as ost
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt

from utils import Config
from OccupancyGrid import generate_occupancy_grid, generate_single_occ_grid
from os.path import join

from scipy.spatial import KDTree
    

def createVoxels(name_scalar, ply_path_horiz):
    config = Config
    print("\n###################")
    print("Creating voxels of {} meters from point cloud {} ...".format(config.voxel_size, config.file_name_read))
    
    file_path_read = join(config.folder_path_in, config.file_name_read)
    pntCloud = ost.read_ply(file_path_read)
    
    # Hole point cloud
    x = pntCloud["x"]
    y = pntCloud["y"]
    z = pntCloud["z"]
    lbl = pntCloud[name_scalar]
    zeros_col = np.zeros(x.shape)
    points = np.c_[x, y, z, lbl, zeros_col, zeros_col, zeros_col]
    
    # Point cloud of facades with normals
    pntCloudFacades = ost.read_ply(ply_path_horiz)
    x = pntCloudFacades["x"]
    y = pntCloudFacades["y"]
    z = pntCloudFacades["z"]
    nx = pntCloudFacades["nx"]
    ny = pntCloudFacades["ny"]
    nz = pntCloudFacades["nz"]
    pointsFacades = np.c_[x, y, z, nx, ny, nz]
    
    del x, y, z, nx, ny, nz, lbl                 # Freeing space, useful when debugging
    
    # Give normals from facades to the same facades' points in the hole point cloud
    kdtree = KDTree(points[:, :3])
    _, indices = kdtree.query(pointsFacades[:, :3], k=1, distance_upper_bound=0.00001, workers=24)
    points[:, 4:][indices] = pointsFacades[:, 3:]
    
    
    ost.write_ply(join(config.folder_path_out, "tempPntCloud.ply"), points, ["x","y","z","lbl","nx","ny","nz"])
    
    # point_cloud=points        # Enlever apres debuggage
    # voxel_size=0.5            # Enlever apres debuggage
    
    # Grille d'occupation
    label_histo_grid, normal_histo_grid, min_coords = generate_occupancy_grid(points, config.voxel_size)
    del points                  # Freeing space, useful when debugging
    
    # Check which label is the most common in each voxel
    label_grid = generate_single_occ_grid(label_histo_grid)
    #normal_grid = generate_single_occ_grid(normal_histo_grid)
    del label_histo_grid        # Freeing space, useful when debugging
    
    # Generate voxel grid coordinates
    x, y, z = np.indices(label_grid.shape)
    
    # Reshape the array into a 1D vector
    lbl = np.reshape(label_grid, (-1,))
    
    # Convert voxel indices to "world" coordinates
    x = x * config.voxel_size + (min_coords[0] + config.voxel_size/2)
    y = y * config.voxel_size + (min_coords[1] + config.voxel_size/2)
    z = z * config.voxel_size + (min_coords[2] + config.voxel_size/2)
    
    x = np.reshape(x, (-1,))
    y = np.reshape(y, (-1,))
    z = np.reshape(z, (-1,))
    
    p = np.c_[x,y,z,lbl]
    
    ply_path_voxels = join(config.folder_path_out, "occGrid.ply")
    ost.write_ply(ply_path_voxels, p, ["x","y","z","lbl"])

    print("Voxels created with success!\nA ply file has been created here: {}".format(ply_path_voxels))
    print("###################")
    
    return ply_path_voxels



def createBlob(voxels_path):
    config = Config
    
    voxels_path = join(config.folder_path_out,"occGrid.ply")
    voxels = ost.read_ply(voxels_path)
    
    x = voxels["x"]
    y = voxels["y"]
    #z = voxels["z"]
    lbl = voxels['lbl']
    voxels3D_flat = np.c_[x, y, lbl]
    
    idx = np.where(6. == lbl)[0]
    voxels3D_flat_6 = voxels3D_flat[idx, :]
    voxels3D_flat_6[:, -1] = True
    voxels3D_flat_6 = np.unique(voxels3D_flat_6, axis=0)
    
    xMin = min(voxels3D_flat_6[:,0])
    yMin = min(voxels3D_flat_6[:,1])
    yMax = max(voxels3D_flat_6[:,1])
    
    xs = np.array((voxels3D_flat_6[:,0]-xMin)*2, dtype = int)
    ys = np.array((voxels3D_flat_6[:,1]-yMin)*2, dtype = int)
    
    ost.write_ply(join(config.folder_path_out,"voxels2Dessay.ply"), voxels3D_flat_6, ["x","y","building"])
    
    sizeX = max(xs)-min(xs)
    sizeX = int(np.ceil(sizeX))
    sizeY = max(ys)-min(ys)
    sizeY = int(np.ceil(sizeY))
    
    num_points = xs.shape[0]
    
    # Create an empty image
    image = Image.new('RGB', (sizeX+1, sizeY+1), (255, 255, 255))
    # Set pixel colors based on point cloud
    pixels = image.load()
    for i in np.arange(num_points-1):
        pixels[xs[i], ys[i]] = (0, 0, 0)  # Set point cloud color (black)
        
    image.save(join(config.folder_path_out,"blob.png"), 'PNG')      # The image will be mirrored because the origin of the image is at the top left corner
    
    return join(config.folder_path_out,"blob.png"), xMin, yMin


def findCenters(img_path):
    config = Config
    #img_path = join(config.folder_path_out,"blob.png")
    # read image through command line
    inputImage = cv2.imread(img_path)
    inputCopy = inputImage.copy()
    
    grayInput = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    _, binaryImage = cv2.threshold(grayInput, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Set kernel (structuring element) size:
    kernelSize = (1, 1)
    
    # Set operation iterations:
    opIterations = 5
    
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    
    # Perform Dilate:
    dilateImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    centroids = []
    
    # Find the contours on the binary image:
    contours, _ = cv2.findContours(dilateImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ratioWH = 0.5
    
    # Look for the outer bounding boxes (no children):
    for _, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)     # returns (x_center, y_center), (width, height), theta
        (x_center, y_center), (width, height), theta = rect
        peri = cv2.arcLength(cnt, True) 
        
        if (125 < area < 3000) and height != 0 and (ratioWH < abs(width/height) < 1/ratioWH):
        # if height != 0:
            box0 = cv2.boxPoints(rect)
            box = np.rint(box0).astype(int)
            
            x_center = int(np.round(x_center))
            y_center = int(np.round(y_center))
            centroids.append([x_center, y_center])
            
            cv2.circle(inputCopy, (x_center, y_center), 4, (255, 255, 0), -1)
            cv2.drawContours(inputCopy,[box],0,(0,0,255),2)
            
    # Uncomment to visialize
    # Note that the image will be mirrored because the origin of the image is at the top left corner
    cv2.imshow("Bounding Rectangles", inputCopy)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    return np.asarray(centroids).astype(float)


def centroidsImgToPnt(centroids_img, xMin, yMin):
    config = Config
    
    centroids_pnt = centroids_img.copy()
    centroids_pnt[:,0] /= 2
    centroids_pnt[:,1] /= 2
    
    centroids_pnt[:,0] += xMin
    centroids_pnt[:,1] += yMin
    
    return centroids_pnt