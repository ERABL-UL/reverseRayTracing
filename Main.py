#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:57:20 2023

@author: willalbert
"""


from utils import Config
from CreateVoxels import createVoxels
import OSToolBox as ost
from os.path import join


if __name__ == "__main__":
    config = Config
    p = createVoxels()
    ost.write_ply(join(config.folder_path, "occGrid.ply"), p, ["x","y","z","occ"])

    print('Finished')