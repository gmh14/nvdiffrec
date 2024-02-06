# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob
import json

import torch
import numpy as np
from PIL import Image

from render import util

from .dataset import Dataset

###############################################################################
# 3D-gen NeRF image based dataset (synthetic)
###############################################################################


class Dataset3dGen(Dataset):
    def __init__(self, file_path, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.examples = examples

        self.all_tgt_imgs = []
        self.all_mvp_mats = []
        self.all_mv = []
        for img_file in glob.glob(os.path.join(file_path, "*.png")):
            tgt_img = np.array(Image.open(img_file)).astype(np.float32) / 255.0
            self.all_tgt_imgs.append(tgt_img)
            img_id = os.path.basename(img_file).split(".")[0].split("_")[-1]
            
            mvp_mat_file = os.path.join(file_path, "mvp_mtx_{}.npy".format(img_id))
            mvp_mat = np.load(mvp_mat_file)
            self.all_mvp_mats.append(mvp_mat)
            
            mv_file = os.path.join(file_path, "mv_{}.npy".format(img_id))
            mv = np.load(mv_file)
            self.all_mv.append(mv)
            
        self.n_images = len(self.all_tgt_imgs)

        # Determine resolution & aspect ratio
        self.resolution = self.all_tgt_imgs[0].shape[:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        if self.FLAGS.local_rank == 0:
            print("Dataset3dGen: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        iter_res = self.FLAGS.train_res
        
        img = torch.tensor(self.all_tgt_imgs[itr], dtype=torch.float32)[None, ...]
        mv = torch.tensor(self.all_mv[itr], dtype=torch.float32)
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp = torch.tensor(self.all_mvp_mats[itr], dtype=torch.float32)
        
        mv = mv[None, ...]
        campos = campos[None, ...]
        mvp = mvp[None, ...]
        
        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : self.FLAGS.spp,
            'img' : img
        }
