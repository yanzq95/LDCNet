#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    datasets.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 8:08 PM

import os
import numpy as np
import glob
import torch
import random
from PIL import Image
import cv2
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as ttf


__all__ = [
    'read_list',
    'kitti',
]


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class kitti(data.Dataset):
    """
    kitti depth completion dataset: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
    """

    def __init__(self, path='/opt/data/private/yanzq/carla_epe_resize', mode='train',
                 height=320, width=640, return_idx=False, return_size=False, transform=None):

        self.mode = mode
        gt_txt = os.getcwd() + '/carla_datalist/carla_' + mode + '.txt'

        rgb_paths = []
        with open(gt_txt, "r") as fo:
            lines = fo.readlines()
            for line in lines:
                line = path + '/rgb/' + line
                line = line.strip()   # remove space in str
                rgb_paths.append(line)
        fo.close()

        self.rgb_paths = rgb_paths
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index):
        # rgb, sparse, target

        rgb_path = self.rgb_paths[index]
        sd_path = rgb_path.replace('/rgb/', '/sparse/').replace('_rgb.png', '_depth.png')
        gt_path = sd_path.replace('sparse', 'semi')

        # read rgb & depth
        assert os.path.exists(rgb_path), '{}'.format(rgb_path)

        rgb = cv2.imread(rgb_path)

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        sd = cv2.imread(sd_path, cv2.IMREAD_ANYDEPTH)
        gt = cv2.imread(gt_path, cv2.IMREAD_ANYDEPTH)

        if self.mode == "train":
            if random.random() > 0.5:
                rgb = cv2.flip(rgb, 1)
                gray = cv2.flip(gray, 1)
                sd = cv2.flip(sd, 1)
                gt = cv2.flip(gt, 1)

        rgb = transforms.ToTensor()(rgb).float()
        gray = transforms.ToTensor()(gray).float()
        sd = torch.from_numpy(np.expand_dims(sd, axis=0)).float()
        gt = torch.from_numpy(np.expand_dims(gt, axis=0)).float()

        name = gt_path.split('/')[-1].replace('_depth.png', '')

        output = [rgb, gray, sd, gt, name]

        return output
