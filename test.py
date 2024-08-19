#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/16 4:47 PM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import yaml
from easydict import EasyDict as edict
import datasets
import encoding
from saver import Saver
import matplotlib.pyplot as plt



def compute_depth_metrics(gt, pred):
    """Computation of metrics between predicted and ground truth depths
    """

    mask = gt > 0

    pred[pred < 0] = 0

    pred = pred[mask]
    gt = gt[mask]

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    mre = torch.mean(torch.abs(gt - pred) / gt)
    mae = torch.mean(torch.abs(gt - pred))

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt) - torch.log10(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    # i_pred = 1 / pred
    # i_gt = 1 / gt
    # irmse = torch.sqrt(((i_gt - i_pred) ** 2).mean())
    # imae = torch.mean(torch.abs(i_gt - i_pred))
    new_pred = pred
    new_pred[new_pred < 1] = 1
    new_gt = gt
    new_gt[new_gt < 1] = 1
    i_pred = 1 / new_pred
    i_gt = 1 / new_gt
    irmse = torch.sqrt(((i_gt - i_pred) ** 2).mean())
    imae = torch.mean(torch.abs(i_gt - i_pred))

    return mre, mae, rmse, rmse_log, a1, a2, a3, irmse, imae

def test():
    net.eval()
    mre_init, mae_init, rmse_init, rmse_log_init, a1_init, a2_init, a3_init, irmse_init, imae_init = 0, 0, 0, 0, 0, 0, 0, 0, 0
    num_init = 0
    for batch_idx, (rgb, gray, lidar, gt, name) in enumerate(testloader):
        with torch.no_grad():
            if config.tta:
                rgbf = torch.flip(rgb, [-1])
                grayf = torch.flip(gray, [-1])
                lidarf = torch.flip(lidar, [-1])
                rgbs = torch.cat([rgb, rgbf], 0)
                grays = torch.cat([gray, grayf], 0)
                lidars = torch.cat([lidar, lidarf], 0)
                rgbs, grays, lidars = rgbs.cuda(), grays.cuda(), lidars.cuda()

                output_dict = net(rgbs, grays, lidars)
                depth_preds = output_dict['pred']

                depth_pred, depth_predf = depth_preds.split(depth_preds.shape[0] // 2)
                depth_predf = torch.flip(depth_predf, [-1])
                depth_pred = (depth_pred + depth_predf) / 2.
            else:
                rgb, gray, lidar = rgb.cuda(), gray.cuda(), lidar.cuda()
                output_dict = net(rgb, gray, lidar)
                depth_pred = output_dict['pred']

        ###############################################################
        # image_folder = os.path.join(os.getcwd(), 'vis')
        # if not os.path.exists(image_folder):
        #     os.makedirs(image_folder)
        # filename = os.path.join(image_folder, '{0:03d}.png'.format(batch_idx))
        #
        # middle = depth_pred
        # img = torch.squeeze(middle.data.cpu()).numpy()
        # # img = (img * 1000).astype('uint16')
        # img = img.astype('uint8')
        # plt.imsave(filename, img, cmap='plasma')

        # filename_rgb = filename.replace('.png', '_rgb_raw.png')
        # rgb = rgb.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
        # plt.imsave(filename_rgb, rgb)
        #
        # filename_rgb_enhance = filename.replace('.png', '_rgb_enhance.png')
        # img_enhance = output_dict['img_enhance']
        # img_enhance = img_enhance.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
        # plt.imsave(filename_rgb_enhance, img_enhance)
        #
        # filename_illu = filename.replace('.png', '_illu.png')
        # illu = output_dict['illu']
        # illu = illu.cpu().numpy().squeeze()
        # plt.imsave(filename_illu, illu)
        ################################################################

        depth_pred = depth_pred.cpu().squeeze(-1)
        gt = gt.cpu().squeeze(-1)

        mre, mae, rmse, rmse_log, a1, a2, a3, irmse, imae = compute_depth_metrics(gt, depth_pred)
        mre_init += mre
        mae_init += mae
        rmse_init += rmse
        rmse_log_init += rmse_log
        a1_init += a1
        a2_init += a2
        a3_init += a3
        irmse_init += irmse
        imae_init += imae
        num_init += 1
        print("testing {} images...".format(batch_idx))

    mre_avg = mre_init / num_init
    mae_avg = mae_init / num_init
    rmse_avg = rmse_init / num_init
    rmse_log_avg = rmse_log_init / num_init
    a1_avg = a1_init / num_init
    a2_avg = a2_init / num_init
    a3_avg = a3_init / num_init
    irmse_avg = irmse_init / num_init
    imae_avg = imae_init / num_init

    file = os.path.join(os.getcwd(), "seven_metrics_for_test.txt")
    with open(file, 'a') as f:
        print(
            "\n " + ("{:>12} | " * 9).format("mre", "mae", "rmse", "rmse_log", "acc1", "acc2", "acc3", "irmse", "imae"),
            file=f)
        print(("&  {: 8.5f} " * 9).format(mre_avg, mae_avg, rmse_avg, rmse_log_avg, a1_avg, a2_avg, a3_avg, irmse_avg,
                                          imae_avg), file=f)

    print(mre_avg, mae_avg, rmse_avg, rmse_log_avg, a1_avg, a2_avg, a3_avg, irmse_avg, imae_avg)


if __name__ == '__main__':
    config_name = 'GNS_test.yaml'
    with open(os.path.join('configs', config_name), 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    config = edict(config_data)
    from utils import *

    transform = init_aug(config.test_aug_configs)
    key, params = config.data_config.popitem()
    dataset = getattr(datasets, key)
    testset = dataset(**params, mode='test', transform=transform, return_idx=True, return_size=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, num_workers=config.num_workers,
                                             shuffle=False, pin_memory=True)
    print('num_test = {}'.format(len(testset)))
    net = init_net(config)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    net.cuda()
    net = encoding.parallel.DataParallelModel(net)
    net = resume_state(config, net)
    my_saver = Saver(os.getcwd())
    test()
