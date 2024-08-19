import os

import numpy as np
import matplotlib.pyplot as plt

import cv2
# import open3d as o3d


def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass


class Saver(object):

    def __init__(self, save_dir):
        self.idx = 0
        self.save_dir = os.path.join(save_dir, "vis")
        if not os.path.exists(self.save_dir):
            mkdirs(self.save_dir)

    # def save_as_point_cloud(self, depth, rgb, path, mask=None):
    #     ######################################
    #     my_mask = np.ones_like(depth)
    #     my_mask[:115, :] = 0
    #     my_mask = my_mask.reshape(-1)
    #     ######################################
    #     h, w = depth.shape
    #     Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    #     Theta = np.repeat(Theta, w, axis=1)
    #     Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
    #     Phi = -np.repeat(Phi, h, axis=0)
    #
    #     X = depth * np.sin(Theta) * np.sin(Phi)
    #     Y = depth * np.cos(Theta)
    #     Z = depth * np.sin(Theta) * np.cos(Phi)
    #
    #     ######################################
    #     X = X.flatten()
    #     Y = Y.flatten()
    #     Z = Z.flatten()
    #     R = rgb[:, :, 0].flatten()
    #     G = rgb[:, :, 1].flatten()
    #     B = rgb[:, :, 2].flatten()
    #     ######################################
    #
    #     # if mask is None:
    #     #     X = X.flatten()
    #     #     Y = Y.flatten()
    #     #     Z = Z.flatten()
    #     #     R = rgb[:, :, 0].flatten()
    #     #     G = rgb[:, :, 1].flatten()
    #     #     B = rgb[:, :, 2].flatten()
    #     # else:
    #     #     X = X[mask]
    #     #     Y = Y[mask]
    #     #     Z = Z[mask]
    #     #     R = rgb[:, :, 0][mask]
    #     #     G = rgb[:, :, 1][mask]
    #     #     B = rgb[:, :, 2][mask]
    #
    #     XYZ = np.stack([X, Y, Z], axis=1)
    #     ############################
    #     XYZ = XYZ[my_mask>0]
    #     ############################
    #     RGB = np.stack([R, G, B], axis=1)
    #     ######################################
    #     RGB = RGB[my_mask>0]
    #     ######################################
    #
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(XYZ)
    #     pcd.colors = o3d.utility.Vector3dVector(RGB)
    #     o3d.io.write_point_cloud(path, pcd)

    def save_samples(self, rgbs, gt_depths, sparse_depths, pred_depths, depth_masks=None):
        """
        Saves samples
        """
        rgbs = rgbs.cpu().numpy().transpose(0, 2, 3, 1)
        depth_preds = pred_depths.cpu().numpy()
        gt_depths = gt_depths.cpu().numpy()
        sparse_depths = sparse_depths.cpu().numpy()
        if depth_masks is None:
            depth_masks = gt_depths != 0
        else:
            depth_masks = depth_masks.cpu().numpy()

        for i in range(rgbs.shape[0]):
            self.idx = self.idx+1
            mkdirs(os.path.join(self.save_dir, '%04d'%(self.idx)))

            # cmap = plt.get_cmap("rainbow_r")
            cmap = plt.get_cmap("plasma_r")

            depth_pred = cmap(depth_preds[i][0].astype(np.float32))
            # depth_pred = depth_preds[i][0].astype(np.float32)
            depth_pred = np.delete(depth_pred, 3, 2)
            path = os.path.join(self.save_dir, '%04d' % (self.idx) ,'_depth_pred.png')
            cv2.imwrite(path, (depth_pred * 1000).astype(np.uint16))
            # path = os.path.join(self.save_dir, '%04d' % (self.idx) ,'_depth_pred.npy')
            # np.save(path, depth_pred)

            depth_gt = cmap(gt_depths[i][0].astype(np.float32))
            depth_gt = np.delete(depth_gt, 3, 2)
            depth_gt[..., 0][~depth_masks[i][0]] = 0
            depth_gt[..., 1][~depth_masks[i][0]] = 0
            depth_gt[..., 2][~depth_masks[i][0]] = 0
            path = os.path.join(self.save_dir, '%04d' % (self.idx), '_depth_gt.png')
            cv2.imwrite(path, (depth_gt * 1000).astype(np.uint16))

            sparse_depths = cmap(sparse_depths[i][0].astype(np.float32))
            sparse_depths = np.delete(sparse_depths, 3, 2)
            path = os.path.join(self.save_dir, '%04d' % (self.idx), '_sparse_depth.png')
            cv2.imwrite(path, (sparse_depths * 1000).astype(np.uint16))

            # rgb = (rgbs[i] * 255).astype(np.uint8)
            # path = os.path.join(self.save_dir, '%04d'%(self.idx) , '_rgb.jpg')
            # cv2.imwrite(path, rgb[:,:,::-1])
    def new_save_samples(self, rgbs, gt_depths, sparse_depths, pred_depths, depth_masks=None):
        """
        Saves samples
        """
        rgbs = rgbs.data.cpu().numpy().transpose(0, 2, 3, 1)
        depth_preds = pred_depths.data.cpu().numpy()
        gt_depths = gt_depths.data.cpu().numpy()
        sparse_depths = sparse_depths.data.cpu().numpy()
        if depth_masks is None:
            depth_masks = gt_depths != 0
        else:
            depth_masks = depth_masks.cpu().numpy()

        print(rgbs.shape[0])
        for i in range(rgbs.shape[0]):
            self.idx = self.idx+1
            mkdirs(os.path.join(self.save_dir, '%04d'%(self.idx)))

            # cmap = plt.get_cmap("rainbow_r")
            cmap = plt.get_cmap("plasma_r")

            depth_pred = cmap(depth_preds[i][0])
            depth_pred = np.delete(depth_pred, 3, 2)
            path = os.path.join(self.save_dir, '%04d' % (self.idx) ,'_depth_pred.png')
            cv2.imwrite(path, (depth_pred * 1000).astype(np.uint16))

            depth_gt = cmap(gt_depths[i][0])
            depth_gt = np.delete(depth_gt, 3, 2)
            depth_gt[..., 0][~depth_masks[i][0]] = 0
            depth_gt[..., 1][~depth_masks[i][0]] = 0
            depth_gt[..., 2][~depth_masks[i][0]] = 0
            path = os.path.join(self.save_dir, '%04d' % (self.idx), '_depth_gt.png')
            cv2.imwrite(path, (depth_gt * 1000).astype(np.uint16))

            sparse_depths = cmap(sparse_depths[i][0])
            sparse_depths = np.delete(sparse_depths, 3, 2)
            path = os.path.join(self.save_dir, '%04d' % (self.idx), '_sparse_depth.png')
            cv2.imwrite(path, (sparse_depths * 1000).astype(np.uint16))
