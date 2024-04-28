import os, sys
import numpy as np
import imageio
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from utils.helpers import *
from utils.direct_pose_model import *
from utils.callbacks import EarlyStopping
from utils.options import config_parser
from models.rendering import render_path
from models.nerfw import to8b
from datasets.load_7Scenes import load_7Scenes_dataloader
from datasets.load_Cambridge import load_Cambridge_dataloader
from .utils.utils import freeze_bn_layer
from feature.direct_feature_matching import train_feature_matching

parser = config_parser()
args = parser.parse_args()
device = torch.device('cuda:0')

def render_test(args, train_dl, val_dl, hwf, start, model, device, render_kwargs_test):
    model.eval()

    if args.render_video_train:
        images_train = []
        poses_train = []
        for img, pose in train_dl:
            predict_pose = inference_pose_regression(args, img, device, model)
            device_cpu = torch.device('cpu')
            predict_pose = predict_pose.to(device_cpu)

            img_val = img.permute(0,2,3,1)
            pose_val = torch.zeros(1,4,4)
            pose_val[0,:3,:4] = predict_pose.reshape(3,4)[:3,:4]
            pose_val[0,3,3] = 1.
            images_train.append(img_val)
            poses_train.append(pose_val)

        images_train = torch.cat(images_train, dim=0).numpy()
        poses_train = torch.cat(poses_train, dim=0)
        print('train poses shape', poses_train.shape)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        with torch.no_grad():
            rgbs, disps = render_path(poses_train.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=None)
        torch.set_default_tensor_type('torch.FloatTensor')
        print('Saving trainset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.model_name, '{}_trainset_{:06d}_'.format(args.model_name, start))
        imageio.mimwrite(moviebase + 'train_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'train_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)

    if args.render_video_test:
        images_val = []
        poses_val = []
        for img, pose in val_dl: 
            predict_pose = inference_pose_regression(args, img, device, model)
            device_cpu = torch.device('cpu')
            predict_pose = predict_pose.to(device_cpu)

            img_val = img.permute(0,2,3,1)
            pose_val = torch.zeros(1,4,4)
            pose_val[0,:3,:4] = predict_pose.reshape(3,4)[:3,:4]
            pose_val[0,3,3] = 1.
            images_val.append(img_val)
            poses_val.append(pose_val)

        images_val = torch.cat(images_val, dim=0).numpy()
        poses_val = torch.cat(poses_val, dim=0)
        print('test poses shape', poses_val.shape)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        with torch.no_grad():
            rgbs, disps = render_path(poses_val.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=None)
        torch.set_default_tensor_type('torch.FloatTensor')
        print('Saving testset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.model_name, '{}_test_{:06d}_'.format(args.model_name, start))
        imageio.mimwrite(moviebase + 'test_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'test_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)
    return

def train():
    print(parser.format_values())
    if args.dataset_type == '7Scenes':
        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader(args)
    elif args.dataset_type == 'Cambridge':
        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_Cambridge_dataloader(args)
    else:
        print("please choose dataset_type: 7Scenes or Cambridge, exiting...")
        sys.exit()

    assert(args.pretrain_model_path != '')
    model = load_exisiting_model(args)

    if args.freezeBN:
        model = freeze_bn_layer(model)
    model.to(device)

    if args.pretrain_featurenet_path == '':
        print('Use the same FeatFormer for Feature Extraction and Pose Regression')
        feat_model = load_exisiting_model(args)
    else: 
        feat_model = load_exisiting_model(args, isFeatureNet=True)

    feat_model.eval()
    feat_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    early_stopping = EarlyStopping(args, patience=args.patience[0], verbose=False)

    if args.dataset_type == '7Scenes':
        train_feature_matching(args, model, feat_model, optimizer, i_split, hwf, near, far, device, early_stopping, train_dl=train_dl, val_dl=val_dl, test_dl=test_dl)
    elif args.dataset_type == 'Cambridge':
        train_feature_matching(args, model, feat_model, optimizer, i_split, hwf, near, far, device, early_stopping, train_dl=train_dl, val_dl=val_dl, test_dl=test_dl)

def eval():
    print(parser.format_values())
    if args.dataset_type == '7Scenes':
        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader(args)
    elif args.dataset_type == 'Cambridge':
        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_Cambridge_dataloader(args)
    else:
        print("please choose dataset_type: 7Scenes or Cambridge, exiting...")
        sys.exit()
    
    model = load_exisiting_model(args)
    if args.freezeBN:
        model = freeze_bn_layer(model)
    model.to(device)

    print(len(test_dl.dataset))

    get_error_in_q(args, test_dl, model, len(test_dl.dataset), device, batch_size=1)

if __name__ == '__main__':
    if args.eval:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        eval()
    else:
        train()
