import time
import pdb
import cv2
from copy import deepcopy
import os
import torch
import torch.nn as nn
import torch.nn.init
import numpy as np
from tqdm import tqdm

from script.dm.helpers import preprocess_data, get_error_in_q
from utils.direct_pose_model import fix_coord_supp
from models.nerfw import create_nerf, to8b, img2mse, mse2psnr
from models.ray_utils import get_rays
from models.rendering import render
from ViTLoc.FeatFormer.utils.utils import freeze_bn_layer_train
from feature.model import PoseNetV2 as FeatureNet
from torchvision.utils import save_image
from ViTLoc.FeatFormer.utils.utils import plot_features, save_image_saliancy

def tmp_plot2(target_in, rgb_in, features_target, features_rgb, i=0):
    '''
    print 1 pair of batch of salient feature map
    :param: target_in [B, 3, H, W]
    :param: rgb_in [B, 3, H, W]
    :param: features_target [B, C, H, W]
    :param: features_rgb [B, C, H, W]
    :param: frame index i of batch
    '''
    print("for debug only...")
    pdb.set_trace()
    save_image(target_in[i], './tmp/target_in.png')
    save_image(rgb_in[i], './tmp/rgb_in.png')
    features_t = features_target[i].clone()[:, None, :, :]
    features_r = features_rgb[i].clone()[:, None, :, :]
    save_image_saliancy(features_t, './tmp/target', True)
    save_image_saliancy(features_r, './tmp/rgb', True)

def preprocess_features_for_loss(feature):
    '''
    transform output features from the network to required shape for computing loss
    :param: feature [L, B, C, H, W] # L stands for level of features (we currently use 3)
    return feature' [B,L*C,H,W]
    '''
    feature = feature.permute(1,0,2,3,4)
    B, L, C, H, W = feature.size()
    feature = feature.reshape((B,L*C,H,W))
    return feature

def disable_model_grad(model):
    ''' set whole model to requires_grad=False, this is for nerf model '''
    for module in model.modules():
        if hasattr(module, 'weight'):
            module.weight.requires_grad_(False)
        if hasattr(module, 'bias'):
            module.bias.requires_grad_(False)
    return model

def inference_pose_regression(args, data, device, model, retFeature=False, isSingleStream=True, return_pose=True):
    """
    Inference the Pose Regression Network
    Inputs:
        args: parsed argument
        data: Input image in shape (batchsize, channels, H, W)
        device: gpu device
        model: PoseNet model
    Outputs:
        pose: Predicted Pose in shape (batchsize, 3, 4)
    """
    inputs = data.to(device)
    _,_,H,W = data.size()
    if args.preprocess_ImgNet:
        inputs = preprocess_data(inputs, device)
    if args.DFNet:
        features, predict_pose = model(inputs, return_feature=retFeature, isSingleStream=isSingleStream, return_pose=return_pose, upsampleH=H, upsampleW=W)
    else:
        features, predict_pose = model(inputs, isTrain=retFeature, isSingleStream=isSingleStream)
    
    if return_pose==False:
        return features, predict_pose

    pose = predict_pose.reshape(inputs.shape[0], 3, 4)

    if args.svd_reg:
        R_torch = pose[:,:3,:3].clone()
        u,s,v=torch.svd(R_torch)
        Rs = torch.matmul(u, v.transpose(-2,-1))
        pose[:,:3,:3] = Rs
    return features, pose

def rgb_loss(rgb, target, extras):
    ''' Compute RGB MSE Loss, original from NeRF Paper '''
    img_loss = img2mse(rgb, target)
    loss = img_loss
    return loss

def normalize_features(tensor, value_range=None, scale_each: bool = False):
    ''' Find unit norm of channel wise feature 
        :param: tensor, img tensor (C,H,W)
    '''
    tensor = tensor.clone()
    C,H,W = tensor.size()

    tensor = tensor.reshape(C, H*W)
    tensor = torch.nn.functional.normalize(tensor)
    return tensor

def feature_loss(feature_rgb, feature_target, img_in=True, per_channel=False):
    ''' Compute Feature MSE Loss 
    :param: feature_rgb, [C,H,W] or [C, N_rand]
    :param: feature_target, [C,H,W] or [C, N_rand]
    :param: img_in, True: input is feature maps, False: input is rays
    :param: random, True: randomly using per pixel or per channel cossimilarity loss
    '''
    if img_in:
        C,H,W = feature_rgb.size()
        fr = feature_rgb.reshape(C, H*W)
        ft = feature_target.reshape(C, H*W)
    else:
        fr = feature_rgb
        ft = feature_target

    if per_channel:
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    else:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = 1 - cos(fr, ft).mean()

    return loss

def PoseLoss(args, pose_, pose, device):
    loss_func = nn.MSELoss()
    predict_pose = pose_.reshape(args.batch_size, 12).to(device)
    pose_loss = loss_func(predict_pose, pose)
    return pose_loss

def prepare_batch_render(args, pose, batch_size, target_, H, W, focal, half_res=True, rand=True):
    ''' Break batch of images into rays '''
    target_ = target_.permute(0, 2, 3, 1).numpy()
    if half_res:
        N_rand = batch_size * (H//2) * (W//2)
        target_half = np.stack([cv2.resize(target_[i], (H//2, W//2), interpolation=cv2.INTER_AREA) for i in range(batch_size)], 0)
        target_half = torch.Tensor(target_half)
        
        rays = torch.stack([torch.stack(get_rays(H//2, W//2, focal/2, pose[i]), 0) for i in range(batch_size)], 0)
        rays_rgb = torch.cat((rays, target_half[:, None, ...]), 1)

    else:
        N_rand = args.N_rand
        target_ = torch.Tensor(target_)
        rays = torch.stack([torch.stack(get_rays(H, W, focal, pose[i]), 0) for i in range(batch_size)], 0)
        rays_rgb = torch.cat([rays, target_[:, None, ...]], 1)
    rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)
    
    rays_rgb = torch.reshape(rays_rgb, (-1, 3, 3))

    if 1:
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]

    batch = rays_rgb[:N_rand].permute(1, 0 , 2)
    batch_rays, target_s = batch[:2], batch[2]

    return batch_rays, target_s

def eval_on_batch(args, data, model, feat_model, pose, img_idx, hwf, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 step of eval'''
    with torch.no_grad():
        H, W, focal = hwf
        _, pose_ = inference_pose_regression(args, data, device, model)
        device_cpu = torch.device('cpu')
        pose_ = pose_.to(device_cpu)
        pose_nerf = pose_.clone()

        if args.NeRFH:

            pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device_cpu)

        half_res=False
        batch_rays, target = prepare_batch_render(args, pose_nerf, args.batch_size, data, H, W, focal, half_res)
        batch_rays = batch_rays.to(device)
        target = target.to(device)
        pose = pose.to(device)
        img_idx = img_idx.to(device)
        pose_nerf = pose_nerf.to(device)

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, img_idx=img_idx, **render_kwargs_test)

        loss = PoseLoss(args, pose_, pose, device)
        psnr = mse2psnr(img2mse(rgb, target))

        torch.set_default_tensor_type('torch.FloatTensor')

        iter_loss = loss.to(device_cpu).detach().numpy()
        iter_loss = np.array([iter_loss])

        iter_psnr = psnr.to(device_cpu).detach().numpy()
    return iter_loss, iter_psnr

def eval_on_epoch(args, data_loaders, model, feat_model, hwf, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 epoch of training with batch '''
    model.eval()
    batch_size = 1
    
    train_dl, val_dl, test_dl = data_loaders

    total_loss = []
    total_psnr = []
    
    for data, pose, img_idx in val_dl:
        loss, psnr = eval_on_batch(args, data, model, feat_model, pose, img_idx, hwf, half_res, device, world_setup_dict, **render_kwargs_test)
        total_loss.append(loss.item())
        total_psnr.append(psnr.item())
    total_loss_mean = np.mean(total_loss)
    total_psnr_mean = np.mean(total_psnr)
    return total_loss_mean, total_psnr_mean

def train_on_feature_batch(args, data, model, feat_model, pose, img_idx, hwf, optimizer, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 step of training using scheme1 '''
    batch_size_iter = data.shape[0]

    H, W, focal = hwf
    data = data.to(device)
    
    _, pose_ = inference_pose_regression(args, data, device, model, retFeature=False)
    pose_nerf = pose_.clone()

    pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device)

    pose = pose.to(device)
    img_idx = img_idx.to(device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    target = data.permute(0,2,3,1)
    rays_o_list=[]
    rays_d_list=[]
    img_idx_list=[]
    N_rand = args.N_rand
    for i in range(pose_nerf.shape[0]):
        rays_o, rays_d = get_rays(H, W, focal, pose_nerf[i])
        rays_o_list.append(rays_o)
        rays_d_list.append(rays_d)
        img_idx_list.append(img_idx[i].repeat(N_rand,1))
    rays_o_batch = torch.stack(rays_o_list)
    rays_d_batch = torch.stack(rays_d_list)
    img_idx_batch = torch.cat(img_idx_list)

    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)
    coords = torch.reshape(coords, [-1,2])
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
    select_coords = coords[select_inds].long()  

    rays_o = rays_o_batch[:, select_coords[:, 0], select_coords[:, 1]]
    rays_d = rays_d_batch[:, select_coords[:, 0], select_coords[:, 1]]
    rays_o = rays_o.reshape(rays_o.shape[0]*rays_o.shape[1], 3) 
    rays_d = rays_d.reshape(rays_d.shape[0]*rays_d.shape[1], 3) 
    batch_rays = torch.stack([rays_o, rays_d], 0)
    target_s = target[:,select_coords[:, 0], select_coords[:, 1]].reshape(batch_size_iter*N_rand,3)  

    rgb_feature, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, img_idx=img_idx_batch, **render_kwargs_test)
    rgb = rgb_feature[...,:3]
    feature = rgb_feature[...,3:].reshape(batch_size_iter, N_rand, args.out_channel_size-3)[None, ...].permute(0,1,3,2)

    target_in = target.permute(0,3,1,2)
    features, _ = feat_model(target_in, True, True, H, W)

    feature_target = features[0][0]
    feature_target = feature_target[None, 0:, :, select_coords[:, 0], select_coords[:, 1]]

    photo_loss = rgb_loss(rgb, target_s, extras)

    feat_loss = feature_loss(feature[0,0], feature_target[0,0], img_in=False, per_channel=args.per_channel)

    if args.combine_loss:
        pose_loss = PoseLoss(args, pose_, pose, device)
        loss = args.combine_loss_w[0] * pose_loss + args.combine_loss_w[1] * photo_loss + args.combine_loss_w[2] * feat_loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    psnr = mse2psnr(img2mse(rgb, target_s))

    torch.set_default_tensor_type('torch.FloatTensor')
    device_cpu = torch.device('cpu')
    iter_loss = loss.to(device_cpu).detach().numpy()
    iter_loss = np.array([iter_loss])

    iter_psnr = psnr.to(device_cpu).detach().numpy()
    return iter_loss, iter_psnr

def train_on_batch(args, data, model, feat_model, pose, img_idx, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 step of training '''

    H, W, focal = hwf
    data = data.to(device)

    _, pose_ = inference_pose_regression(args, data, device, model, retFeature=False)
    pose_nerf = pose_.clone()

    pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device)

    pose = pose.to(device)
    img_idx = img_idx.to(device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if half_res:
        rgb, disp, acc, extras = render(H//4, W//4, focal/4, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb = rgb[None,...].permute(0,3,1,2)
        rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
    else:
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb = rgb[None,...].permute(0,3,1,2)

    feature_list, _ = inference_pose_regression(args, torch.cat([data, rgb]), device, feat_model, retFeature=True, isSingleStream=False, return_pose=False)
    feature_target = feature_list[0]
    feature_rgb = feature_list[1]

    photo_loss = rgb_loss(rgb, data, extras)

    indices = torch.tensor(args.feature_matching_lvl)
    feature_rgb = torch.index_select(feature_rgb, 0, indices)
    feature_target = torch.index_select(feature_target, 0, indices)

    feature_rgb = preprocess_features_for_loss(feature_rgb)
    feature_target = preprocess_features_for_loss(feature_target)

    feat_loss = feature_loss(feature_rgb[0], feature_target[0], per_channel=args.per_channel)

    if args.combine_loss:
        pose_loss = PoseLoss(args, pose_, pose, device)
        loss = args.combine_loss_w[0] * pose_loss + args.combine_loss_w[1] * photo_loss + args.combine_loss_w[2] * feat_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    psnr = mse2psnr(img2mse(rgb, data))

    torch.set_default_tensor_type('torch.FloatTensor')
    device_cpu = torch.device('cpu')
    iter_loss = loss.to(device_cpu).detach().numpy()
    iter_loss = np.array([iter_loss])

    iter_psnr = psnr.to(device_cpu).detach().numpy()
    return iter_loss, iter_psnr

def train_on_epoch(args, data_loaders, model, feat_model, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 epoch of training with batch '''
    model.train()
    model = freeze_bn_layer_train(model)

    train_dl, val_dl, test_dl = data_loaders
    total_loss = []
    total_psnr = []
    
    for data, pose, img_idx in train_dl:
        loss, psnr = train_on_batch(args, data, model, feat_model, pose, img_idx, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test)

        total_loss.append(loss.item())
        total_psnr.append(psnr.item())
    total_loss_mean = np.mean(total_loss)
    total_psnr_mean = np.mean(total_psnr)
    return total_loss_mean, total_psnr_mean

def train_feature_matching(args, model, feat_model, optimizer, i_split, hwf, near, far, device, early_stopping, images=None, poses_train=None, train_dl=None, val_dl=None, test_dl=None):
    ''' finetune pretrained PoseNet using NeRF '''
    half_res = True

    _, render_kwargs_test, start, grad_vars, _ = create_nerf(args)
    global_step = start
    if args.reduce_embedding==2:
        render_kwargs_test['i_epoch'] = global_step

    data_loaders = [train_dl, val_dl, test_dl]
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_test.update(bds_dict)
    i_train, i_val, i_test = i_split

    render_kwargs_test['embedding_a'] = disable_model_grad(render_kwargs_test['embedding_a'])
    render_kwargs_test['embedding_t'] = disable_model_grad(render_kwargs_test['embedding_t'])
    render_kwargs_test['network_fn'] = disable_model_grad(render_kwargs_test['network_fn'])
    render_kwargs_test['network_fine'] = disable_model_grad(render_kwargs_test['network_fine'])

    N_epoch = 2001
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    world_setup_dict = {
        'pose_scale' : train_dl.dataset.pose_scale,
        'pose_scale2' : train_dl.dataset.pose_scale2,
        'move_all_cam_vec' : train_dl.dataset.move_all_cam_vec,
    }

    time0 = time.time()

    model_log = tqdm(total=0, position = 1, bar_format='{desc}')
    for epoch in tqdm(range(N_epoch), desc='epochs'):
        loss, psnr = train_on_epoch(args, data_loaders, model, feat_model, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test)

        val_loss, val_psnr = eval_on_epoch(args, data_loaders, model, feat_model, hwf, half_res, device, world_setup_dict, **render_kwargs_test)


        tqdm.write('At epoch {0:4d} : train loss: {1:.4f}, train psnr: {2:.4f}, val loss: {3:.4f}, val psnr: {4:.4f}'.format(epoch, loss, psnr, val_loss, val_psnr))

        early_stopping(val_loss, model, epoch=epoch, save_multiple=(not args.no_save_multiple), save_all=args.save_all_ckpt, val_psnr=val_psnr)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        model_log.set_description_str(f'Best val loss: {early_stopping.val_loss_min:.4f}')

        if epoch % args.i_eval == 0:
            get_error_in_q(args, val_dl, model, len(val_dl.dataset), device, batch_size=1)
