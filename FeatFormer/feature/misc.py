import torch
from torch import nn
import pdb
# from torchvision import transforms, datasets
import pytorch3d.transforms as transforms
from torchvision.transforms import Resize
import numpy as np
import math
from ViTLoc.FeatFormer.utils.utils import plot_features, save_image_saliancy
from utils.direct_pose_model import fix_coord_supp
from script.dm.helpers import vis_pose
from models.rendering import render, render_path
import time
import os
import imageio
from copy import deepcopy

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype=np.float)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], dtype=np.float)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], dtype=np.float)

rot_psi = lambda psi : np.array([
    [np.cos(psi),-np.sin(psi),0,0],
    [np.sin(psi),np.cos(psi),0,0],
    [0,0,1,0],
    [0,0,0,1]], dtype=np.float)

def compute_error_in_q(args, dl, model, device, results, batch_size=1):
    use_SVD=True
    time_spent = []
    predict_pose_list = []
    gt_pose_list = []
    ang_error_list = []
    i = 0
    for batch in dl:
        if args.NeRFH:
            data, pose, img_idx = batch
        else:
            data, pose = batch
        data = data.to(device) # input
        pose = pose.reshape((batch_size,3,4)).numpy() # label

        if use_SVD:
            with torch.no_grad():
                _, predict_pose = model(data)
                R_torch = predict_pose.reshape((batch_size, 3, 4))[:,:3,:3] # debug
                predict_pose = predict_pose.reshape((batch_size, 3, 4)).cpu().numpy()

                R = predict_pose[:,:3,:3]
                res = R@np.linalg.inv(R)

                u,s,v=torch.svd(R_torch)
                Rs = torch.matmul(u, v.transpose(-2,-1))
            predict_pose[:,:3,:3] = Rs[:,:3,:3].cpu().numpy()
        else:
            start_time = time.time()
            # inference NN
            with torch.no_grad():
                predict_pose = model(data)
                predict_pose = predict_pose.reshape((batch_size, 3, 4)).cpu().numpy()
            time_spent.append(time.time() - start_time)

        pose_q = transforms.matrix_to_quaternion(torch.Tensor(pose[:,:3,:3]))
        pose_x = pose[:, :3, 3]
        predicted_q = transforms.matrix_to_quaternion(torch.Tensor(predict_pose[:,:3,:3]))
        predicted_x = predict_pose[:, :3, 3]
        pose_q = pose_q.squeeze() 
        pose_x = pose_x.squeeze() 
        predicted_q = predicted_q.squeeze() 
        predicted_x = predicted_x.squeeze()
        
        q1 = pose_q / torch.linalg.norm(pose_q)
        q2 = predicted_q / torch.linalg.norm(predicted_q)
        d = torch.abs(torch.sum(torch.matmul(q1,q2))) 
        d = torch.clamp(d, -1., 1.)
        theta = (2 * torch.acos(d) * 180/math.pi).numpy()
        error_x = torch.linalg.norm(torch.Tensor(pose_x-predicted_x)).numpy()
        results[i,:] = [error_x, theta]

        predict_pose_list.append(predicted_x)
        gt_pose_list.append(pose_x)
        ang_error_list.append(theta)
        i += 1
    predict_pose_list = np.array(predict_pose_list)
    gt_pose_list = np.array(gt_pose_list)
    ang_error_list = np.array(ang_error_list)
    vis_info_ret = {"pose": predict_pose_list, "pose_gt": gt_pose_list, "theta": ang_error_list}
    return results, vis_info_ret

def get_error_in_q(args, dl, model, sample_size, device, batch_size=1):
    ''' Convert Rotation matrix to quaternion, then calculate the location errors. original from PoseNet Paper '''
    model.eval()
    
    results = np.zeros((sample_size, 2))
    results, vis_info = compute_error_in_q(args, dl, model, device, results, batch_size)
    median_result = np.median(results,axis=0)
    mean_result = np.mean(results,axis=0)

    print ('Median error {}m and {} degrees.'.format(median_result[0], median_result[1]))
    print ('Mean error {}m and {} degrees.'.format(mean_result[0], mean_result[1]))


def get_render_error_in_q(args, model, sample_size, device, targets, rgbs, poses, batch_size=1):
    ''' use nerf render imgs instead of use real imgs '''
    model.eval()
    
    results = np.zeros((sample_size, 2))
    print("to be implement...")

    predict_pose_list = []
    gt_pose_list = []
    ang_error_list = []

    for i in range(sample_size):
        data = rgbs[i:i+1].permute(0,3,1,2)
        pose = poses[i:i+1].reshape(batch_size, 12)

        data = data.to(device) # input
        pose = pose.reshape((batch_size,3,4)).numpy() # label

        with torch.no_grad():
            _, predict_pose = model(data)
            R_torch = predict_pose.reshape((batch_size, 3, 4))[:,:3,:3] # debug
            predict_pose = predict_pose.reshape((batch_size, 3, 4)).cpu().numpy()

            R = predict_pose[:,:3,:3]
            res = R@np.linalg.inv(R)

            u,s,v=torch.svd(R_torch)
            Rs = torch.matmul(u, v.transpose(-2,-1))
        predict_pose[:,:3,:3] = Rs[:,:3,:3].cpu().numpy()

        pose_q = transforms.matrix_to_quaternion(torch.Tensor(pose[:,:3,:3]))
        pose_x = pose[:, :3, 3] 
        predicted_q = transforms.matrix_to_quaternion(torch.Tensor(predict_pose[:,:3,:3]))
        predicted_x = predict_pose[:, :3, 3]
        pose_q = pose_q.squeeze() 
        pose_x = pose_x.squeeze() 
        predicted_q = predicted_q.squeeze() 
        predicted_x = predicted_x.squeeze()
        
        q1 = pose_q / torch.linalg.norm(pose_q)
        q2 = predicted_q / torch.linalg.norm(predicted_q)
        d = torch.abs(torch.sum(torch.matmul(q1,q2))) 
        d = torch.clamp(d, -1., 1.) # acos can only input [-1~1]
        theta = (2 * torch.acos(d) * 180/math.pi).numpy()
        error_x = torch.linalg.norm(torch.Tensor(pose_x-predicted_x)).numpy()
        results[i,:] = [error_x, theta]

        predict_pose_list.append(predicted_x)
        gt_pose_list.append(pose_x)
        ang_error_list.append(theta)

    predict_pose_list = np.array(predict_pose_list)
    gt_pose_list = np.array(gt_pose_list)
    ang_error_list = np.array(ang_error_list)
    vis_info_ret = {"pose": predict_pose_list, "pose_gt": gt_pose_list, "theta": ang_error_list}

    median_result = np.median(results,axis=0)
    mean_result = np.mean(results,axis=0)

    # standard log
    print ('Median error {}m and {} degrees.'.format(median_result[0], median_result[1]))
    print ('Mean error {}m and {} degrees.'.format(mean_result[0], mean_result[1]))

    return

def render_nerfw_imgs(args, dl, hwf, device, render_kwargs_test, world_setup_dict):
    ''' render nerfw imgs, save unscaled pose and results'''
    H, W, focal = hwf
    target_list = []
    rgb_list = []
    pose_list = []
    img_idx_list = []
    
    for batch_idx, (target, pose, img_idx) in enumerate(dl):
        if batch_idx % 10 == 0:
            print("renders {}/total {}".format(batch_idx, len(dl.dataset)))

        target = target[0].permute(1,2,0).to(device)
        pose = pose.reshape(3,4)

        img_idx = img_idx.to(device)
        pose_nerf = pose.clone()

        pose_nerf = fix_coord_supp(args, pose_nerf[None,...], world_setup_dict)

        with torch.no_grad():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if args.tinyimg:
                rgb, _, _, _ = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=True, img_idx=img_idx, **render_kwargs_test)
                rgb = rgb[None,...].permute(0,3,1,2)
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                rgb = Resize((224,224))(rgb)
                rgb = rgb[0].permute(1,2,0)

            else:
                rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=True, img_idx=img_idx, **render_kwargs_test)
            torch.set_default_tensor_type('torch.FloatTensor')
            # print(rgb.shape)
            
        target_list.append(target.cpu())
        rgb_list.append(rgb.cpu())
        pose_list.append(pose.cpu())
        img_idx_list.append(img_idx.cpu())

    

    targets = torch.stack(target_list).detach()
    rgbs = torch.stack(rgb_list).detach()
    poses = torch.stack(pose_list).detach()
    img_idxs = torch.stack(img_idx_list).detach()
    return targets, rgbs, poses, img_idxs

def render_virtual_imgs(args, pose_perturb, img_idxs, hwf, device, render_kwargs_test, world_setup_dict):
    ''' render nerfw imgs, save unscaled pose and results'''
    H, W, focal = hwf
    rgb_list = []

    for batch_idx in range(pose_perturb.shape[0]):

        pose = pose_perturb[batch_idx]
        img_idx = img_idxs[batch_idx].to(device)
        pose_nerf = pose.clone()

        pose_nerf = fix_coord_supp(args, pose_nerf[None,...].cpu(), world_setup_dict)

        with torch.no_grad():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if args.tinyimg:
                rgb, _, _, _ = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
                rgb = rgb[None,...].permute(0,3,1,2)
                rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
                rgb = rgb[0].permute(1,2,0)

            else:
                rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=False, img_idx=img_idx, **render_kwargs_test)
            torch.set_default_tensor_type('torch.FloatTensor')
        rgb_list.append(rgb.cpu())

    rgbs = torch.stack(rgb_list).detach()
    return rgbs

def PoseLoss(args, pose_, pose, device):
    loss_func = nn.MSELoss()
    predict_pose = pose_.to(device) # maynot need reshape
    pose_loss = loss_func(predict_pose, pose)
    return pose_loss

def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    return x

def masked_loss(criterion, f1, f2, valid_mask):
    ''' 
    compute loss only in masked region
    :param criterion: loss function
    :param f1: [3, batch_size, H, W]
    :param f2: [3, batch_size, H, W]

    :param valid_mask: [batch_size, H, W]
    :return:
        loss
    '''
    f1 = f1 * (valid_mask)
    f2 = f2

    loss = criterion(f1, f2)
    loss = (loss * valid_mask).sum()
    loss = loss / (valid_mask.sum())
    return loss

def triplet_loss(f1, f2, margin=1.):
    ''' 
    naive implementation of triplet loss
    :param criterion: loss function
    :param f1: [lvl, B, C, H, W]
    :param f2: [lvl, B, C, H, W]
    :return:
        loss
    '''
    criterion = nn.TripletMarginLoss(margin=margin, reduction='mean')
    anchor = f1
    positive = f2
    negative = torch.roll(f2, shifts=1, dims=1)
    loss = criterion(anchor, positive, negative)
    return loss

def triplet_loss_hard_negative_mining(f1, f2, margin=1.):
    ''' 
    triplet loss with hard negative mining, inspired by http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf section3.3 
    :param criterion: loss function
    :param f1: [lvl, B, C, H, W]
    :param f2: [lvl, B, C, H, W]
    :return:
        loss
    '''
    criterion = nn.TripletMarginLoss(margin=margin, reduction='mean')
    anchor = f1
    anchor_negative = torch.roll(f1, shifts=1, dims=1)
    positive = f2
    negative = torch.roll(f2, shifts=1, dims=1)

    mse = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        case1 = mse(anchor, negative)
        case2 = mse(positive, anchor_negative)
    
    if case1 < case2:
        loss = criterion(anchor, positive, negative)
    else:
        loss = criterion(positive, anchor, anchor_negative)
    return loss

def triplet_loss_hard_negative_mining_plus(f1, f2, margin=1.):
    ''' 
    triplet loss with hard negative mining, four cases. inspired by http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf section3.3 
    :param criterion: loss function
    :param f1: [lvl, B, C, H, W]
    :param f2: [lvl, B, C, H, W]
    :return:
        loss
    '''
    criterion = nn.TripletMarginLoss(margin=margin, reduction='mean')
    anchor = f1
    anchor_negative = torch.roll(f1, shifts=1, dims=1)
    positive = f2
    negative = torch.roll(f2, shifts=1, dims=1)

    mse = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        case1 = mse(anchor, negative)
        case2 = mse(positive, anchor_negative)
        case3 = mse(anchor, anchor_negative)
        case4 = mse(positive, negative)
        distance_list = torch.stack([case1,case2,case3,case4])
        loss_case = torch.argmin(distance_list)
    
    if loss_case == 0:
        loss = criterion(anchor, positive, negative)
    elif loss_case == 1:
        loss = criterion(positive, anchor, anchor_negative)
    elif loss_case == 2:
        loss = criterion(anchor, positive, anchor_negative)
    elif loss_case == 3:
        loss = criterion(positive, anchor, negative)
    else:
        raise NotImplementedError
    return loss

def perturb_rotation(c2w, theta, phi, psi=0):
    last_row = np.array([[0, 0, 0, 1]])
    c2w = np.concatenate([c2w, last_row], 0)

    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = rot_psi(psi/180.*np.pi) @ c2w
    c2w = c2w[:3,:4]

    return c2w

def perturb_single_render_pose(poses, x, angle):
    """
    Inputs:
        poses: (3, 4)
        x: translational perturb range
        angle: rotation angle perturb range in degrees
    Outputs:
        new_c2w: (N_views, 3, 4) new poses
    """
    c2w=poses
    
    N_views = 1
    new_c2w = np.zeros((N_views, 3, 4))

    for i in range(N_views):
        new_c2w[i] = c2w
        loc = deepcopy(new_c2w[i,:,3])

        rot_rand=np.random.uniform(-angle,angle,3)
        theta, phi, psi = rot_rand

        new_c2w[i] = perturb_rotation(new_c2w[i], theta, phi, psi)

        trans_rand = np.random.uniform(-x,x,3)

        new_c2w[i,:,3] = loc + trans_rand

    return new_c2w

def perturb_single_render_pose_norm(poses, x, angle):
    """
    Inputs:
        poses: (3, 4)
        x: translational perturb range
        angle: rotation angle perturb range in degrees
    Outputs:
        new_c2w: (N_views, 3, 4) new poses
    """
    c2w=poses
    
    N_views = 1
    new_c2w = np.zeros((N_views, 3, 4))

    for i in range(N_views):
        new_c2w[i] = c2w
        trans_rand = np.random.uniform(-x,x,3)

        trans_rand = trans_rand / abs(trans_rand).sum() * x
        new_c2w[i,:,3] = new_c2w[i,:,3] + trans_rand

        theta=np.random.uniform(-angle,angle,1)
        phi=np.random.uniform(-angle,angle,1)
        psi=np.random.uniform(-angle,angle,1)
        
        rot_rand = np.array([theta, phi, psi])
        rot_rand = rot_rand / abs(rot_rand).sum() * angle
        theta, phi, psi = rot_rand

        new_c2w[i] = perturb_rotation(new_c2w[i], theta, phi, psi)
    return new_c2w
