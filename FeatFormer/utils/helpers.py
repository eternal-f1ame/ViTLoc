import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np
from torchvision import models
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt

import math
import time

import pytorch3d.transforms as transforms

def preprocess_data(inputs, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device) # per channel division
    inputs = (inputs - mean[None,:,None,None])/std[None,:,None,None]
    return inputs

def filter_hook(m, g_in, g_out):
    g_filtered = []
    for g in g_in:
        g = g.clone()
        g[g != g] = 0
        g_filtered.append(g)
    return tuple(g_filtered)

def vis_pose(vis_info):
    '''
    visualize predicted pose result vs. gt pose
    '''
    pdb.set_trace()
    pose = vis_info['pose']
    pose_gt = vis_info['pose_gt']
    theta = vis_info['theta']
    ang_threshold=10
    seq_num = theta.shape[0]

    fig = plt.figure(figsize = (8,6))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax1 = fig.add_axes([0, 0.2, 0.9, 0.85], projection='3d')
    ax1.scatter(pose[10:,0],pose[10:,1],zs=pose[10:,2], c='r', s=3**2,depthshade=0) # predict
    ax1.scatter(pose_gt[:,0], pose_gt[:,1], zs=pose_gt[:,2], c='g', s=3**2,depthshade=0) # GT
    ax1.scatter(pose[0:10,0],pose[0:10,1],zs=pose[0:10,2], c='k', s=3**2,depthshade=0) # predict
    ax1.view_init(30, 120)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')

    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)


    ax2 = fig.add_axes([0.1, 0.05, 0.75, 0.2])
    err = theta.reshape(1, seq_num)
    err = np.tile(err, (20, 1))
    ax2.imshow(err, vmin=0,vmax=ang_threshold, aspect=3)
    ax2.set_yticks([])
    ax2.set_xticks([0, seq_num*1/5, seq_num*2/5, seq_num*3/5, seq_num*4/5, seq_num])
    fname = './vis_pose.png'
    plt.savefig(fname, dpi=50)

def compute_error_in_q(args, dl, model, device, results, batch_size=1):
    use_SVD=True 
    time_spent = []
    predict_pose_list = []
    gt_pose_list = []
    ang_error_list = []
    pose_result_raw = []
    pose_GT = []
    i = 0

    for batch in dl:
        if args.NeRFH:
            data, pose, img_idx = batch
        else:
            data, pose = batch
        data = data.to(device) # input
        pose = pose.reshape((batch_size,3,4)).numpy() # label

        if args.preprocess_ImgNet:
            data = preprocess_data(data, device)

        if use_SVD:
            with torch.no_grad():
                if args.featuremetric:
                    _, predict_pose = model(data)
                else:
                    predict_pose = model(data)

                R_torch = predict_pose.reshape((batch_size, 3, 4))[:,:3,:3]
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

        pose_q = transforms.matrix_to_quaternion(torch.Tensor(pose[:,:3,:3]))#.cpu().numpy() # gnd truth in quaternion
        pose_x = pose[:, :3, 3] # gnd truth position
        predicted_q = transforms.matrix_to_quaternion(torch.Tensor(predict_pose[:,:3,:3]))#.cpu().numpy() # predict in quaternion
        predicted_x = predict_pose[:, :3, 3] # predict position
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
        pose_result_raw.append(predict_pose)
        pose_GT.append(pose)
        i += 1

    predict_pose_list = np.array(predict_pose_list)
    gt_pose_list = np.array(gt_pose_list)
    ang_error_list = np.array(ang_error_list)
    pose_result_raw = np.asarray(pose_result_raw)[:,0,:,:]
    pose_GT = np.asarray(pose_GT)[:,0,:,:]
    vis_info_ret = {"pose": predict_pose_list, "pose_gt": gt_pose_list, "theta": ang_error_list, "pose_result_raw": pose_result_raw, "pose_GT": pose_GT}
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
