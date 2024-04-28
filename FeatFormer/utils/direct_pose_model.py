from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

from script.dm.helpers import preprocess_data

from feature.dfnet import DFNet_s as FeatureNet3


''' PoseNet Models '''
from feature.dfnet import DFNet_s as PoseNet3

def inference_pose_regression(args, data, device, model):
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
    if args.preprocess_ImgNet:
        inputs = preprocess_data(inputs, device)
    predict_pose = model(inputs)
    pose = predict_pose.reshape(args.batch_size, 3, 4)

    if args.svd_reg:
        R_torch = pose[:,:3,:3].clone() # debug
        u,s,v=torch.svd(R_torch)
        Rs = torch.matmul(u, v.transpose(-2,-1))
        pose[:,:3,:3] = Rs
    return pose

def PoseLoss(args, pose_, pose, device):
    loss_func = nn.MSELoss()
    predict_pose = pose_.reshape(args.batch_size, 12).to(device)
    pose_loss = loss_func(predict_pose, pose)
    return pose_loss

def load_exisiting_model(args, isFeatureNet=False):
    ''' Load a pretrained DFNet model '''
    if isFeatureNet==False:

        model = PoseNet3()
        model.load_state_dict(torch.load(args.pretrain_model_path))

        return model
    else:

        model = FeatureNet3()
        model.load_state_dict(torch.load(args.pretrain_featurenet_path))

        return model

def fix_coord_supp(args, pose, world_setup_dict, device=None):

    '''supplementary fix_coord() for direct matching
    Inputs:
        args: parsed argument
        pose: pose [N, 3, 4]
        device: cpu or gpu
    Outputs:
        pose: converted Pose in shape [N, 3, 4]
    '''
    sc=world_setup_dict['pose_scale']
    if device is None:
        move_all_cam_vec = torch.Tensor(world_setup_dict['move_all_cam_vec'])
    else:
        move_all_cam_vec = torch.Tensor(world_setup_dict['move_all_cam_vec']).to(device)
    sc2 = world_setup_dict['pose_scale2']
    pose[:,:3,3] *= sc
    pose[:, :3, 3] += move_all_cam_vec
    pose[:,:3,3] *= sc2
    return pose
