import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import copy, pdb
from efficientnet_pytorch import EfficientNet
from typing import List

class PoseNetV2(nn.Module):
    def __init__(self, feat_dim=12):
        super(PoseNetV2, self).__init__()
        self.backbone_net = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = self.backbone_net.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(1280, feat_dim)
    
    def _aggregate_feature(self, x, upsampleH, upsampleW):
        '''
        assume target and nerf rgb are inferenced at the same time,
        slice target batch and nerf batch and aggregate features
        :param x: image blob (2B x C x H x W)
        :param upsampleH: New H
        :param upsampleW: New W
        :return feature: (2 x B x H x W)
        '''
        batch = x.shape[0] # should be target batch_size + rgb batch_size
        feature_t = torch.mean(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(x[:batch//2]), dim=1)
        feature_r = torch.mean(torch.nn.UpsamplingBilinear2d(size=(upsampleH, upsampleW))(x[batch//2:]), dim=1)
        feature = torch.stack([feature_t, feature_r])
        return feature

    def _aggregate_feature2(self, x):
        '''
        assume target and nerf rgb are inferenced at the same time,
        slice target batch and nerf batch and output stacked features
        :param x: image blob (2B x C x H x W)
        :return feature: (2 x B x C x H x W)
        '''
        batch = x.shape[0] # should be target batch_size + rgb batch_size
        feature_t = x[:batch//2]
        feature_r = x[batch//2:]
        feature = torch.stack([feature_t, feature_r])
        return feature

    def forward(self, x, upsampleH=224, upsampleW=224, isTrain=False, isSingleStream=False):
        '''
        Currently under dev.
        :param x: image blob ()
        :param upsampleH: New H obsolete
        :param upsampleW: New W obsolete
        :param isTrain: True to extract features, False only return pose prediction. Really should be isExtractFeature
        :param isSingleStrea: True to inference single img, False to inference two imgs in siemese network fashion
        '''
        feat_out = [] # we only use high level features
        for i in range(len(self.feature_extractor)):
            # print("layer {} encoder layer: {}".format(i, self.feature_extractor[i]))
            x = self.feature_extractor[i](x)

            if isTrain: # collect aggregate features
                if i >= 17 and i <= 17: # 17th block
                    if isSingleStream:
                        feature = torch.stack([x])
                    else:
                        feature = self._aggregate_feature2(x)
                    feat_out.append(feature)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)
        return feat_out, predict
