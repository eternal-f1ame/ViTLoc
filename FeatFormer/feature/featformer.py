import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import List

class AdaptLayers(nn.Module):
    """Small adaptation layers for ViT.
    """

    def __init__(self, hypercolumn_layers: List[str], output_dim: int = 128, encoder=None):
        """Initialize one adaptation layer for every extraction point.

        Args:
            hypercolumn_layers: The list of the hypercolumn layer names.
            output_dim: The output channel dimension.
            encoder: The ViT encoder model.
        """
        super(AdaptLayers, self).__init__()
        self.layers = nn.ModuleList()
        self.output_dim = output_dim
        self.encoder = encoder

        for layer_name in hypercolumn_layers:
            layer_idx = int(layer_name)
            layer = self.encoder.blocks[layer_idx]
            self.layers.append(layer)

    def forward(self, features: List[torch.tensor]):
        """Apply adaptation layers.
        """

        adapted_features = []
        for i, feature in enumerate(features):
            feature = feature
            adapted_feature = self.layers[i](feature)
            adapted_feature = adapted_feature
            adapted_features.append(adapted_feature)
        return adapted_features

class FeatFormer(nn.Module):
    ''' FeatFormer implementation '''
    default_conf = {
        'hypercolumn_layers': ["9", "10", "11"],
        'output_dim': 128,
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, feat_dim=12, places365_model_path=''):
        super(FeatFormer, self).__init__()

        self.encoder = timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=0)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.layer_to_index = {name: idx for idx, (name, _) in enumerate(self.encoder.blocks.named_children())}
        self.hypercolumn_indices = [self.layer_to_index[n] for n in self.default_conf['hypercolumn_layers']]

        self.scales = []
        current_scale = 0
        for i, layer in enumerate(self.encoder.children()):
            if isinstance(layer, timm.models.vision_transformer.PatchEmbed):
                current_scale += 1
            if i in self.hypercolumn_indices:
                self.scales.append(2**current_scale)

        self.adaptation_layers = AdaptLayers(self.default_conf['hypercolumn_layers'], self.default_conf['output_dim'], self.encoder)

        self.avgpool = nn.AdaptiveAvgPool2d((1, self.encoder.num_features))
        self.fc_pose = nn.Linear(self.encoder.num_features, feat_dim)

    def forward(self, x, return_feature=False, isSingleStream=False, return_pose=True, upsampleH=244, upsampleW=244):
        '''
        inference FeatFormer. It can regress camera pose as well as extract intermediate layer features.
            :param x: image blob (2B x C x H x W) two stream or (B x C x H x W) single stream
            :param return_feature: whether to return features as output
            :param isSingleStream: whether it's an single stream inference or siamese network inference
            :param upsampleH: feature upsample size H
            :param upsampleW: feature upsample size W
            :return feature_maps: (2, [B, C, H, W]) or (1, [B, C, H, W]) or None
            :return predict: [2B, 12] or [B, 12]
        '''
        mean, std = x.new_tensor(self.mean), x.new_tensor(self.std)
        x = (x - mean[:, None, None]) / std[:, None, None]

        feature_maps = []
        x = self.encoder.patch_embed(x)
        x = self.encoder.pos_drop(x)
        x = self.encoder.patch_drop(x)
        x = self.encoder.norm_pre(x)
        for i in range(len(self.encoder.blocks)):
            x = self.encoder.blocks[i](x)

            if i in self.hypercolumn_indices:
                feature = x.clone()
                feature_maps.append(feature)

                if i==self.hypercolumn_indices[-1]:
                    if return_pose==False:
                        predict = None
                        break

        if return_feature:
            feature_maps = self.adaptation_layers(feature_maps)
            if isSingleStream:
                feature_stacks = []
                for f in feature_maps:
                    feature_stacks.append(torch.nn.UpsamplingBilinear2d(size=(224, 224))(f))
                feature_maps = [torch.stack(feature_stacks)]
            else:
                feature_stacks_t = []
                feature_stacks_r = []
                for f in feature_maps:
                    deconv = torch.nn.ConvTranspose2d(in_channels=768, out_channels=3, kernel_size=16, stride=16, padding=0, bias=False).to('cuda')
                    f = deconv(f.view(-1, 768, 14, 14))

                    batch = f.shape[0]
                    feature_t = f[:batch//2]
                    feature_r = f[batch//2:]

                    feature_stacks_t.append(torch.nn.UpsamplingBilinear2d(size=(224, 224))(feature_t))
                    feature_stacks_r.append(torch.nn.UpsamplingBilinear2d(size=(224, 224))(feature_r))
                feature_stacks_t = torch.stack(feature_stacks_t)
                feature_stacks_r = torch.stack(feature_stacks_r)
                feature_maps = [feature_stacks_t, feature_stacks_r]
        else:
            feature_maps = None

        if return_pose==False:
            return feature_maps, predict
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)
        return feature_maps, predict
