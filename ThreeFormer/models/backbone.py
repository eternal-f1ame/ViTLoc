import torch.nn.functional as F
from torch import nn
from .pencoder import build_position_encoding
import torch
import timm
import warnings

warnings.filterwarnings("ignore")

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, blocks):
        super().__init__()
        self.body = backbone
        self.blocks = blocks
        self.block_map = {"Block_9": 768, "Block_11": 768}
        self.num_channels = [self.block_map[block] for block in self.blocks]

    def forward(self, tensors : torch.Tensor):
        xs = self.body.patch_embed(tensors)
        xs = self.body.pos_drop(xs)
        xs = self.body.patch_drop(xs)
        xs = self.body.norm_pre(xs)
        out = {}
            
        for idx, block in enumerate(self.body.blocks):
            block_name = f"Block_{idx}"
            xs = block(xs)
            if block_name in self.blocks:
                x = xs.permute(0, 2, 1)
                x = x.view(x.size(0), x.size(1), 14, 14)
                out[block_name] = x
        return out

class Backbone(BackboneBase):
    def __init__(self, str, reduction):
        backbone = timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=0)
        super().__init__(backbone, reduction)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensors : torch.Tensor):
        xs = self[0](tensors)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)

            ret = self[1](x)
            if isinstance(ret, tuple):
                p_emb, m_emb = ret
                pos.append([p_emb.to(x.dtype), m_emb.to(x.dtype)])
            else:
                pos.append(ret.to(x.dtype))

        return out, pos

def build_backbone(config):
    position_embedding = build_position_encoding(config)
    backbone = Backbone(config.get("backbone"), config.get("reduction"))
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
