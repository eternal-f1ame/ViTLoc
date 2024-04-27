"""
Code for the position encoding of TransPoseNet
 code is based on https://github.com/facebookresearch/detr/tree/master/models with the following modifications:
- changed to learn also the position of a learned pose token
"""
import torch
from torch import nn
from typing import Optional
from torch import Tensor
from typing import Optional, List
import torch
from torch import Tensor
import pdb

def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class PositionEmbeddingLearnedWithPoseToken(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(60, num_pos_feats)
        self.col_embed = nn.Embedding(60, num_pos_feats)
        self.pose_token_embed = nn.Embedding(60, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.pose_token_embed.weight)

    def forward(self, tensors : torch.Tensor):
        x = tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device) + 1
        j = torch.arange(h, device=x.device) + 1
        p = i[0]-1
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        p_emb = torch.cat([self.pose_token_embed(p),self.pose_token_embed(p)]).repeat(x.shape[0], 1)

        m_emb = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return p_emb, m_emb

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensors: torch.Tensor):
        x = tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(config):
    hidden_dim = config.get("hidden_dim")
    N_steps = hidden_dim // 2
    learn_embedding_with_pose_token = config.get("learn_embedding_with_pose_token")
    if learn_embedding_with_pose_token:
        position_embedding = PositionEmbeddingLearnedWithPoseToken(N_steps)
    else:
        position_embedding = PositionEmbeddingLearned(N_steps)
    return position_embedding
