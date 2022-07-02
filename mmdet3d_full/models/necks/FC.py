import numpy as np
import torch
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from mmcv.runner import auto_fp16, force_fp32
from torch import nn as nn
import torch.nn.functional as F
from pytorch3d import transforms

from mmdet.models import NECKS
from mmdet3d.models.builder import build_loss

@NECKS.register_module()
class FC(nn.Module):
    """FC for view projection.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
    """

    def __init__(self,
                 in_channels=(512, 1024, 2048),
                 hidden_dim=256,
                 nheads=4,
                 num_encoder_layers=4, 
                 num_decoder_layers=4,
                 encoder_spatial_size=(15, 28),
                 decoder_spatial_size=(256, 256),
                 decoder_upsample=3,
                 encoder_downsample=32):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(FC, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_spatial_size = encoder_spatial_size
        self.decoder_spatial_size = decoder_spatial_size
        self.encoder_row_embed = nn.Parameter(torch.rand(hidden_dim // 2, 50))
        self.encoder_col_embed = nn.Parameter(torch.rand(hidden_dim // 2, 50))
        self.decoder_row_embed = nn.Parameter(torch.rand(hidden_dim // 2, decoder_spatial_size[0]))
        self.decoder_col_embed = nn.Parameter(torch.rand(hidden_dim // 2, decoder_spatial_size[1]))
        self.decoder_upsample = decoder_upsample
        self.encoder_downsample = encoder_downsample
        self.conv = nn.Conv1d(2048, hidden_dim, 1)

        self.transformation_net = nn.Sequential(
            nn.Conv1d(10, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, 1),
        )

        self.feat_merge_net = nn.Sequential(
            nn.Linear(6*hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        self.bev_net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )


    def init_weights(self):
        """Initialize weights of FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif is_norm(m):
                constant_init(m, 1)

    def get_transformation_feats(self, lidar2cam, cam_intrinsic):
        if not self.training:
            lidar2cam = lidar2cam[0]
            cam_intrinsic = cam_intrinsic[0]
        rot = lidar2cam[:, :, :3, :3]
        euler = transforms.matrix_to_euler_angles(rot, convention='XYZ')
        trans = lidar2cam[:, :, :3, 3]

        cam_intrinsic = torch.cat((
            torch.diagonal(cam_intrinsic, offset=0, dim1=-2, dim2=-1)[:, :, :2],
            cam_intrinsic[:, :, :2, 2]
        ), dim=-1) / self.encoder_downsample

        transformation_feats = self.transformation_net(
            torch.cat((euler, trans, cam_intrinsic), dim=-1).permute(0, 2, 1)
        ) # (B, C, N)
        return transformation_feats

    def get_img_feats(self, img_feats):
        B, N, C, H, W = img_feats.size()
        img_feats = img_feats.mean(dim=4).mean(dim=3).permute(0, 2, 1)
        img_feats = self.conv(img_feats) # (B, C, N)
        return img_feats

    @auto_fp16()
    def forward(self, img_feats, lidar2img, lidar2cam, cam_intrinsic):
        """Forward function.

        Args:
            x list[(torch.Tensor)]: 5D Tensor in (B, N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: points (B, N, C).
        """

        img_feats = img_feats[-1]
        B, N, C, H, W = img_feats.size()
        
        oW, oH = self.decoder_spatial_size
        decoder_pos = torch.cat([
            self.decoder_col_embed[:, :oW].unsqueeze(1).repeat(1, oH, 1),
            self.decoder_row_embed[:, :oH].unsqueeze(2).repeat(1, 1, oW),
            ], dim=0).unsqueeze(0) # (hidden_dim, oH, oW) -> (1, hidden_dim, oH, oW)
        
        img_feats = self.get_img_feats(img_feats)
        transformation_feats = self.get_transformation_feats(lidar2cam, cam_intrinsic)

        img_feats = img_feats + transformation_feats  # (B, C, N)
        img_feats = self.feat_merge_net(img_feats.view(B, self.hidden_dim*N))

        img_feats = img_feats.view(B, self.hidden_dim, 1, 1)

        h = img_feats + decoder_pos
        
        h = self.bev_net(h)
        return [h] 
    