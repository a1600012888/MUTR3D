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
class Transformer(nn.Module):
    """Transformer for view projection.

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
                 decoder_spatial_size=(16, 16),
                 decoder_upsample=3,
                 encoder_downsample=32):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_spatial_size = encoder_spatial_size
        self.decoder_spatial_size = decoder_spatial_size
        self.encoder_row_embed = nn.Parameter(torch.rand(hidden_dim // 2, 50))
        self.encoder_col_embed = nn.Parameter(torch.rand(hidden_dim // 2, 50))
        self.decoder_row_embed = nn.Parameter(torch.rand(hidden_dim // 2, 50))
        self.decoder_col_embed = nn.Parameter(torch.rand(hidden_dim // 2, 50))
        self.decoder_upsample = decoder_upsample
        self.encoder_downsample = encoder_downsample
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        self.transformation_net = nn.Sequential(
            nn.Conv1d(10, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, 1),
        )
        self.transformer = nn.Transformer(self.hidden_dim, 
                                          self.nheads,
                                          self.num_encoder_layers,
                                          self.num_decoder_layers)
        upsampler = []
        for _ in range(self.decoder_upsample):
            upsampler.append(nn.ConvTranspose2d(self.hidden_dim, 
                                                self.hidden_dim, 
                                                3, stride=2, 
                                                padding=1, 
                                                output_padding=1,
                                                bias=False))
            upsampler.append(nn.BatchNorm2d(self.hidden_dim))
            upsampler.append(nn.ReLU(inplace=True))
        self.upsampler = nn.Sequential(*upsampler)

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
        ).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1) # (B, N, C, 1, 1)
        return transformation_feats

    def get_img_feats(self, img_feats):
        B, N, C, H, W = img_feats.size()
        img_feats = img_feats.view(B*N, C, H, W)
        img_feats = self.conv(img_feats).view(B, N, self.hidden_dim, H, W) # (B, N, hidden_dim, H, W)
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
        encoder_pos = torch.cat([
            self.encoder_col_embed[:, :W].unsqueeze(1).repeat(1, H, 1),
            self.encoder_row_embed[:, :H].unsqueeze(2).repeat(1, 1, W),
            ], dim=0).unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim, h, W)
        
        oH, oW = self.decoder_spatial_size
        decoder_pos = torch.cat([
            self.decoder_col_embed[:, :oW].unsqueeze(1).repeat(1, oH, 1),
            self.decoder_row_embed[:, :oH].unsqueeze(2).repeat(1, 1, oW),
            ], dim=0).flatten(1, 2).permute(0, 1).unsqueeze(1) # (hidden_dim, oH * oW) -> (oH*oW, 1, hidden_dim)
        decoder_pos = decoder_pos.repeat(1, B, 1)
        img_feats = self.get_img_feats(img_feats)
        transformation_feats = self.get_transformation_feats(lidar2cam, cam_intrinsic)

        img_feats = img_feats + transformation_feats + encoder_pos # (B, N, hidden_dim, H, W)

        img_feats = img_feats.permute(1, 3, 4, 0, 2).reshape(H*W*N, B, self.hidden_dim) # (N*H*W, B, hidden_dim)

        h = self.transformer(img_feats, decoder_pos) # (oH*oW, B, C)
        h = h.permute(1, 2, 0).view(B, self.hidden_dim, oH, oW)
        
        h = self.upsampler(h)
        return [h] 
    