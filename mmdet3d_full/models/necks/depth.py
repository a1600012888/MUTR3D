import numpy as np
import torch
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from mmcv.runner import auto_fp16, force_fp32
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models import NECKS
from torchvision.models.resnet import resnet18
from mmdet3d.models.builder import build_loss


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


@NECKS.register_module()
class Depther(nn.Module):
    """Depther for view projection.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
    """

    def __init__(self,
                 in_channels=(512, 1024, 2048),
                 out_channels=(128, 128),
                 hidden_channels=64,
                 strides=(2, 4, 8),
                 depth_range=(4.5, 45.0, 1.0),
                 feat_size=(120, 224),
                 downsample=4,
                 loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean'),
                 loss_depth=dict(
                     type='L1Loss', reduction='none', loss_weight=1.0),
                 point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 voxel_size=(0.2, 0.2, 8),
                 norm_cfg=dict(type='BN2d')):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(Depther, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.feat_size = feat_size
        self.depth_range = depth_range
        self.hidden_channels = hidden_channels
        self.downsample = downsample
        self.loss_depth = build_loss(loss_depth)
        self.loss_mask = build_loss(loss_mask)
        upsamplers = []
        for i in range(len(strides)):
            if strides[i] == 1:
                upsampler = nn.Sequential(
                    nn.Conv2d(in_channels[i], hidden_channels, kernel_size=3, stride=1, padding=1),
                )
            else:  
                upsampler = nn.Sequential(
                    nn.Conv2d(in_channels[i], hidden_channels, kernel_size=1),
                    nn.Upsample(scale_factor=strides[i], mode='bilinear', align_corners=True),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
                )
            upsamplers.append(upsampler)
        self.upsamplers = nn.ModuleList(upsamplers)
        self.frustum = self.create_frustum()
        self.D = 1
        self.M = 1
        self.depth_net = nn.Sequential(
            nn.Conv2d(len(strides)*hidden_channels, len(strides)*hidden_channels, kernel_size=3, padding=1, stride=1),
            build_norm_layer(norm_cfg, len(strides)*hidden_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(len(strides)*hidden_channels, self.D, kernel_size=1, padding=0)            
        )
        self.mask_net = nn.Sequential(
            nn.Conv2d(len(strides)*hidden_channels, len(strides)*hidden_channels, kernel_size=3, padding=1, stride=1),
            build_norm_layer(norm_cfg, len(strides)*hidden_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(len(strides)*hidden_channels, self.M, kernel_size=1, padding=0)            
        )
        self.feat_net = nn.Sequential(
            nn.Conv2d(len(strides)*hidden_channels, len(strides)*hidden_channels, kernel_size=3, padding=1, stride=1),
            build_norm_layer(norm_cfg, len(strides)*hidden_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(len(strides)*hidden_channels, hidden_channels, kernel_size=1, padding=0)            
        )

    def init_weights(self):
        """Initialize weights of FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif is_norm(m):
                constant_init(m, 1)

    def create_frustum(self):
        H, W = self.feat_size
        xs = torch.linspace(0, W*self.downsample - 1, W, dtype=torch.float).view(1, W).expand(H, W)
        ys = torch.linspace(0, H*self.downsample - 1, H, dtype=torch.float).view(H, 1).expand(H, W)
        # H x W x 2
        frustum = torch.stack((xs, ys), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_cam_feats(self, img_feats):
        img_feats_upsampled = []
        for i, img_feat in enumerate(img_feats):
            B, N, C, H, W = img_feat.size()
            img_feat = img_feat.view(B*N, C, H, W)
            img_feat_upsampled = self.upsamplers[i](img_feat)
            img_feats_upsampled.append(img_feat_upsampled)
        img_feats_concat = torch.cat(img_feats_upsampled, dim=1)
        return img_feats_concat

    def get_points(self, img_feats, img_metas):
        depth = self.depth_net(img_feats)
        depth = F.softplus(depth)
        BN, _, H, W = img_feats.size()
        depth = depth.view(BN, H, W, 1)
        points = self.frustum.view(1, H, W, 2).repeat(BN, 1, 1, 1)
        points = torch.cat((points, depth), -1)
        lidar2img = [np.linalg.inv(l['lidar2img']) for l in img_metas]
        lidar2img = torch.from_numpy(np.asarray(lidar2img).astype(np.float32)).to(self.frustum.device)
        B, N = lidar2img.size()[0:2]
        lidar2img = lidar2img.view(B, N, 1, 1, 4, 4)

        H, W = points.size()[1:3]
        points = points.view(B, N, 1, H, W, 3)

        points = torch.cat((points[..., 0:2] * points[..., 2:3], points[..., 2:3]), -1)
        ones = torch.ones_like(points[..., :1])
        points = torch.cat([points, ones], -1)
        points_lidar = torch.matmul(points, lidar2img.transpose(-2, -1))
        points_lidar = points_lidar[..., :3] 
        depth = depth.view(B, N, H, W)
        return depth, points_lidar

    def get_point_feats(self, img_feats):
        BN = img_feats.size(0)
        img_feats = self.feat_net(img_feats)
        img_feats = img_feats.view(BN // 6, 6, self.hidden_channels, self.feat_size[0], self.feat_size[1])
        img_feats = img_feats.permute(0, 1, 3, 4, 2)
        return img_feats

    def get_point_masks(self, img_feats):
        BN = img_feats.size(0)
        point_masks = self.mask_net(img_feats)
        point_masks = point_masks.view(BN // 6, 6, self.feat_size[0], self.feat_size[1])
        return point_masks

    @auto_fp16()
    def forward(self, img_feats, img_metas, img_depth, img_mask):
        """Forward function.

        Args:
            x list[(torch.Tensor)]: 5D Tensor in (B, N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: points (B, N, 3+C).
        """

        img_feats = self.get_cam_feats(img_feats)

        depth, points = self.get_points(img_feats, img_metas)
        point_feats = self.get_point_feats(img_feats)
        point_masks = self.get_point_masks(img_feats)

        B, N, H, W, C = point_feats.size()

        point_feats = point_feats.reshape(B, N*H*W, C)

        points = points.reshape(B, N*H*W, 3)

        points = torch.cat((points, point_feats), -1)

        masked_points = []
        point_masks_pred = point_masks.reshape(B, N*H*W)
        point_masks_gt = img_mask.reshape(B, N*H*W)

        for i in range(B):
            point = points[i]
            # if self.training:
            #     mask = point_masks_gt[i]
            # else:
            mask = point_masks_pred[i]
            point = point[mask>0, :]
            masked_points.append(point)
        return masked_points, depth, point_masks
    
    @force_fp32(apply_to=('pred_mask', 'pred_depth'))
    def loss(self, gt_depth, gt_mask, pred_depth, pred_mask):
        num = gt_mask.eq(1).float().sum().item()
        loss_mask = self.loss_mask(
            pred_mask.view(-1, 1),
            gt_mask.view(-1, 1).long(),
            avg_factor=max(num, 1))
        loss_depth = self.loss_depth(
            pred_depth,
            gt_depth,
            gt_mask,
            avg_factor=(num + 1e-4)
        )
        loss_dict = dict()
        loss_dict['loss_depth'] = loss_depth
        loss_dict['loss_mask'] = loss_mask
        return loss_dict