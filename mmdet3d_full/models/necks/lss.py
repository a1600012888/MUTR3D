import numpy as np
import torch
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet.models import NECKS
from torchvision.models.resnet import resnet18


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
class LiftSplatShot(nn.Module):
    """LiftSplatShot for view projection.

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
                 strides=(0.5, 1, 2),
                 depth_range=(4.5, 45.0, 1.0),
                 feat_size=(30, 56),
                 downsample=16,
                 point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 voxel_size=(0.2, 0.2, 8),
                 norm_cfg=dict(type='BN2d')):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(LiftSplatShot, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.feat_size = feat_size
        self.depth_range = depth_range
        self.hidden_channels = hidden_channels
        self.downsample = downsample
        upsamplers = []
        for i in range(len(strides)):
            if strides[i] == 0.5:
                upsampler = nn.Sequential(
                    nn.Conv2d(in_channels[i], hidden_channels, kernel_size=3, stride=2, padding=1),
                )
            elif strides[i] == 1:
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
        self.D, _, _, _ = self.frustum.shape
        self.depth_net = nn.Sequential(
            nn.Conv2d(len(strides)*hidden_channels, len(strides)*hidden_channels, kernel_size=3, padding=1, stride=1),
            build_norm_layer(norm_cfg, len(strides)*hidden_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(len(strides)*hidden_channels, self.D+hidden_channels, kernel_size=1, padding=0)            
        )

        dx = torch.Tensor(voxel_size)
        bx = torch.Tensor([point_cloud_range[i]+voxel_size[i]/2 for i in range(len(voxel_size))])
        nx = torch.LongTensor([(point_cloud_range[i+3]-point_cloud_range[i])/voxel_size[i] for i in range(len(voxel_size))])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        trunk = resnet18(pretrained=False, zero_init_residual=True)

        self.bevencode = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            trunk.layer1,
            trunk.layer2
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
        ds = torch.arange(*self.depth_range, dtype=torch.float).view(-1, 1, 1).expand(-1, H, W)
        D, _, _ = ds.shape
        xs = torch.linspace(0, W*self.downsample - 1, W, dtype=torch.float).view(1, 1, W).expand(D, H, W)
        ys = torch.linspace(0, H*self.downsample - 1, H, dtype=torch.float).view(1, H, 1).expand(D, H, W)
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def get_feat(self, img_feats):
        img_feats_upsampled = []
        for i, img_feat in enumerate(img_feats):
            B, N, C, H, W = img_feat.size()
            img_feat = img_feat.view(B*N, C, H, W)
            img_feat_upsampled = self.upsamplers[i](img_feat)
            img_feats_upsampled.append(img_feat_upsampled)
        img_feats_concat = torch.cat(img_feats_upsampled, dim=1)
        img_feats_concat = self.depth_net(img_feats_concat)

        return img_feats_concat

    def get_depth_feat(self, img_feats):
        x = self.get_feat(img_feats)
        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:].unsqueeze(2)
        return depth, new_x

    def get_cam_feats(self, img_feats):
        B, N = img_feats[0].size()[:2]
        _, img_feats = self.get_depth_feat(img_feats)
        img_feats = img_feats.view(B, N, self.hidden_channels, self.D, self.feat_size[0], self.feat_size[1])
        img_feats = img_feats.permute(0, 1, 3, 4, 5, 2)
        return img_feats

    def get_points(self, img_metas):
        lidar2img = [np.linalg.inv(l['lidar2img']) for l in img_metas]
        lidar2img = torch.from_numpy(np.asarray(lidar2img).astype(np.float32)).to(self.frustum.device)
        B, N = lidar2img.size()[0:2]
        lidar2img = lidar2img.view(B, N, 1, 1, 4, 4)
        D, H, W = self.frustum.size()[:3]
        points = self.frustum.view(1, 1, D, H, W, 3)
        points = torch.cat((points[..., 0:2] * points[..., 2:3], points[..., 2:3]), -1)
        ones = torch.ones_like(points[..., :1])
        points = torch.cat([points, ones], -1)
        points = points.repeat(B, N, 1, 1, 1, 1)
        points_lidar = torch.matmul(points, lidar2img.transpose(-2, -1))
        points_lidar = points_lidar[..., :3] 
        return points_lidar
    
    def voxel_pooling(self, points, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W 
        
        x = x.reshape(Nprime, C)
        points = ((points - (self.bx - self.dx/2.)) / self.dx).long()
        points = points = points.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        points = torch.cat((points, batch_ix), 1)

        # filter out points that are outside box
        kept = (points[:, 0] >= 0) & (points[:, 0] < self.nx[0])\
                & (points[:, 1] >= 0) & (points[:, 1] < self.nx[1])\
                & (points[:, 2] >= 0) & (points[:, 2] < self.nx[2])
        x = x[kept]
        points = points[kept]

        # get tensors from the same voxel next to each other
        ranks = points[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + points[:, 1] * (self.nx[2] * B)\
            + points[:, 2] * B\
            + points[:, 3]
        sorts = ranks.argsort()

        x, points, ranks = x[sorts], points[sorts], ranks[sorts]
        x, points = QuickCumsum.apply(x, points, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[points[:, 3], :, points[:, 2], points[:, 0], points[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)
        return final

    @auto_fp16()
    def forward(self, img_feats, img_metas):
        """Forward function.

        Args:
            x list[(torch.Tensor)]: 5D Tensor in (B, N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        img_feats = self.get_cam_feats(img_feats)
        points = self.get_points(img_metas)

        bev_feats = self.voxel_pooling(points, img_feats)
        bev_feats = self.bevencode(bev_feats)

        return [bev_feats]

    