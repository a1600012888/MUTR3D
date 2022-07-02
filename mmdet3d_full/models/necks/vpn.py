import numpy as np
import torch
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet.models import NECKS


@NECKS.register_module()
class VPN(nn.Module):
    """VPN for view projection.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
    """

    def __init__(self,
                 in_channels=(15, 28),
                 out_channels=(128, 128),
                 norm_cfg=dict(type='BN1d')):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(VPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.fcs = nn.ModuleList()

        deblocks = []
        for _ in range(6):
            deblock = nn.Sequential(nn.Linear(in_channels[0]*in_channels[1], 512),
                                    build_norm_layer(norm_cfg, 2048)[1],
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, out_channels[0]*out_channels[1]))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

    def init_weights(self):
        """Initialize weights of FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif is_norm(m):
                constant_init(m, 1)

    @auto_fp16()
    def forward(self, img_feats, img_metas):
        """Forward function.

        Args:
            x list[(torch.Tensor)]: 5D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        x = img_feats[0]
        B, N, C, H, W = x.size()
        x = x.view(B, N, C, H*W)
        xs = torch.chunk(x, 6, dim=1)
        xs = [xsi.view(B, C, H*W) for xsi in xs]
        outs = [deblock(xs[i]) for i, deblock in enumerate(self.deblocks)]
        out = torch.stack(outs, dim=-1)
        out = torch.mean(out, dim=-1)
        out = out.view(B, -1, self.out_channels[0], self.out_channels[1])
        return [out]
