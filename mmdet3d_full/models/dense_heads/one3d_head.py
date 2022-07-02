
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
from mmdet.core import (multi_apply, build_assigner, build_sampler,
                        reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.builder import HEADS, build_loss
from mmdet3d.models import builder
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox.util import encode_bboxes, decode_bboxes


@HEADS.register_module()
class One3DHead(nn.Module):
    """Head of OneNet3D. 

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(self,
                 sync_cls_avg_factor=True,
                 in_channels=[128],
                 tasks=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_cls_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 train_cfg=None,
                 test_cfg=None):
        super(One3DHead, self).__init__()
        self.bg_cls_weight = 0
        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = sum(num_classes)
        self.norm_bbox = norm_bbox
        self.sync_cls_avg_factor = sync_cls_avg_factor
        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        heads = copy.deepcopy(common_heads)
        heads.update(dict(heatmap=(self.num_classes, num_cls_convs)))
        separate_head.update(
            in_channels=share_conv_channel, heads=heads, num_cls=self.num_classes)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        self.det_head = builder.build_head(separate_head)
        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = self.bbox_coder.pc_range
        self.voxel_size = self.bbox_coder.voxel_size
        self.out_size_factor = self.bbox_coder.out_size_factor

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            # assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
            #     'The classification weight for loss and matcher should be' \
            #     'exactly the same.'
            # assert loss_bbox['loss_weight'] == assigner['reg_cost'][
            #     'weight'], 'The regression L1 weight for loss and matcher ' \
            #     'should be exactly the same.'
            # assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
            #     'The regression iou weight for loss and matcher should be' \
            #     'exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.locations = self.create_grid()

    def init_weights(self):
        """Initialize weights."""
        self.det_head.init_weights()

    def create_grid(self):
        x_range = (self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0] / self.out_size_factor
        y_range = (self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1] / self.out_size_factor
        nx = int(x_range)
        ny = int(y_range)
        xs = torch.linspace(0, x_range-1, nx).view(nx, 1).repeat(1, ny)
        ys = torch.linspace(0, y_range-1, ny).view(1, ny).repeat(nx, 1)
        locations = torch.stack((xs, ys), dim=0)
        locations = nn.Parameter(locations, requires_grad=False)
        return locations

    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        x = self.shared_conv(x)   
        outs = self.det_head(x)
        return outs

    def forward(self, feats):
        """Forward function.
        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return self.forward_single(feats[0])

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        cls_scores = preds_dicts['heatmap']
        bbox_preds = torch.cat(
            (preds_dicts['reg'], preds_dicts['height'],
             preds_dicts['dim'], preds_dicts['rot'],
             preds_dicts['vel']), dim=1)
        B, num_cls, H, W = cls_scores.size()
        cls_scores = cls_scores.permute(0, 2, 3, 1)
        bbox_preds = bbox_preds.permute(0, 2, 3, 1)
        cls_scores = cls_scores.reshape(B, H*W, num_cls)
        bbox_preds = bbox_preds.reshape(B, H*W, -1)

        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_bboxes_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, num_cls)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        encoded_bbox_targets = encode_bboxes(bbox_targets, self.voxel_size, self.out_size_factor, self.pc_range)[:, :8]
        isnotnan = torch.isfinite(encoded_bbox_targets).all(dim=-1)
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :8], encoded_bbox_targets[isnotnan], bbox_weights[isnotnan, :8], avg_factor=num_total_pos)

        # loss_iou = self.loss_iou(
        #     decode_bboxes(bbox_preds, self.voxel_size, self.out_size_factor, self.pc_range)[:, :7], 
        #     bbox_targets[:, :7], bbox_weights[:, 0], avg_factor=num_total_pos)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = loss_cls
        loss_dict['loss_bbox'] = loss_bbox
        # loss_dict['loss_iou'] = loss_iou
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # preds_dicts['reg'] += self.locations.unsqueeze(0)
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, 9)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
