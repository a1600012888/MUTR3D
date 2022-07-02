import copy
from collections import OrderedDict
import torch
from mmcv.runner import  load_checkpoint
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models.builder import build_detector

@DETECTORS.register_module()
class Distiller3D(Base3DDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 teacher=None,
                 student=None,
                 train_cfg=None,
                 test_cfg=None,
                 teacher_pretrained=None,
                 student_pretrained=None,
                 pretrained=None):
        super(Distiller3D, self).__init__()
        self.teacher = build_detector(
            teacher, train_cfg=train_cfg, test_cfg=test_cfg)
        self.student = build_detector(
            student, train_cfg=train_cfg, test_cfg=test_cfg)

        self.init_weights_teacher(teacher_pretrained)
        if student_pretrained and train_cfg:
            self.init_weights_student(student_pretrained)

        self.teacher.eval()
        
    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')
    
    def init_weights_student(self, path=None):
        """Load the pretrained model in teacher detector.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.student, path, map_location='cpu')

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        pts_feats = self.extract_pts_feat(points, None, img_metas)
        return (None, pts_feats)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        with torch.no_grad():
            self.teacher.eval()
            teacher_feats = self.teacher.extract_pts_feat(pts, img_feats, img_metas)
        student_feats = self.student.extract_pts_feat(pts, img_feats, img_metas)
        return [teacher_feats, student_feats]

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        teacher_feats, student_feats = pts_feats
        with torch.no_grad():
            self.teacher.eval()
            teacher_outs = self.teacher.pts_bbox_head(teacher_feats[0])
        student_outs = self.student.pts_bbox_head(student_feats[0])
        student_loss_inputs = [gt_bboxes_3d, gt_labels_3d, student_outs, student_feats]
        student_losses = self.student.pts_bbox_head.loss(*student_loss_inputs)
        student_distill_loss_inputs = [teacher_outs, student_outs]
        student_distill_losses = self.student.pts_bbox_head.distill_loss(*student_distill_loss_inputs)
        student_losses.update(student_distill_losses)
        return student_losses

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        _, pts_feats = self.extract_feat(points, img=None, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False):
        return self.student.simple_test(points, img_metas, img=img, rescale=rescale)

    def simple_test_pts(self, x, img_metas, rescale=False):
        return self.student.simple_test_pts(x, img_metas, rescale=rescale)

    def aug_test_pts(self, feats, img_metas, rescale=False):
        return self.student.aug_test_pts(feats, img_metas, rescale=rescale)

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        return self.student.aug_test(points, img_metas, imgs, rescale=rescale)
