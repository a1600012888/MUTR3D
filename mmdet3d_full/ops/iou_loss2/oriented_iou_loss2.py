
import torch
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

from .utils import intersection_area, PolyArea2D


def cvt_box_2_polygon(box):
    """
    :param array: an array of shape [num_conners, 2]
    :return: a shapely.geometry.Polygon object
    """
    # use .buffer(0) to fix a line polygon
    # more infor: https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera
    return Polygon([(box[i, 0], box[i, 1]) for i in range(len(box))]).buffer(0)


def get_corners_vectorize(x, y, w, l, yaw):
    """bev image coordinates format - vectorization
    :param x, y, w, l, yaw: [num_boxes,]
    :return: num_boxes x (x,y) of 4 conners
    """
    device = x.device
    bbox2 = torch.zeros((x.size(0), 4, 2), device=device, dtype=torch.float)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # front left
    bbox2[:, 0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bbox2[:, 0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bbox2[:, 1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bbox2[:, 1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bbox2[:, 2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bbox2[:, 2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bbox2[:, 3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bbox2[:, 3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bbox2


def get_polygons_areas_fix_xy(boxes, fix_xy=100.):
    """
    Args:
        box: (num_boxes, 4) --> w, l, im, re
    """
    device = boxes.device
    n_boxes = boxes.size(0)
    x = torch.full(size=(n_boxes,), fill_value=fix_xy, device=device, dtype=torch.float)
    y = torch.full(size=(n_boxes,), fill_value=fix_xy, device=device, dtype=torch.float)
    w, l, im, re = boxes.t()
    yaw = torch.atan2(im, re)
    boxes_conners = get_corners_vectorize(x, y, w, l, yaw)
    boxes_polygons = [cvt_box_2_polygon(box_) for box_ in boxes_conners]
    boxes_areas = w * l

    return boxes_polygons, boxes_areas


def iou_rotated_boxes_targets_vs_anchors(anchors_polygons, anchors_areas, targets_polygons, targets_areas):
    device = anchors_areas.device
    num_anchors = len(anchors_areas)
    num_targets_boxes = len(targets_areas)

    ious = torch.zeros(size=(num_anchors, num_targets_boxes), device=device, dtype=torch.float)

    for a_idx in range(num_anchors):
        for tg_idx in range(num_targets_boxes):
            intersection = anchors_polygons[a_idx].intersection(targets_polygons[tg_idx]).area
            iou = intersection / (anchors_areas[a_idx] + targets_areas[tg_idx] - intersection + 1e-16)
            ious[a_idx, tg_idx] = iou

    return ious


def iou_pred_vs_target_boxes(pred_boxes, target_boxes, GIoU=True, DIoU=False, CIoU=False):
    assert pred_boxes.size() == target_boxes.size(), "Unmatch size of pred_boxes and target_boxes"
    device = pred_boxes.device
    n_boxes = pred_boxes.size(0)

    t_x, t_y, t_z, t_w, t_h, t_l, t_yaw = target_boxes.t()
    t_conners = get_corners_vectorize(t_x, t_y, t_w, t_h, t_yaw)
    t_areas = t_w * t_h

    t_z2 = t_z + t_l / 2
    t_z1 = t_z - t_l / 2

    p_x, p_y, p_z, p_w, p_h, p_l, p_yaw = pred_boxes.t()
    p_conners = get_corners_vectorize(p_x, p_y, p_w, p_h, p_yaw)
    p_areas = p_w * p_h

    p_z2 = p_z + p_l / 2 
    p_z1 = p_z - p_l / 2

    z_up = torch.minimum(t_z2, p_z2)
    z_bottom = torch.maximum(t_z1, p_z1)
    l_overlap = z_up - z_bottom 
    l_overlap = torch.maximum(l_overlap, torch.zeros_like(l_overlap))

    ious = []
    giou_losses = []
    # Thinking to apply vectorization this step
    for box_idx in range(n_boxes):
        p_cons, t_cons = p_conners[box_idx], t_conners[box_idx]
        if not GIoU:
            p_poly, t_poly = cvt_box_2_polygon(p_cons), cvt_box_2_polygon(t_cons)
            intersection = p_poly.intersection(t_poly).area
        else:
            intersection = intersection_area(p_cons, t_cons)

        intersection = intersection * l_overlap[box_idx]
        p_area, t_area = p_areas[box_idx], t_areas[box_idx]
        union = p_area * p_l[box_idx] + t_area * t_l[box_idx] - intersection
        iou = intersection / (union + 1e-16)

        if GIoU:
            convex_conners = torch.cat((p_cons, t_cons), dim=0)
            hull = ConvexHull(convex_conners.clone().detach().cpu().numpy())  # done on cpu, just need indices output
            convex_conners = convex_conners[hull.vertices]
            convex_area = PolyArea2D(convex_conners)
            giou_loss = 1. - (iou - (convex_area - union) / (convex_area + 1e-16))
        else:
            giou_loss += 1. - iou
        giou_losses.append(giou_loss)
        if DIoU or CIoU:
            raise NotImplementedError

        ious.append(iou)

    return torch.stack(giou_losses, 0), torch.tensor(ious, device=device, dtype=torch.float)