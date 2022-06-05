""" Many parts are borrowed from https://github.com/TuSimple/SimpleTrack
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from copy import deepcopy
import torch
import pyquaternion

def detr3dbox_to_bbox(box_pred, l2g_r, l2g_t):
    """
    box_pred: torch.tensor of shape [N, box_dim]
    01  23  4  5  6    7    8    9
    xy, wl, z, h, sin, cos, vx, vy
    """
    num_box = box_pred.shape[0]
    with torch.no_grad():
        box_pred = box_pred.detach()
        xy = box_pred[..., :2]
        xyz = torch.cat(
            [xy, 
            box_pred[..., [2]]], dim=-1
        )

        xyz = xyz @ l2g_r + l2g_t

        vxyz = torch.cat(
            [box_pred[..., 8:],
            box_pred.new_zeros(num_box, 1)], dim=-1
        )

        vxyz = vxyz @ l2g_r

        box_yaw =  torch.atan2(box_pred[..., 6], box_pred[..., 7])

        box_yaw = box_yaw.detach().cpu().numpy()

        # TODO: check whether this is necessary
        # with dir_offset & dir_limit in the head
        box_yaw = -box_yaw - np.pi / 2
        xyz = xyz.detach().cpu().numpy()
        box_pred = box_pred.detach().cpu().numpy()
        l2g_r = l2g_r.detach().cpu().numpy()
        
    
    bbox_list = []

    for i in range(num_box):

        orientation = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        # quat = pyquaternion.Quaternion(l2g_r)

        # !! what is orientation
        # o_box = quat * orientation
        o_box = box_yaw[i]
        tmp_box = BBox(xyz[i, 0], xyz[i, 1], xyz[i, 2], 
                        box_pred[i, 5], box_pred[i, 2], box_pred[i, 3],
                        o_box)
        
        bbox_list.append(tmp_box)

    
    return bbox_list


def bbox_to_detr3dbox(box_list, g2l_r, l2g_t):
    """
    maybe we only need xyz whl

    Out: numpy array. 
    xyhlzh
    box_detr3d of shape [N, 6] 
    """

    g2l_r = g2l_r.detach().cpu().numpy()
    l2g_t = l2g_t.detach().cpu().numpy()

    box_array = []
    xyz_list = []
    wlh_list = []    
    for i in range(len(box_list)):
        
        bbox = box_list[i]
        xyz_ = np.array([bbox.x, bbox.y, bbox.z])
        wlh_ = np.array([bbox.w, bbox.l, bbox.h])
        xyz_list.append(xyz_)
        wlh_list.append(wlh_)
    

    xyz = np.stack(xyz_list, axis=0)
    wlh = np.stack(wlh_list, axis=0)


    xyz = (xyz - l2g_t) @ g2l_r

    box_detr3d = np.concatenate(
        [
            xyz[:, :2], 
            wlh[:, :2],
            xyz[:, [2]],
            wlh[:, [2]]
        ], axis=-1
    )


    return box_detr3d


# definition of box
class BBox:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, o=None):
        self.x = x      # center x
        self.y = y      # center y
        self.z = z      # center z
        self.h = h      # height
        self.w = w      # width
        self.l = l      # length
        self.o = o      # orientation
        self.s = None   # detection score
    
    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.o, self.l, self.w, self.h, self.s)
    
    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.o}
    
    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def array2bbox(cls, data):
        bbox = BBox()
        bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox
    
    @classmethod
    def dict2bbox(cls, data):
        bbox = BBox()
        bbox.x = data['center_x']
        bbox.y = data['center_y']
        bbox.z = data['center_z']
        bbox.h = data['height']
        bbox.w = data['width']
        bbox.l = data['length']
        bbox.o = data['heading']
        if 'score' in data.keys():
            bbox.s = data['score']
        return bbox
    
    @classmethod
    def copy_bbox(cls, bboxa, bboxb):
        bboxa.x = bboxb.x
        bboxa.y = bboxb.y
        bboxa.z = bboxb.z
        bboxa.l = bboxb.l
        bboxa.w = bboxb.w
        bboxa.h = bboxb.h
        bboxa.o = bboxb.o
        bboxa.s = bboxb.s
        return
    
    @classmethod
    def box2corners2d(cls, bbox):
        """ the coordinates for bottom corners
        """
        bottom_center = np.array([bbox.x, bbox.y, bbox.z - bbox.h / 2])
        cos, sin = np.cos(bbox.o), np.sin(bbox.o)
        pc0 = np.array([bbox.x + cos * bbox.l / 2 + sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 - cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc1 = np.array([bbox.x + cos * bbox.l / 2 - sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 + cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1
    
        return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]
    
    @classmethod
    def box2corners3d(cls, bbox):
        """ the coordinates for bottom corners
        """
        center = np.array([bbox.x, bbox.y, bbox.z])
        bottom_corners = np.array(BBox.box2corners2d(bbox))
        up_corners = 2 * center - bottom_corners
        corners = np.concatenate([up_corners, bottom_corners], axis=0)
        return corners.tolist()
    
    @classmethod
    def motion2bbox(cls, bbox, motion):
        result = deepcopy(bbox)
        result.x += motion[0]
        result.y += motion[1]
        result.z += motion[2]
        result.o += motion[3]
        return result
    
    @classmethod
    def set_bbox_size(cls, bbox, size_array):
        result = deepcopy(bbox)
        result.l, result.w, result.h = size_array
        return result
    
    @classmethod
    def set_bbox_with_states(cls, prev_bbox, state_array):
        prev_array = BBox.bbox2array(prev_bbox)
        prev_array[:4] += state_array[:4]
        prev_array[4:] = state_array[4:]
        bbox = BBox.array2bbox(prev_array)
        return bbox 
    
    @classmethod
    def box_pts2world(cls, ego_matrix, pcs):
        new_pcs = np.concatenate((pcs,
                                  np.ones(pcs.shape[0])[:, np.newaxis]),
                                  axis=1)
        new_pcs = ego_matrix @ new_pcs.T
        new_pcs = new_pcs.T[:, :3]
        return new_pcs
    
    @classmethod
    def edge2yaw(cls, center, edge):
        vec = edge - center
        yaw = np.arccos(vec[0] / np.linalg.norm(vec))
        if vec[1] < 0:
            yaw = -yaw
        return yaw
    
    @classmethod
    def bbox2world(cls, ego_matrix, box):
        # center and corners
        corners = np.array(BBox.box2corners2d(box))
        center = BBox.bbox2array(box)[:3][np.newaxis, :]
        center = BBox.box_pts2world(ego_matrix, center)[0]
        corners = BBox.box_pts2world(ego_matrix, corners)
        # heading
        edge_mid_point = (corners[0] + corners[1]) / 2
        yaw = BBox.edge2yaw(center[:2], edge_mid_point[:2])
        
        result = deepcopy(box)
        result.x, result.y, result.z = center
        result.o = yaw
        return result


class KalmanFilterMotionModel:
    def __init__(self, bbox: BBox, inst_type, time_stamp, covariance='default'):
        # the time stamp of last observation
        self.prev_time_stamp = time_stamp
        self.latest_time_stamp = time_stamp
        # define constant velocity model
        self.score = bbox.s
        self.inst_type = inst_type

        self.kf = KalmanFilter(dim_x=10, dim_z=7) 
        self.kf.x[:7] = BBox.bbox2array(bbox)[:7].reshape((7, 1))
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])     

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0]])
        
        self.kf.B = np.zeros((10, 1))                     # dummy control transition matrix

        # # with angular velocity
        # self.kf = KalmanFilter(dim_x=11, dim_z=7)       
        # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
        #                       [0,1,0,0,0,0,0,0,1,0,0],
        #                       [0,0,1,0,0,0,0,0,0,1,0],
        #                       [0,0,0,1,0,0,0,0,0,0,1],  
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0],
        #                       [0,0,0,0,0,0,0,1,0,0,0],
        #                       [0,0,0,0,0,0,0,0,1,0,0],
        #                       [0,0,0,0,0,0,0,0,0,1,0],
        #                       [0,0,0,0,0,0,0,0,0,0,1]])     

        # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
        #                       [0,1,0,0,0,0,0,0,0,0,0],
        #                       [0,0,1,0,0,0,0,0,0,0,0],
        #                       [0,0,0,1,0,0,0,0,0,0,0],
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0]])

        self.covariance_type = covariance
        # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
        self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.

        # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
        # self.kf.Q[7:, 7:] *= 0.01

        self.history = [bbox]
    
    def predict(self, time_stamp=None):
        """ For the motion prediction, use the get_prediction function.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        return

    def update(self, det_bbox: BBox, aux_info=None): 
        """ 
        Updates the state vector with observed bbox.
        """
        bbox = BBox.bbox2array(det_bbox)[:7]

        # full pipeline of kf, first predict, then update
        self.predict()

        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox[3] = new_theta

        predicted_theta = self.kf.x[3]
        if np.abs(new_theta - predicted_theta) > np.pi / 2.0 and np.abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi       
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if np.abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0: self.kf.x[3] += np.pi * 2
            else: self.kf.x[3] -= np.pi * 2

        #########################     # flip

        self.kf.update(bbox)
        self.prev_time_stamp = self.latest_time_stamp

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        if det_bbox.s is None:
            # self.score = self.score * 0.01
            pass
        else:
            self.score = det_bbox.s
        
        cur_bbox = self.kf.x[:7].reshape(-1).tolist()
        cur_bbox = BBox.array2bbox(cur_bbox + [self.score])
        self.history[-1] = cur_bbox
        return

    def get_prediction(self, time_delta):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        time_lag = time_delta
        self.latest_time_stamp = self.prev_time_stamp + time_delta
        self.kf.F = np.array([[1,0,0,0,0,0,0,time_lag,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,time_lag,0],
                              [0,0,1,0,0,0,0,0,0,time_lag],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])
        pred_x = self.kf.get_prediction()[0]
        if pred_x[3] >= np.pi: pred_x[3] -= np.pi * 2
        if pred_x[3] < -np.pi: pred_x[3] += np.pi * 2
        pred_bbox = BBox.array2bbox(pred_x[:7].reshape(-1))

        self.history.append(pred_bbox)
        return pred_bbox

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.history[-1]
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
    
    def sync_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp
        return
