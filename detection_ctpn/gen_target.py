# -*- coding: utf-8 -*-
# Author: hushukai

from tensorflow.keras import layers
import tensorflow as tf

from detection_ctpn.utils.tf_utils import remove_pad, pad_to_fixed_size


def compute_iou(gt_boxes, anchors):
    """
    Parameter:
        gt_boxes: [M,(x1,y1,x2,y2)]
        anchors:  [N,(x1,y1,x2,y2)]
    Return:
        IoU:[M, N]
    """
    gt_boxes = tf.expand_dims(gt_boxes, axis=1) # [M,1,4]
    anchors = tf.expand_dims(anchors, axis=0)   # [1,N,4]
    
    # 交集
    intersect_w = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 2], anchors[:, :, 2]) -
                             tf.maximum(gt_boxes[:, :, 0], anchors[:, :, 0]))
    intersect_h = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 3], anchors[:, :, 3]) -
                             tf.maximum(gt_boxes[:, :, 1], anchors[:, :, 1]))
    intersect = intersect_h * intersect_w

    # 计算面积
    area_gt = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0]) * (gt_boxes[:, :, 3] - gt_boxes[:, :, 1])
    area_anchor = (anchors[:, :, 2] - anchors[:, :, 0]) * (anchors[:, :, 3] - anchors[:, :, 1])
    
    # 计算并集
    union = area_gt + area_anchor - intersect
    # 交并比
    iou = tf.divide(intersect, union, name='regress_target_iou')
    
    return iou


def ctpn_regress_target(anchors, gt_boxes):
    """
    Parameter:
        anchors: [N,(x1,y1,x2,y2)]
        gt_boxes: [N,(x1,y1,x2,y2)]
    Return:
        regress_target: [N, (dy, dh, dx)]  dx代表侧边改善
    """
    # anchor高度
    h = anchors[:, 3] - anchors[:, 1]
    # gt_box高度
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]

    # anchor中心点y坐标
    center_y = (anchors[:, 1] + anchors[:, 3]) * 0.5
    # gt_box中心点y坐标
    gt_center_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5

    # 计算回归目标
    dy = (gt_center_y - center_y) / h
    dh = tf.log(gt_h / h)
    
    dx = side_regress_target(anchors, gt_boxes)  # 侧边改善
    regress_target = tf.stack([dy, dh, dx], axis=1)
    regress_target /= tf.constant([0.1, 0.2, 0.1])

    return regress_target


def side_regress_target(anchors, gt_boxes):
    """
    Parameter:
        anchors: [N,(x1,y1,x2,y2)]
        gt_boxes: [N,(x1,y1,x2,y2)]
    Return:
        dx: 侧边改善回归目标
    """
    w = anchors[:, 2] - anchors[:, 0]  # 实际是固定长度16
    center_x = (anchors[:, 0] + anchors[:, 2]) * 0.5
    gt_center_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    
    # 侧边框移动到gt的侧边，相当于中心点偏移的两倍;不是侧边的anchor 偏移为0. ???
    dx = (gt_center_x - center_x) * 2 / w
    
    return dx


def ctpn_target(gt_boxes, valid_anchors, valid_indices, train_anchors_num=256, positive_ratio=0.5):
    """
    生成单张图像的ctpn回归目标
        a)正样本：与gt_box iou大于0.7的anchor, 或者与gt_box iou最大的那个anchor
        b)保证所有gt_box都有anchor与之对应
    Note: gt_boxes shape [gt_num, ( x1, y1, x2, y2, class_id, padding_flag)]
    """
    # 获取真正的GT,去除标签位
    gt_boxes = remove_pad(gt_boxes)
    gt_boxes, gt_cls = gt_boxes[:,:4], gt_boxes[:,4]
    gt_cls = tf.ones_like(gt_cls) # fg:1
    
    gt_num = tf.shape(gt_boxes)[0]
    
    # 计算iou值
    iou = compute_iou(gt_boxes, valid_anchors)

    # 每个anchor与所有gt_box最大的iou, iou>0.7时为正样本
    anchors_iou_max = tf.reduce_max(iou, axis=0, keep_dims=True)
    anchors_iou_max = tf.where(tf.greater_equal(anchors_iou_max, 0.7), anchors_iou_max, -1)
    anchors_iou_max_bool = tf.equal(iou, anchors_iou_max)
    
    anchors_no_goal = tf.where(tf.less(anchors_iou_max, 0.7), True, False)
    
    # 与gt_box iou最大的anchor是正样本(可能有多个)
    gt_iou_max = tf.reduce_max(iou, axis=1, keep_dims=True)
    gt_iou_max_bool = tf.equal(iou, gt_iou_max)
    
    gt_iou_max_bool = tf.logical_and(gt_iou_max_bool, anchors_no_goal)
    
    # 合并两部分正样本索引
    positive_bool_matrix = tf.logical_or(anchors_iou_max_bool, gt_iou_max_bool)
    
    # 获取iou值用于度量, 用作度量的必须是浮点类型
    gt_match_min_iou = tf.reduce_min(tf.boolean_mask(iou, positive_bool_matrix))    # 标量
    gt_match_mean_iou = tf.reduce_mean(tf.boolean_mask(iou, positive_bool_matrix))  # 标量
    
    # 正样本索引
    positive_indices = tf.where(positive_bool_matrix)  # [positive_num, (gt索引号, anchor索引号)]
    
    # 采样正样本
    positive_num = tf.minimum(tf.shape(positive_indices)[0], int(train_anchors_num * positive_ratio))
    positive_indices = tf.random_shuffle(positive_indices)[:positive_num]
    
    # 获取正样本anchor和对应的gt_box
    positive_gt_indices = positive_indices[:, 0]
    positive_anchor_indices = positive_indices[:, 1]
    
    positive_anchors = tf.gather(valid_anchors, positive_anchor_indices)
    positive_gt_boxes = tf.gather(gt_boxes, positive_gt_indices)
    positive_gt_cls = tf.gather(gt_cls, positive_gt_indices)
    
    # 计算回归目标
    deltas = ctpn_regress_target(positive_anchors, positive_gt_boxes)
    
    # 获取负样本 iou < 0.5
    negative_bool = tf.less(tf.reduce_max(iou, axis=0), 0.5)
    positive_bool = tf.reduce_any(positive_bool_matrix, axis=0)
    negative_bool = tf.logical_and(negative_bool, tf.logical_not(positive_bool))

    # 采样负样本
    negative_indices = tf.where(negative_bool)  # [negative_num, (anchor索引号)]
    negative_num = tf.minimum(tf.shape(negative_indices)[0], train_anchors_num - positive_num)
    negative_indices = tf.random_shuffle(negative_indices[:, 0])[:negative_num]

    negative_gt_cls = tf.zeros([negative_num])  # bg:0
    negative_deltas = tf.zeros([negative_num, 3])

    # 合并正负样本
    deltas = tf.concat([deltas, negative_deltas], axis=0, name='ctpn_target_deltas')
    class_ids = tf.concat([positive_gt_cls, negative_gt_cls], axis=0, name='ctpn_target_class_ids')
    anchor_indices_sampled = tf.concat([positive_anchor_indices, negative_indices], axis=0, name='ctpn_train_anchor_indices')
    anchor_indices_sampled = tf.gather(valid_indices, anchor_indices_sampled)  # 对应到全局索引
    
    # padding
    deltas = pad_to_fixed_size(deltas, train_anchors_num)
    class_ids = pad_to_fixed_size(tf.expand_dims(class_ids, 1), train_anchors_num)
    anchor_indices_sampled = pad_to_fixed_size(tf.expand_dims(anchor_indices_sampled, 1), train_anchors_num)

    # 用作度量的必须是浮点类型
    return [deltas, class_ids, anchor_indices_sampled,
            tf.cast(gt_num, dtype=tf.float32),
            tf.cast(positive_num, dtype=tf.float32),
            tf.cast(negative_num, dtype=tf.float32),
            gt_match_min_iou,
            gt_match_mean_iou]


class CtpnTarget(layers.Layer):
    def __init__(self, train_anchors_num=256, positive_ratio=0.5, **kwargs):
        self.train_anchors_num = train_anchors_num
        self.positive_ratio = positive_ratio
        super(CtpnTarget, self).__init__(**kwargs)

    def __call__(self, inputs, **kwargs):
        """
        parameter: inputs
            gt_boxes: [batch_size, num_gt_boxes,( x1, y1, x2, y2, class_id, padding_flag)]
            valid_anchors: [anchor_num, (x1, y1, x2, y2)]
            valid_indices: [anchor_num]
        """
        gt_boxes, valid_anchors, valid_indices = inputs
        
        options = {"valid_anchors": valid_anchors,
                   "valid_indices": valid_indices,
                   "train_anchors_num": self.train_anchors_num,
                   "positive_ratio": self.positive_ratio}
        
        outputs = tf.map_fn(fn=lambda i_gt_boxes: ctpn_target(i_gt_boxes, **options),
                            elems=gt_boxes,
                            dtype=tf.float32)
        
        return outputs
    
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.train_anchors_num, 4),  # deltas (dy, dh, dx, padding_flag)
                (input_shape[0], self.train_anchors_num, 2),  # cls (class_id, padding_flag)
                (input_shape[0], self.train_anchors_num, 2),  # samples_indices (samples_index, padding_flag)
                (input_shape[0],),  # gt_num
                (input_shape[0],),  # positive_num
                (input_shape[0],),  # negative_num
                (input_shape[0],),  # gt_match_min_iou
                (input_shape[0],)]  # gt_match_mean_iou