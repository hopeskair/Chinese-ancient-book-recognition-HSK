# -*- coding: utf-8 -*-
# Author: hushukai

import tensorflow as tf
from tensorflow.keras import layers

from detection_ctpn.utils.tf_utils import pad_to_fixed_size


def apply_regress(deltas, side_deltas, anchors, use_side_refine=False):
    """应用回归目标到边框, 垂直中心点偏移和高度缩放
    Parameter:
        deltas: 回归目标 [num_anchors,(dy,dh)]
        side_deltas: 回归目标 [num_anchors,(dx)]
        anchors: [num_anchors, (x1, y1, x2, y2)]
        use_side_refine: 是否应用侧边回归
    """
    # 高度和宽度
    w = anchors[:, 2] - anchors[:, 0]
    h = anchors[:, 3] - anchors[:, 1]

    # 中心点坐标
    cx = (anchors[:, 0] + anchors[:, 2]) * 0.5
    cy = (anchors[:, 1] + anchors[:, 3]) * 0.5

    deltas = tf.concat([deltas, side_deltas], axis=1)
    
    # 回归系数
    deltas *= tf.constant([0.1, 0.2, 0.1])
    dy, dh, dx = deltas[:, 0], deltas[:, 1], deltas[:, 2]

    # 中心坐标回归
    cy += dy * h
    # 侧边精调
    cx += dx * w
    # 高度回归
    h *= tf.exp(dh)

    # 转为x1, y1, x2, y2
    x1 = tf.maximum(cx - w * 0.5, 0.)  # 限制在窗口内,修复后继节点找不到对应的前驱节点
    x2 = cx + w * 0.5
    y1 = cy - h * 0.5
    y2 = cy + h * 0.5

    if use_side_refine:
        return tf.stack([x1, y1, x2, y2], axis=1)
    else:
        return tf.stack([anchors[:, 0], y1, anchors[:, 2], y2], axis=1)


def get_valid_predicts(deltas, side_deltas, class_logits, valid_anchors_indices):
    
    deltas = tf.gather(deltas, valid_anchors_indices)
    side_deltas = tf.gather(side_deltas, valid_anchors_indices)
    class_logits = tf.gather(class_logits, valid_anchors_indices)
    
    return [deltas, side_deltas, class_logits]


def nms(boxes, scores, class_logits, max_outputs=2000, iou_thresh=0.5, score_thresh=0.1, name=None):
    """非极大抑制
    Parameter:
        boxes: [num_boxes, 4], 浮点型Tensor
        scores: [num_boxes], 浮点型Tensor
        class_logits: [num_boxes, num_classes]
        max_outputs: 非极大抑制选择的框的最大数量
        iou_thresh: IOU 阈值
        score_thresh:  过滤掉低于阈值的边框
    """
    indices = tf.image.non_max_suppression(boxes, scores,
                                           max_output_size=max_outputs,
                                           iou_threshold=iou_thresh,
                                           score_threshol=score_thresh,
                                           name=name)
    
    output_boxes = tf.gather(boxes, indices)
    box_scores = tf.gather(scores, indices)
    class_logits = tf.gather(class_logits, indices)

    # padding到固定大小
    output_boxes = pad_to_fixed_size(output_boxes, max_outputs),
    box_scores = pad_to_fixed_size(box_scores[:, tf.newaxis], max_outputs),
    class_logits = pad_to_fixed_size(class_logits, max_outputs)
    
    return [output_boxes, box_scores, class_logits]


class TextProposal(layers.Layer):
    """生成候选框"""
    def __init__(self, nms_max_outputs=2000, cls_score_thresh=0.7, iou_thresh=0.3, use_side_refine=False, **kwargs):
        self.nms_max_outputs = nms_max_outputs
        self.cls_score_thresh = cls_score_thresh
        self.iou_thresh = iou_thresh
        self.use_side_refine = use_side_refine
        super(TextProposal, self).__init__(**kwargs)

    def __call__(self, inputs, **kwargs):
        """用回归值生成边框，并使用nms筛选
        Parameter: inputs:
            deltas: [batch_size, N, (dy,dh)],  N是所有的anchors数量
            side_deltas: [batch_size, N, (dx)]
            class_logits: [batch_size, N, num_classes]
            valid_anchors: [anchor_num, (x1, y1, x2, y2)]
            valid_indices: [anchor_num]
        """
        deltas, side_deltas, class_logits, valid_anchors, valid_indices = inputs
        
        # 只考虑有效anchor的预测结果
        options = {"valid_anchors_indices": valid_indices}
        deltas, side_deltas, class_logits = tf.map_fn(fn=lambda args: get_valid_predicts(*args, **options),
                                                      elems=[deltas, side_deltas, class_logits],
                                                      dtype=[tf.float32, tf.float32, tf.float32])
        
        # 转化为分类score
        class_scores = tf.nn.softmax(logits=class_logits, axis=-1)
        fg_scores = tf.reduce_max(class_scores[..., 1:], axis=-1)   # bg_pos:0, fg_pos:1

        # 应用边框回归
        options = {"anchors": valid_indices, "use_side_refine":self.use_side_refine}
        proposals = tf.map_fn(fn=lambda args: apply_regress(*args, anchors=valid_anchors),
                              elems=[deltas, side_deltas],
                              dtype=[tf.float32, tf.float32])
        
        # 非极大抑制
        options = {"max_outputs": self.nms_max_outputs,
                   "iou_thresh": self.iou_thresh,
                   "score_thresh": self.cls_score_thresh,}
        outputs = tf.map_fn(fn=lambda x: nms(*x, **options),
                            elems=[proposals, fg_scores, class_logits],
                            dtype=[tf.float32] * 3)
        
        return outputs

    def compute_output_shape(self, input_shape):
        """多输出，__call__返回值必须是列表"""
        return [(input_shape[0][0], self.nms_max_outputs, 4 + 1),    # proposal_boxes, (x1,y1,x2,y2, padding_flag)
                (input_shape[0][0], self.nms_max_outputs, 1 + 1),     # proposal_boxes, (score, padding_flag)
                (input_shape[0][0], self.nms_max_outputs, input_shape[2][-1])]
