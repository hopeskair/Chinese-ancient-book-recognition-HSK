# -- coding: utf-8 --
# Author: hushukai
# Reference: https://github.com/qqwweee/keras-yolo3

from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, layers, models, regularizers

from networks.resnet import ResNet76V2_for_yolo as ResNet_for_yolo
from networks.resnext import ResNeXt76_for_yolo as ResNeXt_for_yolo
from networks.densenet import DenseNet73_for_yolo as DenseNet_for_yolo


def CNN(inputs, scope="densenet"):
    """cnn for YOLO_V3"""
    
    if "resnet" in scope:
        features_list = ResNet_for_yolo(inputs, scope)  # 1/8 size
    elif "resnext" in scope:
        features_list = ResNeXt_for_yolo(inputs, scope)  # 1/8 size
    elif "densenet" in scope:
        features_list = DenseNet_for_yolo(inputs, scope)  # 1/8 size
    else:
        ValueError("Optional CNN scope: 'resnet*', 'resnext*', 'densenet*'.")
    
    return features_list


def compose_funcs(*funcs):
    """
    Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    if funcs:
        return reduce(lambda f1, f2: lambda *args, **kwargs: f2(f1(*args, **kwargs)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


def Conv2D_BN_Leaky(*args):
    return compose_funcs(
        layers.Conv2D(*args, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(5e-4)),
        layers.BatchNormalization(epsilon=1.001e-5),
        layers.LeakyReLU(alpha=0.1))


def make_last_layers(x, num_filters, out_filters):
    x = compose_funcs(
        Conv2D_BN_Leaky(num_filters, (1, 1)),
        Conv2D_BN_Leaky(num_filters*2, (3, 3)),
        Conv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose_funcs(
        Conv2D_BN_Leaky(num_filters*2, (3, 3)),
        layers.Conv2D(out_filters, (1, 1), padding="same", kernel_regularizer=regularizers.l2(5e-4)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes, model_struc="densenet"):
    """Create YOLO_V3 model body in keras."""
    std_inputs = inputs / 255
    feature1, feature2, feature3 = CNN(std_inputs, scope=model_struc)   # Size 1/8, 1/16, 1/32
    
    x, y3 = make_last_layers(feature3, 512, num_anchors*(num_classes+5))  # Size 1/32
    
    x = compose_funcs(Conv2D_BN_Leaky(256, (1, 1)), layers.UpSampling2D(2))(x)
    x = layers.Concatenate()([x, feature2])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))         # Size 1/16

    x = compose_funcs(Conv2D_BN_Leaky(128, (1, 1)), layers.UpSampling2D(2))(x)
    x = layers.Concatenate()([x, feature1])
    _, y1 = make_last_layers(x, 128, num_anchors*(num_classes+5))         # Size 1/8
    
    return models.Model(inputs, [y1, y2, y3])


def yolo_head(y_pred, anchors, num_classes, input_shape, calc_loss=False):
    h, w = backend.shape(y_pred)[1:3]
    num_anchors = len(anchors)

    # Reshape to (batch, height, width, num_anchors, 4+1+num_classes)
    y_pred = backend.reshape(y_pred, [-1, h, w, num_anchors, 4 + 1 + num_classes])
    
    anchors = backend.reshape(backend.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_y = backend.tile(backend.reshape(backend.arange(h), [-1, 1, 1, 1]), [1, w, 1, 1])
    grid_x = backend.tile(backend.reshape(backend.arange(w), [1, -1, 1, 1]), [h, 1, 1, 1])
    grid = backend.concatenate([grid_x, grid_y], axis=-1)
    grid = backend.cast(grid, dtype="float32")
    
    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (backend.sigmoid(y_pred[..., 0:2]) + grid) / backend.cast([w, h], dtype="float32")         # 归一化值
    box_wh = backend.exp(y_pred[..., 2:4]) * anchors / backend.cast(input_shape[::-1], dtype="float32") # 归一化值
    box_confidence = backend.sigmoid(y_pred[..., 4:5])
    box_pred_probs = backend.sigmoid(y_pred[..., 5:])

    if calc_loss:
        return y_pred, grid, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_pred_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape):
    """Get corrected boxes"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = backend.cast(input_shape, "float32")

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = backend.concatenate([box_mins[..., 0:1],  # y_min
                                 box_mins[..., 1:2],  # x_min
                                 box_maxes[..., 0:1], # y_max
                                 box_maxes[..., 1:2]  # x_max
                                ])
    boxes *= backend.concatenate([input_shape, input_shape])
    return boxes


def yolo_boxes_and_scores(y_out, anchors, num_classes, input_shape):
    """Process Conv layer output.
    Note that a batch contains only one image when predicting.
    """
    box_xy, box_wh, box_confidence, box_pred_probs = yolo_head(y_out, anchors, num_classes, input_shape)
    
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape)
    boxes = backend.reshape(boxes, [-1, 4])
    
    box_scores = box_confidence * box_pred_probs
    box_scores = backend.reshape(box_scores, [-1, num_classes])
    
    return boxes, box_scores


def yolo_eval(y_outs, anchors, num_classes, score_thresh=0.6, iou_thresh=0.5, max_boxes=50):
    """Evaluate YOLO model on given inputs and return filtered boxes.
    Note that a batch contains only one image when predicting.
    """
    y_num = len(y_outs)
    anchor_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    input_shape = backend.shape(y_outs[0])[1:3] * 8
    
    boxes = []
    box_scores = []
    for i in range(y_num):
        _boxes, _scores = yolo_boxes_and_scores(y_outs[i], anchors[anchor_indices[i]], num_classes, input_shape)
        boxes.append(_boxes)
        box_scores.append(_scores)
    boxes = backend.concatenate(boxes, axis=0)              # (-1, 4)
    box_scores = backend.concatenate(box_scores, axis=0)    # (-1, num_classes)
    
    mask = box_scores >= score_thresh
    max_boxes_tensor = backend.constant(max_boxes, dtype='int32')
    boxes_selected = []
    scores_selected = []
    classes_selected = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        
        nms_indices = tf.image.non_max_suppression(class_boxes, class_scores, max_boxes_tensor, iou_thresh)
        
        class_boxes = backend.gather(class_boxes, nms_indices)
        class_scores = backend.gather(class_scores, nms_indices)
        classes = backend.ones_like(class_scores, 'int32') * c
        
        boxes_selected.append(class_boxes)
        scores_selected.append(class_scores)
        classes_selected.append(classes)
    
    boxes_selected = backend.concatenate(boxes_selected, axis=0)
    scores_selected = backend.concatenate(scores_selected, axis=0)
    classes_selected = backend.concatenate(classes_selected, axis=0)
    
    return boxes_selected, scores_selected, classes_selected


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """Pre-process true boxes to training input format.
    
    Parameters:
        true_boxes: array, shape=(bsize, num_boxes, 5)
            The last dimension is [x_min, y_min, x_max, y_max, class_id] of box.
        input_shape: array-like, hw, multiples of 32.
        anchors: array, shape=(n, 2), wh
        num_classes: integer

    Returns:
        y_true: list of array, shape like yolo_outputs, xywh are relative value.
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class_id must be less than num_classes'
    assert len(anchors) == 9
    y_num = len(anchors) // 3
    anchor_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    true_boxes = np.array(true_boxes, dtype=np.float32)
    input_shape = np.array(input_shape, dtype=np.int32)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]     # relative value
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]     # relative value

    bsize = true_boxes.shape[0]
    grid_shapes = [input_shape // [8, 16, 32][i] for i in range(y_num)]
    y_true = [np.zeros(shape=(bsize, grid_shapes[i][0], grid_shapes[i][1], 3, 4+1+num_classes), dtype=np.float32)
              for i in range(y_num)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)    # (1, num_anchors, 2), wh
    anchor_maxes = anchors / 2.             # relative to the center point
    anchor_mins = -anchor_maxes             # relative to the center point
    valid_mask = boxes_wh[..., 0] > 0       # padded invalid box looks like [0, 0, 0, 0, 0] in true_boxes.

    for i in range(bsize):
        # Discard zero rows.
        valid_wh = boxes_wh[i, valid_mask[i]]
        if len(valid_wh) == 0: continue
        
        # Expand dim to apply broadcasting.
        valid_wh = np.expand_dims(valid_wh, -2)   # (num_valid, 1, 2)
        box_maxes = valid_wh / 2.                 # relative to the center point
        box_mins = -box_maxes                     # relative to the center point
        
        # 计算目标box与每一个anchor的iou，确定应由哪一个anchor来预测这个box
        intersect_mins = np.maximum(box_mins, anchor_mins)                  # (num_valid, num_anchors, 2)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]        # (num_valid, num_anchors)
        box_area = valid_wh[..., 0] * valid_wh[..., 1]                      # (num_valid, 1)
        anchor_area = anchors[..., 0] * anchors[..., 1]                     # (1, num_anchors)
        iou = intersect_area / (box_area + anchor_area - intersect_area)    # (num_valid, num_anchors)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)                               # (num_valid,)

        for j in range(len(valid_wh)):
            for k in range(y_num):
                if best_anchor[j] in anchor_indices[k]:
                    _x = np.floor(true_boxes[i, j, 0] * grid_shapes[k][1]).astype('int32')
                    _y = np.floor(true_boxes[i, j, 1] * grid_shapes[k][0]).astype('int32')
                    _z = anchor_indices[k].index(best_anchor[j])
                    class_id = true_boxes[i, j, 4].astype('int32')
                    y_true[k][i, _y, _x, _z, 0:4] = true_boxes[i, j, 0:4]
                    y_true[k][i, _y, _x, _z, 4] = 1
                    y_true[k][i, _y, _x, _z, 5+class_id] = 1

    return y_true


def box_iou(pred_boxes, true_boxes):
    """Calculate iou value of pred_boxes and true_boxes.

    Parameters:
        pred_boxes: tensor, shape=(height, width, anchors, 4), xywh of pred_box are reletive value.
        true_boxes: tensor, shape=(n, 4),  xywh of true_box are reletive value.

    Returns:
        iou: tensor, shape=(height, width, anchors, n)
    """
    # Expand dim to apply broadcasting.
    pred_boxes = backend.expand_dims(pred_boxes, -2)    # (height, width, anchors, 1, 4)
    pred_boxes_xy = pred_boxes[..., 0:2]
    pred_boxes_wh = pred_boxes[..., 2:4]
    pred_boxes_wh_half = pred_boxes_wh / 2.
    pred_boxes_mins = pred_boxes_xy - pred_boxes_wh_half
    pred_boxes_maxes = pred_boxes_xy + pred_boxes_wh_half
    
    # Expand dim to apply broadcasting.
    true_boxes = backend.expand_dims(true_boxes, 0)     # (1, n, 4)
    true_boxes_xy = true_boxes[..., 0:2]
    true_boxes_wh = true_boxes[..., 2:4]
    true_boxes_wh_half = true_boxes_wh / 2.
    true_boxes_mins = true_boxes_xy - true_boxes_wh_half
    true_boxes_maxes = true_boxes_xy + true_boxes_wh_half

    intersect_mins = backend.maximum(pred_boxes_mins, true_boxes_mins)
    intersect_maxes = backend.minimum(pred_boxes_maxes, true_boxes_maxes)
    intersect_wh = backend.maximum(intersect_maxes - intersect_mins, 0.)        # (height, width, anchors, n, 2)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]                # (height, width, anchors, n)
    pred_boxes_area = pred_boxes_wh[..., 0] * pred_boxes_wh[..., 1]             # (height, width, anchors, 1)
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]             # (1, n)
    iou = intersect_area / (pred_boxes_area + true_boxes_area - intersect_area) # (height, width, anchors, n)

    return iou


def yolo_loss(args, anchors, num_classes, iou_thresh=0.5, print_loss=True):
    """ Calculate yolo loss.
    
    Parameters:
        args: list of *y_pred and *y_true.
            y_pred: the output of yolo_body
            y_true: the output of preprocess_true_boxes
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        iou_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns:
        loss: tensor, shape=(1,)
    """
    assert len(args) == 6 and len(anchors) == 9
    y_num = len(args) // 2  # 3
    y_outs = args[:y_num]   # Size 1/8, 1/16, 1/32
    y_true = args[y_num:]   # Size 1/8, 1/16, 1/32
    anchor_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    
    input_shape = backend.shape(y_outs[0])[1:3] * 8  # 原始的输入尺寸(h, w), tensor
    grid_shapes = [backend.shape(y_outs[i])[1:3] for i in range(y_num)]  # 划分网格
    bsize = backend.shape(y_outs[0])[0]  # batch size, tensor
    bsize = backend.cast(bsize, dtype="float32")
    
    loss = 0
    for i in range(y_num):
        true_mask  = y_true[i][..., 4:5]
        true_probs = y_true[i][..., 5:]

        y_pred, grid, pred_xy, pred_wh = yolo_head(y_pred=y_outs[i],
                                                   anchors=anchors[anchor_indices[i]],
                                                   num_classes=num_classes,
                                                   input_shape=input_shape,
                                                   calc_loss=True)
        pred_box = backend.concatenate([pred_xy, pred_wh])
        
        # Raw box to calculate loss.
        # Fix bug: grid_shapes重复相乘，另一个在preprocess_true_boxes中
        raw_true_xy = y_true[i][..., 0:2] * grid_shapes[i][::-1] - grid                                 # 计算偏移值(对应y_pred_yx的sigmoid结果)，y_true_yx为归一化值
        raw_true_wh = backend.log(y_true[i][..., 2:4] * input_shape[::-1] / anchors[anchor_indices[i]]) # 计算y_pred_hw的真值，y_true_hw为归一化值
        raw_true_wh = backend.switch(true_mask, raw_true_wh, backend.zeros_like(raw_true_wh))           # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]                                  # 2-h*w

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(dtype="float32", size=1, dynamic_size=True)
        true_mask_bool = backend.cast(true_mask, dtype="bool")
        
        def loop_condition(j, ignore_mask_unused):
            return j < bsize
        
        def loop_body(j, ignore_mask):
            true_box = tf.boolean_mask(y_true[i][j, ..., 0:4], true_mask_bool[j, ..., 0])    # shape=(-1, 4)
            iou = box_iou(pred_box[j], true_box)                                             # (height, width, anchors, n), pred_box和true_box均为归一化值
            best_iou = backend.max(iou, axis=-1)                                             # (height, width, anchors)
            ignore_mask = ignore_mask.write(j, backend.cast(best_iou<iou_thresh, "float32"))
            return j+1, ignore_mask
        
        _, ignore_mask = tf.while_loop(cond=loop_condition,
                                       body=loop_body,
                                       loop_vars=[0, ignore_mask])
        
        ignore_mask = ignore_mask.stack()                                                    # (bsize, height, width, anchors)
        ignore_mask = backend.expand_dims(ignore_mask, -1)                                   # (bsize, height, width, anchors, 1)
        
        # backend.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = true_mask * box_loss_scale * backend.binary_crossentropy(raw_true_xy, y_pred[..., 0:2], from_logits=True)
        wh_loss = true_mask * box_loss_scale * 0.5 * backend.square(raw_true_wh - y_pred[..., 2:4])
        confidence_loss = true_mask * backend.binary_crossentropy(true_mask, y_pred[..., 4:5], from_logits=True) + \
                          (1-true_mask) * backend.binary_crossentropy(true_mask, y_pred[..., 4:5], from_logits=True) * ignore_mask
        class_loss = true_mask * backend.binary_crossentropy(true_probs, y_pred[..., 5:], from_logits=True)

        xy_loss = backend.sum(xy_loss) / bsize
        wh_loss = backend.sum(wh_loss) / bsize
        confidence_loss = backend.sum(confidence_loss) / bsize
        class_loss = backend.sum(class_loss) / bsize
        
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        
        if print_loss:
            tf.print("Feature%d losses: total %.5f, xy %.5f, wh %.5f, confidence %.5f, class %.5f"
                     % (i+1, loss, xy_loss, wh_loss, confidence_loss, class_loss))
        
    # Here is necessary because of the slicing op in the process of
    # model.compile calculating losses: array_ops.shape(y_pred)[-1]
    loss = tf.expand_dims(loss, axis=-1)
        
    return loss
