# -*- encoding: utf-8 -*-
# Author: hushukai

import numpy as np
from PIL import Image

from config import TEXT_LINE_SIZE


def sparse_tensor_when_cpu(raw_batch_labels, dtype=np.int32):
    """
        Inspired from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    """
    indices = []
    values = []
    num_rows = 0
    num_cols = 0
    
    for batch_id, labels in enumerate(raw_batch_labels):
        indices.extend(zip([batch_id] * len(labels), [col_id for col_id in range(len(labels))]))
        values.extend(labels)
        num_rows += 1
        num_cols = max(num_cols, len(labels))
    
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    dense_shape = np.asarray([num_rows, num_cols], dtype=np.int64)
    sparse_tensor = (indices, values, dense_shape)

    return sparse_tensor


def dense_tensor_when_gpu(raw_batch_labels, dtype=np.int32, pad_value=0):
    
    indices, values, dense_shape = sparse_tensor_when_cpu(raw_batch_labels, dtype=dtype)

    dense_tensor = np.empty(dense_shape, dtype=dtype)
    dense_tensor.fill(value=pad_value)
    dense_tensor[indices[:,0], indices[:,1]] = values
    
    return dense_tensor
    

def resize_text_image(PIL_img, obj_size=TEXT_LINE_SIZE, type="horizontal"):
    """Resize text image to the obj_size.
    """
    if type.lower() in ("h", "horizontal"):
        type = "h"
    elif type.lower() in ("v", "vertical"):
        type = "v"
    else:
        ValueError("Optional types: 'h', 'horizontal', 'v', 'vertical'.")
    
    w, h = PIL_img.size
    
    if type == "h":
        ratio = obj_size/h
        PIL_img = PIL_img.resize(size=(int(w*ratio), obj_size), resample=Image.BICUBIC)
    else:
        ratio = obj_size/w
        PIL_img = PIL_img.resize(size=(obj_size, int(h*ratio)), resample=Image.BICUBIC)
    
    return PIL_img
