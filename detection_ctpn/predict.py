# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K

from . import visualize
from .model import ctpn_net
from .utils import np_utils
from .text_connector.detectors import TextDetector
from .data_pipeline import change_text_vertical_to_horisontal
from .data_pipeline import restore_text_horizontal_to_vertical

from util import check_or_makedirs
from config import CTPN_ROOT_DIR

def load_images(img_path):
    if os.path.isfile(img_path) and os.path.splitext(img_path)[1] in ("jpg", "png", "gif"):
        img_paths = [img_path,]
    if os.path.isdir(img_path):
        img_paths = [os.path.join(img_path, file) for file in os.listdir(img_path)
                     if os.path.splitext(file)[1] in ("jpg", "png", "gif")]
    for img_path in img_paths:
        PIL_img = Image.open(img_path)
        if PIL_img.mode != "L":
            PIL_img = PIL_img.convert("L")
        np_img = np.array(PIL_img)
        h, w = np_img.shape[:2]
        np.pad(np_img, pad_width=((0, -h%16), (0, -w%16)), mode="maximum")
        yield np_img, os.path.basename(img_path)
        

def main(img_path, dest_dir, model_type="vertical", weights_path=""):
    check_or_makedirs(dest_dir)
    K.set_learning_phase(False)
    assert os.path.exists(weights_path) and model_type in weights_path

    # 加载模型
    ctpn_model = ctpn_net("predict")
    ctpn_model.load_weights(weights_path, by_name=True)
    # m.summary()
    
    for np_img, img_name in load_images(img_path):
        if model_type.lower() in ("v", "vertical"):
            np_img, _ = change_text_vertical_to_horisontal(np_img)
        
        batch_images = np_img[np.newaxis, :, :, np.newaxis]
        
        # 模型预测
        boxes, scores = ctpn_model.predict(x=batch_images)
        
        boxes = np_utils.remove_pad(boxes[0])[:, :4]
        scores = np_utils.remove_pad(scores[0])[:, 0]
        
        # 文本行检测器
        textdetector = TextDetector(DETECT_MODE='H')
        text_lines = textdetector.detect(boxes, scores, np_img.shape[:2])
        
        # 可视化保存图像
        np_img = visualize.draw_text_lines(np_img, text_lines)
        if model_type.lower() in ("v", "vertical"):
            np_img = restore_text_horizontal_to_vertical(boxes=np_img)
        
        PIL_img = Image.fromarray(np_img)
        PIL_img.save(os.path.join(dest_dir, img_name), format="jpeg")


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_path", type=str, default="", help="image path")
    parse.add_argument("--weight_path", type=str, default="", help="weight path")
    parse.add_argument("--use_side_refine", type=int, default=0, help="1: use side refine; 0 not use")
    argments = parse.parse_args(sys.argv[1:])

    dest_dir = os.path.join(CTPN_ROOT_DIR, "samples")
    main(img_path, dest_dir, model_type="vertical", weights_path="")
