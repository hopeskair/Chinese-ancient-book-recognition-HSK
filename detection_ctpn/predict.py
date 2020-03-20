# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K

from . import visualize
from .model import work_net
from .utils import np_utils
from .text_connector.detectors import TextDetector
from .data_pipeline import restore_text_horizontal_to_vertical
from .data_pipeline import adjust_img_into_model

from util import check_or_makedirs
from config import CTPN_ROOT_DIR, CTPN_CKPT_DIR
from config import BOOK_PAGE_FIXED_SIZE


def load_images(img_path):
    if os.path.isfile(img_path) and os.path.splitext(img_path)[1] in (".jpg", ".png", ".gif"):
        img_paths = [img_path,]
    if os.path.isdir(img_path):
        img_paths = [os.path.join(img_path, file) for file in os.listdir(img_path)
                     if os.path.splitext(file)[1] in (".jpg", ".png", ".gif")]
    for img_path in img_paths:
        PIL_img = Image.open(img_path)
        if PIL_img.mode != "RGB":
            PIL_img = PIL_img.convert("RGB")
        np_img = np.array(PIL_img)
        
        yield np_img, os.path.basename(img_path)
        

TRAIN_FINISHED_WEIGHTS = os.path.join(CTPN_CKPT_DIR, "densenet_gru_vertical_ctpn_finished.h5")


def main(img_path, dest_dir, text_type="vertical", weights_path=TRAIN_FINISHED_WEIGHTS):
    check_or_makedirs(dest_dir)
    K.set_learning_phase(False)
    assert os.path.exists(weights_path) and text_type in weights_path
    
    # 加载模型
    ctpn_model = work_net("predict", batch_size=1, text_type=text_type, model_struc="densenet_gru")
    ctpn_model.load_weights(weights_path, by_name=True)
    print("\nLoad model weights from %s\n" % weights_path)
    # ctpn_model.summary()
    
    count = 0
    for np_img, img_name in load_images(img_path):
        count += 1
        
        np_img = adjust_img_into_model(np_img, text_type=text_type, fixed_size=BOOK_PAGE_FIXED_SIZE)
        batch_images = np_img[np.newaxis, :, :, :]
        
        # 模型预测
        boxes, scores = ctpn_model.predict(x=batch_images)
        
        boxes = np_utils.remove_pad(boxes[0])
        scores = np_utils.remove_pad(scores[0])[:, 0]
        
        # 文本行检测器
        textdetector = TextDetector(DETECT_MODE='H')
        text_lines = textdetector.detect(boxes, scores, np_img.shape[:2])
        
        # 可视化
        np_img = visualize.draw_text_lines(np_img, text_lines)
        
        if text_type.lower() in ("v", "vertical"):
            np_img = restore_text_horizontal_to_vertical(np_img)
        
        PIL_img = Image.fromarray(np_img)
        dest_path = os.path.join(dest_dir, os.path.splitext(img_name)[0] + ".jpg")
        PIL_img.save(dest_path, format="jpeg")
        print(count, "Finished: " + dest_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_path", type=str, default="", help="image path")
    parse.add_argument("--dest_path", type=str, default="", help="detected result path")
    parse.add_argument("--text_type", type=str, default="vertical", help="horizontal or vertical text")
    parse.add_argument("--weight_path", type=str, default="", help="model weight path")
    parse.add_argument("--use_side_refine", type=int, default=1, help="1: use side refine; 0 not use")
    args = parse.parse_args(sys.argv[1:])
    
    dest_dir = os.path.join(CTPN_ROOT_DIR, "samples")
    main(img_path=args.image_path, dest_dir=dest_dir, text_type="vertical", weights_path=TRAIN_FINISHED_WEIGHTS)
