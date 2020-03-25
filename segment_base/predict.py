# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K

from segment_base import visualize
from segment_base.model import work_net
from segment_base.data_pipeline import adjust_img_to_fixed_height, restore_original_angle
from segment_base.utils import get_segment_task_path, get_segment_task_params

from util import check_or_makedirs


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


def main(img_path, dest_dir, segment_task="book_page", text_type="horizontal", model_struc="densenet_gru", weights_path=""):
    check_or_makedirs(dest_dir)
    K.set_learning_phase(False)
    _, fixed_h, _ = get_segment_task_params(segment_task)
    _, ckpt_dir, logs_dir = get_segment_task_path(segment_task)
    if not os.path.exists(weights_path):
        weights_path = os.path.join(ckpt_dir, model_struc + "_ctpn_finished.h5")
        assert os.path.exists(weights_path)
    
    # 加载模型
    segment_model = work_net(stage="predict", segment_task=segment_task, text_type=text_type, model_struc=model_struc)
    segment_model.load_weights(weights_path, by_name=True)
    print("\nLoad model weights from %s\n" % weights_path)
    # ctpn_model.summary()
    
    count = 0
    for raw_np_img, img_name in load_images(img_path):
        count += 1

        np_img, _, scale_ratio = adjust_img_to_fixed_height(raw_np_img, fixed_h=fixed_h, segment_task="book_page", text_type="horizontal")
        batch_images = np_img[np.newaxis, :, :, :]
        
        split_positions, scores = segment_model.predict(x=batch_images)         # 模型预测

        text_type = text_type[0].lower()
        if (segment_task, text_type) in (("book_page", "h"), ("double_line", "h"), ("text_line", "v"), ("mix_line", "v")):
            _, split_positions = restore_original_angle(np_img=None, pred_split_positions=split_positions)
        
        split_positions = split_positions / scale_ratio
        image = visualize.draw_split_lines(raw_np_img, split_positions, scores)  # 可视化
        
        PIL_img = Image.fromarray(image)
        dest_path = os.path.join(dest_dir, os.path.splitext(img_name)[0] + ".jpg")
        PIL_img.save(dest_path, format="jpeg")
        print(count, "Finished: " + dest_path)


if __name__ == '__main__':
    # parse = argparse.ArgumentParser()
    # parse.add_argument("--image_path", type=str, default="", help="image path")
    # parse.add_argument("--dest_path", type=str, default="", help="detected result path")
    # parse.add_argument("--text_type", type=str, default="vertical", help="horizontal or vertical text")
    # parse.add_argument("--weight_path", type=str, default="", help="model weight path")
    # parse.add_argument("--use_side_refine", type=int, default=1, help="1: use side refine; 0 not use")
    # args = parse.parse_args(sys.argv[1:])

    segment_task = "book_page"
    root_dir, _, _ = get_segment_task_path(segment_task)
    dest_dir = os.path.join(root_dir, "samples")
    main(img_path, dest_dir, segment_task="book_page", text_type="horizontal")
