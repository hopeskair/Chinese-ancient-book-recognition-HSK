# -*- coding: utf-8 -*-
# Author: hushukai

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K

from segment_base import visualize
from segment_base.model import work_net
from segment_base.data_pipeline import adjust_img_to_fixed_height, pack_a_batch
from segment_base.data_pipeline import rotate_90_degrees, restore_original_angle
from segment_base.utils import remove_pad_np
from segment_base.utils import get_segment_task_path, get_segment_task_params

from util import check_or_makedirs


def convert_images(images):
    if isinstance(images, np.ndarray):
        np_img_list = [images,]
    elif isinstance(images, Image.Image):
        np_img_list = [np.array(images),]
    elif isinstance(images, list):
        if isinstance(images[0], np.ndarray):
            np_img_list = images
        elif isinstance(images[0], Image.Image):
            np_img_list = [np.array(PIL_img) for PIL_img in images]
        else:
            ValueError("The element type of images list is wrong!")
    else:
        ValueError("The type of images is wrong!")
    return np_img_list


def load_images(img_paths):
    if isinstance(img_paths, str):
        if os.path.isfile(img_paths):
            assert os.path.splitext(img_paths)[1] in (".jpg", ".png", ".gif")
            img_paths = [img_paths,]
        else:
            assert os.path.isdir(img_paths)
            img_paths = [os.path.join(img_paths, file) for file in os.listdir(img_paths)
                         if os.path.splitext(file)[1] in (".jpg", ".png", ".gif")]
    else:
        assert isinstance(img_paths, list)
        img_paths = [img_path for img_path in img_paths
                     if os.path.splitext(img_path)[1] in (".jpg", ".png", ".gif")]
    
    np_img_list, img_name_list = [], []
    for img_path in img_paths:
        PIL_img = Image.open(img_path)
        if PIL_img.mode != "RGB":
            PIL_img = PIL_img.convert("RGB")
        np_img = np.array(PIL_img)
        np_img_list.append(np_img)
        img_name_list.append(os.path.basename(img_path))
    
    return np_img_list, img_name_list


def model_weights_path(weights, segment_task, model_struc="densenet_gru"):
    _, ckpt_dir, _ = get_segment_task_path(segment_task)
    
    if isinstance(weights, str):
        if os.path.exists(weights):
            weights_path = weights
        elif os.path.exists(os.path.join(ckpt_dir, weights)):
            weights_name = weights
            weights_path = os.path.join(ckpt_dir, weights_name)
        else:
            files = os.listdir(ckpt_dir)
            if len(files) == 1 and files[0].endswith(".h5"):
                weights_path = os.path.join(ckpt_dir, files[0])
            else:
                weights_path = os.path.join(ckpt_dir, segment_task + "_segment_" + model_struc + "_finished.h5")
                assert os.path.exists(weights_path)
    else:
        assert isinstance(weights, int)
        weights_id = "{:04d}".format(weights)
        weights_path = os.path.join(ckpt_dir, segment_task + "_segment_" + model_struc + "_" + weights_id + ".h5")
        assert os.path.exists(weights_path)
    
    return weights_path


def segment_predict(images=None,
                    img_paths=None,
                    dest_dir=None,
                    segment_model=None,
                    segment_task="book_page",
                    text_type="horizontal",
                    model_struc="densenet_gru",
                    weights=""):
    
    # images
    if images is not None:
        np_img_list = convert_images(images)
        img_name_list = [str(i)+".jpg" for i in range(len(np_img_list))]
    else:
        assert img_paths is not None
        np_img_list, img_name_list = load_images(img_paths)
    
    # model
    if segment_model is None:
        K.set_learning_phase(False)
        weights_path = model_weights_path(weights, segment_task, model_struc)
        
        # 加载模型
        segment_model = work_net(stage="predict", segment_task=segment_task, text_type=text_type, model_struc=model_struc)
        segment_model.load_weights(weights_path, by_name=True)
        print("\nLoad model weights from %s\n" % weights_path)
        # segment_model.summary()
    
    # predict
    batch_size, fixed_h, feat_stride = get_segment_task_params(segment_task)
    text_type = text_type[0].lower()
    split_positions_list, scores_list = [], []
    for i in range(0, len(np_img_list), batch_size):
        _images_list, _scale_ratio_list = [], []
        for np_img in np_img_list[i:i+batch_size]:
            np_img, _, scale_ratio = adjust_img_to_fixed_height(np_img, None, fixed_h, segment_task, text_type)
            _images_list.append(np_img)
            _scale_ratio_list.append(scale_ratio)
        batch_images, real_images_width, _ = pack_a_batch(_images_list, None, feat_stride, background="white")

        nms_split_positions, nms_scores = segment_model.predict(x=[batch_images, real_images_width])  # 模型预测
        
        for j in range(len(batch_images)):
            scores = remove_pad_np(nms_scores[j])[:, 0]
            split_positions = remove_pad_np(nms_split_positions[j])
            split_positions = split_positions / _scale_ratio_list[j]
            if (segment_task, text_type) in (("book_page", "h"), ("double_line", "h"), ("text_line", "v"), ("mix_line", "v")):
                _, split_positions = restore_original_angle(np_img=None, pred_split_positions=split_positions)
            split_positions_list.append(split_positions)
            scores_list.append(scores)
    
    # draw
    if dest_dir is not None:
        check_or_makedirs(dest_dir)
        for i in range(len(np_img_list)):
            if (segment_task, text_type) in (("book_page", "h"), ("double_line", "h"), ("text_line", "v"), ("mix_line", "v")):
                np_img, split_positions = rotate_90_degrees(np_img_list[i], split_positions_list[i])
            
            np_img = visualize.draw_split_lines(np_img, split_positions, scores_list[i])  # 可视化

            if (segment_task, text_type) in (("book_page", "h"), ("double_line", "h"), ("text_line", "v"), ("mix_line", "v")):
                np_img, _ = restore_original_angle(np_img)
            
            PIL_img = Image.fromarray(np_img)
            dest_path = os.path.join(dest_dir, os.path.splitext(img_name_list[i])[0] + ".jpg")
            PIL_img.save(dest_path, format="jpeg")
            print(i, "Finished: " + dest_path)
    
    return np_img_list, img_name_list, split_positions_list, scores_list


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
    segment_predict(img_paths="", dest_dir=dest_dir, segment_task="book_page", text_type="horizontal")
