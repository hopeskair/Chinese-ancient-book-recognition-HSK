# -*- encoding: utf-8 -*-
# Author: hushukai

import json
import random
import numpy as np
from PIL import Image
from multiprocessing import Queue, Process

from recognition_crnn.util import resize_text_image
from recognition_crnn.util import dense_tensor_from_list
from data_generator.generate_text_lines import create_text_line

from utils import CHAR2ID_DICT
from config import CRNN_TEXT_LINE_TAGS_FILE_H, CRNN_TEXT_LINE_TAGS_FILE_V
from config import TEXT_LINE_SIZE
from config import BATCH_SIZE_TEXT_LINE


TAGS_FILE_LIST = [(CRNN_TEXT_LINE_TAGS_FILE_H, "horizontal"), (CRNN_TEXT_LINE_TAGS_FILE_V, "vertical")]


def load_text_lines_batch(tags_file_list=TAGS_FILE_LIST, type="horizontal", batch_size=BATCH_SIZE_TEXT_LINE):
    img_label_list = []
    while True:
        for tags_file, text_type in tags_file_list:
            if text_type != type:
                continue
            
            with open(tags_file, "r", encoding="utf-8") as fr:
                for line in fr:
                    img_path, ids_str, chars = line.strip().split("\t")
                    ids_list = json.loads(ids_str)
                    PIL_img = Image.open(img_path)
                    PIL_img = PIL_img if PIL_img.mode == "L" else PIL_img.convert("L")
                    PIL_img = resize_text_image(PIL_img, obj_size=TEXT_LINE_SIZE, type=type)
                    np_img = np.asarray(PIL_img)
                    multiple_imgs = [(np_img, ids_list)] # * 3
                    img_label_list.extend(multiple_imgs)
                    
                    if len(img_label_list) > 1000:
                        random.shuffle(img_label_list)
                        while len(img_label_list) > 500:
                            yield pack_text_lines(img_label_list, batch_size, type, "white")


def create_text_lines_batch(type="horizontal", batch_size=BATCH_SIZE_TEXT_LINE):
    img_label_list = []
    while True:
        random_size = random.randint(5*TEXT_LINE_SIZE, 20*TEXT_LINE_SIZE)
        text_shape = (TEXT_LINE_SIZE, random_size) if type == "horizontal" else (random_size, TEXT_LINE_SIZE)
        
        PIL_text, chinese_char_and_box_list = create_text_line(text_shape, type=type)
        
        np_img = np.asarray(PIL_text)
        ids_list = [CHAR2ID_DICT[char] for char, box in chinese_char_and_box_list]
        multiple_imgs = [(np_img, ids_list)] * 3
        img_label_list.extend(multiple_imgs)

        if len(img_label_list) > 1000:
            random.shuffle(img_label_list)
            while len(img_label_list) > 500:
                yield pack_text_lines(img_label_list, batch_size, type, "white")


def pack_text_lines(img_label_list, batch_size, type, background="white"):
    raw_np_imgs = []
    raw_labels = []
    for _ in range(batch_size):
        np_img, ids_list = img_label_list.pop()
        raw_np_imgs.append(np_img)
        raw_labels.append(ids_list)
    
    img_shape = [np_img.shape[:2] for np_img in raw_np_imgs]
    max_h = max([h for (h, w) in img_shape])
    max_w = max([w for (h, w) in img_shape])
    label_len = [len(ids_list) for ids_list in raw_labels]
    
    if type in ("h", "horizontal"):
        assert max_h == TEXT_LINE_SIZE
        img_len = [w for (h, w) in img_shape]
    else:
        assert max_w == TEXT_LINE_SIZE
        img_len = [h for (h, w) in img_shape]
        
    batch_imgs = np.empty(shape=(batch_size, max_h, max_w), dtype=np.float32)
    if background == "white":
        batch_imgs.fill(255)
    elif background == "black":
        batch_imgs.fill(0)
    else:
        ValueError("Optional image background: 'white', 'black'.")
    
    for i, np_img in enumerate(raw_np_imgs):
        img_h, img_w = np_img.shape[:2]
        batch_imgs[i, :img_h, :img_w] = np_img
    
    batch_imgs = np.expand_dims(batch_imgs, axis=-1)
    img_len = np.asarray(img_len, dtype=np.int32)
    batch_labels = dense_tensor_from_list(raw_labels, dtype=np.int32, pad_value=0)
    label_len = np.asarray(label_len, dtype=np.int32)
    
    train_inputs = [batch_imgs, img_len, batch_labels, label_len]
    train_target = batch_labels
    
    return (train_inputs, train_target)
