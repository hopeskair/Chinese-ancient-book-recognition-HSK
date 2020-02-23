# -*- encoding: utf-8 -*-
# Author: hushukai

import json
import random
import numpy as np
from PIL import Image
from multiprocessing import Queue, Process

from recognition_crnn.utils import resize_text_image
from recognition_crnn.utils import sparse_tensor_when_cpu, dense_tensor_when_gpu
from data_generator.generate_text_lines import create_text_line

from utils import CHAR2ID_DICT
from config import CRNN_TEXT_LINE_TAGS_FILE_H, CRNN_TEXT_LINE_TAGS_FILE_V
from config import TEXT_LINE_SIZE
from config import BATCH_SIZE_TEXT_LINE


def load_text_lines_batch(tags_file_list, queue, batch_size=BATCH_SIZE_TEXT_LINE):
    h_imgs_list = []
    v_imgs_list = []
    while True:
        for tags_file, type in tags_file_list:
            if type in ("h", "horizontal"):
                imgs_list = h_imgs_list
            elif type in ("v", "vertical"):
                imgs_list = v_imgs_list
            else:
                ValueError("Optional text types: 'h', 'horizontal', 'v', 'vertical'.")
            
            with open(tags_file, "r", encoding="utf-8") as fr:
                for line in fr:
                    img_path, ids_str, chars = line.strip().split("\t")
                    ids_list = json.loads(ids_str)
                    PIL_img = Image.open(img_path)
                    PIL_img = PIL_img if PIL_img.mode == "L" else PIL_img.convert("L")
                    PIL_img = resize_text_image(PIL_img, obj_size=TEXT_LINE_SIZE, type=type)
                    np_img = np.asarray(PIL_img)
                    multiple_imgs = [(np_img, ids_list)] * 3
                    imgs_list.extend(multiple_imgs)
                    
                    if len(imgs_list) > 10000:
                        random.shuffle(imgs_list)
                        while len(imgs_list) > 5000:
                            queue.put(pack_text_lines(imgs_list, batch_size, type, "white"))


def create_text_lines_batch(queue, batch_size=BATCH_SIZE_TEXT_LINE):
    h_imgs_list = []
    v_imgs_list = []
    while True:
        type = random.choice(["horizontal", "vertical"])
        random_size = random.randint(5*TEXT_LINE_SIZE, 20*TEXT_LINE_SIZE)
        text_shape = (TEXT_LINE_SIZE, random_size) if type == "horizontal" else (random_size, TEXT_LINE_SIZE)
        imgs_list = h_imgs_list if type == "horizontal" else v_imgs_list
        
        PIL_text, chinese_char_and_box_list = create_text_line(text_shape, type=type)
        
        np_img = np.asarray(PIL_text)
        ids_list = [CHAR2ID_DICT(char) for char, box in chinese_char_and_box_list]
        multiple_imgs = [(np_img, ids_list)] * 2
        imgs_list.extend(multiple_imgs)

        if len(imgs_list) > 10000:
            random.shuffle(imgs_list)
            while len(imgs_list) > 5000:
                queue.put(pack_text_lines(imgs_list, batch_size, type, "white"))


def pack_text_lines(img_label_tuples, batch_size, type, background="white"):
    raw_np_imgs = []
    raw_labels = []
    max_h = max_w = 0
    for _ in range(batch_size):
        np_img, ids_list = img_label_tuples.pop()
        raw_np_imgs.append(np_img)
        raw_labels.append(ids_list)
        max_h = max(max_h, np_img.shape[0])
        max_w = max(max_w, np_img.shape[1])
    
    if type in ("h", "horizontal"):
        assert max_h == TEXT_LINE_SIZE
    else:
        assert max_w == TEXT_LINE_SIZE
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
    batch_labels = dense_tensor_when_gpu(raw_labels, dtype=np.int32, pad_value=0)
    
    return batch_imgs, batch_labels


def text_lines_batch_generator(method="load", batch_size=BATCH_SIZE_TEXT_LINE):
    queue = Queue()
    
    if method == "load":
        batch_generator = load_text_lines_batch
        args = (
            [(CRNN_TEXT_LINE_TAGS_FILE_H, "horizontal"), (CRNN_TEXT_LINE_TAGS_FILE_V, "vertical")],
            queue,
            batch_size
        )
    elif method == "create":
        batch_generator = create_text_lines_batch
        args = (
            queue,
            batch_size
        )
    else:
        ValueError("Optional generator method : 'load', 'create'.")

    writer = Process(target=batch_generator, args=args)
    writer.start()
    
    while True:
        yield queue.get()
