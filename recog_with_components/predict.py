# -*- coding: utf-8 -*-
# Author: hushukai

import os
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K

from .model import work_net
from .data_pipeline import adjust_img_to_fixed_shape

from util import NUM_COMPO
from util import NUM_CHARS_TASK2 as NUM_CHARS
from util import ID2CHAR_DICT_TASK2 as ID2CHAR_DICT
from config import CHAR_IMG_SIZE
from config import CHAR_RECOG_CKPT_DIR


def load_images(img_path):
    if os.path.isfile(img_path) and os.path.splitext(img_path)[1] in (".jpg", ".png", ".gif"):
        img_paths = [img_path, ]
    if os.path.isdir(img_path):
        img_paths = [os.path.join(img_path, file) for file in os.listdir(img_path)
                     if os.path.splitext(file)[1] in (".jpg", ".png", ".gif")]
    for img_path in img_paths:
        PIL_img = Image.open(img_path)
        yield PIL_img, os.path.basename(img_path)


def main(img_path, model_struc="densenet", weights_path=""):
    K.set_learning_phase(False)
    
    if not os.path.exists(weights_path):
        weights_path = os.path.join(CHAR_RECOG_CKPT_DIR, "char_recog_with_components_" + model_struc + "_finished.h5")
        assert os.path.exists(weights_path)
    
    # 加载模型
    pred_model = work_net(NUM_CHARS, NUM_COMPO, stage="predict", img_size=CHAR_IMG_SIZE, model_struc=model_struc)
    pred_model.load_weights(weights_path, by_name=True)
    print("\nLoad model weights from %s\n" % weights_path)
    # ctpn_model.summary()
    
    count = 0
    for PIL_img, img_name in load_images(img_path):
        count += 1
        chinese_char = img_name[0]
        
        PIL_img = adjust_img_to_fixed_shape(PIL_img=PIL_img, img_shape=(CHAR_IMG_SIZE, CHAR_IMG_SIZE))
        np_img = np.array(PIL_img, dtype=np.uint8)
        batch_images = np_img[np.newaxis, :, :, :]

        # 模型预测
        class_indices, _, compo_hit_indices, _, combined_pred1, combined_pred2 = pred_model.predict(x=batch_images)
        pred_indices = [class_indices[0, 0], compo_hit_indices[0, 0], combined_pred1[0], combined_pred2[0]]
        pred_chars = [ID2CHAR_DICT[pred_index] for pred_index in pred_indices]
        
        print("Target {}: cls_pred {}, compo_pred {}, comb_pred1 {}, comb_pred2 {}".format(chinese_char, *pred_chars))
        

if __name__ == '__main__':
    print("Done !")