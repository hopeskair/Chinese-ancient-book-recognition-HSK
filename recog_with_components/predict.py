# -*- coding: utf-8 -*-
# Author: hushukai

import os
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K

from .model import work_net
from .data_pipeline import adjust_img_to_fixed_shape

from config import CHAR_IMG_SIZE
from config import CHAR_RECOG_CKPT_DIR
from config import ID_TO_CHAR_STRUC
from util import COMPO_SEQ_TO_CHAR


def load_images(img_path):
    if os.path.isfile(img_path) and os.path.splitext(img_path)[1] in (".jpg", ".png", ".gif"):
        img_paths = [img_path, ]
    if os.path.isdir(img_path):
        img_paths = [os.path.join(img_path, file) for file in os.listdir(img_path)
                     if os.path.splitext(file)[1] in (".jpg", ".png", ".gif")]
    for img_path in img_paths:
        PIL_img = Image.open(img_path)
        yield PIL_img, os.path.basename(img_path)


def main(img_path, model_struc="densenet_gru", weights_path=""):
    K.set_learning_phase(False)
    
    if not os.path.exists(weights_path):
        weights_path = os.path.join(CHAR_RECOG_CKPT_DIR, "char_recog_with_compo_" + model_struc + "_finished.h5")
        assert os.path.exists(weights_path)
    
    # 加载模型
    pred_model = work_net(stage="predict", img_size=CHAR_IMG_SIZE, model_struc=model_struc)
    pred_model.load_weights(weights_path, by_name=True)
    print("\nLoad model weights from %s\n" % weights_path)
    # pred_model.summary()
    
    count = 0
    for PIL_img, img_name in load_images(img_path):
        count += 1
        chinese_char = img_name[0]
        
        PIL_img = adjust_img_to_fixed_shape(PIL_img=PIL_img, img_shape=(CHAR_IMG_SIZE, CHAR_IMG_SIZE))
        np_img = np.array(PIL_img, dtype=np.uint8)
        batch_images = np_img[np.newaxis, :, :, :]
        
        # 模型预测
        pred_char_struc, pred_results = pred_model.predict(x=batch_images)
        
        topk = min(5, np.shape(pred_results)[1])
        pred_chars_topk = ""
        for j in range(topk):
            compo_seq = pred_results[0, j]
            if pred_char_struc[0] == 0:
                compo_info = "s" + str(compo_seq[0])    # simple char
            else:
                # pred_char_struc[0] == 1, left-right char.
                zero_indices = np.where(compo_seq == 0)[0]
                first_zero_pos = zero_indices[0] if zero_indices.size > 0 else compo_seq.size
                compo_str_seq = [str(compo_id) for compo_id in compo_seq[:first_zero_pos]]
                compo_info = "⿰" + ",".join(compo_str_seq)
            
            pred_chars_topk += COMPO_SEQ_TO_CHAR.get(compo_info, "")
        
        print("Target {} - {} Prediction.".format(chinese_char, pred_chars_topk))
        

if __name__ == '__main__':
    print("Done !")