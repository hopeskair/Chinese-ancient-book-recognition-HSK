# -*- coding: utf-8 -*-
# Author: hushukai

import os
import numpy as np
from PIL import Image
from tensorflow.keras import backend as K

from .model import work_net
from .data_pipeline import adjust_img_to_fixed_shape
from segment_base.predict import convert_images, load_images

from config import CHAR_IMG_SIZE, CHAR_RECOG_BATCH_SIZE
from config import CHAR_RECOG_CKPT_DIR
from config import ID_TO_CHAR_STRUC
from util import COMPO_SEQ_TO_CHAR


def model_weights_path(weights, model_struc="densenet_gru"):
    ckpt_dir = CHAR_RECOG_CKPT_DIR
    
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
                weights_path = os.path.join(ckpt_dir, "char_recog_with_compo_" + model_struc + "_finished.h5")
                assert os.path.exists(weights_path)
    else:
        assert isinstance(weights, int)
        weights_id = "{:04d}".format(weights)
        weights_path = os.path.join(ckpt_dir, "char_recog_with_compo_" + model_struc + "_" + weights_id + ".h5")
        assert os.path.exists(weights_path)
    
    return weights_path


def char_predict(images=None, img_paths=None, recog_model=None, model_struc="densenet_gru", weights="", to_print=False):
    
    # images
    if images is not None:
        np_img_list = convert_images(images)
        img_name_list = [str(i) + ".jpg" for i in range(len(np_img_list))]
    else:
        assert img_paths is not None
        np_img_list, img_name_list = load_images(img_paths)

    # model
    if recog_model is None:
        K.set_learning_phase(False)
        weights_path = model_weights_path(weights, model_struc)

        # 加载模型
        pred_model = work_net(stage="predict", img_size=CHAR_IMG_SIZE, model_struc=model_struc)
        pred_model.load_weights(weights_path, by_name=True)
        print("\nLoad model weights from %s\n" % weights_path)
        # pred_model.summary()
    
    # predict
    batch_size = CHAR_RECOG_BATCH_SIZE
    pred_topk_chars_list = []
    for i in range(0, len(np_img_list), batch_size):
        _images_list = []
        for np_img in np_img_list[i:i + batch_size]:
            PIL_img = adjust_img_to_fixed_shape(np_img=np_img, img_shape=(CHAR_IMG_SIZE, CHAR_IMG_SIZE))
            np_img = np.array(PIL_img, dtype=np.uint8)
            _images_list.append(np_img)
        batch_images = np.array(_images_list, dtype=np.float32)
        
        pred_char_struc, pred_results = pred_model.predict(x=batch_images)  # 模型预测

        topk = min(5, np.shape(pred_results)[1])
        for j in range(len(batch_images)):
            pred_chars_topk = ""
            for k in range(topk):
                compo_seq = pred_results[j, k]
                if pred_char_struc[j] == 0:
                    compo_info = "s" + str(compo_seq[0])  # simple char
                else:
                    # pred_char_struc[0] == 1, left-right char.
                    zero_indices = np.where(compo_seq == 0)[0]
                    first_zero_pos = zero_indices[0] if zero_indices.size > 0 else compo_seq.size
                    compo_str_seq = [str(compo_id) for compo_id in compo_seq[:first_zero_pos]]
                    compo_info = "⿰" + ",".join(compo_str_seq)
                pred_chars_topk += COMPO_SEQ_TO_CHAR.get(compo_info, "")
            pred_topk_chars_list.append(pred_chars_topk)
    
    if to_print:
        for i, img_name in enumerate(img_name_list):
            chinese_char = img_name[0]
            print("Target {} - {} Prediction.".format(chinese_char, pred_topk_chars_list[i]))
    
    return np_img_list, img_name_list, pred_topk_chars_list
        

if __name__ == '__main__':
    print("Done !")