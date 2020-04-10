# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import numpy as np
from PIL import Image
from tensorflow.keras import backend

from recog_with_crnn.model import CRNN
from recog_with_crnn.utils import resize_text_image
from recog_with_crnn.utils import sparse_tensor_to_list

from config import TEXT_LINE_SIZE
from config import CRNN_CKPT_DIR
from util import ID2CHAR_DICT


def predict(imgs_dir, model_epoch, model_type="horizontal", model_struc="resnet_lstm"):
    backend.set_learning_phase(False)
    
    crnn = CRNN(model_type=model_type, model_struc=model_struc)
    model = crnn.model_for_predicting()
    
    weights_prefix = os.path.join(CRNN_CKPT_DIR, "vertical_densenet_gru_crnn_weights_00003_162.33.tf")
    model.load_weights(filepath=weights_prefix)
    
    for file in os.listdir(imgs_dir):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".gif"):
            img_path = os.path.join(imgs_dir, file)
            PIL_img = Image.open(img_path)
            PIL_img = PIL_img if PIL_img.mode == "L" else PIL_img.convert("L")
            PIL_img = resize_text_image(PIL_img, obj_size=TEXT_LINE_SIZE, type=model_type)
            np_img = np.asarray(PIL_img)
            
            batch_imgs = np_img[np.newaxis, :, :, np.newaxis]
            img_len = np_img.shape[2] if model_type in ("h", "horizontal") else np_img.shape[1]
            img_len = np.array([img_len], dtype=np.int32)

            np_indices, np_values = model.predict(x=[batch_imgs, img_len])
            print(np_indices, np_values)
            
            batch_labels = sparse_tensor_to_list(np_indices, np_values)
            labels = batch_labels[0]
            
            result_str = "".join([ID2CHAR_DICT[id] for id in labels])
            print(file + "\t" + result_str)
            

if __name__ == '__main__':
    
    print("Done !")
