# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import numpy as np
from PIL import Image
from tensorflow.keras import backend

from recognition_crnn.crnn import CRNN
from recognition_crnn.util import resize_text_image
from recognition_crnn.util import sparse_tensor_to_list

from config import TEXT_LINE_SIZE
from config import CRNN_CKPT_DIR
from utils import ID2CHAR_DICT


def predict(imgs_dir, type, model_epoch, model_type="horizontal", model_struc="resnet_lstm"):
    backend.set_learning_phase(False)
    
    crnn = CRNN(model_type=model_type, model_struc=model_struc)
    model = crnn.model_for_predicting()
    
    weights_prefix = os.path.join(CRNN_CKPT_DIR, model_type + model_struc + "_crnn_weights_%05d_" % model_epoch)
    model.load_weights(filepath=weights_prefix)
    
    for file in os.listdir(imgs_dir):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".gif"):
            img_path = os.path.join(imgs_dir, file)
            PIL_img = Image.open(img_path)
            PIL_img = PIL_img if PIL_img.mode == "L" else PIL_img.convert("L")
            PIL_img = resize_text_image(PIL_img, obj_size=TEXT_LINE_SIZE, type=type)
            np_img = np.asarray(PIL_img)
            
            batch_imgs = np_img[np.newaxis, :, :, np.newaxis]
            img_len_ratio = np.array([1.0], dtype=np.float32)

            np_indices, np_values = model.predict(x=[batch_imgs, img_len_ratio])
            
            batch_labels = sparse_tensor_to_list(np_indices, np_values)
            labels = batch_labels[0]
            
            result_str = "".join([ID2CHAR_DICT[id] for id in labels])
            print(file + "\t" + result_str)
            

if __name__ == '__main__':
    predict()
    
    print("Done !")
