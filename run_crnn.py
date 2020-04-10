# -*- encoding: utf-8 -*-
# Author: hushukai

from recog_with_crnn.train import train
from recog_with_crnn.predict import predict
from config import ONE_TEXT_LINE_IMGS_H, ONE_TEXT_LINE_IMGS_V


if __name__ == '__main__':
    # train(num_epochs=200, start_epoch=0, model_type="vertical", model_struc="densenet_gru")
    predict(imgs_dir=ONE_TEXT_LINE_IMGS_V, model_epoch=3, model_type="vertical", model_struc="densenet_gru")
    
    print("Done !")
