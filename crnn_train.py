# -*- encoding: utf-8 -*-
# Author: hushukai

from recognition_crnn.train import train

if __name__ == '__main__':
    train(num_epochs=10, start_epoch=0, model_struc="resnet_lstm")
    
    print("Done !")
