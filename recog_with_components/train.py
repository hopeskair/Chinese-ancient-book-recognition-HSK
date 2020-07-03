# -*- coding: utf-8 -*-
# Author: hushukai

import os
import tensorflow as tf
from tensorflow.keras import backend as K

from .model import work_net, compile, add_metrics
from .callback import tf_config, get_callbacks
from .data_pipeline import data_generator

from config import CHAR_IMG_SIZE
from config import CHAR_RECOG_CKPT_DIR, CHAR_RECOG_LOGS_DIR
from config import CHAR_IMAGE_PATHS_FILE, CHAR_TFRECORDS_PATHS_FILE


def train(data_file, src_type, epochs, init_epochs=0, model_struc="densenet_gru", weights_path=""):
    tf_config()
    K.set_learning_phase(True)
    
    # 加载模型
    train_model = work_net(stage="train", img_size=CHAR_IMG_SIZE, model_struc=model_struc)
    compile(train_model, loss_names=["char_struc_loss", "sc_char_loss", "lr_compo_loss"])
    
    # 增加度量汇总
    metrics_summary = train_model.get_layer('summary_fn').output
    add_metrics(train_model,
                metric_name_list=["char_struc_acc", "sc_acc", "sc_top3", "sc_top5",
                                  "lr_acc", "lr_top3", "lr_top5",
                                  "correct_lr_acc", "correct_lr_top3", "correct_lr_top5",
                                  "total_acc", "total_top3", "total_top5"],
                metric_val_list=metrics_summary)
    train_model.summary()
    
    # for layer in train_model.layers:
    #     print(layer.name, " trainable: ", layer.trainable)
    
    # load model
    load_path = os.path.join(CHAR_RECOG_CKPT_DIR, "char_recog_with_compo_" + model_struc + "_{:04d}.h5".format(init_epochs))
    weights_path = weights_path if os.path.exists(weights_path) else load_path
    if os.path.exists(weights_path):
        train_model.load_weights(weights_path, by_name=True)
        print("\nLoad model weights from %s\n" % weights_path)
    
    training_generator, validation_generator = data_generator(data_file=data_file, src_type=src_type)
    
    # 开始训练
    train_model.fit_generator(generator=training_generator,
                              steps_per_epoch=500,
                              epochs=epochs + init_epochs,
                              initial_epoch=init_epochs,
                              verbose=1,
                              validation_data=validation_generator,
                              validation_steps=20,
                              callbacks=get_callbacks(model_struc),
                              max_queue_size=100)
    
    # 保存模型
    train_model.save_weights(os.path.join(CHAR_RECOG_CKPT_DIR, "char_recog_with_compo_" + model_struc + "_finished.h5"))
    

def main():
    train(data_file=CHAR_TFRECORDS_PATHS_FILE,
          src_type="tfrecords",
          epochs=50*10,
          init_epochs=0,
          model_struc="densenet_gru",
          weights_path="")


if __name__ == "__main__":
    print("Done !")
