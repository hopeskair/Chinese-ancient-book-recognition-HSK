# -*- coding: utf-8 -*-
# Author: hushukai

import os
import tensorflow as tf
from tensorflow.keras import backend as K

from .model import work_net, compile, add_metrics
from .callback import tf_config, get_callbacks
from .data_pipeline import data_generator

from util import NUM_CHARS_TASK2 as NUM_CHARS
from util import NUM_COMPO
from config import CHAR_IMG_SIZE
from config import CHAR_RECOG_CKPT_DIR, CHAR_RECOG_LOGS_DIR


def main(data_file, src_type, epochs, init_epochs=0, model_struc="densenet", weights_path=""):
    tf_config()
    K.set_learning_phase(True)
    
    # 加载模型
    train_model = work_net(NUM_CHARS, NUM_COMPO, stage="train", img_size=CHAR_IMG_SIZE, model_struc=model_struc)
    compile(train_model, loss_names=['chinese_cls_loss', 'chinese_compo_loss'])
    
    # 增加度量汇总
    summary_metrics = train_model.get_layer('summary_fn').output
    add_metrics(train_model,
                metric_name_list=['class_acc', 'top3_cls_acc', 'top5_cls_acc',
                                  'compo_hit1_ratio', 'compo_hit_acc', 'com_pred1_acc', 'com_pred2_acc'],
                metric_val_list=summary_metrics[:7])
    train_model.summary()
    
    # for layer in train_model.layers:
    #     print(layer.name, " trainable: ", layer.trainable)
    
    # load model
    load_path = os.path.join(CHAR_RECOG_CKPT_DIR, "char_recog_with_components_" + model_struc + "_{:04d}.h5".format(init_epochs))
    weights_path = weights_path if os.path.exists(weights_path) else load_path
    if os.path.exists(weights_path):
        train_model.load_weights(weights_path, by_name=True)
        print("\nLoad model weights from %s\n" % weights_path)
    
    training_generator, validation_generator = data_generator(data_file=data_file, src_type=src_type)
    
    # 开始训练
    train_model.fit_generator(generator=training_generator,
                              steps_per_epoch=200,
                              epochs=epochs + init_epochs,
                              initial_epoch=init_epochs,
                              verbose=1,
                              validation_data=validation_generator,
                              validation_steps=10,
                              callbacks=get_callbacks(model_struc),
                              max_queue_size=100)

    # 保存模型
    train_model.save_weights(os.path.join(CHAR_RECOG_CKPT_DIR, "char_recog_with_components_" + model_struc + "_finished.h5"))
    
    print("Done !")
