# -*- coding: utf-8 -*-
# Author: hushukai

import os
import tensorflow as tf
from tensorflow.keras import backend as K

from segment_base.model import work_net, compile, add_metrics
from segment_base.data_pipeline import data_generator
from segment_base.callback import tf_config, get_callbacks
from segment_base.utils import get_segment_task_path


def main(data_file, src_type, text_type, segment_task, epochs, init_epochs=0, model_struc="densenet_gru", weights_path=""):
    _, ckpt_dir, logs_dir = get_segment_task_path(segment_task)
    tf_config()
    K.set_learning_phase(True)
    
    # 加载模型
    train_model, val_model = work_net(stage="train", segment_task=segment_task, text_type=text_type, model_struc=model_struc)
    compile(train_model, loss_names=['segment_class_loss', 'segment_regress_loss'])
    
    # 增加度量汇总
    total_acc, pos_acc, neg_acc = train_model.get_layer('accuracy').output
    num_pos, num_neg = train_model.get_layer('segment_target').output[4:]
    add_metrics(train_model,
                metric_name_list=['total_acc', 'pos_acc', 'neg_acc', 'num_pos', 'num_neg'],
                metric_val_list=[total_acc, pos_acc, neg_acc, num_pos, num_neg])
    train_model.summary()
    
    # for layer in train_model.layers:
    #     print(layer.name, " trainable: ", layer.trainable)
    
    # load model
    load_path = os.path.join(ckpt_dir, segment_task + "_segment_" + model_struc + "_{:04d}.h5".format(init_epochs))
    weights_path = weights_path if os.path.exists(weights_path) else load_path
    if os.path.exists(weights_path):
        train_model.load_weights(load_path, by_name=True)
        print("\nLoad model weights from %s\n"%weights_path)
    
    training_generator, validation_generator = data_generator(data_file=data_file,
                                                              segment_task=segment_task,
                                                              src_type=src_type,
                                                              text_type=text_type)
    
    summary_writer = tf.summary.create_file_writer(logs_dir)
    callbacks = get_callbacks(segment_task, model_struc)
    steps_per_epoch = 200
    for epoch in range(init_epochs, init_epochs + epochs, 5):
        # 开始训练
        train_model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epoch+5,
                                  initial_epoch=epoch,
                                  verbose=1,
                                  validation_data=validation_generator,
                                  validation_steps=10,
                                  callbacks=callbacks,
                                  max_queue_size=100)
        
        for i in range(5):  # 汇总图片
            x = next(validation_generator) if src_type == "images" else validation_generator
            summary_images = val_model.predict_on_batch(x=x).numpy()
            with summary_writer.as_default():
                tf.summary.image("image_%d"%i, summary_images.astype("uint8"), step=epoch * steps_per_epoch, max_outputs=20)
        summary_writer.flush()
    
    summary_writer.close()
    train_model.save_weights(os.path.join(ckpt_dir, segment_task + "_segment_" + model_struc + "_finished.h5"))    # 保存模型
    
    print("Done !")
