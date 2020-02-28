# -*- encoding: utf-8 -*-
# Author: hushukai

import os
from tensorflow.keras import backend, optimizers, callbacks

from recognition_crnn.crnn import CRNN
from recognition_crnn.data_pipeline import text_lines_batch_generator

from config import CRNN_CKPT_DIR, BATCH_SIZE_TEXT_LINE


def train(num_epochs, start_epoch=0, model_type="horizontal", model_struc="resnet_lstm"):
    backend.set_learning_phase(True)
    
    crnn = CRNN(model_type=model_type, model_struc=model_struc)
    model = crnn.model_for_training()
    model.compile(optimizer=optimizers.Adagrad(learning_rate=0.01),
                  loss={"ctc_loss": lambda y_true, out_loss: out_loss})
    
    if start_epoch > 0:
        weights_prefix = os.path.join(CRNN_CKPT_DIR, model_struc + "_crnn_weights_%05d_" % start_epoch)
        model.load_weights(filepath=weights_prefix)

    ckpt_path = os.path.join(CRNN_CKPT_DIR, model_type + model_struc + "_crnn_weights_{epoch:05d}_{val_loss:.5f}.h5")
    checkpoint = callbacks.ModelCheckpoint(filepath=ckpt_path,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True,
                                           mode="min")

    model.fit_generator(generator=text_lines_batch_generator("create", model_type, BATCH_SIZE_TEXT_LINE),
                        steps_per_epoch=10000,
                        epochs=start_epoch + num_epochs,
                        verbose=1,
                        callbacks=[checkpoint],
                        validation_data=text_lines_batch_generator("load", model_type, BATCH_SIZE_TEXT_LINE),
                        validation_steps=10,
                        initial_epoch=start_epoch + 1)
    
    
if __name__ == "__main__":
    train(num_epochs=10, start_epoch=0, model_type="horizontal", model_struc="resnet_lstm")
    
    print("Done !")