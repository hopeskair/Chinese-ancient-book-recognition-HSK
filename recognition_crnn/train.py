# -*- encoding: utf-8 -*-
# Author: hushukai

import os
from tensorflow.keras import backend, optimizers, callbacks

from recognition_crnn.model import CRNN
from recognition_crnn.data_pipeline import create_text_lines_batch
from recognition_crnn.data_pipeline import load_text_lines_batch

from config import CRNN_CKPT_DIR, BATCH_SIZE_TEXT_LINE
from util import check_or_makedirs


def train(num_epochs, start_epoch=0, model_type="horizontal", model_struc="resnet_lstm"):
    backend.set_learning_phase(True)
    
    crnn = CRNN(model_type=model_type, model_struc=model_struc)
    model = crnn.model_for_training()
    model.compile(optimizer=optimizers.Adagrad(learning_rate=0.01),
                  loss={"ctc_loss": lambda y_true, out_loss: out_loss})
    
    if start_epoch > 0:
        weights_prefix = os.path.join(CRNN_CKPT_DIR, model_type + "_" + model_struc + "_crnn_weights_%05d_" % start_epoch)
        model.load_weights(filepath=weights_prefix)
    
    check_or_makedirs(CRNN_CKPT_DIR)
    ckpt_path = os.path.join(CRNN_CKPT_DIR, model_type + "_" + model_struc + "_crnn_weights_{epoch:05d}_{val_loss:.2f}.tf")
    checkpoint = callbacks.ModelCheckpoint(filepath=ckpt_path,
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True,
                                           mode="min")

    model.fit_generator(generator=create_text_lines_batch(type=model_type, batch_size=BATCH_SIZE_TEXT_LINE),
                        steps_per_epoch=100,
                        epochs=start_epoch+num_epochs,
                        verbose=1,
                        callbacks=[checkpoint],
                        validation_data=load_text_lines_batch(type=model_type, batch_size=BATCH_SIZE_TEXT_LINE),
                        validation_steps=50,
                        max_queue_size=50,
                        workers=2,
                        use_multiprocessing=True,
                        initial_epoch=start_epoch)
    
    
if __name__ == "__main__":
    train(num_epochs=100, start_epoch=0, model_type="horizontal", model_struc="densenet_lstm")
    
    print("Done !")