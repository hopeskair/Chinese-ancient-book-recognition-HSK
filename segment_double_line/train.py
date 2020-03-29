# -*- encoding: utf-8 -*-
# Author: hushukai

from segment_base.train import main as train

from config import SEGMENT_DOUBLE_LINE_TAGS_FILE_H, SEGMENT_DOUBLE_LINE_TAGS_FILE_V
from config import SEGMENT_DOUBLE_LINE_TFRECORDS_H, SEGMENT_DOUBLE_LINE_TFRECORDS_V

def main():
    train(data_file=SEGMENT_DOUBLE_LINE_TFRECORDS_H,
          src_type="tfrecords",
          text_type="horizontal",
          segment_task="double_line",
          epochs=100,
          init_epochs=0,
          model_struc="densenet_gru",
          weights_path="")


if __name__ == '__main__':
    main()
    print("Done !")
