# -*- encoding: utf-8 -*-
# Author: hushukai

from segment_base.train import main as train

from config import SEGMENT_MIX_LINE_TAGS_FILE_H, SEGMENT_MIX_LINE_TAGS_FILE_V
from config import SEGMENT_MIX_LINE_TFRECORDS_H, SEGMENT_MIX_LINE_TFRECORDS_V

def main():
    train(data_file=SEGMENT_MIX_LINE_TFRECORDS_V,
          src_type="tfrecords",
          text_type="vertical",
          segment_task="mix_line",
          epochs=100,
          init_epochs=59,
          model_struc="densenet_gru",
          weights_path="")


if __name__ == '__main__':
    main()
    print("Done !")
