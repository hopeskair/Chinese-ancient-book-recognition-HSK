# -*- encoding: utf-8 -*-
# Author: hushukai

from segment_base.train import main as train

from config import SEGMENT_BOOK_PAGE_TAGS_FILE_H, SEGMENT_BOOK_PAGE_TAGS_FILE_V
from config import SEGMENT_BOOK_PAGE_TFRECORDS_H, SEGMENT_BOOK_PAGE_TFRECORDS_V

def main():
    train(data_file=SEGMENT_BOOK_PAGE_TFRECORDS_V,
          src_type="tfrecords",
          text_type="vertical",
          segment_task="book_page",
          epochs=300,
          init_epochs=40,
          model_struc="densenet_gru",
          weights_path="")


if __name__ == '__main__':
    main()
    print("Done !")
