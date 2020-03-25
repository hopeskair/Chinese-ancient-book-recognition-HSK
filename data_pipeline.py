# -*- encoding: utf-8 -*-
# Author: hushukai

import os
from segment_base.data_pipeline import data_generator


if __name__ == "__main__":
    from config import SEGMENT_BOOK_PAGE_TFRECORDS_H
    training_generator, validation_generator = \
        data_generator(data_file=SEGMENT_BOOK_PAGE_TFRECORDS_H,
                       src_type="tfrecords",
                       segment_task="book_page",
                       text_type="horizontal")
    
    for inputs_dict in training_generator:
        print(inputs_dict)
    
    print("Done !")