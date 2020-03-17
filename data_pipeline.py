# -*- encoding: utf-8 -*-
# Author: hushukai

import os
from detection_ctpn.data_pipeline import data_generator


if __name__ == "__main__":
    from config import BOOK_PAGE_TFRECORDS_V
    training_generator, validation_generator = \
        data_generator(data_file=os.path.join(BOOK_PAGE_TFRECORDS_V, "..", "book_pages_tags_ctpn.txt"),
                       batch_size=1,
                       src_type="tfrecords",
                       text_type="vertical")
    
    for inputs_dict, b in training_generator:
        print(inputs_dict)
    
    print("Done !")