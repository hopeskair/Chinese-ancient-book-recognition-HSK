# -*- encoding: utf-8 -*-
# Author: hushukai

from recog_with_components.train import main as char_recog_main
from recog_with_components.data_pipeline import data_generator

from config import CHAR_IMAGE_PATHS_FILE, CHAR_TFRECORDS_PATHS_FILE


if __name__ == '__main__':
    char_recog_main()
    
    # training_dataset, _ = data_generator(data_file=CHAR_TFRECORDS_PATHS_FILE, src_type="tfrecords")
    # for output_dict in training_dataset:
    #     print(output_dict)
    
    print("Done !")
