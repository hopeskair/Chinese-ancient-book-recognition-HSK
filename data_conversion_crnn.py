# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import json

from config import TEXT_LINE_TAGS_FILE_H, TEXT_LINE_IMGS_H
from config import TEXT_LINE_TAGS_FILE_V, TEXT_LINE_IMGS_V
from config import CRNN_TEXT_LINE_TAGS_FILE_H, CRNN_TEXT_LINE_TAGS_FILE_V
from data_generator.data_conversion_crnn import convert_annotation


if __name__ == '__main__':
    convert_annotation(src_list=[(TEXT_LINE_TAGS_FILE_H, TEXT_LINE_IMGS_H)],
                       dest_file=CRNN_TEXT_LINE_TAGS_FILE_H)
    convert_annotation(src_list=[(TEXT_LINE_TAGS_FILE_V, TEXT_LINE_IMGS_V)],
                       dest_file=CRNN_TEXT_LINE_TAGS_FILE_V)
    
    print("Done !")
