# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import json

from config import TEXT_LINE_TAGS_FILE_H, TEXT_LINE_IMGS_H
from config import TEXT_LINE_TAGS_FILE_V, TEXT_LINE_IMGS_V
from config import CRNN_TEXT_LINE_TAGS_FILE_H, CRNN_TEXT_LINE_TAGS_FILE_V
from utils import CHAR2ID_DICT


def convert_annotation(src_list, dest_file):
    with open(dest_file, "w", encoding="utf-8") as fw:
        for src_file, root_dir in src_list:
            with open(src_file, "r", encoding="utf-8") as fr:
                for line in fr:
                    img_name, boxes_str = line.strip().split("\t")
                    img_path = os.path.join(root_dir, img_name)
                    boxes_list = json.loads(boxes_str)
                    chars = "".join([char for char, bounding_box in boxes_list])
                    ids_list = [CHAR2ID_DICT[char] for char in chars]
                    fw.write(img_path + "\t" + json.dumps(ids_list) + "\t" + chars + "\n")


if __name__ == '__main__':
    convert_annotation(src_list=[(TEXT_LINE_TAGS_FILE_H, TEXT_LINE_IMGS_H)],
                       dest_file=CRNN_TEXT_LINE_TAGS_FILE_H)
    convert_annotation(src_list=[(TEXT_LINE_TAGS_FILE_V, TEXT_LINE_IMGS_V)],
                       dest_file=CRNN_TEXT_LINE_TAGS_FILE_V)
    
    print("Done !")
