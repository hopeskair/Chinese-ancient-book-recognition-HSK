# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import json

from config import BOOK_PAGE_TAGS_FILE_H, BOOK_PAGE_IMGS_H
from config import BOOK_PAGE_TAGS_FILE_V, BOOK_PAGE_IMGS_V
from config import YOLO3_BOOK_PAGE_TAGS_FILE


def convert_annotation(src_list, dest_file):
    with open(dest_file, "w", encoding="utf-8") as fw:
        for src_file, root_dir in src_list:
            with open(src_file, "r", encoding="utf-8") as fr:
                for line in fr:
                    img_name, boxes_str = line.strip().split("\t")
                    img_path = os.path.join(root_dir, img_name)
                    boxes = json.loads(boxes_str)
                    cls_id = 0
                    boxes_str = " ".join(["%d,%d,%d,%d,%d"%(x1, y1, x2, y2, cls_id) for x1, y1, x2, y2 in boxes])
                    fw.write(img_path + "\t" + boxes_str + "\n")


if __name__ == '__main__':
    convert_annotation(src_list=[(BOOK_PAGE_TAGS_FILE_H, BOOK_PAGE_IMGS_H),
                                 (BOOK_PAGE_TAGS_FILE_V, BOOK_PAGE_IMGS_V)],
                       dest_file=YOLO3_BOOK_PAGE_TAGS_FILE)
    
    print("Done !")
