# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import json

from config import BOOK_PAGE_TAGS_FILE_H, BOOK_PAGE_IMGS_H
from config import BOOK_PAGE_TAGS_FILE_V, BOOK_PAGE_IMGS_V
from config import BOOK_PAGE_TFRECORDS_H, BOOK_PAGE_TFRECORDS_V
from config import CTPN_BOOK_PAGE_TAGS_FILE


def convert_annotation(img_sources=None, tfrecords_dir=None, dest_file=CTPN_BOOK_PAGE_TAGS_FILE):
    assert [img_sources, tfrecords_dir].count(None) == 1
    
    with open(dest_file, "w", encoding="utf-8") as fw:
        if img_sources is not None:
            for src_file, root_dir in img_sources:
                with open(src_file, "r", encoding="utf-8") as fr:
                    for line in fr:
                        img_name, boxes_str = line.strip().split("\t")
                        img_path = os.path.join(root_dir, img_name)
                        boxes = json.loads(boxes_str)
                        cls_id = 1  # fg:1, bg:0
                        boxes_str = " ".join(["%d,%d,%d,%d,%d"%(x1, y1, x2, y2, cls_id) for x1, y1, x2, y2 in boxes])
                        fw.write(img_path + "\t" + boxes_str + "\n")
        
        elif tfrecords_dir is not None:
            assert os.path.exists(tfrecords_dir)
            for file in os.listdir(tfrecords_dir):
                if file.endswith(".tfrecords"):
                    file_path = os.path.join(tfrecords_dir, file)
                    fw.write(file_path + "\n")


if __name__ == '__main__':
    # convert_annotation(img_sources=[(BOOK_PAGE_TAGS_FILE_V, BOOK_PAGE_IMGS_V)])
    convert_annotation(tfrecords_dir=BOOK_PAGE_TFRECORDS_V)
    
    print("Done !")
