# -*- encoding: utf-8 -*-
# Author: hushukai

import os
from PIL import Image

from segment_base.data_pipeline import get_image_and_split_pos
from segment_base.data_pipeline import rotate_90_degrees
from segment_base.visualize import draw_split_lines

from util import check_or_makedirs
from config import SEGMENT_DOUBLE_LINE_ROOT_DIR
from config import TWO_TEXT_LINE_IMGS_H, TWO_TEXT_LINE_TAGS_FILE_H
from config import TWO_TEXT_LINE_IMGS_V, TWO_TEXT_LINE_TAGS_FILE_V
from config import TWO_TEXT_LINE_TFRECORDS_H, TWO_TEXT_LINE_TFRECORDS_V
from config import SEGMENT_DOUBLE_LINE_TAGS_FILE_H, SEGMENT_DOUBLE_LINE_TAGS_FILE_V
from config import SEGMENT_DOUBLE_LINE_TFRECORDS_H, SEGMENT_DOUBLE_LINE_TFRECORDS_V


def convert_annotation(img_sources=None, tfrecords_dir=None, dest_file=None):
    assert [img_sources, tfrecords_dir].count(None) == 1
    
    check_or_makedirs(os.path.dirname(dest_file))
    with open(dest_file, "w", encoding="utf-8") as fw:
        if img_sources is not None:
            for src_file, root_dir in img_sources:
                with open(src_file, "r", encoding="utf-8") as fr:
                    for line in fr:
                        img_name, tags_str = line.strip().split("\t")
                        img_path = os.path.join(root_dir, img_name)
                        fw.write(img_path + "\t" + tags_str + "\n")
        
        elif tfrecords_dir is not None:
            assert os.path.exists(tfrecords_dir)
            for file in os.listdir(tfrecords_dir):
                if file.endswith(".tfrecords"):
                    file_path = os.path.join(tfrecords_dir, file)
                    fw.write(file_path + "\n")


def check_tags(tags_file, segment_task, text_type):
    with open(tags_file, "r", encoding="utf8") as fr:
        lines = [line.strip() for line in fr.readlines()]
    
    save_path = os.path.join(SEGMENT_DOUBLE_LINE_ROOT_DIR, "samples")
    check_or_makedirs(save_path)
    
    for i, line in enumerate(lines):
        np_img, split_pos = get_image_and_split_pos(line, segment_task="mix_line")

        text_type = text_type[0].lower()
        if (segment_task, text_type) in (("book_page", "h"), ("double_line", "h"), ("text_line", "v"), ("mix_line", "v")):
            np_img, split_pos = rotate_90_degrees(np_img, split_pos)

        np_img = draw_split_lines(np_img, split_pos)
        PIL_img = Image.fromarray(np_img)
        PIL_img.save(os.path.join(save_path, str(i) + ".jpg"))


def main():
    convert_annotation(img_sources=[(TWO_TEXT_LINE_TAGS_FILE_H, TWO_TEXT_LINE_IMGS_H)], dest_file=SEGMENT_DOUBLE_LINE_TAGS_FILE_H)
    convert_annotation(img_sources=[(TWO_TEXT_LINE_TAGS_FILE_V, TWO_TEXT_LINE_IMGS_V)], dest_file=SEGMENT_DOUBLE_LINE_TAGS_FILE_V)
    # convert_annotation(tfrecords_dir=TWO_TEXT_LINE_TFRECORDS_H, dest_file=SEGMENT_DOUBLE_LINE_TFRECORDS_H)
    # convert_annotation(tfrecords_dir=TWO_TEXT_LINE_TFRECORDS_V, dest_file=SEGMENT_DOUBLE_LINE_TFRECORDS_V)
    
    # check_tags(tags_file=SEGMENT_DOUBLE_LINE_TAGS_FILE_H, segment_task="double_line", text_type="horizontal")
    check_tags(tags_file=SEGMENT_DOUBLE_LINE_TAGS_FILE_V, segment_task="double_line", text_type="vertical")


if __name__ == '__main__':
    print("Done !")
