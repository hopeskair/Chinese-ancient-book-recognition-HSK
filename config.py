# -*- encoding: utf-8 -*-
# Author: hushukai

import os


# ************************ basic configuration ***************************
# os.getcwd() returns the current working directory
CURR_DIR = os.path.dirname(__file__)
CHINESE_LABEL_FILE = os.path.join(CURR_DIR, "chinese_labels", "chinese_labels_test.txt")
IGNORABLE_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "ignorable_chars.txt")
IMPORTANT_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "important_chars.txt")
# ************************ basic configuration ***************************


LOG_DIR = os.path.join(CURR_DIR, "_log")
CHECKPOINT_DIR = os.path.join(CURR_DIR, "_ckpt")


# ************************ generate image data ***************************
DATA_DIR = os.path.join(CURR_DIR, "data")
CHAR_IMGS_DIR = os.path.join(DATA_DIR, "char_images")
TEXT_LINE_IMGS_DIR = os.path.join(DATA_DIR, "text_line_images")
BOOK_PAGE_IMGS_DIR = os.path.join(DATA_DIR, "book_page_images")

FONT_FILE_DIR = os.path.join(CURR_DIR, "chinese_fonts")
FONT_FINISHED_DIR = os.path.join(CURR_DIR, "chinese_fonts_finished")
# EXTERNEL_IMAGES_DIR = os.path.join(CURR_DIR, "../ziku_images")
EXTERNEL_IMAGES_DIR = "E:/pycharm_project/ziku_images"

CHAR_IMG_SIZE = 64
MAX_ROTATE_ANGLE = 10
NUM_IMAGES_PER_FONT = 10
# ************************ generate image data ***************************
