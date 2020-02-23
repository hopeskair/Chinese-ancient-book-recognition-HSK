# -*- encoding: utf-8 -*-
# Author: hushukai

import os


# ************************ basic configuration ***************************
# os.getcwd() returns the current working directory
CURR_DIR = os.path.dirname(__file__).replace("/", os.sep)
CHINESE_LABEL_FILE = os.path.join(CURR_DIR, "chinese_labels", "chinese_labels_test.txt")
IGNORABLE_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "ignorable_chars.txt")
IMPORTANT_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "important_chars.txt")
# ************************ basic configuration ***************************


LOG_DIR = os.path.join(CURR_DIR, "_log")
CHECKPOINT_DIR = os.path.join(CURR_DIR, "_ckpt")


# ************************ generate image data ***************************
DATA_DIR = os.path.join(CURR_DIR, "data")

CHAR_IMGS_DIR = os.path.join(DATA_DIR, "chars", "imgs")
CHAR_TFRECORDS_DIR = os.path.join(DATA_DIR, "chars", "tfrecords")

TEXT_LINE_IMGS_H = os.path.join(DATA_DIR, "text_lines", "imgs_horizontal")
TEXT_LINE_IMGS_V = os.path.join(DATA_DIR, "text_lines", "imgs_vertical")
TEXT_LINE_TAGS_FILE_H = os.path.join(DATA_DIR, "text_lines", "text_lines_tags_horizontal.txt")
TEXT_LINE_TAGS_FILE_V = os.path.join(DATA_DIR, "text_lines", "text_lines_tags_vertical.txt")
TEXT_LINE_TFRECORDS_H = os.path.join(DATA_DIR, "text_lines", "tfrecords_horizontal")
TEXT_LINE_TFRECORDS_V = os.path.join(DATA_DIR, "text_lines", "tfrecords_vertical")

BOOK_PAGE_IMGS_H = os.path.join(DATA_DIR, "book_pages", "imgs_horizontal")
BOOK_PAGE_IMGS_V = os.path.join(DATA_DIR, "book_pages", "imgs_vertical")
BOOK_PAGE_TAGS_FILE_H = os.path.join(DATA_DIR, "book_pages", "book_pages_tags_horizontal.txt")
BOOK_PAGE_TAGS_FILE_V = os.path.join(DATA_DIR, "book_pages", "book_pages_tags_vertical.txt")
BOOK_PAGE_TFRECORDS_H = os.path.join(DATA_DIR, "book_pages", "tfrecords_horizontal")
BOOK_PAGE_TFRECORDS_V = os.path.join(DATA_DIR, "book_pages", "tfrecords_vertical")

FONT_FILE_DIR = os.path.join(CURR_DIR, "chinese_fonts")
FONT_FINISHED_DIR = os.path.join(CURR_DIR, "chinese_fonts_finished")

# EXTERNEL_IMAGES_DIR = os.path.join(CURR_DIR, "../ziku_images")
EXTERNEL_IMAGES_DIR = "E:/pycharm_project/ziku_images"

# chinese char images
CHAR_IMG_SIZE = 64
MAX_ROTATE_ANGLE = 10
NUM_IMAGES_PER_FONT = 10

# text line images
TEXT_LINE_SIZE = CHAR_IMG_SIZE
# ************************ generate image data ***************************


# *********************** data format conversion *************************
YOLO3_BOOK_PAGE_TAGS_FILE = os.path.join(DATA_DIR, "book_pages", "book_pages_tags_yolo3.txt")
CRNN_TEXT_LINE_TAGS_FILE_H = os.path.join(DATA_DIR, "text_lines", "text_lines_tags_yolo3_horizontal.txt")
CRNN_TEXT_LINE_TAGS_FILE_V = os.path.join(DATA_DIR, "text_lines", "text_lines_tags_yolo3_vertical.txt")
# *********************** data format conversion *************************


# ***************************** Train ************************************
# text line recognition
BATCH_SIZE_TEXT_LINE = 16
# ***************************** Train ************************************