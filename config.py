# -*- encoding: utf-8 -*-
# Author: hushukai

import os


# ************************ basic configuration ***************************
# os.getcwd() returns the current working directory
CURR_DIR = os.path.dirname(__file__).replace("/", os.sep)
CHINESE_LABEL_FILE = os.path.join(CURR_DIR, "chinese_labels", "chinese_labels_simplified_level1.txt")
IGNORABLE_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "ignorable_chars.txt")
IMPORTANT_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "important_chars.txt")
# ************************ basic configuration ***************************


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
CRNN_TEXT_LINE_TAGS_FILE_H = os.path.join(DATA_DIR, "text_lines", "text_lines_tags_crnn_horizontal.txt")
CRNN_TEXT_LINE_TAGS_FILE_V = os.path.join(DATA_DIR, "text_lines", "text_lines_tags_crnn_vertical.txt")
# *********************** data format conversion *************************


# ***************************** Model ************************************
VALIDATION_SPLIT = 0.1

# book page detection
BOX_CLASSES_ON_BOOK = ["text",]
BATCH_SIZE_BOOK_PAGE = 2
YOLO3_CLASS_SCORE_THRESH = 0.6
YOLO3_NMS_IOU_THRESH = 0.45
YOLO3_NMS_MAX_BOXES_NUM = 50
YOLO3_ROOT_DIR = os.path.join(CURR_DIR, "detection_yolo3")
YOLO3_CKPT_DIR = os.path.join(CURR_DIR, "detection_yolo3", "ckpt")
YOLO3_LOGS_DIR = os.path.join(CURR_DIR, "detection_yolo3", "logs")
YOLO3_ANCHORS_FILE = os.path.join(CURR_DIR, "detection_yolo3", "anchors_with_kmeans.txt")

# text line recognition
BATCH_SIZE_TEXT_LINE = 8
CRNN_CKPT_DIR = os.path.join(CURR_DIR, "recognition_crnn", "ckpt")
CRNN_LOGS_DIR = os.path.join(CURR_DIR, "recognition_crnn", "logs")
# ***************************** Train ************************************