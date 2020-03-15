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
# ************************ generate image data ***************************


# ************************* Commonly used ********************************
# chinese char images
CHAR_IMG_SIZE = 64
MAX_ROTATE_ANGLE = 10
NUM_IMAGES_PER_FONT = 10

# text line images
TEXT_LINE_SIZE = CHAR_IMG_SIZE

# validation data
VALIDATION_SPLIT = 0.1

# book page detection
BOX_CLASSES_ON_BOOK = ["text",]
BATCH_SIZE_BOOK_PAGE = 2

# text line recognition crnn
BATCH_SIZE_TEXT_LINE = 8
# ************************* Commonly used ********************************


# *********************** data format conversion *************************
CPTN_BOOK_PAGE_TAGS_FILE = os.path.join(DATA_DIR, "book_pages", "book_pages_tags_cptn.txt")
YOLO3_BOOK_PAGE_TAGS_FILE = os.path.join(DATA_DIR, "book_pages", "book_pages_tags_yolo3.txt")
CRNN_TEXT_LINE_TAGS_FILE_H = os.path.join(DATA_DIR, "text_lines", "text_lines_tags_crnn_horizontal.txt")
CRNN_TEXT_LINE_TAGS_FILE_V = os.path.join(DATA_DIR, "text_lines", "text_lines_tags_crnn_vertical.txt")
# *********************** data format conversion *************************


# **************************** Model yolo3 *******************************
# book page detection yolo3
YOLO3_ROOT_DIR = os.path.join(CURR_DIR, "detection_yolo3")
YOLO3_CKPT_DIR = os.path.join(CURR_DIR, "detection_yolo3", "ckpt")
YOLO3_LOGS_DIR = os.path.join(CURR_DIR, "detection_yolo3", "logs")
YOLO3_ANCHORS_FILE = os.path.join(CURR_DIR, "detection_yolo3", "anchors_with_kmeans.txt")

YOLO3_CLASS_SCORE_THRESH = 0.6
YOLO3_NMS_IOU_THRESH = 0.45
YOLO3_NMS_MAX_BOXES_NUM = 50
# **************************** Model yolo3 *******************************


# **************************** Model ctpn ********************************
# book page detection ctpn
CTPN_ROOT_DIR = os.path.join(CURR_DIR, "detection_ctpn")
CTPN_CKPT_DIR = os.path.join(CURR_DIR, "detection_ctpn", "ckpt")
CTPN_LOGS_DIR = os.path.join(CURR_DIR, "detection_ctpn", "logs")

CTPN_CLASS_MAPPING = {'bg': 0, 'text': 1}
CTPN_NET_STRIDE = 16
CTPN_ANCHORS_WIDTH = 16
CTPN_ANCHORS_HEIGHTS = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]

# 训练样本
CTPN_TRAIN_ANCHORS_PER_IMAGE = 1024
CTPN_ANCHOR_POSITIVE_RATIO = 0.5

# text proposal
CTPN_PROPOSALS_MIN_SCORE = 0.7
CTPN_PROPOSALS_NMS_THRESH = 0.3
CTPN_PROPOSALS_MAX_NUM = 1024
CTPN_PROPOSALS_WIDTH = CTPN_ANCHORS_WIDTH
CTPN_USE_SIDE_REFINE = False

# 训练超参数
CTPN_INIT_LEARNING_RATE = 0.01
CTPN_LEARNING_MOMENTUM = 0.9
CTPN_GRADIENT_CLIP_NORM = 5.0

# 权重衰减
CTPN_WEIGHT_DECAY = 0.0005,

CTPN_LOSS_WEIGHTS = {
    "ctpn_class_loss": 1.,
    "ctpn_regress_loss": 1.,
    "side_regress_loss": 1.
}

# text line boxes
CTPN_MAX_HORIZONTAL_GAP = 50
CTPN_MIN_V_OVERLAPS = 0.7
CTPN_MIN_SIZE_SIM = 0.7
CTPN_TEXT_LINE_MIN_SCORE = 0.7
CTPN_MIN_NUM_PROPOSALS = 1
CTPN_TEXT_LINE_NMS_THRESH = 0.3
# **************************** Model ctpn ********************************


# **************************** Model crnn ********************************
# text line recognition
CRNN_CKPT_DIR = os.path.join(CURR_DIR, "recognition_crnn", "ckpt")
CRNN_LOGS_DIR = os.path.join(CURR_DIR, "recognition_crnn", "logs")
# **************************** Model crnn ********************************