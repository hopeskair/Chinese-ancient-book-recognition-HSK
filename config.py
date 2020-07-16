# -*- encoding: utf-8 -*-
# Author: hushukai

import os


# ************************ basic configuration ***************************
# os.getcwd() returns the current working directory
CURR_DIR = os.path.dirname(__file__)    # .replace("/", os.sep)
CHINESE_LABEL_FILE = os.path.join(CURR_DIR, "chinese_labels", "chinese_labels_all.txt")
TRADITION_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "chars_Big5_all_traditional_13053.txt")
IGNORABLE_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "ignorable_chars.txt")
IMPORTANT_CHARS_FILE = os.path.join(CURR_DIR, "chinese_labels", "important_chars.txt")
# ************************ basic configuration ***************************


# ************************ generate image data ***************************
DATA_DIR = os.path.join(CURR_DIR, "data")

CHAR_IMGS_DIR = os.path.join(DATA_DIR, "chars", "imgs")
CHAR_TFRECORDS_DIR = os.path.join(DATA_DIR, "chars", "tfrecords")

ONE_TEXT_LINE_IMGS_H = os.path.join(DATA_DIR, "one_text_lines", "imgs_horizontal")
ONE_TEXT_LINE_IMGS_V = os.path.join(DATA_DIR, "one_text_lines", "imgs_vertical")
ONE_TEXT_LINE_TAGS_FILE_H = os.path.join(DATA_DIR, "one_text_lines", "text_lines_tags_horizontal.txt")
ONE_TEXT_LINE_TAGS_FILE_V = os.path.join(DATA_DIR, "one_text_lines", "text_lines_tags_vertical.txt")
ONE_TEXT_LINE_TFRECORDS_H = os.path.join(DATA_DIR, "one_text_lines", "tfrecords_horizontal")
ONE_TEXT_LINE_TFRECORDS_V = os.path.join(DATA_DIR, "one_text_lines", "tfrecords_vertical")

TWO_TEXT_LINE_IMGS_H = os.path.join(DATA_DIR, "two_text_lines", "imgs_horizontal")
TWO_TEXT_LINE_IMGS_V = os.path.join(DATA_DIR, "two_text_lines", "imgs_vertical")
TWO_TEXT_LINE_TAGS_FILE_H = os.path.join(DATA_DIR, "two_text_lines", "text_lines_tags_horizontal.txt")
TWO_TEXT_LINE_TAGS_FILE_V = os.path.join(DATA_DIR, "two_text_lines", "text_lines_tags_vertical.txt")
TWO_TEXT_LINE_TFRECORDS_H = os.path.join(DATA_DIR, "two_text_lines", "tfrecords_horizontal")
TWO_TEXT_LINE_TFRECORDS_V = os.path.join(DATA_DIR, "two_text_lines", "tfrecords_vertical")

MIX_TEXT_LINE_IMGS_H = os.path.join(DATA_DIR, "mix_text_lines", "imgs_horizontal")
MIX_TEXT_LINE_IMGS_V = os.path.join(DATA_DIR, "mix_text_lines", "imgs_vertical")
MIX_TEXT_LINE_TAGS_FILE_H = os.path.join(DATA_DIR, "mix_text_lines", "text_lines_tags_horizontal.txt")
MIX_TEXT_LINE_TAGS_FILE_V = os.path.join(DATA_DIR, "mix_text_lines", "text_lines_tags_vertical.txt")
MIX_TEXT_LINE_TFRECORDS_H = os.path.join(DATA_DIR, "mix_text_lines", "tfrecords_horizontal")
MIX_TEXT_LINE_TFRECORDS_V = os.path.join(DATA_DIR, "mix_text_lines", "tfrecords_vertical")

BOOK_PAGE_IMGS_H = os.path.join(DATA_DIR, "book_pages", "imgs_horizontal")
BOOK_PAGE_IMGS_V = os.path.join(DATA_DIR, "book_pages", "imgs_vertical")
BOOK_PAGE_TAGS_FILE_H = os.path.join(DATA_DIR, "book_pages", "book_pages_tags_horizontal.txt")
BOOK_PAGE_TAGS_FILE_V = os.path.join(DATA_DIR, "book_pages", "book_pages_tags_vertical_3.txt")
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
BATCH_SIZE_BOOK_PAGE = 1
BOOK_PAGE_FIXED_SIZE = (640, 640)
BOOK_PAGE_MAX_GT_BOXES = 1500

# text line recognition crnn
BATCH_SIZE_TEXT_LINE = 8

# 训练超参数
INIT_LEARNING_RATE = 0.01
SGD_LEARNING_MOMENTUM = 0.9
SGD_GRADIENT_CLIP_NORM = 5.0

# 标签平滑
LABEL_SMOOTHING = 0.1

# 权重衰减
L2_WEIGHT_DECAY = 0.0005,
# ************************* Commonly used ********************************


# *********************** data format conversion *************************
CTPN_BOOK_PAGE_TAGS_FILE = os.path.join(DATA_DIR, "book_pages", "book_pages_tags_ctpn.txt")
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
CTPN_POS_ANCHOR_IOU = 0.65
CTPN_NEG_ANCHOR_IOU = 0.35
CTPN_PROPOSALS_MIN_SCORE = 0.7
CTPN_PROPOSALS_NMS_THRESH = 0.3
CTPN_PROPOSALS_MAX_NUM = 1500
CTPN_PROPOSALS_WIDTH = CTPN_ANCHORS_WIDTH
CTPN_USE_SIDE_REFINE = True

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
CTPN_MAX_HORIZONTAL_GAP = 40
CTPN_MIN_V_OVERLAPS = 0.7
CTPN_MIN_SIZE_SIM = 0.7
CTPN_TEXT_LINE_MIN_SCORE = 0.7
CTPN_MIN_NUM_PROPOSALS = 1
CTPN_TEXT_LINE_NMS_THRESH = 0.3
# **************************** Model ctpn ********************************


# **************************** Model crnn ********************************
# text line recognition
CRNN_CKPT_DIR = os.path.join(CURR_DIR, "recog_with_crnn", "ckpt")
CRNN_LOGS_DIR = os.path.join(CURR_DIR, "recog_with_crnn", "logs")
# **************************** Model crnn ********************************


# ************************** Segment model *******************************
SEGMENT_BASE_ROOT_DIR = os.path.join(CURR_DIR, "segment_base")

SEGMENT_BOOK_PAGE_ROOT_DIR = os.path.join(CURR_DIR, "segment_book_page")
SEGMENT_BOOK_PAGE_CKPT_DIR = os.path.join(CURR_DIR, "segment_book_page", "ckpt")
SEGMENT_BOOK_PAGE_LOGS_DIR = os.path.join(CURR_DIR, "segment_book_page", "logs")
SEGMENT_BOOK_PAGE_TAGS_FILE_H = os.path.join(CURR_DIR, "segment_book_page", "files", "book_page_tags_h.txt")
SEGMENT_BOOK_PAGE_TAGS_FILE_V = os.path.join(CURR_DIR, "segment_book_page", "files", "book_page_tags_v.txt")
SEGMENT_BOOK_PAGE_TFRECORDS_H = os.path.join(CURR_DIR, "segment_book_page", "files", "book_page_tfrecords_h.txt")
SEGMENT_BOOK_PAGE_TFRECORDS_V = os.path.join(CURR_DIR, "segment_book_page", "files", "book_page_tfrecords_v.txt")

SEGMENT_TEXT_LINE_ROOT_DIR = os.path.join(CURR_DIR, "segment_text_line")
SEGMENT_TEXT_LINE_CKPT_DIR = os.path.join(CURR_DIR, "segment_text_line", "ckpt")
SEGMENT_TEXT_LINE_LOGS_DIR = os.path.join(CURR_DIR, "segment_text_line", "logs")
SEGMENT_TEXT_LINE_TAGS_FILE_H = os.path.join(CURR_DIR, "segment_text_line", "files", "text_line_tags_h.txt")
SEGMENT_TEXT_LINE_TAGS_FILE_V = os.path.join(CURR_DIR, "segment_text_line", "files", "text_line_tags_v.txt")
SEGMENT_TEXT_LINE_TFRECORDS_H = os.path.join(CURR_DIR, "segment_text_line", "files", "text_line_tfrecords_h.txt")
SEGMENT_TEXT_LINE_TFRECORDS_V = os.path.join(CURR_DIR, "segment_text_line", "files", "text_line_tfrecords_v.txt")

SEGMENT_MIX_LINE_ROOT_DIR = os.path.join(CURR_DIR, "segment_mix_line")
SEGMENT_MIX_LINE_CKPT_DIR = os.path.join(CURR_DIR, "segment_mix_line", "ckpt")
SEGMENT_MIX_LINE_LOGS_DIR = os.path.join(CURR_DIR, "segment_mix_line", "logs")
SEGMENT_MIX_LINE_TAGS_FILE_H = os.path.join(CURR_DIR, "segment_mix_line", "files", "mix_line_tags_h.txt")
SEGMENT_MIX_LINE_TAGS_FILE_V = os.path.join(CURR_DIR, "segment_mix_line", "files", "mix_line_tags_v.txt")
SEGMENT_MIX_LINE_TFRECORDS_H = os.path.join(CURR_DIR, "segment_mix_line", "files", "mix_line_tfrecords_h.txt")
SEGMENT_MIX_LINE_TFRECORDS_V = os.path.join(CURR_DIR, "segment_mix_line", "files", "mix_line_tfrecords_v.txt")

SEGMENT_DOUBLE_LINE_ROOT_DIR = os.path.join(CURR_DIR, "segment_double_line")
SEGMENT_DOUBLE_LINE_CKPT_DIR = os.path.join(CURR_DIR, "segment_double_line", "ckpt")
SEGMENT_DOUBLE_LINE_LOGS_DIR = os.path.join(CURR_DIR, "segment_double_line", "logs")
SEGMENT_DOUBLE_LINE_TAGS_FILE_H = os.path.join(CURR_DIR, "segment_double_line", "files", "double_line_tags_h.txt")
SEGMENT_DOUBLE_LINE_TAGS_FILE_V = os.path.join(CURR_DIR, "segment_double_line", "files", "double_line_tags_v.txt")
SEGMENT_DOUBLE_LINE_TFRECORDS_H = os.path.join(CURR_DIR, "segment_double_line", "files", "double_line_tfrecords_h.txt")
SEGMENT_DOUBLE_LINE_TFRECORDS_V = os.path.join(CURR_DIR, "segment_double_line", "files", "double_line_tfrecords_v.txt")

SEGMENT_ROOT_DIR = {
    "book_page": SEGMENT_BOOK_PAGE_ROOT_DIR,
    "double_line": SEGMENT_DOUBLE_LINE_ROOT_DIR,
    "mix_line": SEGMENT_MIX_LINE_ROOT_DIR,
    "text_line": SEGMENT_TEXT_LINE_ROOT_DIR
}

SEGMENT_CKPT_DIR = {
    "book_page": SEGMENT_BOOK_PAGE_CKPT_DIR,
    "double_line": SEGMENT_DOUBLE_LINE_CKPT_DIR,
    "mix_line": SEGMENT_MIX_LINE_CKPT_DIR,
    "text_line": SEGMENT_TEXT_LINE_CKPT_DIR
}

SEGMENT_LOGS_DIR = {
    "book_page": SEGMENT_BOOK_PAGE_LOGS_DIR,
    "double_line": SEGMENT_DOUBLE_LINE_LOGS_DIR,
    "mix_line": SEGMENT_MIX_LINE_LOGS_DIR,
    "text_line": SEGMENT_TEXT_LINE_LOGS_DIR
}


SEGMENT_TASK_ID = {
    "book_page": 0,
    "mix_line": 1,
    "double_line": 2,
    "text_line": 3
}

SEGMENT_ID_TO_TASK = {
    0: "book_page",
    1: "mix_line",
    2: "double_line",
    3: "text_line"
}

SEGMENT_BATCH_SIZE = {
    "book_page": 1,
    "mix_line": 12,
    "double_line": 12,
    "text_line": 36
}

SEGMENT_FIXED_HEIGHT = {
    "book_page": 1024,
    "mix_line": 96,
    "double_line": 400, # 等宽缩放
    "text_line": 64
}

SEGMENT_MAX_INCLINATION = {
    "book_page": 28,
    "mix_line": 12,
    "double_line": 12,
    "text_line": 8
}

SEGMENT_FEATURE_STRIDE = {
    "book_page": 16,
    "mix_line": 16,
    "double_line": 16,
    "text_line": 16
}

SEGMENT_CLS_SCORE_THRESH = {
    "book_page": 0.7,
    "mix_line": 0.7,
    "double_line": 0.7,
    "text_line": 0.7
}

SEGMENT_DISTANCE_THRESH = {
    "book_page": 28,
    "mix_line": 20,
    "double_line": 16,
    "text_line": 0.6    # distance threshold ratio
}

SEGMENT_NMS_MAX_OUTPUTS = {
    "book_page": 50,
    "mix_line": 10,
    "double_line": 5,
    "text_line": 80
}

SEGMENT_LOSS_WEIGHTS = {
    "segment_class_loss": 1.,
    "segment_regress_loss": 1.
}

SEGMENT_LINE_WEIGHTS = {
    "split_line": 1.,  # book_page:1, mix_line:1, double_line:1, text_line:1
    "other_space": 1.,
    "pad_space": 1.
}
# ************************** Segment model *******************************


# ********** Chinese character & components recognition model ************
CHINESE_COMPO_ROOT_DIR = os.path.join(CURR_DIR, "chinese_components")
CHINESE_SPLIT_FILE = os.path.join(CHINESE_COMPO_ROOT_DIR, "chinese_split_basic.txt")
COMPO_SUMMARY_FILE = os.path.join(CHINESE_COMPO_ROOT_DIR, "chinese_compo_summary.txt")

CHAR_RECOG_ROOT_DIR = os.path.join(CURR_DIR, "recog_with_components")
CHAR_RECOG_CKPT_DIR = os.path.join(CURR_DIR, "recog_with_components", "ckpt")
CHAR_RECOG_LOGS_DIR = os.path.join(CURR_DIR, "recog_with_components", "logs")

CHAR_IMAGE_PATHS_FILE = os.path.join(CURR_DIR, "recog_with_components", "files", "char_image_paths.txt")
CHAR_TFRECORDS_PATHS_FILE = os.path.join(CURR_DIR, "recog_with_components", "files", "char_tfrecords_paths.txt")

CHAR_RECOG_BATCH_SIZE = 256

CHAR_RECOG_FEAT_STRIDE = 16
COMPO_SEQ_LENGTH = CHAR_IMG_SIZE // CHAR_RECOG_FEAT_STRIDE

CHAR_STRUC_TO_ID = {"s": 0, "⿰": 1,
                    # "⿱": 2
                    }

ID_TO_CHAR_STRUC = {0: "s", 1: "⿰",
                    # 2: "⿱"
                    }

TOP_K_TO_PRED = 10

CHAE_RECOG_LOSS_WEIGHTS = {
    "char_struc_loss": 1.,
    "sc_char_loss": 1.,
    "lr_compo_loss": 1.,
    # "ul_compo_loss": 1.
}
# ********** Chinese character & components recognition model ************