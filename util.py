# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import shutil

from chinese_components.char2compo import chinese_components_info

from config import CHINESE_LABEL_FILE, TRADITION_CHARS_FILE
from config import IGNORABLE_CHARS_FILE, IMPORTANT_CHARS_FILE


def check_or_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def remove_then_makedirs(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


# json字典的key只能是字符串，python字典的key可以是 str, int, float, tuple
def chinese_labels_dict():
    assert os.path.exists(CHINESE_LABEL_FILE), "Label file does not exist!"
    
    id2char_dict = {}
    char2id_dict = {}
    with open(CHINESE_LABEL_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            label_id, char = line.strip().split()[:2]
            label_id = int(label_id)
            if label_id not in id2char_dict and char not in char2id_dict:
                id2char_dict[label_id] = char
                char2id_dict[char] = label_id
            elif label_id in id2char_dict:
                raise ValueError("The label must be unique, but multiple label %d."%label_id)
            else:
                raise ValueError("The character must be unique, but multiple character %s."%char)
    
    num_chars = len(char2id_dict)
    
    return id2char_dict, char2id_dict, num_chars


def ignorable_chars():
    chars = set()
    with open(IGNORABLE_CHARS_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            chinese_char = line.strip()[0]
            chars.add(chinese_char)
    return "".join(chars)


def important_chars():
    chars = set()
    with open(IMPORTANT_CHARS_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            chinese_char = line.strip()[0]
            chars.add(chinese_char)
    return "".join(chars)


# General tasks
ID2CHAR_DICT, CHAR2ID_DICT, NUM_CHARS = chinese_labels_dict()
BLANK_CHAR = ID2CHAR_DICT[0]
IGNORABLE_CHARS = ignorable_chars()
IMPORTANT_CHARS = important_chars()


def traditional_chars():
    with open(TRADITION_CHARS_FILE, "r", encoding="utf-8") as fr:
        tradition_chars = fr.read()
        tradition_chars = tradition_chars.strip()
    tradition_chars = "".join([c for c in tradition_chars if c in CHAR2ID_DICT])
    return tradition_chars


TRADITION_CHARS = traditional_chars()

# Chinese components recognition task
CHAR_TO_COMPO_SEQ, COMPO_SEQ_TO_CHAR, NUM_CHAR_STRUC, NUM_SIMPLE_CHAR, NUM_LR_COMPO = chinese_components_info()


if __name__ == '__main__':
    # print(ignorable_chars())
    print("Done !")