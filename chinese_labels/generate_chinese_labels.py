# -*- encoding: utf-8 -*-
# Author: hushukai

import os
CURR_DIR = os.path.dirname(__file__)


def generate_labels(label_file, chars_file_list):
    label_file_path = os.path.join(CURR_DIR, label_file)
    chars_file_paths = [os.path.join(CURR_DIR, chars_file) for chars_file in chars_file_list]

    # In tf.nn.ctc_loss, default blank label is 0. Here use "*" as blank.
    id2char_dict = {0: "*"}
    if os.path.exists(label_file_path):
        with open(label_file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                label_id, char = line.strip().split()[:2]
                label_id = int(label_id)
                if label_id not in id2char_dict:
                    id2char_dict[label_id] = char
                elif label_id == 0:
                    pass
                else:
                    raise ValueError("The label must be unique, but multiple label %d."%label_id)
    
    char2id_dict = { char:label_id for label_id, char in id2char_dict.items()}
    for chars_file in chars_file_paths:
        with open(chars_file, "r", encoding="utf-8") as fr:
            chars = fr.readline().strip()
            for char in chars:
                if char not in char2id_dict:
                    char2id_dict[char] = len(char2id_dict)
    
    id2char_dict = { label_id:char for char, label_id in char2id_dict.items()}
    dict_size = len(char2id_dict)
    with open(label_file_path, "w", encoding="utf-8") as fw:
        for label_id in range(dict_size):
            assert label_id in id2char_dict
            fw.write(str(label_id) + "\t" + id2char_dict[label_id] + "\n")


def generate_chinese_labels_simplified_level1():
    generate_labels(label_file="chinese_labels_simplified_level1.txt",
                    chars_file_list=["chars_GB2312_level1_3755.txt"])


def generate_chinese_labels_simplified_level2():
    generate_labels(label_file="chinese_labels_simplified_level2.txt",
                    chars_file_list=["chars_GB2312_both_levels_6763.txt"])
    

def generate_chinese_labels_traditional_common():
    generate_labels(label_file="chinese_labels_traditional_common.txt",
                    chars_file_list=["chars_Big5_common_traditional_5401.txt"])
    
    
def generate_chinese_labels_traditional_all():
    generate_labels(label_file="chinese_labels_traditional_all.txt",
                    chars_file_list=["chars_Big5_all_traditional_13053.txt"])
    

def generate_chinese_labels_mixed_common():
    generate_labels(label_file="chinese_labels_mixed_common.txt",
                    chars_file_list=["chars_GB2312_level1_3755.txt", "chars_Big5_common_traditional_5401.txt"])
    

def generate_chinese_labels_mixed_all():
    generate_labels(label_file="chinese_labels_mixed_all.txt",
                    chars_file_list=["chars_GB2312_both_levels_6763.txt", "chars_Big5_all_traditional_13053.txt"])


if __name__ == '__main__':
    # generate_chinese_labels_simplified_level1()
    # generate_chinese_labels_simplified_level2()
    # generate_chinese_labels_traditional_common()
    # generate_chinese_labels_traditional_all()
    # generate_chinese_labels_mixed_common()
    # generate_chinese_labels_mixed_all()
    
    print("Done !")
