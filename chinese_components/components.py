# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import numpy as np

from chinese_components.crawler import get_char_split_info

from config import CHINESE_COMPO_ROOT_DIR


CHINESE_COMPONENTS_FILE      = os.path.join(CHINESE_COMPO_ROOT_DIR, "char_split_table_normal.txt")
MISSING_CHARS_FILE           = os.path.join(CHINESE_COMPO_ROOT_DIR, "missing_chars_in_split_table.txt")
CHINESE_COMPONENTS_CRAWLED   = os.path.join(CHINESE_COMPO_ROOT_DIR, "char_split_table_crawled.txt")
CHINESE_SPLIT_TABLE          = os.path.join(CHINESE_COMPO_ROOT_DIR, "chinese_split_table.txt")
SIMILAR_COMPONENTS_FILE      = os.path.join(CHINESE_COMPO_ROOT_DIR, "similar_components.txt")
CHINESE_SPLIT_TABLE_REPLACED = os.path.join(CHINESE_COMPO_ROOT_DIR, "chinese_split_table_replaced.txt")
COMPONENTS_SUMMARY_FILE      = os.path.join(CHINESE_COMPO_ROOT_DIR, "chinese_compo_summary.txt")
CHINESE_STRUCTURES           = {'⿱', '⿰', '⿳', '⿲', '⿸', '⿹', '⿺', '⿶', '⿵', '⿷', '⿴', '⿻'}


def missing_chars_in_split_table():
    chinese_split_set = set()
    with open(CHINESE_COMPONENTS_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            _, u_code, _, components = line.strip().split("\t")[:4]
            u_code = int(u_code, base=16)
            chinese_split_set.add(u_code)

    missing_unicodes = []
    for i in range(0x4E00, 0x9FA5 + 1):
        if i not in chinese_split_set:
            print(i, chr(i))
            missing_unicodes.append(hex(i)[2:])
            
    with open(MISSING_CHARS_FILE, "w", encoding="utf-8") as fw:
        fw.write("\n".join(missing_unicodes))


def crawle_missing_split_info():
    with open(MISSING_CHARS_FILE, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
    unicodes = [int(line.strip(), base=16) for line in lines]
    get_char_split_info(unicode_list=unicodes)


def combine_split_table():
    chinese_split_dict = {}
    with open(CHINESE_COMPONENTS_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            _, u_code, _, components = line.strip().split("\t")[:4]
            u_code = int(u_code, base=16)
            chinese_split_dict.update({u_code: components})
    
    with open(CHINESE_COMPONENTS_CRAWLED, "r", encoding="utf-8") as fr:
        for line in fr:
            u_code, _, components = line.strip().split("\t")[:3]
            u_code = int(u_code, base=16)
            chinese_split_dict.update({u_code: components})
    
    with open(CHINESE_SPLIT_TABLE, "w", encoding="utf-8") as fw:
        for u_code in range(0x4E00, 0x9FA5 + 1):
            assert u_code in chinese_split_dict
            chinese_char = chr(u_code)
            line_str = hex(u_code).upper()[2:] + "\t" + chinese_char + "\t" + chinese_split_dict[u_code] + "\n"
            fw.write(line_str)


def replace_similar_components():
    chinese_split_dict = {}
    with open(CHINESE_SPLIT_TABLE, "r", encoding="utf-8") as fr:
        for line in fr:
            u_code, _, components = line.strip().split("\t")[:4]
            u_code = int(u_code, base=16)
            chinese_split_dict.update({u_code: components})
    
    similar_components = {}
    with open(SIMILAR_COMPONENTS_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            raw_ccode, _, dest_ccode = line.strip().split()[:3]
            similar_components.update({raw_ccode: dest_ccode})
    
    with open(CHINESE_SPLIT_TABLE_REPLACED, "w", encoding="utf-8") as fw:
        for u_code in range(0x4E00, 0x9FA5 + 1):
            assert u_code in chinese_split_dict
            char_id = str(u_code - 0x4E00)
            chinese_char = chr(u_code)
            components = chinese_split_dict[u_code]
            components = components.split("/")
            components = [similar_components[c] if c in similar_components else c
                          for c in components]
            components = "/".join(set(components))
            u_code = hex(u_code).upper()[2:]
            line_str = char_id + "\t" + u_code + "\t" + chinese_char + "\t" + components + "\n"
            fw.write(line_str)


def encode_components():
    components_list = []
    component_to_chinese_dict = {}
    component_to_unicode_dict = {}
    with open(CHINESE_SPLIT_TABLE_REPLACED, "r", encoding="utf-8") as fr:
        for line in fr:
            _, u_code, _, components = line.strip().split("\t")[:4]
            chinese_char = chr(int(u_code, base=16))
            components = components.split("/")
            for c in components:
                if c not in components_list:
                    components_list.append(c)
                    component_to_chinese_dict[c] = [chinese_char,]
                    component_to_unicode_dict[c] = [u_code,]
                else:
                    component_to_chinese_dict[c].append(chinese_char)
                    component_to_unicode_dict[c].append(u_code)

    with open(COMPONENTS_SUMMARY_FILE, "w", encoding="utf-8") as fw:
        for compo_id, component in enumerate(components_list):
            chinese_chars = "".join(component_to_chinese_dict[component])
            unicodes = "/".join(component_to_unicode_dict[component])
            line_str = str(compo_id) + "\t" + component + "\t" + chinese_chars + "\t" + unicodes + "\n"
            fw.write(line_str)


def co_occurrence_prob_of_components(char_id2compo_indices, num_compo):
    compo_counter = np.zeros([num_compo, ], dtype=np.float32)
    co_occurrences = np.zeros([num_compo, num_compo], dtype=np.float32)
    
    for compo_indices in char_id2compo_indices.values():
        compo_counter[compo_indices] += 1
        num_indices = len(compo_indices)
        if num_indices == 1: continue
        for i in range(0, num_indices - 1):
            for j in range(i + 1, num_indices):
                compo1 = compo_indices[i]
                compo2 = compo_indices[j]
                co_occurrences[compo1, compo2] += 1
                co_occurrences[compo2, compo1] += 1
    
    compo_counter = np.expand_dims(compo_counter, axis=-1)
    co_occurrence_prob = co_occurrences / compo_counter
    
    return co_occurrence_prob


def chinese_components_recognition_dicts():
    assert os.path.exists(CHINESE_SPLIT_TABLE_REPLACED)
    assert os.path.exists(COMPONENTS_SUMMARY_FILE)
    
    id2char_dict = {}
    char2id_dict = {}
    id2compo_dict = {}
    with open(CHINESE_SPLIT_TABLE_REPLACED, "r", encoding="utf-8") as fr:
        for line in fr:
            label_id, _, chinese_char, components = line.strip().split("\t")[:4]
            label_id = int(label_id)
            if label_id not in id2char_dict and chinese_char not in char2id_dict:
                id2char_dict[label_id] = chinese_char
                char2id_dict[chinese_char] = label_id
                id2compo_dict[label_id] = components
            elif label_id in id2char_dict:
                raise ValueError("The label must be unique, but multiple label %d." % label_id)
            else:
                raise ValueError("The character must be unique, but multiple character %s." % chinese_char)
    
    cid2compo_dict = {}
    compo2cid_dict = {}
    cid2chars_dict = {}
    with open(COMPONENTS_SUMMARY_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            cid, component, chinese_chars = line.strip().split("\t")[:3]
            cid = int(cid)
            if cid not in cid2compo_dict and component not in compo2cid_dict:
                cid2compo_dict[cid] = component
                compo2cid_dict[component] = cid
                cid2chars_dict[cid] = chinese_chars
            elif cid in cid2compo_dict:
                raise ValueError("The component id must be unique, but multiple id %d." % cid)
            else:
                raise ValueError("The component must be unique, but multiple component %s." % component)
    
    num_chars = len(id2char_dict)
    num_compo = len(cid2compo_dict)
    
    id2compo_indices = {}
    for char_id, components in id2compo_dict.items():
        components = components.split("/")
        compo_indices = [compo2cid_dict[c] for c in components]
        id2compo_indices[char_id] = compo_indices

    cid2char_indices = {}
    for cid, chinese_chars in cid2chars_dict.items():
        char_indices = [char2id_dict[char] for char in chinese_chars]
        cid2char_indices[cid] = char_indices

    compo_co_occurrence_prob = co_occurrence_prob_of_components(id2compo_indices, num_compo)
    
    return id2char_dict, char2id_dict, num_chars, num_compo, id2compo_indices, cid2char_indices, compo_co_occurrence_prob


if __name__ == '__main__':
    # missing_chars_in_split_table()
    # crawle_missing_split_info()
    # combine_split_table()
    # replace_similar_components()
    # encode_components()
    chinese_components_recognition_dicts()
    
    print("Done !")
