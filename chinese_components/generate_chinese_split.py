# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import re
import numpy as np

from chinese_components.convert_chinese_number import convert_chinese_number

from config import CHINESE_COMPO_ROOT_DIR

CHINESE_STROKES_RAW_FILE     = os.path.join(CHINESE_COMPO_ROOT_DIR, "CJK统一汉字表(按笔画数排序).txt")
CHINESE_STROKES_FILE         = os.path.join(CHINESE_COMPO_ROOT_DIR, "chinese_strokes_num.txt")
CHINESE_SPLIT_RAW_FILE       = os.path.join(CHINESE_COMPO_ROOT_DIR, "IDS-UCS-Basic.txt")
CHINESE_SPLIT_FILE           = os.path.join(CHINESE_COMPO_ROOT_DIR, "chinese_split_basic.txt")
SIMILAR_COMPONENTS_FILE      = os.path.join(CHINESE_COMPO_ROOT_DIR, "similar_components.txt")
CHINESE_SPLIT_TABLE_REPLACED = os.path.join(CHINESE_COMPO_ROOT_DIR, "chinese_split_table_replaced.txt")
COMPONENTS_SUMMARY_FILE      = os.path.join(CHINESE_COMPO_ROOT_DIR, "chinese_compo_summary.txt")
CHINESE_STRUCTURES           = {'⿱', '⿰', '⿳', '⿲', '⿸', '⿹', '⿺', '⿶', '⿵', '⿷', '⿴', '⿻'}


def process_chinese_strokes():
    strokes_dict = {}
    with open(CHINESE_STROKES_RAW_FILE, "r", encoding="utf8") as fr:
        curr_strokes_num = None
        for line in fr:
            m = re.search(pattern=r"【(?P<strokes_num>.*?)画】", string=line)
            if m:
                curr_strokes_num = convert_chinese_number(chinese_str=m.group("strokes_num"))
                strokes_dict[curr_strokes_num] = ""
                continue
            if curr_strokes_num:
                chinese_chars = line.strip()[3:]
                strokes_dict[curr_strokes_num] += chinese_chars
    
    with open(CHINESE_STROKES_FILE, "w", encoding="utf8") as fw:
        for strokes_num, chinese_chars in strokes_dict.items():
            fw.write(str(strokes_num) + "\t" + chinese_chars + "\n")


def summarize_split_info():
    chinese_split_dict = {}
    with open(CHINESE_SPLIT_RAW_FILE, "r", encoding="utf-8") as fr:
        for line in fr:
            _, chinese_char, split_info = line.strip().split()[:3]
            chinese_split_dict[chinese_char] = split_info
    
    chinese_strokes_dict = {}
    with open(CHINESE_STROKES_FILE, "r", encoding="utf8") as fr:
        for line in fr:
            strokes_num, chinese_chars = line.strip().split()[:2]
            for char in chinese_chars:
                chinese_strokes_dict[char] = int(strokes_num)
    
    assert len(chinese_split_dict) == 20902 and len(chinese_strokes_dict) == 20902

    start, end = 0x4e00, 0x9fa5
    with open(CHINESE_SPLIT_FILE, "w", encoding="utf8") as fw:
        for i in range(start, end+1):
            chinese_char = chr(i)
            label = str(i - start)
            u_code = hex(i).upper()[2:]
            assert chinese_char in chinese_split_dict and chinese_char in chinese_strokes_dict
            strokes_num = str(chinese_strokes_dict[chinese_char])
            split_info = chinese_split_dict[chinese_char]
            fw.write(label + "\t" + u_code + "\t" + chinese_char + "\t" + strokes_num + "\t" + split_info + "\n")


def get_sub_compo(c, chinese_split_dict):
    if c not in chinese_split_dict or chinese_split_dict[c] == [c]:
        return [c]
    else:
        components = chinese_split_dict[c]
        sub_compo = []
        [sub_compo.extend(get_sub_compo(c, chinese_split_dict)) for c in components]
        return sub_compo


def define_chinese_components():
    chinese_split_dict = {}
    chinese_strokes_dict = {}
    with open(CHINESE_SPLIT_FILE, "r", encoding="utf-8") as fr:
        pattern = re.compile(r"&.*?;|.")
        for line in fr:
            _, u_code, chinese_char, strokes_num, split_info = line.strip().split()[:5]
            # u_code = int(u_code, base=16)
            components = re.findall(pattern, split_info)
            chinese_split_dict[chinese_char] = components
            chinese_strokes_dict[chinese_char] = int(strokes_num)
    
    real_split_dict = {}
    for chinese_char in chinese_split_dict.keys():
        real_split_dict[chinese_char] = get_sub_compo(chinese_char, chinese_split_dict)
        # print(chinese_char, ":", real_split_dict[chinese_char])
    
    components_set = set()
    [components_set.update(components) for components in real_split_dict.values()]
    components_set = list(components_set)
    components_set.sort()
    print(len(components_set), components_set)
    
    compo_chinese_dict = {}
    for chinese_char, components in real_split_dict.items():
        for compo in components:
            if compo not in compo_chinese_dict:
                compo_chinese_dict[compo] = set(chinese_char)
            else:
                compo_chinese_dict[compo].add(chinese_char)

    for compo in components_set:
        char_set = compo_chinese_dict[compo]
        print(compo, ":", len(char_set), char_set)


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
    # process_chinese_strokes()
    # summarize_split_info()
    define_chinese_components()
    # replace_similar_components()
    # encode_components()
    # chinese_components_recognition_dicts()
    
    print("Done !")
