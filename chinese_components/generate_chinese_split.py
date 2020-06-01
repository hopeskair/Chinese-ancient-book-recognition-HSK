# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import re
import json

from config import CHINESE_COMPO_ROOT_DIR


CHINESE_STROKES_RAW_FILE    = os.path.join(CHINESE_COMPO_ROOT_DIR, "UCS_strokes.txt")
CHINESE_STROKES_FILE        = os.path.join(CHINESE_COMPO_ROOT_DIR, "chinese_strokes_num.txt")
CHINESE_SPLIT_RAW_FILE      = os.path.join(CHINESE_COMPO_ROOT_DIR, "UCS_IDS.txt")
CHINESE_SPLIT_CORRECT_FILE  = os.path.join(CHINESE_COMPO_ROOT_DIR, "UCS_IDS_to_correct.txt")
CHINESE_SPLIT_FILE          = os.path.join(CHINESE_COMPO_ROOT_DIR, "chinese_split_basic.txt")
CHINESE_STRUCTURES          = {'⿱', '⿰', '⿳', '⿲', '⿸', '⿹', '⿺', '⿶', '⿵', '⿷', '⿴', '⿻'}
UNICODE_HEAD, UNICODE_TAIL  = 0x4e00, 0x9fa5


def process_chinese_strokes():
    strokes_dict = {}
    with open(CHINESE_STROKES_RAW_FILE, "r", encoding="utf8") as fr:
        for line in fr:
            if line.startswith("U+") or line.startswith("CDP-"):
                _, char, strokes_num = line.strip().split()[:3]
                if len(char) == 1 and UNICODE_HEAD <= ord(char) <= UNICODE_TAIL:
                    if "," in strokes_num:
                        strokes_num = strokes_num.split(",")[0]
                    strokes_num = int(strokes_num)
                    if strokes_num not in strokes_dict:
                        strokes_dict[strokes_num] = char
                    else:
                        strokes_dict[strokes_num] += char
    
    with open(CHINESE_STROKES_FILE, "w", encoding="utf8") as fw:
        for strokes_num, chinese_chars in strokes_dict.items():
            fw.write(str(strokes_num) + "\t" + chinese_chars + "\n")


def parse_split_seq(split_seq):
    assert split_seq[0] in CHINESE_STRUCTURES
    curr_struc = split_seq[0]
    compo_num = 3 if curr_struc in "⿳⿲" else 2
    
    result_seq = [curr_struc,]
    curr_pos = 1
    for _ in range(compo_num):
        curr_e = split_seq[curr_pos]
        if curr_e in CHINESE_STRUCTURES:
            composite_compo, composite_num = parse_split_seq(split_seq[curr_pos:])
            result_seq.append(composite_compo)
            curr_pos += composite_num
        else:
            single_compo = curr_e
            result_seq.append(single_compo)
            curr_pos += 1
    
    return result_seq, curr_pos


def list_to_str(composite_compo):
    result_str = ""
    for e in composite_compo:
        if isinstance(e, str):
            result_str += e
        else:
            assert isinstance(e, list)
            result_str += list_to_str(e)
    return result_str
    

def get_all_compo(c, split_dict):
    if c not in split_dict or split_dict[c] == [c]:
        return [c]
    elif len(split_dict[c]) == 1:
        return split_dict[c]
    else:
        components = split_dict[c]
        sub_compo = []
        [sub_compo.extend(get_all_compo(c, split_dict)) for c in components]
        return sub_compo + components


def get_sub_compo_by_struc(c, split_dict=None, base_struc=None):
    if isinstance(c, str):
        # single component
        assert split_dict is not None
        if c not in split_dict or split_dict[c] == [c]:
            return [c]
        elif len(split_dict[c]) == 1:
            return split_dict[c]
        split_seq = split_dict[c]
        if base_struc is None:
            if split_seq[0] == "⿻":
                return [c]
            elif split_seq[0] in "⿰⿲⿺":
                base_struc = "⿰"
            elif split_seq[0] in "⿱⿳⿸":
                base_struc = "⿱"
            else:
                # enclosed structure
                # base_struc = "⿴"
                return [c]
    else:
        # composite component
        assert isinstance(c, list)
        split_seq = c
    
    curr_struc = split_seq[0]
    if curr_struc in "⿰⿲⿺":
        curr_struc = "⿰"
    elif curr_struc in "⿱⿳⿸":
        curr_struc = "⿱"
    else:
        curr_struc = "other"
    
    if curr_struc != base_struc:
        if isinstance(c, str):
            return [c]
        else:
            assert isinstance(c, list)
            c_str = list_to_str(composite_compo=c)
            return [c_str]
    else:
        # curr_struc == base_struc
        compo_list = []
        for sub_compo in split_seq[1:]:
            compo_list += get_sub_compo_by_struc(sub_compo, split_dict, base_struc=base_struc)
        return compo_list


def compo_strokes(compo, strokes_dict):
    if len(compo) == 1 or compo.startswith("&CDP-"):
        return strokes_dict[compo]
    else:
        pattern2 = re.compile(r"&.*?;|.")
        compo_seq = re.findall(pattern2, compo)
        total_strokes = 0
        for c in compo_seq:
            if c in CHINESE_STRUCTURES: continue
            if len(c) == 1 or c.startswith("&CDP-"):
                total_strokes += strokes_dict[c]
            elif c.startswith("&") and c.endswith(";"):
                total_strokes += sum([strokes_dict[_c] for _c in c[1:-1]])
        return total_strokes


def to_sort(compo_seq, strokes_dict):
    str_type = isinstance(compo_seq, str)
    compo_seq = list(compo_seq)
    # compo_seq.sort()
    compo_seq.sort(key=lambda c: compo_strokes(c, strokes_dict))
    if str_type:
        compo_seq = "".join(compo_seq)
    return compo_seq


def extract_split_info():
    # extract char strokes_num
    strokes_dict = {}
    with open(CHINESE_STROKES_RAW_FILE, "r", encoding="utf8") as fr:
        for line in fr:
            if line.startswith("U+") or line.startswith("CDP-"):
                _, char, strokes_num = line.strip().split()[:3]
                if "," in strokes_num:
                    strokes_num = strokes_num.split(",")[0]
                strokes_dict[char] = int(strokes_num)
    
    # extract char split_info
    split_dict = {}
    pattern1 = re.compile(r"^.*(?=\[.*?\]$)")
    pattern2 = re.compile(r"&.*?;|.")
    with open(CHINESE_SPLIT_RAW_FILE, "r", encoding="utf8") as fr:
        for line in fr:
            if line.startswith("U+") or line.startswith("CDP-"):
                _, char, split_info = line.strip().split()[:3]
                m = re.search(pattern1, split_info)
                if m:
                    split_info = m.group(0)
                split_seq = re.findall(pattern2, split_info)
                split_dict[char] = split_seq
    
    # # list all char and compo by their first sub-compo
    # temp_dict = {}
    # temp_set = set()
    # for i in range(UNICODE_HEAD, UNICODE_TAIL + 1):
    #     chinese_char = chr(i)
    #     # label = str(i - UNICODE_HEAD)
    #     # u_code = hex(i).upper()[2:]
    #     assert chinese_char in split_dict and chinese_char in strokes_dict
    #     char_compo = get_all_compo(chinese_char, split_dict)
    #     temp_set.update(char_compo)
    #     temp_set.add(chinese_char)
    # _all = set()
    # for char in temp_set:
    #     if char in CHINESE_STRUCTURES: continue
    #     char_compo = split_dict.get(char, "-")
    #     if char_compo[0] in "⿱⿳":
    #         _all.add(char)
    #         c = char_compo[1] if char_compo[1] not in CHINESE_STRUCTURES else char_compo[2]
    #         # for c in char_compo:
    #         if c not in temp_dict:
    #             temp_dict[c] = {char}
    #         else:
    #             temp_dict[c].add(char)
    # print(len(_all), _all)
    # all_compo = list(temp_dict.keys())
    # all_compo.sort(key=lambda c: strokes_dict.get(c, -1))
    # # all_compo.sort(key=lambda c: strokes_dict.get(c, -1))
    # for c in all_compo:
    #     chars = list(temp_dict[c])
    #     chars.sort(key=lambda c: (len(c), c))
    #     print(c+":", "".join(chars))
    
    # correct char split_info
    with open(CHINESE_SPLIT_CORRECT_FILE, "r", encoding="utf8") as fr:
        for line in fr:
            line = line.strip()
            if line.startswith("#"): continue
            line = line.split()
            if len(line) == 1:
                chars = re.findall(pattern2, line[0])
                for char in chars:
                    split_dict[char] = [char]
            if len(line) == 4:
                char, old_split_info, _, new_split_info = line
                new_split_seq = re.findall(pattern2, new_split_info)
                split_dict[char] = new_split_seq

    # convert split_dict format
    for char, split_seq in split_dict.items():
        if len(split_seq) > 1:
            init_length = len(split_seq)
            split_seq, last_pos = parse_split_seq(split_seq)
            assert last_pos == init_length
            split_dict[char] = split_seq

    # save components sequence
    simple_chars, nested_chars, enclosed_chars = "", "", ""
    lr_compo_set, ul_compo_set = set(), set()
    lr_compo_dict, ul_compo_dict = dict(), dict()
    with open(CHINESE_SPLIT_FILE, "w", encoding="utf8") as fw:
        for i in range(UNICODE_HEAD, UNICODE_TAIL+1):
            chinese_char = chr(i)
            label = str(i - UNICODE_HEAD)
            u_code = hex(i).upper()[2:]
            assert chinese_char in split_dict and chinese_char in strokes_dict
            strokes_num = str(strokes_dict[chinese_char])
            
            split_seq = split_dict[chinese_char]
            split_json = None
            if len(split_seq) == 1:
                simple_chars += chinese_char
                # split_json = json.dumps([chinese_char])
                split_json = chinese_char
            else:
                base_struc = split_seq[0]
                assert base_struc in CHINESE_STRUCTURES

                if base_struc == "⿻":
                    nested_chars += chinese_char
                elif base_struc in "⿰⿲⿺":
                    base_struc = "⿰"
                    sub_components = get_sub_compo_by_struc(chinese_char, split_dict, base_struc=base_struc)
                    # split_json = json.dumps([base_struc] + sub_components)
                    split_json = ",".join([base_struc] + sub_components)
                    lr_compo_set.update(sub_components)
                    for c in set(sub_components):
                        if c not in lr_compo_dict:
                            lr_compo_dict[c] = chinese_char
                        else:
                            lr_compo_dict[c] += chinese_char
                elif base_struc in "⿱⿳⿸":
                    base_struc = "⿱"
                    sub_components = get_sub_compo_by_struc(chinese_char, split_dict, base_struc=base_struc)
                    # split_json = json.dumps([base_struc] + sub_components)
                    split_json = ",".join([base_struc] + sub_components)
                    ul_compo_set.update(sub_components)
                    for c in set(sub_components):
                        if c not in ul_compo_dict:
                            ul_compo_dict[c] = chinese_char
                        else:
                            ul_compo_dict[c] += chinese_char
                else:
                    # enclosed structure
                    enclosed_chars += chinese_char
            
            assert split_json is not None
            fw.write(label + "\t" + u_code + "\t" + chinese_char + "\t" + strokes_num + "\t" + split_json + "\n")
        
        # print(len(simple_chars), to_sort(simple_chars, strokes_dict))
        # print(len(nested_chars), to_sort(nested_chars, strokes_dict))
        # print(len(enclosed_chars), to_sort(enclosed_chars, strokes_dict))
        # print(len(ul_compo_set), to_sort(ul_compo_set, strokes_dict))
        # print(len(lr_compo_set), to_sort(lr_compo_set, strokes_dict))
        # for c in to_sort(lr_compo_set, strokes_dict):
        #     print(c, ":", to_sort(lr_compo_dict[c], strokes_dict))


if __name__ == '__main__':
    # process_chinese_strokes()
    extract_split_info()
    
    print("Done !")
