# -*- encoding: utf-8 -*-
# Author: hushukai

import os
from pprint import pprint

from config import CHINESE_SPLIT_FILE, COMPO_SUMMARY_FILE


def encode_components():
    simple_chars = ""
    lr_compo_set, ul_compo_set = set(), set()
    lr_compo_dict, ul_compo_dict = dict(), dict()
    with open(CHINESE_SPLIT_FILE, "r", encoding="utf8") as fr:
        for line in fr:
            _, u_code, _, _, components = line.strip().split("\t")[:5]
            chinese_char = chr(int(u_code, base=16))
            components = components.split(",")
            
            if components[0] == "⿰":
                lr_compo_set.update(components[1:])
                for c in components[1:]:
                    if c not in lr_compo_dict:
                        lr_compo_dict[c] = chinese_char
                    elif chinese_char not in lr_compo_dict[c]:
                        lr_compo_dict[c] += chinese_char
            elif components[0] == "⿱":
                ul_compo_set.update(components[1:])
                for c in components[1:]:
                    if c not in ul_compo_dict:
                        ul_compo_dict[c] = chinese_char
                    elif chinese_char not in ul_compo_dict[c]:
                        ul_compo_dict[c] += chinese_char
            else:
                assert len(components) == 1
                simple_chars += chinese_char

    lr_compo_set, ul_compo_set = list(lr_compo_set), list(ul_compo_set)
    lr_compo_set.sort(key=lambda c: (len(c), c))
    ul_compo_set.sort(key=lambda c: (len(c), c))
    lr_compo_set.insert(0, "EOC")   # End of Character
    ul_compo_set.insert(0, "EOC")   # End of Character
    
    with open(COMPO_SUMMARY_FILE, "w", encoding="utf8") as fw:
        
        fw.write("# simple chinese characters\n")
        for s_cid, s_char in enumerate(simple_chars):
            line_str = str(s_cid) + "\t" + s_char + "\n"
            fw.write(line_str)
        
        fw.write("\n\n# ⿰, components of left-right chinese characters\n")
        for lr_cid, lr_compo in enumerate(lr_compo_set):
            line_str = str(lr_cid) + "\t" + lr_compo + "\t" + lr_compo_dict.get(lr_compo, "EOC") + "\n"
            fw.write(line_str)
        
        # fw.write("\n\n# ⿱, components of upper-lower chinese characters\n")
        # for ul_cid, ul_compo in enumerate(ul_compo_set):
        #     line_str = str(ul_cid) + "\t" + ul_compo + "\t" + ul_compo_dict.get(ul_compo, "EOC") + "\n"
        #     fw.write(line_str)


def get_compo2cid_dict():
    sc_char2cid, lr_compo2cid = dict(), dict()
    with open(COMPO_SUMMARY_FILE, "r", encoding="utf8") as fr:
        work_dict = None
        for line in fr:
            line = line.strip()
            if len(line) == 0: continue
            if line.startswith("#"):
                if "simple" in line: work_dict = sc_char2cid
                if "⿰" in line: work_dict = lr_compo2cid
                # if "⿱" in line: work_dict = ul_compo2cid
                continue
            cid, c = line.split()[:2]
            work_dict[c] = int(cid)
    return sc_char2cid, lr_compo2cid


def chinese_components_info():
    assert os.path.exists(CHINESE_SPLIT_FILE)
    if not os.path.exists(COMPO_SUMMARY_FILE):
        encode_components()

    num_char_struc = 2  # simple, left-right
    sc_char2cid, lr_compo2cid = get_compo2cid_dict()
    num_simple_char, num_lr_compo = len(sc_char2cid), len(lr_compo2cid)
    
    char_to_compo_seq, compo_seq_to_char = dict(), dict()
    with open(CHINESE_SPLIT_FILE, "r", encoding="utf8") as fr:
        for line in fr:
            _, u_code, _, _, components = line.strip().split("\t")[:5]
            chinese_char = chr(int(u_code, base=16))
            components = components.split(",")
            
            if components[0] == "⿰":
                compo_info = "⿰" + ",".join([str(lr_compo2cid[c]) for c in components[1:]])
            # elif components[0] == "⿱":
            else:
                assert len(components) == 1
                compo_info = "s" + str(sc_char2cid[chinese_char])
            
            char_to_compo_seq[chinese_char] = compo_info
            compo_seq_to_char[compo_info] = chinese_char
    
    # pprint(char_to_compo_seq)
    # pprint(compo_seq_to_char)
    return char_to_compo_seq, compo_seq_to_char, num_char_struc, num_simple_char, num_lr_compo


if __name__ == '__main__':
    # encode_components()
    chinese_components_info()
    
    print("Done !")
