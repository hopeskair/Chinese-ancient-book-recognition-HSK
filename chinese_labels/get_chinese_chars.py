# -*- encoding: utf-8 -*-
# Author: hushukai

import os
CURR_DIR = os.path.dirname(__file__)


def print_chars_to_file(chars, obj_file):
    with open(os.path.join(CURR_DIR, obj_file), "w", encoding="utf-8") as fw:
        fw.write(chars)
    return


def get_unicode_chars(intervals_list=[(0x4e00, 0x9fa5)]):
    # Unicode编码，基本汉字编码范围 4E00-9FA5，共20902个
    
    chars = ""
    for start_unicode, end_unicode in intervals_list:
        curr_unicode = start_unicode
        while curr_unicode <= end_unicode:
            chars += chr(curr_unicode)
            curr_unicode = curr_unicode + 1

    print_chars_to_file(chars, "chars_Unicode_all_chinese_20902.txt")
    return


def get_chars_by_encoding(intervals_list, encoding):
    chars = ""
    
    for start, end in intervals_list:
        curr = start
        while curr <= end:
            try:
                chars += bytes().fromhex(hex(curr)[2:]).decode(encoding=encoding)
            except UnicodeDecodeError:
                pass
            curr = curr + 1
    
    return chars


def get_big5_common_chinese_chars(big5_intervals_list=[(0xa440, 0xc67e)]):
    # Big5编码，常用汉字编码范围 A440-C67E，共5401个
    chars = get_chars_by_encoding(big5_intervals_list, encoding="big5")
    print_chars_to_file(chars, "chars_Big5_common_traditional_5401.txt")
    

def get_big5_all_chinese_chars(big5_intervals_list=[(0xa440, 0xc67e), (0xc940, 0xf9dc)]):
    # Big5编码，汉字编码范围 A440-F9DC，其中包含日语字符，故用分段区间表示
    chars = get_chars_by_encoding(big5_intervals_list, encoding="big5")
    print_chars_to_file(chars, "chars_Big5_all_traditional_13053.txt")
    

if __name__ == '__main__':
    # get_unicode_chars()
    get_big5_common_chinese_chars()
    # get_big5_all_chinese_chars()
    
    print("Done !")
