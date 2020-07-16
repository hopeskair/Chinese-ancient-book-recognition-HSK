# -*- encoding: utf-8 -*-
# Author: hushukai

from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import math
import os
import sys

try:
    from skimage.filters import threshold_adaptive
except:
    from skimage.filters import threshold_local as threshold_adaptive
from skimage.filters import threshold_otsu
from skimage import io, transform, color, util


def adjust_image_contrast(np_page):
    # 求foreground和background的分割阈值, 全局阈值
    thresh = threshold_otsu(np_page)
    
    if book_page_too_dark(np_page, thresh=thresh):
        x1, y1, x2, y2 = find_dark_page_box(np_page, thresh=thresh)
        np_page = np_page[y1:y2 + 1, x1:x2 + 1]
        thresh = threshold_otsu(np_page)

    greater_v = np_page * 1.5
    if (greater_v >= 255).astype(np.int16).sum() / (np_page > thresh).astype(np.int16).sum() < 0.8:
        greater_v *= 1.5
    greater_img = np.where(greater_v > 255, 255, greater_v).astype(dtype=np.uint8)

    # # np_img-(255-np_img)*0.5 -> np_img*1.5-128 -> greater_v-128
    # less_v = greater_v - 128
    # less_img = np.where(less_v < 0, 0, less_v).astype(dtype=np.uint8)

    np_page = np.where(np_page > thresh, greater_img, 0).astype(dtype=np.uint8)

    return np_page


def book_page_too_dark(np_page, thresh):
    page_height, page_width = np_page.shape[:2]

    ksize = round(min(page_height, page_width) * 0.25)
    stride = 30
    for i in range(0, page_height, stride):
        for j in range(0, page_width, stride):
            if i + ksize > page_height or j + ksize > page_width:
                continue

            kernel = np_page[i:i + ksize, j:j + ksize]
            if (kernel <= thresh).astype(np.int16).sum() == kernel.size:
                return True

    return False


def find_dark_page_box(np_page, thresh):
    page_height, page_width = np_page.shape[:2]
    all_black_blocks = find_out_black_blocks(np_page, thresh=thresh)

    blocks_size = [len(block) for block in all_black_blocks]
    max_block = np.array(all_black_blocks[blocks_size.index(max(blocks_size))])

    # 假设black_block近似为矩形，接下来寻找其最大内接矩形
    block_y, block_x = max_block[:, 0], max_block[:, 1]

    center_y = np.mean(block_y).round().astype(np.int32)
    center_x = np.mean(block_x).round().astype(np.int32)

    center_h = np.where(block_x == center_x)[0].size
    center_w = np.where(block_y == center_y)[0].size

    dh = round(center_h * 0.7 * 0.5)
    dw = round(center_w * 0.7 * 0.5)

    x = center_x
    while x >= 0 and (np_page[center_y - dh:center_y + dh, x] <= thresh).astype(np.int16).sum() == 2 * dh:
        x -= 1
    x1 = x + 1

    x = center_x
    while x < page_width and (np_page[center_y - dh:center_y + dh, x] <= thresh).astype(np.int16).sum() == 2 * dh:
        x += 1
    x2 = x - 1

    y = center_y
    while y >= 0 and (np_page[y, center_x - dw:center_x + dw] <= thresh).astype(np.int16).sum() == 2 * dw:
        y -= 1
    y1 = y + 1

    y = center_y
    while y < page_height and (np_page[y, center_x - dw:center_x + dw] <= thresh).astype(np.int16).sum() == 2 * dw:
        y += 1
    y2 = y - 1

    page_box = (x1, y1, x2, y2)

    return page_box


def find_out_black_blocks(np_page, thresh):
    page_height, page_width = np_page.shape[:2]

    all_black_blocks = []
    ksize = round(min(page_height, page_width) * 0.25)
    stride = 30
    flag = np.zeros_like(np_page, dtype=np.int8)
    for i in range(0, page_height, stride):
        for j in range(0, page_width, stride):

            if flag[i, j] == 1:
                continue
            flag[i, j] = 1

            kernel = np_page[i:i + ksize, j:j + ksize]
            if (kernel <= thresh).astype(np.int16).sum() < kernel.size:
                continue

            black_block = BFS(i, j, np_page, flag, thresh)
            all_black_blocks.append(black_block)

    return all_black_blocks


def BFS(i, j, np_page, flag, thresh):
    # 广度优先搜索
    assert np_page[i, j] < thresh and flag[i, j] == 1
    black_block = [(i, j), ]

    page_height, page_width = np_page.shape[:2]

    queue = [(i, j), ]
    while len(queue) > 0:
        i, j = queue.pop(0)
        for new_i, new_j in [(i, j - 1), (i - 1, j), (i, j + 1), (i + 1, j)]:
            if new_i < 0 or new_i >= page_height or new_j < 0 or new_j >= page_width:
                continue

            if flag[new_i, new_j] == 1:
                continue

            if np_page[new_i, new_j] <= thresh:
                black_block.append((new_i, new_j))
                queue.append((new_i, new_j))

            flag[new_i, new_j] = 1

    return black_block


def binarization(src_image):
    if len(src_image.shape) == 3:
        image = (src_image.sum(axis=2) / 3).astype('ubyte')
    else:
        image = src_image

    # 求foreground和background的分割阈值, 全局阈值
    thresh = threshold_otsu(image=image)
    binary = (image > thresh).astype('ubyte')

    # 局部阈值
    # thresh_arr = threshold_adaptive(image, block_size=45, offset=0)
    # binary = (image > thresh_arr).astype('ubyte')

    # temp_img = ((image > thresh).astype('ubyte') * 255).astype('ubyte')
    # Image.fromarray(temp_img).show()

    return binary


def crop_page_margin(np_page):
    binary = binarization(src_image=np_page)
    binary = (1 - binary).astype('ubyte')

    # 裁剪上下边缘
    rows_sum = binary.sum(axis=1)
    crop_up, crop_down = find_optimal_crop_margin(values_seq=rows_sum)

    # 裁剪左右边缘
    cols_sum = binary[crop_up:crop_down + 1, :].sum(axis=0)
    crop_left, crop_right = find_optimal_crop_margin(values_seq=cols_sum)

    crop_page = np_page[crop_up:crop_down + 1, crop_left:crop_right + 1]

    # bar_chart_by_matplotlib(values_seq=cols_sum)

    return crop_page


def find_optimal_crop_margin(values_seq):
    pieces_list = piecewise_by_extreme_value(values_seq)
    maximums_and_indices = [(np.max(values_seq[piece_left:piece_right + 1]), (piece_left + piece_right) // 2)
                            for piece_left, piece_right in pieces_list]

    length = len(values_seq)
    safety_zone_start, safety_zone_end = round(0.25 * length), round(0.75 * length)
    avg_maximum = np.mean([maximum for maximum, index in maximums_and_indices
                           if safety_zone_start < index < safety_zone_end])

    valid_indices = np.where(values_seq > 0)[0]
    first_index = valid_indices[0]
    last_index = valid_indices[-1]

    # 小端裁剪
    crop_small = first_index
    for i in range(len(pieces_list)):
        piece_left, piece_right = pieces_list[i]
        maximum, index = maximums_and_indices[i]
        piece_width = piece_right - piece_left + 1
        if piece_width > 20 and maximum >= 0.45 * avg_maximum:
            crop_small = piece_left
            break

    # 大端裁剪
    crop_large = last_index
    for i in range(len(pieces_list) - 1, -1, -1):
        piece_left, piece_right = pieces_list[i]
        maximum, index = maximums_and_indices[i]
        piece_width = piece_right - piece_left + 1
        if piece_width > 20 and maximum >= 0.45 * avg_maximum:
            crop_large = piece_right
            break

    return crop_small, crop_large


def piecewise_by_extreme_value(values_seq):
    maximums_and_indices = find_extreme_value_by_window(values_seq, window_size=10)

    valid_indices = np.where(values_seq > 0)[0]
    first_index = valid_indices[0]
    last_index = valid_indices[-1]

    minimum_indices_in_gaps = []
    prev_index = first_index
    extend_indices = [index for _, index in maximums_and_indices] + [last_index]
    for i, curr_index in enumerate(extend_indices):
        minimum, head_index = find_minimum_value_in_range(values_seq, start=prev_index, end=curr_index, reverse=False)
        minimum, tail_index = find_minimum_value_in_range(values_seq, start=prev_index, end=curr_index, reverse=True)
        if i == 0 or i == len(extend_indices) - 1:
            if minimum <= 1:
                minimum_indices_in_gaps.append(head_index)
                minimum_indices_in_gaps.append(tail_index)
        else:
            if minimum < 8:
                minimum_indices_in_gaps.append(head_index)
                minimum_indices_in_gaps.append(tail_index)
        prev_index = curr_index

    minimum_indices_in_gaps.insert(0, first_index)
    minimum_indices_in_gaps.append(last_index)

    pieces_list = []
    for i in range(0, len(minimum_indices_in_gaps), 2):
        piece_left, piece_right = minimum_indices_in_gaps[i:i + 2]
        pieces_list.append((piece_left, piece_right))

    return pieces_list


def find_extreme_value_by_window(values_seq, window_size=10):
    # Find several maximums of values sequence
    # The type of values is list or 1d-numpy-array

    maximums_in_window = [np.max(values_seq[max(0, i - window_size): min(i + window_size + 1, len(values_seq))])
                          for i in range(len(values_seq))]

    maximums_counter = []
    prev_value = maximums_in_window[0]
    start_index = end_index = 0
    for curr_value in maximums_in_window[1:]:
        if curr_value == prev_value:
            end_index += 1
        else:
            count = end_index - start_index + 1
            maximum_index = (start_index + end_index) // 2
            maximums_counter.append((prev_value, count, maximum_index))
            prev_value = curr_value
            start_index = end_index + 1
            end_index = start_index

    count = end_index - start_index + 1
    maximum_index = (start_index + end_index) // 2
    maximums_counter.append((prev_value, count, maximum_index))

    maximums_list = [(maximum, index) for maximum, count, index in maximums_counter
                     if maximum > 0 and count >= 2 * window_size + 1]

    return maximums_list


def find_minimum_value_in_range(values_seq, start=None, end=None, reverse=False):
    if start is None:
        start = 0
    if end is None:
        end = len(values_seq) - 1

    obj_seq = values_seq[start: end + 1]
    minimum = np.min(obj_seq)
    if reverse is False:
        minimum_index = np.argmin(obj_seq) + start
    else:
        reverse_seq = obj_seq[::-1]
        minimum_index = len(obj_seq) - 1 - np.argmin(reverse_seq) + start

    return minimum, minimum_index
    

def book_page_pre_processing(np_page_list):
    np_pages = []
    for np_page in np_page_list:
        np_page = color.rgb2gray(np_page)   # float32 [0, 1.]
        np_page = util.img_as_ubyte(np_page)
        np_page = adjust_image_contrast(np_page=np_page)
        # np_page = crop_page_margin(np_page=np_page)
        np_page = color.gray2rgb(np_page)   # uint8
        np_pages.append(np_page)
    return np_pages


if __name__ == '__main__':
    PIL_img = Image.open("D:/feidegenggao/Desktop/test/版刻图像_史记1.jpg")
    np_img = np.array(PIL_img, dtype=np.uint8)
    np_img = book_page_pre_processing([np_img,])[0]
    PIL_img = Image.fromarray(np_img)
    print(PIL_img.mode)
    PIL_img.show()
    print("Done !")
