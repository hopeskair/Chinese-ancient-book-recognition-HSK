# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.keras import models
from tensorflow.keras import backend as K

from segment_base.model import work_net as segment_work_net
from segment_base.predict import model_weights_path as segment_model_weights_path
from segment_base.predict import segment_predict
from segment_base.gen_prediction import ExtractSplitPosition
from recog_with_components.model import work_net as recog_work_net
from recog_with_components.predict import model_weights_path as recog_model_weights_path
from recog_with_components.predict import char_predict

from util import check_or_makedirs
from config import CHAR_IMG_SIZE


def load_segment_model(segment_task="book_page", text_type="vertical", model_struc="densenet_gru", weights=""):
    segment_model = segment_work_net(stage="predict", segment_task=segment_task, text_type=text_type, model_struc=model_struc)
    
    weights_path = segment_model_weights_path(weights=weights, segment_task=segment_task, model_struc=model_struc)
    segment_model.load_weights(weights_path, by_name=True)
    print("\nLoad %s model weights from %s\n" % (segment_task, weights_path))
    
    return segment_model, weights_path


def load_recog_model(model_struc="densenet_gru", weights=""):
    recog_model = recog_work_net(stage="predict", img_size=CHAR_IMG_SIZE, model_struc=model_struc)

    weights_path = recog_model_weights_path(weights, model_struc)
    recog_model.load_weights(weights_path, by_name=True)
    print("\nLoad model weights from %s\n" % weights_path)
    
    return recog_model, weights_path


class SegmentModel:

    def __init__(self, segment_task, text_type="vertical", model_struc="densenet_gru", weights=""):
        if segment_task not in ("book_page", "mix_line", "double_line", "text_line"):
            ValueError("The task type is wrong !")
        self.segment_task = segment_task
        self.text_type = text_type
        self.model_struc = model_struc
        self.weights = weights
        _model, self.weights_path = load_segment_model(self.segment_task, self.text_type, self.model_struc, self.weights)
        # self.yaml_str = _model.to_json()
    
    def segment_predict(self, images=None, img_paths=None, dest_dir=None, reuse=False):
        if reuse:
            # _model = models.model_from_json(self.yaml_str, custom_objects={'ExtractSplitPosition': ExtractSplitPosition})
            # _model.load_weights(self.weights_path, by_name=True)
            # return segment_predict(images, img_paths, dest_dir, segment_model=_model)
            ValueError("There are some unresolved errors in this branch !")
        else:
            K.clear_session()
            return segment_predict(images, img_paths, dest_dir, segment_model=None, segment_task=self.segment_task,
                                   text_type=self.text_type, model_struc=self.model_struc, weights=self.weights)


class CharRecogModel:
    
    def __init__(self, model_struc="densenet_gru", weights=""):
        self.model_struc = model_struc
        self.weights = weights
        self.model, self.weights_path = load_recog_model(self.model_struc, self.weights)
    
    def char_predict(self, images=None, img_paths=None, reuse=False, to_print=False):
        if reuse:
            ValueError("There are some unresolved errors in this branch !")
        else:
            K.clear_session()
            return char_predict(images, img_paths, recog_model=None,
                                model_struc=self.model_struc, weights=self.weights, to_print=to_print)


def extract_slices(np_text, split_positions, relative_coord, segment_task="book_page", text_type="vertical"):
    PIL_text = Image.fromarray(np_text)
    split_positions = split_positions.astype(np.int32)
    split_num = len(split_positions)
    
    np_sub_text_list = []
    split_lines_list = []
    
    def _extract_quad_block(x0, y0, x1, y1, x2, y2, x3, y3, sub_text_width, sub_text_height):
        # quad_coordinates: 8-tuple (x0, y0, x1, y1, x2, y2, x3, y3), they are
        # upper left, lower left, lower right, and upper right corner of the source quadrilateral.
        quad_coordinates = (x0, y0, x1, y1, x2, y2, x3, y3)
        size = (sub_text_width, sub_text_height)
        print("***************", size, quad_coordinates)
        PIL_sub_text = PIL_text.transform(size=size, method=Image.QUAD, data=quad_coordinates)
        np_sub_text_list.append(np.array(PIL_sub_text, dtype=np.uint8))
    
    text_type = text_type[0].lower()
    task_type = (segment_task, text_type)
    x0, y0, x1, y1, x2, y2, x3, y3 = [None] * 8
    print(split_positions)
    if task_type in (("book_page", "v"), ("double_line", "v")):
        for i in range(split_num-1, 0, -1):
            x0, x1 = split_positions[i-1]
            x3, x2 = split_positions[i]
            sub_text_width = (x3 - x0 + x2 - x1) // 2
            sub_text_height = np_text.shape[0]
            y0, y1, y2, y3 = 0, sub_text_height - 1, sub_text_height - 1, 0
            _extract_quad_block(x0, y0, x1, y1, x2, y2, x3, y3, sub_text_width, sub_text_height)
            split_lines_list.append((x3, y3, x2, y2))
        split_lines_list.append((x0, y0, x1, y1))
    
    elif task_type in (("text_line", "v"), ("mix_line", "v"), ("book_page", "h")):
        for i in range(split_num-1):
            y0, y3 = split_positions[i]
            y1, y2 = split_positions[i+1]
            sub_text_height = (y1 - y0 + y2 - y3) // 2
            sub_text_width = np_text.shape[1]
            x0, x1, x2, x3 = 0, 0, sub_text_width - 1, sub_text_width - 1
            _extract_quad_block(x0, y0, x1, y1, x2, y2, x3, y3, sub_text_width, sub_text_height)
            split_lines_list.append((x0, y0, x3, y3))
        split_lines_list.append((x1, y1, x2, y2))
    
    elif task_type in (("text_line", "h"),):
        for i in range(split_num-1):
            x0, x1 = split_positions[i]
            x3, x2 = split_positions[i+1]
            sub_text_width = (x3 - x0 + x2 - x1) // 2
            sub_text_height = np_text.shape[0]
            y0, y1, y2, y3 = 0, sub_text_height - 1, sub_text_height - 1, 0
            _extract_quad_block(x0, y0, x1, y1, x2, y2, x3, y3, sub_text_width, sub_text_height)
            split_lines_list.append((x0, y0, x1, y1))
        split_lines_list.append((x3, y3, x2, y2))
            
    else:
        ValueError("Impossible task_type: (%s, %s)."%(segment_task, text_type))
    
    if split_num == 0 or split_num == 1:
        split_lines = np.empty(shape=[0, 4], dtype=np.int32)
    else:
        relative_coord = np.hstack([relative_coord, relative_coord])[np.newaxis, :]
        split_lines = np.array(split_lines_list, dtype=np.int32)
        split_lines += relative_coord
    
    return np_sub_text_list, split_lines


def check_and_correct_double_split(double_split_pos_list, double_scores_list, img_w):
    total_num = len(double_split_pos_list)
    # assert len(double_scores_list) == total_num
    # assert all([len(double_split_pos_list[i]) == len(double_scores_list[i]) for i in range(total_num)])
    
    split_num_list = [len(double_split_pos_list[i]) for i in range(total_num)]
    if all([s in (2, 3) for s in split_num_list]):
        if total_num == 1:
            return double_split_pos_list, double_scores_list
        elif all([split_num_list[i] + split_num_list[i + 1] == 5 for i in range(total_num - 1)]):  # total_num > 1
            return double_split_pos_list, double_scores_list
        else:
            if total_num >= 3:
                for i in range(1, total_num-1):
                    if split_num_list[i] == split_num_list[i-1] and split_num_list[i] == split_num_list[i+1]:
                        _split_pos, _split_score = double_split_pos_list[i], double_scores_list[i]
                        if split_num_list[i] == 2:
                            # mid_line = np.mean(_split_pos, axis=0).astype(np.int32)
                            mid_line = np.array([img_w/2, img_w/2], dtype=np.int32)
                            double_split_pos_list[i] = np.stack([_split_pos[0], mid_line, _split_pos[1]])
                            double_scores_list[i] = np.array([_split_score[0], -1, _split_score[1]], dtype=np.float32)
                            split_num_list[i] = 3
                        else:
                            # split_num_list[i] == 3
                            double_split_pos_list[i] = np.stack([_split_pos[0], _split_pos[2]])
                            double_scores_list[i] = np.array([_split_score[0], _split_score[2]], dtype=np.float32)
                            split_num_list[i] = 2
    else:
        for i, s in enumerate(split_num_list):
            if s > 3:
                _split_pos, _split_score = double_split_pos_list[i], double_scores_list[i]
                # mid_line = (_split_pos[0] + _split_pos[-1]) // 2
                mid_line = np.array([img_w / 2, img_w / 2], dtype=np.int32)
                double_split_pos_list[i] = np.stack([_split_pos[0], mid_line, _split_pos[-1]])
                double_scores_list[i] = np.array([_split_score[0], -1, _split_score[-1]], dtype=np.float32)
                split_num_list[i] = 3
            else:
                # s < 2
                double_split_pos_list[i] = np.array([[0, 0], [img_w-1, img_w-1]], dtype=np.int32)
                double_scores_list[i] = np.array([-1, -1], dtype=np.float32)
                split_num_list[i] = 2
    
    return double_split_pos_list, double_scores_list


def main(book_page_dir, dest_dir=None, is_mix_line=False, text_type="vertical", model_struc="densenet_gru"):
    if dest_dir is not None: check_or_makedirs(dest_dir)
    K.set_learning_phase(False)
    
    # 加载模型
    segment_book_page_model = SegmentModel("book_page", text_type, model_struc, weights=99)
    segment_mix_line_model = SegmentModel("mix_line", text_type, model_struc, weights=65)
    segment_double_line_model = SegmentModel("double_line", text_type, model_struc, weights=72)
    segment_text_line_model = SegmentModel("text_line", text_type, model_struc, weights=42)
    recog_model = CharRecogModel(model_struc, weights=121)
    
    # 切分书页
    np_page_list, page_name_list, page_split_pos_list, page_scores_list = segment_book_page_model.segment_predict(img_paths=book_page_dir)
    for i in range(len(np_page_list)):
        try:
            split_line_dict = {"page":[], "mix":[], "double":[], "single":[]}
            start_coord = np.array([0, 0], dtype=np.int32)
            
            np_line_list, page_split_lines = \
                extract_slices(np_page_list[i], page_split_pos_list[i], start_coord, segment_task="book_page", text_type=text_type)
            split_line_dict["page"].append(page_split_lines)
            
            text_list = []
            if text_type in ("v", "vertical") and is_mix_line:
                np_mix_line_list = np_line_list
                
                # 切分单双行
                _, _, mix_split_pos_list, mix_scores_list = segment_mix_line_model.segment_predict(images=np_mix_line_list)
    
                for j in range(len(np_mix_line_list)):
                    np_double_line_list, mix_split_lines = \
                        extract_slices(np_mix_line_list[j], mix_split_pos_list[j], page_split_lines[j+1, :2], segment_task="mix_line", text_type=text_type)
                    split_line_dict["mix"].append(mix_split_lines)
                    
                    # 切分双行
                    _, _, double_split_pos_list, double_scores_list = segment_double_line_model.segment_predict(images=np_double_line_list)
                    
                    img_w = np_mix_line_list[j].shape[1]
                    double_split_pos_list, double_scores_list = check_and_correct_double_split(double_split_pos_list, double_scores_list, img_w)
                    text1, text2 = "", ""
                    for k in range(len(np_double_line_list)):
                        np_text_line_list, double_split_lines = \
                            extract_slices(np_double_line_list[k], double_split_pos_list[k], mix_split_lines[k, :2], segment_task="double_line", text_type=text_type)
                        split_line_dict["double"].append(double_split_lines)
                        
                        # 切分单行（文本行）
                        _, _, char_split_pos_list, char_scores_list = segment_text_line_model.segment_predict(images=np_text_line_list)
                        
                        assert len(np_text_line_list) in (1, 2)
                        sub_text1, sub_text2 = "", ""
                        for t in range(len(np_text_line_list)):
                            np_char_list, single_split_lines = \
                                extract_slices(np_text_line_list[t], char_split_pos_list[t], double_split_lines[t+1, :2], segment_task="text_line", text_type=text_type)
                            split_line_dict["single"].append(single_split_lines)
                            
                    #         # 单字识别
                    #         _, _, pred_topk_chars_list = recog_model.char_predict(images=np_char_list)
                    #
                    #         # 识别结果
                    #         text_str = "".join([chars[0] if len(chars) > 0 else "？" for chars in pred_topk_chars_list])
                    #         if t == 0:
                    #             sub_text1 = text_str
                    #         else:
                    #             sub_text2 = text_str
                    #
                    #     # 等长调整
                    #     len_1, len_2 = len(sub_text1), len(sub_text2)
                    #     max_len = max(len_1, len_2)
                    #     text1 += sub_text1 + "　" * (max_len - len_1)
                    #     text2 += sub_text2 + "　" * (max_len - len_2)
                    #
                    # # 保存当前单双行文本
                    # text_list.extend([text1, text2, "\n"])
            
    
            elif text_type in ("v", "vertical", "h", "horizontal") and not is_mix_line:
                np_text_line_list = np_line_list
    
                # 切分单行（文本行）
                _, _, char_split_pos_list, char_scores_list = segment_text_line_model.segment_predict(images=np_text_line_list)
                
                for t in range(len(np_text_line_list)):
                    _t = t+1 if text_type in ("v", "vertical") else t
                    np_char_list, single_split_lines = \
                        extract_slices(np_text_line_list[t], char_split_pos_list[t], page_split_lines[_t, :2], segment_task="text_line", text_type=text_type)
                    split_line_dict["single"].append(single_split_lines)
                    
                    # 单字识别
                    _, _, pred_topk_chars_list = recog_model.char_predict(images=np_char_list)
                    
                    # 识别结果
                    text_str = "".join([chars[0] if len(chars) > 0 else "？" for chars in pred_topk_chars_list])
                    text_list.extend([text_str, "\n"])  # 保存
            
            else:
                ValueError("Horizontal book page should not exist single-double text line.")
            
            # save
            if dest_dir is not None:
                PIL_page_drawn = draw_split_lines(np_page=np_page_list[i], split_line_dict=split_line_dict)  # draw
                page_name = os.path.splitext(page_name_list[i])[0]
                PIL_page_drawn.save(os.path.join(dest_dir, page_name + ".jpg"), format="jpeg")
                with open(os.path.join(dest_dir, page_name + ".txt"), "w", encoding="utf8") as fw:
                    fw.write("\n".join(text_list))
            
            # print
            print("\n*******************", page_name_list[i], "*******************\n")
            print("\n".join(text_list))
        
        except:
            continue
        

def draw_split_lines(np_page, split_line_dict):
    colors = {"page":"red", "mix":"orange", "double":"green", "single":"blue"}
    
    # np_page = np_page.astype(np.uint8)
    PIL_page = Image.fromarray(np_page)
    if PIL_page.mode != "RGB":
        PIL_page = PIL_page.convert(mode="RGB")
    
    draw = ImageDraw.Draw(PIL_page)
    for split_type, split_line_list in split_line_dict.items():
        for np_split_lines in split_line_list:
            if split_type == "double":
                if len(np_split_lines) < 2: continue
                np_split_lines = np_split_lines[1:-1]
            for x1, y1, x2, y2 in np_split_lines:
                draw.line([x1, y1, x2, y2], fill=colors[split_type], width=2)  # 画线
    
    return PIL_page


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--img_dir", type=str, default="", help="root path of images.")
    parse.add_argument("--dest_dir", type=str, default="_recog_result", help="root path of recognition result.")
    parse.add_argument("--text_type", type=str, default="vertical", help="horizontal or vertical text")
    parse.add_argument("--mix_line", type=bool, default=True, help="Whether or not the text is single-double line type.")
    args = parse.parse_args(sys.argv[1:])
    
    book_pages_dir = args.img_dir if args.img_dir != "" else "_samples"
    
    main(book_pages_dir, dest_dir=args.dest_dir , is_mix_line=args.mix_line, text_type=args.text_type)
    
    print("Done !")
