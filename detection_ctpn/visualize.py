# -- coding: utf-8 --

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os

from config import CTPN_ROOT_DIR

def draw_text_lines(np_img, text_lines):
    np_img = np_img.astype(np.uint8)
    h, w = np_img.shape[:2]
    
    PIL_img = Image.fromarray(np_img)
    if PIL_img.mode != "RGB":
        PIL_img = PIL_img.convert(mode="RGB")
    
    draw = ImageDraw.Draw(PIL_img)
    font_path = os.path.join(CTPN_ROOT_DIR, "font", "FiraMono-Medium.otf")
    font_object = ImageFont.truetype(font=font_path, size=int(3e-2 * h))  # 字体
    
    for i, text_line in enumerate(text_lines):
        score = text_line[8]  # 得分
        
        label = '{:.2f}'.format(score)  # 标签
        # label_size = draw.textsize(label, font_object)  # 标签文字
        
        # top, left, bottom, right = text_line
        x1, y1, x2, y2, x3, y3, x4, y4 = np.round(text_line[:8]).astype(np.int32)
        
        text_origin = np.array([x1 + 1, y1 + 1])
        
        # draw.rectangle([left, top, right, bottom], outline=colors[c], width=2)  # 画框
        draw.polygon(xy=[x1, y1, x2, y2, x3, y3, x4, y4], fill="blue")
        # draw.rectangle([tuple(text_origin), tuple(text_origin+label_size)], fill=self.colors[c])    # 文字背景
        draw.text(text_origin, label, fill="blue", font=font_object)  # 文案
    
    np_img = np.array(PIL_img, dtype=np.uint8)
    
    return np_img