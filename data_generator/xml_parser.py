# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import json
from lxml import etree


extracted_tags_file = "extracted_tags.txt"


def parse_PASCAL_VOC_file(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    # root.find() or root.findall()
    annotations = root.xpath("//object")
    
    bounding_boxes = []
    for annotation in annotations:
        bbox = annotation.xpath("./bndbox")[0]
        x1, y1 = int(bbox.find("xmin").text), int(bbox.find("ymin").text)
        x2, y2 = int(bbox.find("xmax").text), int(bbox.find("ymax").text)
        bounding_boxes.append((x1, y1, x2, y2))
    
    return bounding_boxes


def extract_split_lines(bounding_boxes):
    bounding_boxes.sort(key=lambda tup: (tup[0] + tup[2]) / 2)
    
    split_pos = []
    
    x1, _, _, _ = bounding_boxes[0]
    split_pos.append(x1)
    
    for i in range(len(bounding_boxes)-1):
        _, _, prev_x2, _ = bounding_boxes[i]
        next_x1, _, _, _ = bounding_boxes[i+1]
        x_cent = round((prev_x2+next_x1)/2)
        split_pos.append(x_cent)
    
    _, _, x2, _ = bounding_boxes[0]
    split_pos.append(x2)
    
    return split_pos


def convert_anntation(root_path):
    assert os.path.exists(root_path)

    tags_path = os.path.join(root_path, extracted_tags_file)
    with open(tags_path, "w", encoding="utf8") as fw:
        for root, dirs, files in os.walk(root_path):
            for img_name in files:
                _img_name, extension = os.path.splitext(img_name)
                xml_path = os.path.join(root, _img_name + ".xml")
                
                if extension not in (".jpg", ".png", "gif") or not os.path.exists(xml_path):
                    continue
                bounding_boxes = parse_PASCAL_VOC_file(xml_path)
                split_pos_list = extract_split_lines(bounding_boxes)
    
                image_tags = {"text_bbox_list": bounding_boxes, "split_pos_list": split_pos_list}
                fw.write(img_name + "\t" + json.dumps(image_tags) + "\n")
            

if __name__ == '__main__':
    # print(parse_PASCAL_VOC_file("D:/feidegenggao/Desktop/test/版刻图像_史记1.xml"))
    convert_anntation("D:/feidegenggao/Desktop/test")
    print("Done !")
