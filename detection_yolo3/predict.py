# -- coding: utf-8 --

import os
from timeit import default_timer as timer

import numpy as np
from PIL import Image
from tensorflow.keras import backend, layers, models

from detection_yolo3.model import yolo_eval, yolo_body
from detection_yolo3.utils import get_anchors, draw_boxes

from util import check_or_makedirs
from config import BOX_CLASSES_ON_BOOK
from config import YOLO3_ANCHORS_FILE
from config import YOLO3_CLASS_SCORE_THRESH
from config import YOLO3_NMS_IOU_THRESH, YOLO3_NMS_MAX_BOXES_NUM


class YOLO(object):
    
    def __init__(self, model_struc="densenet", model_path=""):
        self.model_struc = model_struc
        self.model_path = model_path
        self.classes = BOX_CLASSES_ON_BOOK
        self.anchors = get_anchors(YOLO3_ANCHORS_FILE)
        
        self.score_thresh = YOLO3_CLASS_SCORE_THRESH
        self.nms_iou_thresh = YOLO3_NMS_IOU_THRESH
        self.max_boxes_num = YOLO3_NMS_MAX_BOXES_NUM
        
        self.predict_model = self.build_and_load_model()
        
    def build_and_load_model(self):
        assert self.model_path.endswith('.h5'), "Keras model or weights must be a .h5 file."

        image_input = layers.Input(shape=(None, None, 1), dtype="float32")  # 图片输入格式
        raw_img_shape = layers.Input(shape=(2,), dtype="int32")
        
        num_anchors = len(self.anchors)  # anchor的数量
        num_classes = len(self.classes)  # 类别数
        
        self.yolo_model = yolo_body(image_input, num_anchors//3, num_classes, self.model_struc)
        self.yolo_model.load_weights(self.model_path)  # 加载模型参数
        print('{} model, {} anchors, and {} classes loaded.'.format(self.model_path, num_anchors, num_classes))
        
        # 处理模型的输出，提取模型的预测结果。注意这里的设定：一个batch只包含一张图片
        boxes, scores, classes = layers.Lambada(yolo_eval,
                                                name='yolo_eval',
                                                arguments={'anchors':self.anchors,
                                                           'num_classes':num_classes,
                                                           'score_thresh':self.score_thresh,
                                                           'iou_thresh':self.nms_iou_thresh,
                                                           'max_boxes':self.max_boxes_num}
                                                )(self.yolo_model.outputs)
                                                
        return models.Model(self.yolo_model.inputs, outputs=[boxes, scores, classes])
    
    def detect_image(self, img_path, dest_dir, background="white"):
        if not os.path.exists(img_path): return
        img_name = os.path.basename(img_path)
        check_or_makedirs(dest_dir)
        
        PIL_img = Image.open(img_path)
        if PIL_img.mode != "L":
            PIL_img = PIL_img.convert("L")
        np_img = np.array(PIL_img, dtype=np.uint8)
        
        h, w = np_img.shape[:2]
        new_h = -h % 32 + h
        new_w = -w % 32 + w
        batch_imgs = np.empty(shape=(1, new_h, new_w), dtype=np.float32)
        if background == "white":
            batch_imgs.fill(255)
        elif background == "black":
            batch_imgs.fill(0)
        else:
            ValueError("Optional image background: 'white', 'black'.")
        batch_imgs[0, :h, :w] = np_img
        batch_imgs = np.expand_dims(batch_imgs, axis=-1)

        start = timer()  # 起始时间
        out_boxes, out_scores, out_classes = self.predict_model.predict(x=batch_imgs)
        print('Time {:.2f}s, found {} boxes in {}'.format(timer()-start, len(out_boxes), img_name))

        np_img_rgb = draw_boxes(np_img, out_boxes, out_scores, out_classes)
        PIL_img = Image.fromarray(np_img_rgb)
        PIL_img.save(os.path.join(dest_dir, img_name), format="jpeg")
    
    def detect_images(self, src_dir, dest_dir, background="white"):
        assert os.path.exists(src_dir)
        img_paths = [os.path.join(src_dir, file) for file in os.listdir(src_dir)
                     if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".gif")]
        for img_path in img_paths:
            self.detect_image(img_path, dest_dir, background)


if __name__ == '__main__':
    print("Done !")
