# -*- encoding: utf-8 -*-
# Author: hushukai

from PIL import Image
from yolo import YOLO


if __name__ == '__main__':
    yolo=YOLO()
    path = '/data/work/tensorflow/data/panda_test/1.jpg'
    try:
        image = Image.open(path)
    except:
        print('Open Error! Try again!')
    else:
        r_image, _ = yolo.detect_image(image)
        r_image.show()
    yolo.close_session()

    print("Done !")
