# -*- encoding: utf-8 -*-
# Author: hushukai

import os
import sys
import argparse

from detection_ctpn.train import main as train
from detection_ctpn.predict import main as predict, TRAIN_FINISHED_WEIGHTS

from config import CTPN_BOOK_PAGE_TAGS_FILE, CTPN_ROOT_DIR


if __name__ == '__main__':
    # train
    # train(data_file=CTPN_BOOK_PAGE_TAGS_FILE, src_type="tfrecords", text_type="vertical", epochs=500, init_epochs=9)
    
    # predict
    parse = argparse.ArgumentParser()
    parse.add_argument("--image_path", type=str, default="", help="image path")
    parse.add_argument("--dest_path", type=str, default="", help="detected result path")
    parse.add_argument("--text_type", type=str, default="vertical", help="horizontal or vertical text")
    parse.add_argument("--weight_path", type=str, default="", help="model weight path")
    parse.add_argument("--use_side_refine", type=int, default=1, help="1: use side refine; 0 not use")
    args = parse.parse_args(sys.argv[1:])

    dest_dir = os.path.join(CTPN_ROOT_DIR, "samples")
    predict(img_path=args.image_path, dest_dir=dest_dir, text_type="vertical", weights_path=TRAIN_FINISHED_WEIGHTS)
    
    print("Done !")
