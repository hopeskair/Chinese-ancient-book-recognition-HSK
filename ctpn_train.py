# -*- encoding: utf-8 -*-
# Author: hushukai
import os
from detection_ctpn.train import main

from config import CTPN_BOOK_PAGE_TAGS_FILE
from config import CTPN_CKPT_DIR


if __name__ == '__main__':
    weights_path = os.path.join(CTPN_CKPT_DIR, "densenet_gru_ctpn_003.h5")
    main(data_file=CTPN_BOOK_PAGE_TAGS_FILE, src_type="tfrecords", model_type="vertical", epochs=100, weights_path=weights_path)
    print("Done !")
