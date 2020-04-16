# -*- encoding: utf-8 -*-
# Author: hushukai

import os

from util import check_or_makedirs
from config import CHAR_IMGS_DIR, CHAR_TFRECORDS_DIR
from config import CHAR_IMAGE_PATHS_FILE, CHAR_TFRECORDS_PATHS_FILE


def extract_annotation(imgs_dir=None, tfrecords_dir=None, dest_file=None):
    assert [imgs_dir, tfrecords_dir].count(None) == 1
    
    check_or_makedirs(os.path.dirname(dest_file))
    with open(dest_file, "w", encoding="utf-8") as fw:
        if imgs_dir is not None:
            for root, dirs, files_list in os.walk(imgs_dir):
                if len(files_list) > 0:
                    for file_name in files_list:
                        if file_name.lower()[-4:] in (".gif", ".jpg", ".png"):
                            image_path = os.path.join(root, file_name)
                            fw.write(image_path + "\n")
        
        elif tfrecords_dir is not None:
            assert os.path.exists(tfrecords_dir)
            for file in os.listdir(tfrecords_dir):
                if file.endswith(".tfrecords"):
                    file_path = os.path.join(tfrecords_dir, file)
                    fw.write(file_path + "\n")


def main():
    # extract_annotation(imgs_dir=CHAR_IMGS_DIR, dest_file=CHAR_IMAGE_PATHS_FILE)
    extract_annotation(tfrecords_dir=CHAR_TFRECORDS_DIR, dest_file=CHAR_TFRECORDS_PATHS_FILE)


if __name__ == '__main__':
    print("Done !")
