# -*- encoding: utf-8 -*-
# Author: hushukai

from data_generator.generate_text_lines import generate_text_line_imgs
from data_generator.generate_book_pages import generate_book_page_tfrecords


if __name__ == '__main__':
    # generate_text_line_imgs(obj_num=10000, type="horizontal")
    # generate_text_line_imgs(obj_num=10000, type="vertical")
    # generate_text_line_tfrecords(obj_num=100, type="horizontal")
    # generate_text_line_tfrecords(obj_num=100, type="vertical")

    # generate_book_pages(obj_num=100, type="horizontal", page_shape=(416, 416))
    # generate_book_pages(obj_num=100, type="vertical", page_shape=(416, 416))
    # generate_book_page_tfrecords(obj_num=100, type="horizontal")
    generate_book_page_tfrecords(obj_num=8000, type="vertical")
    
    # display_tfrecords(os.path.join(TEXT_LINE_TFRECORDS_H, "text_lines_0.tfrecords"))
    
    print("Done !")
