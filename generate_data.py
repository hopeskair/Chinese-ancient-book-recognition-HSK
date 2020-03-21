# -*- encoding: utf-8 -*-
# Author: hushukai

from data_generator.generate_text_lines import generate_one_text_line_imgs, generate_one_text_line_tfrecords
from data_generator.generate_text_lines import generate_two_text_line_imgs, generate_two_text_line_tfrecords
from data_generator.generate_text_lines import generate_mix_text_line_imgs, generate_mix_text_line_tfrecords
from data_generator.generate_book_pages import generate_book_page_imgs, generate_book_page_tfrecords


if __name__ == '__main__':
    # generate_one_text_line_imgs(obj_num=100, text_type="horizontal")
    # generate_one_text_line_imgs(obj_num=100, text_type="vertical")
    generate_one_text_line_tfrecords(obj_num=20000, text_type="horizontal")
    generate_one_text_line_tfrecords(obj_num=25000, text_type="vertical")

    # generate_two_text_line_imgs(obj_num=100, text_type="horizontal")
    # generate_two_text_line_imgs(obj_num=100, text_type="vertical")
    generate_two_text_line_tfrecords(obj_num=20000, text_type="horizontal")
    generate_two_text_line_tfrecords(obj_num=25000, text_type="vertical")

    # generate_mix_text_line_imgs(obj_num=100, text_type="horizontal")
    # generate_mix_text_line_imgs(obj_num=100, text_type="vertical")
    generate_mix_text_line_tfrecords(obj_num=20000, text_type="horizontal")
    generate_mix_text_line_tfrecords(obj_num=25000, text_type="vertical")
    #
    # generate_book_page_imgs(obj_num=10, text_type="horizontal", page_shape=(416, 416))
    # generate_book_page_imgs(obj_num=10, text_type="vertical", page_shape=(416, 416))
    generate_book_page_tfrecords(obj_num=6500, text_type="horizontal")
    generate_book_page_tfrecords(obj_num=8000, text_type="vertical")
    
    print("Done !")
