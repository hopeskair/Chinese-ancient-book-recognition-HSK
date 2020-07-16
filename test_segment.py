# -*- encoding: utf-8 -*-
# Author: hushukai

from segment_book_page.predict import segment_book_page_predict
from segment_mix_line.predict import segment_mix_line_predict
from segment_text_line.predict import segment_text_line_predict
from segment_double_line.predict import segment_double_line_predict
from recog_with_components.predict import main as char_predict

if __name__ == '__main__':
    # segment_book_page_predict()
    # segment_mix_line_predict()
    # segment_double_line_predict()
    # segment_text_line_predict()
    char_predict()
    
    print("Done !")
