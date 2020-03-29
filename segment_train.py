# -*- encoding: utf-8 -*-
# Author: hushukai

from segment_book_page.train import main as book_page_train
from segment_mix_line.train import main as mix_line_train
from segment_text_line.train import main as text_line_train


if __name__ == '__main__':
    # book_page_train()
    # mix_line_train()
    text_line_train()
    
    print("Done !")
