# -*- encoding: utf-8 -*-
# Author: hushukai

from segment_base.predict import segment_predict


def segment_mix_line_predict():
    segment_predict(images=None,
                    img_paths="_mix",
                    dest_dir="_mix_",
                    segment_model=None,
                    segment_task="mix_line",
                    text_type="vertical",
                    model_struc="densenet_gru",
                    weights=65)


if __name__ == '__main__':
    print("Done !")
