# coding:utf-8

import numpy as np

from detection_ctpn.utils.np_utils import quadrangle_nms
from .text_proposal_connector import TextProposalConnector
from .text_proposal_connector_oriented import TextProposalConnector as TextProposalConnectorOriented

from config import CTPN_TEXT_LINE_MIN_SCORE, CTPN_PROPOSALS_WIDTH, CTPN_MIN_NUM_PROPOSALS
from config import CTPN_TEXT_LINE_NMS_THRESH


class TextDetector:
    def __init__(self, DETECT_MODE="H"):
        self.mode = DETECT_MODE
        if self.mode == "H":
            self.text_proposal_connector = TextProposalConnector()
        elif self.mode == "O":
            self.text_proposal_connector = TextProposalConnectorOriented()

    def detect(self, text_proposals, scores, im_size):
        # 获取检测结果
        text_lines = self.text_proposal_connector.get_text_lines(text_proposals, scores, im_size)
        keep_indices = self.filter_boxes(text_lines)
        text_lines = text_lines[keep_indices]
        
        # 文本行nms
        if text_lines.shape[0] != 0:
            keep_indices = quadrangle_nms(text_lines[:, :8], text_lines[:, 8], CTPN_TEXT_LINE_NMS_THRESH)
            text_lines = text_lines[keep_indices]

        text_lines = self.clip_text_lines(text_lines, im_size)
        
        return text_lines

    def filter_boxes(self, text_lines):
        widths = (text_lines[:, 2] - text_lines[:, 0] + text_lines[:, 4] - text_lines[:, 6])/2
        scores = text_lines[:, -1]
        return np.where((scores > CTPN_TEXT_LINE_MIN_SCORE) &
                        (widths > CTPN_PROPOSALS_WIDTH * CTPN_MIN_NUM_PROPOSALS))[0]

    def clip_text_lines(self, text_lines, im_size):
        """裁剪边框到图像内
        Parameter:
            text_lines: [N, (x1,y1,x2,y2,x3,y3,x4,y4)]
        """
        h, w = im_size[:2]
        
        text_lines[text_lines[:, 0] < 0, 0] = 0
        text_lines[text_lines[:, 1] < 0, 1] = 0
        text_lines[text_lines[:, 2] > w-1, 2] = w-1
        text_lines[text_lines[:, 3] < 0, 3] = 0
        text_lines[text_lines[:, 4] > w-1, 4] = w-1
        text_lines[text_lines[:, 5] > h-1, 5] = h-1
        text_lines[text_lines[:, 6] < 0, 6] = 0
        text_lines[text_lines[:, 7] > h-1, 7] = h-1
        
        return text_lines
