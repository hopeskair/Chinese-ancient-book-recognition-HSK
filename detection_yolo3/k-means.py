# -- coding: utf-8 --
# Author: hushukai

from sklearn.cluster import KMeans
import numpy as np

from config import YOLO3_BOOK_PAGE_TAGS_FILE
from config import YOLO3_ANCHORS_FILE


def get_anchors_with_kmeans(src_file, dest_file):
    boxes_wh = []
    with open(src_file, "r", encoding="utf-8") as fr:
        for line in fr:
            boxes = line.strip().split()[1:]
            boxes = [list(map(int, box.split(",")[:4])) for box in boxes]
            wh = [(x2-x1, y2-y1) for x1, y1, x2, y2 in boxes]
            boxes_wh.extend(wh)
    
    boxes_wh = np.array(boxes_wh, dtype=np.float32)
    kmeans = KMeans(n_clusters=9)
    kmeans.fit(boxes_wh)
    
    anchors = np.round(kmeans.cluster_centers_).astype(dtype=np.int32)
    anchors = anchors.tolist()
    anchors_str = ", ".join([str(w)+","+str(h) for w, h in anchors])
    
    with open(dest_file, "w", encoding="utf-8") as fw:
        fw.write(anchors_str)


if __name__ == '__main__':
    get_anchors_with_kmeans(src_file=YOLO3_BOOK_PAGE_TAGS_FILE, dest_file=YOLO3_ANCHORS_FILE)
    print("Done !")
