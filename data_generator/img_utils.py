# -*- encoding: utf-8 -*-
# Author: hushukai

import platform
import sys
import random
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from config import CHAR_IMG_SIZE, MAX_ROTATE_ANGLE


# 先生成比目标size稍大的图片，图片处理(旋转裁剪)之后，再缩放至目标大小
# 如果直接生成目标大小的图片，生成的字会更小，经过其他处理后，再放大时图片失真严重
# 这样也与处理外部图片的方式更加相似
def generate_bigger_image_by_font(chinese_char, font_file, image_size=int(CHAR_IMG_SIZE*1.2)):
    height = width = image_size

    # 生成灰度图像，黑底白字，黑色像素值为0，白色像素值为255
    # 之所以先生成黑底白字，是因为这样能简化后面的一些运算
    # 判断字体文件是否有效 和 寻找字的边界 时，利用"像素值>0"即可。若是白底黑字，则不好处理
    # 当需要白底黑字的图像时，用"255-像素值"转换即可
    bigger_PIL_img = Image.new(mode="L", size=(width, height), color="black")  # 黑色背景
    draw = ImageDraw.Draw(bigger_PIL_img, mode="L")

    font_size = int(0.85 * width)

    # Windows系统不能正确解析路径中的汉字时，将其按系统默认编码方式编码即可
    if "windows" in platform.architecture()[1].lower():
        font_file = font_file.encode(sys.getdefaultencoding())
    font_object = ImageFont.truetype(font_file, font_size)

    start_x = (width - font_size) // 2
    start_y = (height - font_size) // 2
    draw.text((start_x, start_y), text=chinese_char, fill=255, font=font_object)  # 白色字体

    # bigger_PIL_img.show()
    # print(np.array(bigger_PIL_img).tolist())
    return bigger_PIL_img


def rotate_PIL_image(PIL_img, rotate_angle=0):
    if rotate_angle == 0:
        return PIL_img
    else:
        rotated_img = PIL_img.rotate(rotate_angle, fillcolor=0)  # 旋转后在边角处填充黑色
        return rotated_img


# 图片增强, 添加点噪声
def add_noise(np_img):
    # 随机添加若干个点噪声
    for i in range(random.randint(10, 20)):
        pos_h = np.random.randint(0, np_img.shape[0])
        pos_w = np.random.randint(0, np_img.shape[1])
        point_size = random.randint(2, 3)
        np_img[pos_h:pos_h+point_size, pos_w:pos_w+point_size] = \
            255 - np_img[pos_h:pos_h+point_size, pos_w:pos_w+point_size]
    return np_img


# 图片增强，淡化局部
def add_local_dim(np_img):
    # 随机淡化某个区域
    pos_h = np.random.randint(0, np_img.shape[0])
    pos_w = np.random.randint(0, np_img.shape[1])
    radius = random.randint(5, 10)
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            if i**2 + j**2 <= radius**2 and 0 <= pos_h+i < np_img.shape[0] and 0 <= pos_w+j < np_img.shape[1]:
                np_img[pos_h + i, pos_w + j] = 0
    return np_img


# 图片增强, 腐蚀操作
def add_erode(np_img):
    # kernel_size = random.randint(2, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    np_img = cv2.erode(np_img, kernel)
    return np_img


# 图片增强, 膨胀操作
def add_dilate(np_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    np_img = cv2.dilate(np_img, kernel)
    return np_img


# 图片增强，所有相关函数打包其中
def image_augmentation(np_img, noise=True, dilate=True, erode=True):
    if random.random() < 0.6:
        np_img = add_local_dim(np_img)

    if noise and random.random() < 0.6:
        np_img = add_noise(np_img)

    if dilate and random.random() < 0.2:
        np_img = add_dilate(np_img)
    elif erode and random.random() < 0.5:
        np_img = add_erode(np_img)

    return np_img


# 查找字体的最小包含矩形, 黑底白字
def find_min_bound_box(np_img):
    height, width = np_img.shape[:2]
    column_sum = np.sum(np_img, axis=0)
    row_sum = np.sum(np_img, axis=1)

    left = 0
    right = width - 1
    top = 0
    low = height - 1

    # 从左往右扫描，遇到非零像素点就以此为字体的左边界
    for i in range(width):
        if column_sum[i] > 0:
            left = i
            break
    # 从右往左扫描，遇到非零像素点就以此为字体的右边界
    for i in range(width - 1, -1, -1):
        if column_sum[i] > 0:
            right = i
            break
    # 从上往下扫描，遇到非零像素点就以此为字体的上边界
    for i in range(height):
        if row_sum[i] > 0:
            top = i
            break
    # 从下往上扫描，遇到非零像素点就以此为字体的下边界
    for i in range(height - 1, -1, -1):
        if row_sum[i] > 0:
            low = i
            break
    return (left, right, top, low)


# 对字体图像做等比例缩放
def resize_image_keep_ratio(np_img, background_size=CHAR_IMG_SIZE):
    obj_height = obj_width = background_size
    cur_height, cur_width = np_img.shape[:2]

    width_ratio = obj_width / cur_width
    height_ratio = obj_height / cur_height
    scale_ratio = min(width_ratio, height_ratio)

    new_size = ( math.floor(cur_width*scale_ratio), math.floor(cur_height*scale_ratio))
    new_size = (max(new_size[0], 1), max(new_size[1], 1))

    # cv.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)
    # dsize为目标图像大小，(fx, fy)为(横, 纵)方向的缩放比例，参数dsize和参数(fx, fy)不必同时传值
    # interpolation为插值方法，共有5种：INTER_NEAREST 最近邻插值法，INTER_LINEAR 双线性插值法(默认)，
    # INTER_AREA 基于局部像素的重采样，INTER_CUBIC 基于4x4像素邻域的3次插值法，INTER_LANCZOS4 基于8x8像素邻域的Lanczos插值
    # 如果是缩小图片，效果最好的是INTER_AREA；如果是放大图片，效果最好的是INTER_CUBIC(slow)或INTER_LINEAR(faster but still looks OK)
    if scale_ratio==1:
        return np_img
    elif scale_ratio<1:
        interpolation = cv2.INTER_AREA
    elif scale_ratio>1:
        interpolation = cv2.INTER_CUBIC
    resized_np_img = cv2.resize(np_img, dsize=new_size, interpolation=interpolation)

    return resized_np_img


def put_img_in_center(small_np_img, large_np_img):
    small_height, small_width = small_np_img.shape[:2]
    large_height, large_width = large_np_img.shape[:2]

    if small_height > large_height:
        raise ValueError("small_height > large_height")
    if small_width > large_width:
        raise ValueError("small_width > large_width")

    start_x = (large_width - small_width) // 2
    start_y = (large_height - small_height) // 2
    large_np_img[start_y:start_y + small_height, start_x:start_x + small_width] = small_np_img

    return large_np_img


def adjust_img_and_put_into_background(np_img, background_size=CHAR_IMG_SIZE):
    obj_height = obj_width = background_size

    if len(np_img.shape) > 2:
        pixel_dim = np_img.shape[2]
    else:
        pixel_dim = None

    adjusted_np_img = resize_image_keep_ratio(np_img, background_size=background_size)

    adjusted_height, adjusted_width = adjusted_np_img.shape[:2]
    if adjusted_width == obj_width and adjusted_height == obj_height:
        obj_np_img = adjusted_np_img
    else:
        if pixel_dim is not None:
            background_shape = (obj_height, obj_width, pixel_dim)
        else:
            background_shape = (obj_height, obj_width)

        # 背景图片为全黑色，即全部像素值为0
        background_np_img = np.zeros(shape=background_shape, dtype=np.uint8)
        # 将缩放后的字体图像置于背景图像中央
        obj_np_img = put_img_in_center(small_np_img=adjusted_np_img, large_np_img=background_np_img)

    return obj_np_img


# 图片颜色反转，将黑底白字转换为白底黑字
def reverse_image_color(np_img=None, PIL_img=None):
    if np_img is not None:
        np_img = 255 - np_img
        return np_img
    elif PIL_img is not None:
        np_img = np.array(PIL_img, dtype=np.uint8)
        np_img = 255 - np_img
        PIL_img = Image.fromarray(np_img)
        return PIL_img
    else:
        return None


# 将生成的稍大的图片缩放至目标大小，图片颜色反转
def get_standard_image(PIL_img, obj_size=CHAR_IMG_SIZE, reverse_color=False):

    # 转化为numpy.ndarray格式的图片
    np_img = np.array(PIL_img, dtype='uint8')

    # 查找字体的最小包含矩形
    left, right, top, low = find_min_bound_box(np_img)
    np_img = np_img[top:low + 1, left:right + 1]

    # 把字体图像等比例缩放,使之撑满固定大小的背景图像
    np_img = adjust_img_and_put_into_background(np_img, background_size=obj_size)

    # 将黑底白字的汉字图片转换为白底黑字
    if reverse_color:
        np_img = reverse_image_color(np_img=np_img)

    # 转化为PIL.Image格式的图片
    std_PIL_img = Image.fromarray(np_img)

    return std_PIL_img


# 图片旋转，将生成的稍大的图片缩放至目标大小
# 对汉字图片进行增强及图片颜色反转
def get_augmented_image(PIL_img, obj_size=CHAR_IMG_SIZE, rotation=True, noise=True, dilate=True, erode=True, reverse_color=False):

    # 图像旋转一个角度
    if rotation:
        rotate_angle = random.choice(range(-MAX_ROTATE_ANGLE, MAX_ROTATE_ANGLE+1))
        PIL_img = rotate_PIL_image(PIL_img, rotate_angle)

    # 转化为numpy.ndarray格式的图片
    np_img = np.array(PIL_img, dtype='uint8')

    # 查找字体的最小包含矩形
    left, right, top, low = find_min_bound_box(np_img)
    np_img = np_img[top:low + 1, left:right + 1]

    # 把字体图像等比例缩放,使之撑满固定大小的背景图像
    np_img = adjust_img_and_put_into_background(np_img, background_size=obj_size)

    # 图像增强
    np_img = image_augmentation(np_img, noise=noise, dilate=dilate, erode=erode)

    # 将黑底白字的汉字图片转换为白底黑字
    if reverse_color:
        np_img = reverse_image_color(np_img=np_img)

    # 转化为PIL.Image格式的图片
    augmented_PIL_img = Image.fromarray(np_img)

    return augmented_PIL_img


# 加载外部图片，将图片调整为正方形
# 为了保证图片旋转时不丢失信息，生成的图片应该比本来的图片稍微bigger
# 为了方便图片的后续处理，图片必须加载为黑底白字，可以用reverse_color来调整
def load_external_image_bigger(img_path, white_background=True, reverse_color=True):
    PIL_img = Image.open(img_path)
    # print(PIL_img.size)
    # print(PIL_img.mode)   # 'P'
    if PIL_img.mode != "L":
        PIL_img = PIL_img.convert(mode="L")

    img_height, img_width = PIL_img.size

    np_img = np.array(PIL_img, dtype=np.uint8)
    # print(np_img.tolist())

    # 将图片转为黑底白字
    if white_background:
        np_img = reverse_image_color(np_img=np_img)

    obj_size = int(max(img_height, img_width) * 1.2)

    # 背景图片为全黑色，即全部像素值为0
    background_np_img = np.zeros(shape=(obj_size, obj_size), dtype=np.uint8)
    # 将原图像置于更大的背景图像中央
    bigger_np_img = put_img_in_center(small_np_img=np_img, large_np_img=background_np_img)

    # 判断是否需要再次反转颜色，异或运算
    if white_background^reverse_color:
        bigger_np_img = reverse_image_color(np_img=bigger_np_img)

    bigger_PIL_img = Image.fromarray(bigger_np_img)
    # print(bigger_PIL_img.size)
    # print(bigger_PIL_img.mode)
    
    return bigger_PIL_img


if __name__ == "__main__":

    # PIL_img = generate_bigger_image_by_font(chinese_char="龥", font_file="../chinese_fonts/mingliu.ttc")
    PIL_img = load_external_image_bigger("E:/pycharm_project/ziku_images/張即之/行.gif", white_background=True, reverse_color=True)
    # PIL_img = rotate_PIL_image(PIL_img, rotate_angle=30)
    # np_img = np.array(PIL_img, dtype=np.uint8)
    # print(np_img.tolist())
    # np_img = image_Augmentation(np_img)
    # np_img = adjust_img_and_put_into_Background(np_img, obj_size=(196,196))

    # # 图片去噪
    # np_img[np_img < 10] = 0
    # np_img[np_img > 240] = 255
    # PIL_img = Image.fromarray(np_img)

    # PIL_img = get_standard_image(PIL_img, obj_size=max(PIL_img.height, PIL_img.width), reverse_color=True)
    PIL_img = get_augmented_image(PIL_img, obj_size=max(PIL_img.height, PIL_img.width), rotation=False, noise=False, dilate=False,erode=False, reverse_color=True)
    PIL_img.show()

    print("Done !")