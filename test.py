import tensorflow as tf
from PIL import Image
import numpy as np
import os
from util import check_or_makedirs

im = Image.open("1.jpg")
print(im.mode, im.size)

np_im = np.array(im)
tf_im = tf.constant(np_im)
print(tf_im.dtype)

img = tf.image.grayscale_to_rgb(tf_im[:,:,tf.newaxis])

# scale image to fixed size
fixed_size = tf.constant([640, 640], dtype=tf.float32)  # 16的倍数
raw_shape = tf.cast(tf.shape(img)[:2], tf.float32)
scale_ratio = tf.reduce_min(fixed_size / raw_shape)
new_size = tf.cast(raw_shape * scale_ratio, dtype=tf.int32)
img = tf.image.resize(img, size=new_size)
delta = tf.cast(fixed_size, tf.int32) - new_size
dh, dw = delta[0], delta[1]
img = tf.pad(img, paddings=[[0, dh], [0, dw], [0, 0]], mode='CONSTANT', constant_values=255) # fixed_size, 白底黑字

# image = tf.image.random_brightness(img, max_delta=0.5)
# image = tf.image.random_contrast(image, lower=0.5, upper=2.)
# image = tf.image.random_hue(image, max_delta=0.4)
# image = tf.image.random_jpeg_quality(image, min_jpeg_quality=20, max_jpeg_quality=80)
# image = tf.image.random_saturation(image, lower=0.5, upper=5)

# check_or_makedirs(os.path.join("..", "summary"))
# summary_writer = tf.summary.create_file_writer(os.path.join("..", "summary"))
# with summary_writer.as_default():
#     print(np_im.dtype)
#     tf.summary.image("image", np_im.reshape((1, 897, 708, 1)).astype("float32")/255, step=0)
# summary_writer.flush()

noise = tf.random.normal(img.shape, mean=0.0, stddev=30.0)

img = img + noise
img = tf.where(img<0, 0, img)
img = tf.where(img>255, 255, img)
img = tf.cast(img, tf.uint8)

for i in range(100):
    print(i, img.dtype)
    
    # ****************************
    delta = -1+i*2/100
    im = tf.image.adjust_brightness(img, delta=delta)
    print(im.dtype)
    np_im = im.numpy().astype(np.uint8)
    p_im = Image.fromarray(np_im)
    check_or_makedirs(os.path.join("..", "tf_image", "brightness"))
    im_path = os.path.join("..", "tf_image", "brightness", "delta_" + str(delta) + ".jpg")
    p_im.save(im_path, format="jpeg")

    # ****************************
    contrast_factor = 0.3 + i * 1.5 / 100
    im = tf.image.adjust_contrast(img, contrast_factor=contrast_factor)
    print(im.dtype)
    np_im = im.numpy().astype(np.uint8)
    p_im = Image.fromarray(np_im)
    check_or_makedirs(os.path.join("..", "tf_image", "contrast"))
    im_path = os.path.join("..", "tf_image", "contrast", "contrast_factor_" + str(contrast_factor) + ".jpg")
    p_im.save(im_path, format="jpeg")

    # ****************************
    delta = -1 + i * 2 / 100
    im = tf.image.adjust_hue(img, delta=delta)
    print(im.dtype)
    np_im = im.numpy().astype(np.uint8)
    p_im = Image.fromarray(np_im)
    check_or_makedirs(os.path.join("..", "tf_image", "hue"))
    im_path = os.path.join("..", "tf_image", "hue", "delta_" + str(delta) + ".jpg")
    p_im.save(im_path, format="jpeg")

    # ****************************
    jpeg_quality = 0 + i
    im = tf.image.adjust_jpeg_quality(img, jpeg_quality=jpeg_quality)
    print(im.dtype)
    np_im = (im.numpy()*255).astype(np.uint8)
    # print(np_im)
    p_im = Image.fromarray(np_im)
    check_or_makedirs(os.path.join("..", "tf_image", "jpeg_quality"))
    im_path = os.path.join("..", "tf_image", "jpeg_quality", "quality_" + str(jpeg_quality) + ".jpg")
    p_im.save(im_path, format="jpeg")

    # ****************************
    saturation_factor = 0 + i*100/100
    im = tf.image.adjust_saturation(img, saturation_factor=saturation_factor)
    print(im.dtype)
    np_im = im.numpy().astype(np.uint8)
    p_im = Image.fromarray(np_im)
    check_or_makedirs(os.path.join("..", "tf_image", "saturation"))
    im_path = os.path.join("..", "tf_image", "saturation", "saturation_factor_" + str(saturation_factor) + ".jpg")
    p_im.save(im_path, format="jpeg")
    
    
    