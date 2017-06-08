from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from PIL import Image

import numpy as np


def preprocess_array(img_array, target_size=(299, 299)):
    img = Image.fromarray(np.uint8(img_array))
    img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return np.squeeze(np.array(x, dtype=np.float32))


def read_img(img_path, target_size=None):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return np.squeeze(np.array(x, dtype=np.float32))
