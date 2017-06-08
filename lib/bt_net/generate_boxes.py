"""
Library to generate boxes
"""
from multiprocessing import Pool
from skimage.transform import resize
import sys, random, math
import itertools
import numpy as np

def sliding_window(num_boxes, image, windowSize, workers=3):
    image_batch = None
    #'''Guarantee that middle of picture is always covered'''
    #data = get_middle_box(image, windowSize)
    #img = image[data[1]:data[3],data[0]:data[2]]
    #image_batch = img[np.newaxis]

    for i in range((num_boxes / workers)):
        y = random.randrange(0, image.shape[0] - windowSize[0])
        x = random.randrange(0, image.shape[1] - windowSize[1])
        img = image[y:y + windowSize[1], x:x + windowSize[0]]
        img = resize(img, (299, 299))

        if image_batch is None:
            image_batch = img[np.newaxis]
        else:
            image_batch = np.append(image_batch, img[np.newaxis], axis=0)
    return image_batch

def sliding_window2(num_boxes, image, windowSize, workers=3):
    image_batch = []
    for i in range(num_boxes / workers):
        y = random.randrange(0, image.shape[0] - windowSize[0])
        x = random.randrange(0, image.shape[1] - windowSize[1])
        img = image[y:y + windowSize[1], x:x + windowSize[0]]
        img = resize(img, (299, 299))
        image_batch.append(img[np.newaxis])

    return image_batch


def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return sliding_window(*a_b)


def get_middle_box(image, windowSize):
    x = int((image.shape[0] - windowSize[0]) / 2)
    y = int((image.shape[1] - windowSize[1]) / 2)
    return (x, y, x + windowSize[0], y + windowSize[1])

def generate_n_boxes(image, num_boxes=30, temp_size = (500, 700), pool=None):
    height = image.shape[0]
    width =  image.shape[1]
    if width > height:
        #image = resize(image, temp_size)
        windowSize_1 = (int(math.floor(image.shape[0] / 2)), int(math.floor(image.shape[1] / 2)))
        windowSize_2 = (int(math.floor(image.shape[0] / 3)), int(math.floor(image.shape[1] / 3)))
        windowSize_3 = (int(math.floor(image.shape[0] / 4)), int(math.floor(image.shape[1] / 4)))
    else:
        #image = resize(image, (temp_size[1], temp_size[0]))
        windowSize_1 = (int(math.floor(image.shape[0] / 2)), int(math.floor(image.shape[1] / 2)))
        windowSize_2 = (int(math.floor(image.shape[0] / 3)), int(math.floor(image.shape[1] / 3)))
        windowSize_3 = (int(math.floor(image.shape[0] / 4)), int(math.floor(image.shape[1] / 4)))

   # windowSize_1 = (int(math.floor(image.shape[0] / 2)), int(math.floor(image.shape[1] / 2)))
   # windowSize_2 = (int(math.floor(image.shape[0] / 4)), int(math.floor(image.shape[1] / 4)))
   # windowSize_3 = (int(math.floor(image.shape[0] / 6)), int(math.floor(image.shape[1] / 6)))
    windowSizes = [windowSize_1, windowSize_2, windowSize_3]
    num_boxes, image, workers = num_boxes, image, len(windowSizes)
    pool_none = False
    if pool is None:
        pool = Pool(4)
        pool_none = True
    im1, im2, im3 = pool.map(func_star, itertools.izip(itertools.repeat(num_boxes),
                                                itertools.repeat(image),
                                                windowSizes ,
                                                itertools.repeat(workers)))
    if pool_none:
        pool.close()
        pool.join()
        pool.terminate()
    result = im1
    result = np.append(result, im2, axis=0)
    result = np.append(result, im3, axis=0)
    #result = im1 + im2 + im3
    #print result[0]
    #print len(result)
    return result
