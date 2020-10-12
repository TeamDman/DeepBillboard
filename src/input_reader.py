import glob
import os
import logging
import numpy as np
import cv2
import keras.preprocessing.image as image_preprocessor
from keras.applications.imagenet_utils import preprocess_input

LOGGER = logging.getLogger(__name__)

from collections import namedtuple

ImageSize = namedtuple("ImageSize", "width height")

def get_data_from_images(filelist):
    images = []
    image_size = None

    # Load image data
    for file in sorted(filelist):
        image = image_preprocessor.load_img(file)
        data = image_preprocessor.img_to_array(image)
        if image_size is None:
            shape = np.shape(data)
            image_size = ImageSize(shape[0], shape[1])
        data = cv2.resize(data, dsize=(100,100))
        data = np.expand_dims(data, axis=0)
        data = preprocess_input(data)
        images.append(data)
    return images, image_size

def get_coordinate_data(path, image_size: ImageSize):
    start_points = []
    occl_sizes = []

    with open(path) as file:
        for line in file:
            row = [float(x) for x in line.split()]
            start_points.append([
                round(float(row[2]) * 100.0 / image_size.width),
                round(float(row[1]) * 100.0 / image_size.height),
            ])
            occl_sizes.append([
                max(1, round((row[8] - row[2]) * 100.0 / image_size.width)) * 2,
                max(1, round((row[7] - row[1]) * 100.0 / image_size.width)) * 2,
            ])
    return start_points, occl_sizes

def get_input_images(path, extension):
    files = glob.glob(os.path.join(path, "*." + extension))
    LOGGER.info(f"Found {len(files)} images")

    images, image_size = get_data_from_images(files)

    # Load coordinate data
    start_points, occl_sizes = get_coordinate_data(
        os.path.join(path, "coordinates.txt"),
        image_size
    )
    return images, image_size, start_points, occl_sizes
