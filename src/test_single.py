import common
import logging
import input_reader
import glob
import os.path

common.setup()

LOGGER = logging.getLogger(__name__)

files = glob.glob(os.path.join("resources", "new", "*.jpg"))
images, image_size = input_reader.get_data_from_images(files)


model, iterate = common.get_model()

predictions = common.get_predictions(model, images)

for file, prediction in zip(files, predictions):
    print(file, end="\t")
    print(prediction)