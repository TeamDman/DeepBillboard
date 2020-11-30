import logging
from pathlib import Path

import tensorflow as tf
import tensorflow.keras.backend as K

import old.driving_models
import old.utils

LOGGER = logging.getLogger(__name__)

from collections import namedtuple

ImageSize = namedtuple("ImageSize", "width height")


def display_image(image):
    """
    https://gist.github.com/ctmakro/3ae3cd9538390b706820cd01dac6861f
    """
    import cv2
    import IPython
    _, ret = cv2.imencode(".jpg", image)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)


def setup():
    logging.basicConfig(level=logging.INFO)

    tf.compat.v1.disable_eager_execution()
    Path("out/").mkdir(exist_ok=True)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    # load multiple models sharing same input tensor
    K.set_learning_phase(0)


def get_model(target_model: int = 0, direction: str = "left", weight_diff: int = 1):
    # define input tensor as a placeholder
    # shape is input image dimensions, (rows, columns, colours)
    input_tensor = tf.keras.layers.Input(shape=(100, 100, 3))

    model = old.driving_models.Dave_orig(
        input_tensor=input_tensor,
        load_weights=True,
    )

    model_layer_dict = old.utils.init_coverage_tables2(model)
    loss_func = None
    # construct joint loss function
    if target_model == 0:
        loss_func = weight_diff * K.mean(model.get_layer('prediction').output)
    elif target_model == 1 or target_model == 2:
        loss_func = K.mean(model.get_layer('before_prediction').output[..., 0])
    else:
        print(f"Unknown model {target_model}")
        exit(1)

    # for adversarial image generation
    final_loss = None
    if direction == "left":
        final_loss = K.mean(loss_func)
    elif direction == "right":
        final_loss = -K.mean(loss_func)
    else:
        LOGGER.error(f"Unknown direction \"{direction}\"")
        exit()

    # we compute the gradient of the input picture wrt this loss
    grads = old.utils.normalize(K.gradients(final_loss, input_tensor)[0])
    neuron_layer, neuron_index = old.utils.neuron_to_cover(model_layer_dict)
    loss_neuron = K.mean(
        model.get_layer(neuron_layer).output[..., neuron_index]
    )

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss_func, loss_neuron, grads])

    return model, iterate


def get_predictions(model, images):
    rtn = []
    for image in images:
        rtn.append(model.predict(image)[0])
    return rtn


playing_for_benchmarks_billboard_colour = [150, 20, 20]