import logging
import math
from pathlib import Path

import cv2
import numpy as np
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


def get_model(target: int = 0, direction: str = "left", weight_diff: int = 1):
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
    if target == 0:
        loss_func = weight_diff * K.mean(model.get_layer('prediction').output)
    elif target == 1 or target == 2:
        loss_func = K.mean(model.get_layer('before_prediction').output[..., 0])
    else:
        print(f"Unknown model {target}")
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


def train(
    iterate: callable,
    batch_size: int,
    images: list,
    optimal: bool,
    angle_labels: list,
    model,
    gradient_descent_iterations: int,
    gradient_descent_step_size: int,
    greedy_strategy: str,
    percent_fixed_images: float,
    logo_size: ImageSize,
    overlay_strategy: str,
    start_points: list,
    occl_sizes: list,
    transformation: str,
    jsma_enabled: bool,
    jsma_count: int,
    simulated_annealing_k: int,
    simulated_annealing_b: int,
    simulated_annealing_enabled: bool,
):
    logo = np.zeros((logo_size.height, logo_size.width, 3))
    indices = np.arange(len(images), dtype=np.int32)

    imgs = np.array(images.copy())
    tmp_imgs = np.array(images.copy())
    # last_diff : the total difference in last mini batch
    last_diff = 0
    # change_times : the total change times of logo in ONE ITERATION
    change_times = 0
    bad_change_times = 0
    # we run gradient ascent for 20 steps
    fixed_pixels = np.zeros_like(logo)
    if optimal:
        logo[:, :] = old.utils.gen_optimal(imgs, model, angle_labels,
                                           start_points,
                                           occl_sizes)

    for iters in range(gradient_descent_iterations):
        fixed_pixels = np.zeros_like(logo)
        change_times = 0
        bad_change_times = 0
        if greedy_strategy != 'sequence_fix':
            np.random.shuffle(indices)
        for i in range(0, len(imgs), batch_size):
            if (
                (
                    greedy_strategy == 'sequence_fix'
                    or 'random_fix'
                    or 'highest_fix'
                )
                and i > percent_fixed_images * len(imgs)
            ):
                break
            if i <= len(imgs) - batch_size:
                minibatch = [imgs[indices[j]] for j in range(i, i + batch_size)]
            else:
                minibatch = [imgs[indices[j]] for j in range(i, len(imgs))]
            logo_data = np.zeros(
                (batch_size, logo_size.height, logo_size.width, 3))
            count = 0
            for gen_img in minibatch:
                loss_value1, loss_neuron1, grads_value = iterate([gen_img])
                if transformation == 'light':
                    # constraint the gradients value
                    grads_value = old.utils.constraint_light(grads_value)
                elif transformation == 'occl':
                    # print(np.shape(grads_value),start_points[indexs[i+count]],occl_sizes[indexs[i+count]])
                    # constraint the gradients value
                    grads_value = old.utils.constraint_occl(
                        grads_value,
                        start_points[indices[i + count]],
                        occl_sizes[indices[i + count]]
                    )
                elif transformation == 'blackout':
                    # constraint the  gradients value
                    grads_value = old.utils.constraint_black(grads_value)
                if jsma_enabled:
                    k_th_value = old.utils.find_kth_max(grads_value, jsma_count)
                    super_threshold_indices = abs(grads_value) < k_th_value
                    grads_value[super_threshold_indices] = 0
                # if the selected image's change make a positive reflection
                # (diff in one image > 0.1)
                # then we will count the image
                # (add the image's gradient into the logo_data)
                # if angle_diverged3(angle3[indexs[i+count]],model1.predict(tmp_img)[0]):
                logo_data = old.utils.transform_occl3(
                    grads_value, start_points[indices[i + count]],
                    occl_sizes[indices[i + count]], logo_data, count
                )
                # print(i,count,np.array_equal(np.sum(logo_data,axis = 0),np.zeros_like(np.sum(logo_data,axis = 0))))
                # random_fix and sequence fix is almost same
                # except that the indexes are shuffled or not
                if (
                    greedy_strategy == 'random_fix'
                    or greedy_strategy == 'sequence_fix'
                ):
                    # grads_value will only be adopted if the pixel is not fixed
                    logo_data[count] = cv2.multiply(
                        logo_data[count], 1 - fixed_pixels
                    )
                    grads_value = np.array(
                        logo_data[count],
                        dtype=np.bool
                    )
                    grads_value = np.array(
                        grads_value,
                        dtype=np.int
                    )
                    fixed_pixels += grads_value
                count += 1
            if overlay_strategy == 'sum':
                logo_data = np.sum(logo_data, axis=0)
            if overlay_strategy == 'max':
                index = np.argmax(
                    np.absolute(logo_data),
                    axis=0
                )
                shp = np.array(logo_data.shape)
                dim_idx = [index]
                dim_idx += list(np.ix_(*[np.arange(i) for i in shp[1:]]))
                logo_data = logo_data[dim_idx]

            tmp_logo = logo_data * gradient_descent_step_size + logo
            tmp_logo = old.utils.control_bound(tmp_logo)
            tmp_imgs = old.utils.update_image(
                tmp_imgs,
                tmp_logo,
                start_points,
                occl_sizes
            )
            # If this mini batch generates a
            # higher total difference we will consider this one.
            this_diff = old.utils.total_diff(tmp_imgs, model, angle_labels)
            if this_diff > last_diff:
                logo += logo_data * gradient_descent_step_size
                logo = old.utils.control_bound(logo)
                imgs = old.utils.update_image(
                    imgs,
                    logo,
                    start_points,
                    occl_sizes
                )
                last_diff = this_diff
                change_times += 1
            elif simulated_annealing_enabled:
                if (
                    old.utils.random.random()
                    <
                    pow(
                        math.e,
                        simulated_annealing_k * (this_diff - last_diff)
                        / (pow(simulated_annealing_b, iters))
                    )
                    and this_diff != last_diff
                ):
                    logo += logo_data * gradient_descent_step_size
                    logo = old.utils.control_bound(logo)
                    imgs = old.utils.update_image(
                        imgs,
                        logo,
                        start_points,
                        occl_sizes
                    )
                    last_diff = this_diff
                    bad_change_times += 1
        angle_diff = 0
        gray_angle_diff = 0
        for i in range(len(imgs)):
            prediction = model.predict(imgs[i])[0]
            gray_angle_diff += abs(prediction - angle_labels[i])
            # if(i==30):
            # gen_img_deprocessed = draw_arrow3(deprocess_image(imgs[i]),angle3[i],angle1)
            # imsave('./generated_inputs/' +str(iters) + '_iter.png', gen_img_deprocessed)
        if iters % 5 == 0:
            LOGGER.info(
                f"iteration {iters}. diff between raw and adversarial {gray_angle_diff / len(imgs) * (180 / math.pi)}. change time is {change_times}. bad_change_times, {bad_change_times}")

    decal = old.utils.deprocess_image(
        logo,
        shape=(logo_size.height, logo_size.width, 3)
    )
    LOGGER.info(f"Generating substitutes")
    log = ""
    out_images = []
    for i in range(len(imgs)):
        label = str(float(angle_labels[i]))
        prediction = model.predict(imgs[i])[0]
        line = f"image {i} label {label} guess {prediction}\n"
        log += line
        draw_angle = min(max(angle_labels[i], -math.pi / 2), math.pi / 2)
        out_image = old.utils.draw_arrow3(
            old.utils.deprocess_image(imgs[i]),
            draw_angle,
            prediction
        )
        out_images.append(out_image)
    return decal, out_images, log
