import logging

logging.basicConfig(level=logging.INFO)

from pathlib import Path
from imageio import imwrite as imsave
from old.driving_models import *
from old.utils import *

from argument_parser import args
from input_reader import get_input_images

#####################
## SETUP
#####################

LOGGER = logging.getLogger(__name__)

tf.compat.v1.disable_eager_execution()
Path("out/").mkdir(exist_ok=True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3

# load multiple models sharing same input tensor
K.set_learning_phase(0)


#####################
## MODEL
#####################

# noinspection PyShadowingNames
def get_model(target):
    # define input tensor as a placeholder
    # shape is input image dimensions, (rows, columns, colours)
    input_tensor = Input(shape=(100, 100, 3))

    model = Dave_orig(input_tensor=input_tensor, load_weights=True)
    model_layer_dict = init_coverage_tables2(model)
    loss_func = None
    # construct joint loss function
    if target == 0:
        loss_func = args.weight_diff * K.mean(
            model.get_layer('prediction').output)
    elif target == 1 or target == 2:
        loss_func = K.mean(model.get_layer('before_prediction').output[..., 0])
    else:
        print(f"Unknown model {target}")
        exit(1)


    # for adversarial image generation
    final_loss = K.mean(loss_func)
    if (args.direction == "left"):
        final_loss = K.mean(loss_func)
    elif (args.direction == "right"):
        final_loss = -K.mean(loss_func)
    else:
        LOGGER.error(f"Unknown direction \"{args.direction}\"")
        exit()

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])
    neuron_layer, neuron_index = neuron_to_cover(model_layer_dict)
    loss_neuron = K.mean(model.get_layer(neuron_layer).output[..., neuron_index])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss_func, loss_neuron, grads])

    return model, iterate

model, \
iterate = get_model(args.target_model)


#####################
## LOAD
#####################

imgs, \
img_size, \
start_points, \
occl_sizes = get_input_images(args.path, args.type)


# noinspection PyShadowingNames
def get_predictions(model, images):
    rtn = []
    for image in images:
        rtn.append(model.predict(image)[0])
    return rtn


angle_labels = get_predictions(model, imgs)


logo_width = 600
logo_height = 400

batch = 5
indexs = np.arange(len(imgs), dtype=np.int32)
imgs = np.array(imgs)


def get_temp_images():
    global imgs
    return imgs.copy()


def train(iteration):
    LOGGER.info(f"Training iteration {iteration}")
    logo = np.zeros((logo_height, logo_width, 3))

    imgs = get_temp_images()
    tmp_imgs = get_temp_images()
    # last_diff : the total difference in last minibatch
    last_diff = 0
    # change_times : the total change times of logo in ONE ITERATION
    change_times = 0
    bad_change_times = 0
    # we run gradient ascent for 20 steps
    fixed_pixels = np.zeros_like(logo)
    if (args.op):
        logo[:, :] = gen_optimal(imgs, model, angle_labels, start_points,
                                 occl_sizes)

    for iters in range(args.grad_iterations):
        fixed_pixels = np.zeros_like(logo)
        change_times = 0
        bad_change_times = 0
        if (args.greedy_stratage != 'sequence_fix'):
            np.random.shuffle(indexs)
        for i in range(0, len(imgs), batch):
            if ((
                args.greedy_stratage == 'sequence_fix' or 'random_fix' or 'highest_fix') and i > args.fix_p * len(
                imgs)):
                break
            if i <= len(imgs) - batch:
                minibatch = [imgs[indexs[j]] for j in range(i, i + batch)]
            else:
                minibatch = [imgs[indexs[j]] for j in range(i, len(imgs))]
            logo_data = np.zeros((batch, logo_height, logo_width, 3))
            count = 0
            for gen_img in minibatch:
                loss_value1, loss_neuron1, grads_value = iterate([gen_img])
                if args.transformation == 'light':
                    # constraint the gradients value
                    grads_value = constraint_light(grads_value)
                elif args.transformation == 'occl':
                    # print(np.shape(grads_value),start_points[indexs[i+count]],occl_sizes[indexs[i+count]])
                    grads_value = constraint_occl(grads_value,
                                                  start_points[
                                                      indexs[i + count]],
                                                  occl_sizes[indexs[
                                                      i + count]])  # constraint the gradients value
                elif args.transformation == 'blackout':
                    # constraint the  gradients value
                    grads_value = constraint_black(grads_value)
                if (args.jsma):
                    k_th_value = find_kth_max(grads_value, args.jsma_n)
                    super_threshold_indices = abs(grads_value) < k_th_value
                    grads_value[super_threshold_indices] = 0
                # IF the selected image's change make a positive reflection (diff in one image > 0.1) then
                #  we will count the image(add the image's gradient into the logo_data)
                # if angle_diverged3(angle3[indexs[i+count]],model1.predict(tmp_img)[0]):
                logo_data = transform_occl3(
                    grads_value, start_points[indexs[i + count]],
                    occl_sizes[indexs[i + count]], logo_data, count)
                # print(i,count,np.array_equal(np.sum(logo_data,axis = 0),np.zeros_like(np.sum(logo_data,axis = 0))))
                # random_fix and sequence fix is almost same except that the indexes are shuffled or not
                if (
                    args.greedy_stratage == 'random_fix' or args.greedy_stratage == 'sequence_fix'):
                    # grads_value will only be adopted if the pixel is not fixed
                    logo_data[count] = cv2.multiply(
                        logo_data[count], 1 - fixed_pixels)
                    grads_value = np.array(logo_data[count], dtype=np.bool)
                    grads_value = np.array(grads_value, dtype=np.int)
                    fixed_pixels += grads_value
                count += 1
            if (args.overlap_stratage == 'sum'):
                logo_data = np.sum(logo_data, axis=0)
            if (args.overlap_stratage == 'max'):
                index = np.argmax(np.absolute(logo_data), axis=0)
                shp = np.array(logo_data.shape)
                dim_idx = []
                dim_idx.append(index)
                dim_idx += list(np.ix_(*[np.arange(i) for i in shp[1:]]))
                logo_data = logo_data[dim_idx]
            # TODO1: ADAM May be adapted.
            # TODO2: Smooth box constait
            # TODO3: Consider the angle increase or decrease direction (the gradient should be positive or negative)

            tmp_logo = logo_data * args.step + logo
            tmp_logo = control_bound(tmp_logo)
            tmp_imgs = update_image(tmp_imgs, tmp_logo, start_points,
                                    occl_sizes)
            # If this minibatch generates a higher total difference we will consider this one.
            this_diff = total_diff(tmp_imgs, model, angle_labels)
            # print("iteration ",iters,". batch count ",i,". this time diff ",this_diff,". last time diff ", last_diff)
            if (this_diff > last_diff):
                logo += logo_data * args.step
                logo = control_bound(logo)
                imgs = update_image(imgs, logo, start_points, occl_sizes)
                last_diff = this_diff
                change_times += 1
            else:
                # simulated_annealing is applied in current version. DATE: 26/07
                if (args.simulated_annealing):
                    # if(this_diff != last_diff):
                    # print(i,"probability = ",pow(math.e,args.sa_k * (this_diff-last_diff)/(pow(args.sa_b,iters))),". this diff ",this_diff,". last diff ", last_diff)
                    if (random.random() < pow(math.e,
                                              args.sa_k * (
                                                  this_diff - last_diff) / (
                                                  pow(args.sa_b,
                                                      iters))) and this_diff != last_diff):
                        logo += logo_data * args.step
                        logo = control_bound(logo)
                        imgs = update_image(imgs, logo, start_points,
                                            occl_sizes)
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
        if (iters % 5 == 0):
            LOGGER.info(
                f"iteration {iters}. diff between raw and adversarial {gray_angle_diff / len(imgs) * (180 / math.pi)}. change time is {change_times}. bad_change_times, {bad_change_times}")

    deprocessed = deprocess_image(logo, shape=(logo_height, logo_width, 3))
    imsave(f"out/decal_{iteration}.png", deprocessed)
    LOGGER.info(f"Training iteration {iteration} done")
    LOGGER.info(f"Generating substitutes")
    output = open('out/' + "Output.txt", "a")
    for i in range(len(imgs)):
        label = str(float(angle_labels[i]))
        prediction = model.predict(imgs[i])[0]
        line = f"iteration {iteration} image {i} label {label} guess {prediction}\n"
        output.write(line)
        draw_angle = min(max(angle_labels[i], -math.pi / 2), math.pi / 2)
        out_image = draw_arrow3(deprocess_image(imgs[i]), draw_angle,
                                prediction)
        imsave(f"out/sub_iter_{iteration}_img_{i}.png", out_image)
    output.close()


for i in range(0, 100):
    train(i)