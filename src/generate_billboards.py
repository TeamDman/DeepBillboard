import logging

from common import get_predictions

logging.basicConfig(level=logging.INFO)

from pathlib import Path

import tensorflow as tf
import tensorflow.keras.backend as K

import common

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

model, \
iterate = common.get_model(args.target_model, args.direction, args.weight_diff)

#####################
## LOAD
#####################

images, \
image_size, \
start_points, \
occl_sizes = get_input_images(args.path, args.type)

angle_labels = get_predictions(model, images)

logo_width = 600
logo_height = 400

#####################
## TRAIN
#####################

if __name__ == "__main__":
    for i in range(0, 100):
        common.train(
            iteration=i,
            iterate=iterate,
            batch_size=5,
            images=images,
            optimal=args.op,
            model=model,
            gradient_descent_iterations=args.grad_iterations,
            greedy_strategy=args.greedy_stratage,
            logo_size=common.ImageSize(logo_width, logo_height),
            angle_labels=angle_labels,
            percent_fixed_images=args.fix_p,
            overlay_strategy=args.overlap_stratage,
            start_points=start_points,
            occl_sizes=occl_sizes,
            transformation=args.transformation,
            jsma_enabled=args.jsma,
            jsma_count=args.jsma_n,
            gradient_descent_step_size=args.step,
            simulated_annealing_b=args.sa_b,
            simulated_annealing_k=args.sa_k,
            simulated_annealing_enabled=args.simulated_annealing
        )
