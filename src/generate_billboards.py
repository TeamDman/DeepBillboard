#####################
## SETUP
#####################
import logging
import common
import training

common.setup()
LOGGER = logging.getLogger(__name__)


#####################
## MODEL
#####################
from argument_parser import args

model, \
iterate = common.get_model(args.target_model, args.direction, args.weight_diff)

#####################
## LOAD
#####################
from input_reader import get_input_images

images, \
image_size, \
start_points, \
occl_sizes = get_input_images(args.path, args.type)

angle_labels = common.get_predictions(model, images)

#####################
## TRAIN
#####################
from imageio import imwrite as imsave

if __name__ == "__main__":
    for i in range(0, 100):
        decal, output_images, log = training.train(
            iterate=iterate,
            batch_size=5,
            images=images,
            optimal=args.op,
            model=model,
            gradient_descent_iterations=args.grad_iterations,
            greedy_strategy=args.greedy_stratage,
            logo_size=common.ImageSize(600, 400),
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
            simulated_annealing_enabled=args.simulated_annealing,
        )
        imsave(f"out/decal_{i}.png", decal)
        for j, img in enumerate(output_images):
            imsave(f"out/sub_iter_{i}_img_{j}.png", img)
