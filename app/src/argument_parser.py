# read the parameter
# argument parsing
import argparse

parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in Driving dataset')
parser.add_argument(
    '-tr',
    '--transformation',
    help="realistic transformation type",
    choices=[
        'light',
        'occl',
        'blackout'
    ],
    default='occl')
parser.add_argument(
    '-diff',
    '--weight_diff',
    help="weight hyperparm to control differential behavior",
    default=1,
    type=float)
parser.add_argument(
    '-w',
    '--weight_nc',
    help="weight hyperparm to control neuron coverage",
    default=0.1,
    type=float)
parser.add_argument(
    '-st',
    '--step', help="step size of gradient descent",
    default=5,
    type=float)
parser.add_argument(
    '-sd',
    '--seeds', help="number of seeds of input",
    default=1,
    type=int)
parser.add_argument(
    '-grad',
    '--grad_iterations',
    help="number of iterations of gradient descent",
    default=25,
    type=int)
parser.add_argument(
    '-th',
    '--threshold',
    help="threshold for determining neuron activated",
    default=0,
    type=float)
parser.add_argument(
    '-t',
    '--target_model',
    help="target model that we want it predicts differently",
    choices=[0, 1, 2],
    default=0,
    type=int)
parser.add_argument(
    '-sp',
    '--start_point',
    help="occlusion upper left corner coordinate",
    default=(50, 50),
    type=tuple)
parser.add_argument(
    '-occl_size',
    '--occlusion_size',
    help="occlusion size",
    default=(50, 50),
    type=tuple)
parser.add_argument(
    '-overlap_stra',
    '--overlap_stratage',
    help='max:select maximum gradient value. sum:..., highest: detect the influence',
    choices=[
        'max',
        'sum',
        'highest_influence'
    ],
    default='sum')
parser.add_argument(
    '-greedy_stra',
    '--greedy_stratage',
    help='random_fix:fix pixels if it is changed in one iteration and the img is selected randomly. dynamic:pixels can be changed in each iteration. highest: detect the influence',
    choices=[
        'random_fix',
        'sequence_fix',
        'dynamic',
        'hightest_fix'
    ],
    default='dynamic')
parser.add_argument(
    '-fix_p',
    '--fix_p',
    help='parameter p(percetage) of fixed images(batches), only p of 1 images will be selected',
    default=1.0,
    type=float)
parser.add_argument(
    '-jsma',
    '--jsma',
    help='Using jsma or not',
    default=False,
    type=bool)
parser.add_argument(
    '-jsma_n',
    '--jsma_n',
    help='parameter n of jsma, top n pixels in gradients of board will be selected',
    default=5,
    type=int)
parser.add_argument(
    '-sa',
    '--simulated_annealing',
    help='Simulated annealing to control the logo update',
    default=True,
    type=bool)
parser.add_argument(
    '-sa_k',
    '--sa_k',
    help='parameter k of simulated annealing, higher = less changing probability',
    default=30,
    type=int)
parser.add_argument(
    '-sa_b',
    '--sa_b',
    help='parameter b in range (0,1) of simulated annealing, higher = higher changing probability. p = pow(e,k*diff/pow(b,iter))',
    default=0.96,
    type=float)
parser.add_argument(
    '-path',
    '--path',
    default="../Digital/digital_Dave_curve1",
    type=str)
parser.add_argument(
    '-direction',
    '--direction',
    default='left',
    type=str)
parser.add_argument(
    '-batch',
    '--batch',
    default=5,
    type=int)
parser.add_argument(
    '-type',
    '--type',
    default="jpg",
    choices=["jpg", "png"])
parser.add_argument(
    '-op',
    '--op',
    default=False,
    type=bool)
args = parser.parse_args()
