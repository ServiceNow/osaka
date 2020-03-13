"""
Defines the command-line arguments used in main.py.
"""

import argparse
from typing import List, Optional, Union
import yaml


def str2bool(value: Union[str, bool]) -> bool:
    """Parses a `bool` value from a string.

    Can be used as the `type` argument to the `add_argument()` function for easy
    parsing of flags/boolean values.

    Taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Examples:
    >>> str2bool("1")
    True
    >>> str2bool("0")
    False
    >>> str2bool("false")
    False
    >>> str2bool("true")
    True
    >>> str2bool("yes")
    True
    >>> str2bool("no")
    False
    """
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected for argument, received '{value}'")

def parse_args():
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser('MAML', formatter_class=help_formatter)

    # General args

    group = parser.add_argument_group("General Settings")
    #group.add_argument('--mode', default='train', choices=['train', 'test'])
    datasets = ['sinusoid', 'omniglot', 'miniimagenet', 'tiered-imagenet', "harmonics", "synbols"]
    group.add_argument('--dataset', choices=datasets, default='omniglot', help='Name of the dataset.')  # NOTE: removed "-da" option.
    group.add_argument('--folder',         type=str, default='Data', help='Path to the folder the data is downloaded to.')
    group.add_argument('--wandb',          type=str, default=None,   help='Wandb project name. If None, no wandb logging')
    group.add_argument('--name',           type=str, default=None,   help='Wandb run name. If None, name will be random')
    group.add_argument('--wandb_key',      type=str, default=None,   help='Wandb token key for login. If None, shell login assumed')
    group.add_argument('--output-folder',  type=str, default=None,   help='Path to the output folder to save the model.')
    group.add_argument('--pretrain-model', type=str, default=None,   help='Path to the pretrained model (if any)')
    group.add_argument('--num_ways',       type=int, default=5,      help='Number of classes per task (N in "N-way").')
    group.add_argument('--num_shots',      type=int, default=5,      help='Number of training example per class (k in "k-shot").')
    group.add_argument('--num_shots-test', type=int, default=15,     help='Number of test example per class. If negative, same as the number of training examples `--num_shots`')
    group.add_argument('--seed',           type=int, default=100,    help='Seed')

    # Model
    group = parser.add_argument_group("Model Settings")
    group.add_argument('--model_name', type=str, default='ours', choices=['ours', 'online_sgd', 'fine_tuning', 'MetaCOG', 'MetaBGD', 'MAML', 'ANIL', 'BGD'])
    group.add_argument('-hs', '--hidden_size', type=int, default=64, help='Number of channels in each convolution layer of the VGG network or hidden size of an MLP. If None, kept to default')
    group.add_argument('-de', '--deeper', type=int, default=0, help='number of layers after the convs and before the classifier')
    group.add_argument('-nclml', '--no_cl_meta_learning',  type=int,   default=0,    help='turn off meta-learning at cl time')
    group.add_argument('--freeze_visual_features',         type=int,   default=0,    help='for MRCL, freeze all conv layers at cl time')
    group.add_argument('-cl_s',  '--cl_strategy',          type=str,   default=None, choices=['always_retrain', 'never_retrain', 'acc', 'loss'])
    group.add_argument('-cl_st', '--cl_strategy_thres',    type=float, default=0,    help='threshold for training on the incoming data or not')
    group.add_argument('-cl_tt', '--cl_tbd_thres',         type=float, default=-1,   help='threshold for task boundary detection (-1 to turn on off)')

    # ModularMAML
    group = parser.add_argument_group("ModularMAML", "Settings related to the Modular MAML model.")
    group.add_argument('-m', '--method', type=str, default='MAML', choices=['MAML','ModularMAML', 'DynamicModularMAML'])
    group.add_argument('-mo', '--modularity', type=str, default='param_wise', choices=['param_wise'], help='dont mind this for now')
    group.add_argument('-ma','--mask_activation', type=str, default='None', choices=['None','sigmoid','ReLU', 'hardshrink'], help='activation before applying the masks')
    group.add_argument('-lr', '--l1_reg', type=float, default=0.0, help='regularization strenght to encourage sparsity')
    group.add_argument('-kr', '--kl_reg', type=float, default=0.0, help='regularization strength to encourage hard sparsity')
    group.add_argument('-bp', '--bern_prior', type=float, default=0.2, help='bernouili prior')
    group.add_argument('-mi', '--masks_init', type=float, default=0.5, help='masks initialization')
    group.add_argument('-hm', '--hard_masks', type=int, default=0, help='{0,1} masking')

    # Optimization
    group = parser.add_argument_group("Optimization", "Settings for the optimizers and learning-rate schedules")
    group.add_argument('--batch_size', type=int, default=25, help='Number of tasks in a batch of tasks (default: 25).')
    group.add_argument('--num_epochs', type=int, default=50, help='Number of epochs of meta-training (default: 50).')
    group.add_argument('--patience', type=int, default=5, help='Number of epochs without a valid loss decrease we can wait')
    group.add_argument('--num_batches', type=int, default=100, help='Number of batch of tasks per epoch (default: 100).')
    group.add_argument('-ns', '--num_steps', type=int, default=1, help='Number of inner updates')
    group.add_argument('-ss', '--step_size', type=float, default=0.1, help='Size of the fast adaptation step, ie. learning rate in the gradient descent update (default: 0.1).')
    group.add_argument('-lss', '--learn_step_size', type=bool, default=False, help='Weither or not to learn the (inner loop) step-size')
    group.add_argument('-ppss', '--per_param_step_size', type=bool, default=False, help='Weither ot not to learn param specific step-size')
    group.add_argument('--first_order', type=int, default=0, help='Use the first order approximation, do not use higher-order derivatives during meta-optimization.')
    group.add_argument('--meta_lr', type=float, default=0.001, help='Learning rate for the meta-optimizer (optimization of the outer loss). The default optimizer is Adam (default: 1e-3).')

    # CL
    group = parser.add_argument_group("CL", "Settings specific to the continual learning setting.")
    group.add_argument('--model_config', type=str, default="config/ours.yaml", help="Path to a yaml config file.")
    group.add_argument('--n_runs',     type=int, default=1,     help='number of runs for cl experiment')
    group.add_argument('--timesteps',  type=int, default=10000, help='number of timesteps for the CL exp')
    group.add_argument('--prob_statio',                    type=float, default=0.98, help='probability to stay in the same task')
    group.add_argument("--task_sequence",    type=str, choices=["train", "test", "ood"], default=None, nargs="*", help="predefined task sequence for the dataloader to serve in a loop.")
    group.add_argument("--n_steps_per_task", type=int, default=1, help="Number of steps per task in the sequence.")

    #BGD for CL
    group = parser.add_argument_group("BGD", "Settings related to the BGD optimizer")
    group.add_argument('-bgd', '--bgd_optimizer', type=int, default=0, help='set to "True" if BGD optimizer should be used')
    group.add_argument('-tr_mc', '--train_mc_iters', default=5, type=int, help='Number of MonteCarlo samples during training with BGD optimizer')
    group.add_argument('--mean_eta', default=0.001, type=float, help='Eta for mean step')
    group.add_argument('--std_init', default=0.01, type=float, help='STD init value')

    # Misc
    group = parser.add_argument_group("Misc", "Miscelaneous settings")
    group.add_argument('--num-workers', type=int, default=1, help='Number of workers to use for data-loading (default: 1).')
    group.add_argument('-d', '--debug', action='store_true', help='enable debug mode to go faster')
    group.add_argument('-v', '--verbose', action='store_true')
    group.add_argument('--note', type=str, default=None, help='to ease hparam search analysis')


    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    if args.model_name != "ours" and args.model_config == "config/ours.yaml":
        # use the custom config file if a different model_name was passed.
        args.model_config = f"config/{args.model_name}.yaml"

    # Load a set of pre-configured arguments.
    if args.model_config:
        with open(args.model_config) as f:
            file_args = yaml.load(f, Loader=yaml.FullLoader)
        # overwrite the default values with the values from the file.
        args_dict = vars(args)
        args_dict.update(vars(file_args))
        args = argparse.Namespace(**args_dict)

    if args.debug:
        print('\nDEBUGGING\n')
        args.batch_size = 8
        args.num_epochs = 1
        args.num_batches = 10
        args.first_order = 1
        args.timesteps = 100
        args.prob_statio = 0.9
        args.n_runs = 2

    return args

if __name__ == "__main__":
    import doctest
    doctest.testmod()
