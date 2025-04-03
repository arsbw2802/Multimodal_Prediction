from arguments import parse_args
from trainer import learn_model
from utils import set_all_seeds
import torch
import os
from ray import tune
from functools import partial


def mini_main(config, args):
    args.input_data_percentage = config['idp']
    args.random_seed = config['seed']
    args.fold_val = config['fold']
    args.learning_rate = 0.005
    args.batch_size = 4096

    set_all_seeds(args.random_seed)
    # Starting the pre-training
    print("Starting the pre-training!")
    learn_model(args=args)

    return 1

# ------------------------------------------------------------------------------
if __name__ == '__main__':


    # DO NOT CHANGE
    #input_data_percent_array = [50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.390625, 0.1953125]
    #random_seeds_array = [42, 5, 31, 73, 58]
    # KEEP IN MIND THE EXECUTION TIME!
    input_data_percent_array = [50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.390625, 0.1953125] # Just for testing purposes
    random_seeds_array = [42, 5, 31, 73, 58] # Just for testing purposes
    fold_array = [0, 1, 2, 3, 4]

    print("GPU Availability Flag")
    print(torch.cuda.is_available())
    print('test 1')
    print(os.path.isdir('./Aug'))

    args = parse_args()
    set_all_seeds(args.random_seed)

    #Define hyper parameter serach space
    config = {
        'idp' : tune.grid_search([50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.390625, 0.1953125]), #loguniform, range:[1e-4,0.1], quantization level:1e-4
        'seed' : tune.grid_search([42, 5, 31, 73, 58]),
        'fold' : tune.grid_search([0, 1, 2, 3, 4]),
    }

    #(Optional), you can specify metrics that will be shown in the raytune progress table
    #reporter = CLIReporter( metric_columns = ["train_loss", "val_loss"] )

    #Tune launch
    experiment = tune.run(
        partial(mini_main, args=args), #Trainable function
        #name = 'raytune_example', #(Optional) Experiment Name
        config = config, #Hyper parameter search space
        local_dir = 'ray_result', #raytune result save directory
        resources_per_trial = {'cpu': 1, 'gpu': 0.25}, #CPU&GPU resources that each trial can use
        num_samples = 1, #How many trials will be perforemd
        #scheduler = asha_scheduler, #raytune scheduler
        fail_fast = True, #If at least one trial reports error, immediately quit every trial
        log_to_file = True,
        #progress_reporter = reporter
        )


    print('------ Pre-training complete! ------')
