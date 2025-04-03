import os
import socket
from functools import partial
from ray import tune, train
from ray.tune.search.basic_variant import BasicVariantGenerator

from arguments import parse_args
from utils import set_all_seeds
from evaluate_with_classifier  import  evaluate_with_classifier

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Loading the arguments necessary
    args = parse_args()
    set_all_seeds(args.random_seed)
    print(args)

   
    # adding an exp name for easy access
    args.notes = ''

    hostname = socket.gethostname()
    args.hostname = hostname

    param_space = {
       # pre-training
        # "classifier_lr": tune.choice([1e-3, 1e-4, 5e-4]),
        # "classifier_wd": tune.choice([0.0, 1e-4, 1e-5]),
        
        # "classifier_lr":0.001,
        # "classifier_wd":0.0,
        "classifier_batch_size": 64,

        
        "embedding_type":tune.grid_search(["v1", 'v1-1', 'gpt_v1', 'gpt_v1-1']),
    

        "dataset": tune.grid_search(["aruba", "milan", "cairo", "kyoto7"]),
       
        "fold": tune.grid_search([1,2,3]),


        "exp_name": tune.sample_from(
            lambda spec: '{}'.format('tdost_har_new_enc')),
        
        "sentence_encoder_name": tune.grid_search([ 'all-distilroberta-v1', 'sentence-t5-base']),
    }

    tuner = tune.Tuner(
        tune.with_resources(
            partial(evaluate_with_classifier, args=args),
            resources={"cpu": 6.0, "gpu": 1.0}
        ),
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(constant_grid_search=True),
            num_samples=1
        ),
        run_config=train.RunConfig(
            local_dir="/coc/pcba1/mthukral3/gt/TDOST/HAR/code/ray_results"
        ),
        param_space=param_space,
    )

    results = tuner.fit()

    print('---------Training complete!---------')
