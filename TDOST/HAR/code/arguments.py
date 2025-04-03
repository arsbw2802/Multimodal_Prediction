import argparse

import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def parse_args(**kwargs):
    parser = argparse.ArgumentParser(
        description='Parameters for the autoencoder pre-training and '
                    'classification')

    # Data loading parameters
    parser.add_argument('--fold',        type=int, default=kwargs.get('fold'   ,     2))
    parser.add_argument('--input_size',  type=int, default=kwargs.get('input_size' , 768))
    parser.add_argument('--dataset',     type=str, default=kwargs.get('dataset'   , 'aruba'))
    parser.add_argument('--root_dir',    type=str, default=kwargs.get('root_dir', '/coc/pcba1/mthukral3/gt/TDOST/folds/pre-segmented/'))
    parser.add_argument('--num_classes', type=int, default=kwargs.get('num_classes', 9))
    parser.add_argument('--num_epochs',  type=int, default=kwargs.get('num_epochs',  75))

    parser.add_argument('--sentence_encoder_name', type=str, default='sentence-t5-base', 
                        choices=["all-MiniLM-L6-v2", "all-distilroberta-v1",'sentence-t5-base'],
                        help='name of the sentence encoder')
   
    # -----------------------------------------------------------

    # Encoder and decoder params
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='The filter size for the conv layers')
    parser.add_argument('--padding', type=int, default=1,
                        help='The padding size for the conv layers')

    # -----------------------------------------------------------

    # Classification parameters
    parser.add_argument('--classifier_lr', type=float, default=5e-3,
                        help='Learning rate for the classifier')
    parser.add_argument('--classifier_wd', type=float, default=1e-4,
                        help='L2 norm for the classifier')
    parser.add_argument('--classifier_batch_size', type=int, default=64,
                        help='Batch size for the classifier')
    parser.add_argument('--learning_schedule', type=str, default='all_layers',
                        choices=['last_layer', 'all_layers', 'last_conv'],
                        help='whether to train all layers or the last layer')
    parser.add_argument('--drop_last', type=str, default='False',
                        help='To drop the last batch if necessary')
    parser.add_argument('--classification_model', type=str, default='mlp',
                        choices=['linear', 'mlp'],
                        help='Choosing the classifier: linear is a single FC '
                             'layer whereas MLP is the 3-layer network with '
                             'BN and ReLU and dropout.')

   

    # ------------------------------------------------------------
    # Random seed setting
    parser.add_argument('--random_seed', type=int, default=42)

    # -----------------------------------------------------------
    
    parser.add_argument('--embedding_type', type=str, default="v1")
    parser.add_argument('--gpu_id', type=int, default=0)
    
    
    args, _ = parser.parse_known_args()

    if args.dataset == 'milan':
        args.num_classes = 10
    elif args.dataset == 'aruba':
        args.num_classes = 9
    elif args.dataset == 'cairo':
        args.num_classes = 7
    elif args.dataset == 'aware':
        args.num_classes = 10
    elif args.dataset == 'kyoto7':
        args.num_classes = 7

    args.device = torch.device(
        "cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")

    args.padding = int(args.kernel_size // 2)

    # ------------------------------------------------------------
    # Getting the dataset locs based on args.dataset

    return args
