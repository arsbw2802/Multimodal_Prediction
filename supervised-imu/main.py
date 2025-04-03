from model import Encoder
from model import Classifier
from model import Net
from dataset import MyDataset
import torch
import typer
import numpy as np
import os
from supervised_loop import (
    loop_over_all_datapoints,
    loop_over_all_epochs,
)
from torch import nn
import omegaconf
from utils import *
import copy
import matplotlib.pyplot as plt

import optuna
from optuna.integration.tensorboard import TensorBoardCallback

# Set up the TensorBoard log directory
log_dir = "./tb_logs"

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Params for the resnet model')
    parser.add_argument(
        '--base_data_filepath',
        type=str,
        default="C:\\Users\\LENOVO\\Desktop\\supervised\\all_data\\Nov-16-2024\\",
        help='Location of a joblib file. EG "C:\\Users\\LENOVO\\Desktop\\supervised\\all_data\\Oct-29-2024\\"'
        )
    
    parser.add_argument('--model_save_path', type=str, default='C:\\Users\\LENOVO\\Desktop\\supervised',
                        help='The location/dir for where the model will be saved after training/tuning. EG:"C:\\Users\\LENOVO\\Desktop\\supervised"')
    
    parser.add_argument(
        '--n_folds',
        type=int,
        default=3,
        help='The number of folds that the dataset has been split to.'
        )
    
    parser.add_argument(
        '--working_fold',
        type=int,
        default=0,
        help='fold to work on'
        )
    parser.add_argument(
        '--num_of_classes',
        type=int,
        default=0,
        help='number of classes (unique activities) in the dataset'
        )
    

    args = parser.parse_args()
    return args

def objective(trial):
    # Define the hyperparameter search space
    exp_config = {
        "num_of_features": 3, 
        "num_of_classes": args.num_of_classes,    
        "encoder_type": "resnet", 
        # "base_data_filepath": args.base_data_filepath, 
        "model_save_path": args.model_save_path,
        "data_augmentation": False,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "batch_size": 250,
        "gpu_device": 1,
        "num_of_epochs": 10,
        "embedding_dim": 512,
        "resnet_type": "resnet_1_block_conv_5",  
    }

    fold_test_losses = []

    for fold_num in range(args.working_fold,args.working_fold+1):
        exp_config["base_data_filepath"] = os.path.join(args.base_data_filepath, f'hhar_watch_sr_50_fold_{fold_num}.joblib')
        
        # Train model and calculate validation metrics
        num_of_features = exp_config["num_of_features"]
        num_of_classes = exp_config["num_of_classes"]
        encoder_type = exp_config["encoder_type"]
        model_save_path = exp_config["model_save_path"]
        data_augmentation = exp_config["data_augmentation"]
        learning_rate = exp_config["learning_rate"]
        weight_decay = exp_config["weight_decay"]
        batch_size = exp_config["batch_size"]
        num_of_epochs = exp_config["num_of_epochs"]
        embedding_dim = exp_config["embedding_dim"]
        additional_params = {"resnet_type": exp_config["resnet_type"]}
        
        set_all_seeds(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model initialization
        encoder = Encoder(num_of_features, embedding_dim, encoder_type, additional_params)
        classifier = Classifier(embedding_dim, num_of_classes, encoder_type)
        net = Net([encoder, classifier]).to(device)

        initial_sd = copy.deepcopy(net.state_dict())
        optimizer = define_optimizer([net], learning_rate, weight_decay)

        # Load initial state dict before training
        net.load_state_dict(initial_sd)

        # Data loading
        train_dataset = MyDataset(phase="train", filepath=exp_config["base_data_filepath"])
        if data_augmentation:
            train_dataset.X, train_dataset.y = apply_augmentations(train_dataset)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        val_dataset = MyDataset(phase="val", filepath=exp_config["base_data_filepath"])
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Train & Validate
        loss, acc, f1macro, f1weighted = loop_over_all_epochs(
            dataloaders=[trainloader, valloader],
            num_of_epochs=num_of_epochs,
            net=net,
            device=device,
            phase="train",
            model_save_path=model_save_path,
            optimizer=optimizer,
            base_data_filepath=exp_config["base_data_filepath"],
        )

        # Run test phase after training
        test_dataset = MyDataset(phase="test", filepath=exp_config["base_data_filepath"])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loss, test_acc, test_f1macro, test_f1weighted = loop_over_all_epochs(
            dataloaders=[test_loader],
            num_of_epochs=1,  # Only one test run needed
            net=net,
            device=device,
            phase="test",
            model_save_path=model_save_path,
            base_data_filepath=exp_config["base_data_filepath"]
        )
        
        train_epoch_losses = test_loss[0]
        val_epoch_losses = test_loss[1]
        plt.plot(train_epoch_losses)
        plt.plot(val_epoch_losses)

        fig_filepath = model_save_path + "/loss.png"
        plt.title("Train & Val Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(fig_filepath)
        plt.clf()
        # Return the best validation loss as the objective to minimize
        fold_test_losses.append(test_loss[0])

    avg_test_loss = np.mean(fold_test_losses) 
    print(f'fold_test_losses: {fold_test_losses}')
    print(f'fold_test_losses avg: {avg_test_loss}')
    return fold_test_losses[0]

if __name__ == "__main__":
    tensorboard_callback = TensorBoardCallback(dirname=log_dir, metric_name="Fold 0: F1 Weighted")
    args = parse_arguments()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, callbacks=[tensorboard_callback])
    print('##')
    print('##')
    print("Best hyperparameters:", study.best_params)
    print('##')
    print('##')

# To view hyperparm results, run `tensorboard --logdir=./tb_logs` in the terminal/cmd
