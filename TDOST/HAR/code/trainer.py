import copy
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_dataset
from meter import RunningMeter, BestMeter
from model import AutoEncoder
from tqdm.auto import tqdm
from utils import set_all_seeds, compute_best_metrics, update_loss, save_meter, \
    save_model, update_args


def learn_model(config, args=None):
    print('Inside pre-train')
    set_all_seeds(args.random_seed)
    
    args = update_args(config, args)


    print(args)

    # Data loaders
    data_loaders, dataset_sizes = load_dataset(args)
    

    # Tracking meter
    running_meter = RunningMeter(args=args)
    best_meter = BestMeter()

    model = AutoEncoder(args).to(args.device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    trigger_times = 0

    for epoch in range(0, args.num_epochs):
        since = time()

        # Training
        model, optimizer = train(model,
                                 data_loaders["train"],
                                 criterion,
                                 optimizer,
                                 args,
                                 epoch,
                                 dataset_sizes["train"],
                                 running_meter)

        scheduler.step()

        # Evaluating on the validation data
        evaluate(model,
                 data_loaders["val"],
                 args,
                 criterion,
                 epoch,
                 phase="val",
                 dataset_size=dataset_sizes["val"],
                 running_meter=running_meter)

        # Saving the logs
        save_meter(args, running_meter)

        # Doing the early stopping check
        if epoch >= 2:
            if running_meter.loss['val'][-1] > best_meter.loss["val"]:
                trigger_times += 1
                print('Trigger times: {}'.format(trigger_times))

                if trigger_times >= args.patience:
                    print('Early stopping the model at epoch: {}. The '
                          'validation loss has not improved for {}'.format(
                        epoch, trigger_times))
                    break
            else:
                trigger_times = 0
                print('Resetting the trigger counter for early stopping')

        # Updating the best weights
        if running_meter.loss["val"][-1] < best_meter.loss["val"]:
            print('Updating the best val loss at epoch: {}, since {} < '
                  '{}'.format(epoch, running_meter.loss["val"][-1],
                              best_meter.loss["val"]))
            best_meter = compute_best_metrics(running_meter, best_meter)
            running_meter.update_best_meter(best_meter)

            best_model_wts = copy.deepcopy(model.state_dict())

            # Saving the logs
            save_meter(args, running_meter)

        # Printing the time taken
        time_elapsed = time() - since
        print('Epoch {} completed in {:.0f}m {:.0f}s'
              .format(epoch, time_elapsed // 60, time_elapsed % 60))

    # Printing the best metrics
    best_meter.display()

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Saving the best performing model
    print('Updating the best trained model at epoch: {}!'.format(epoch))
    save_model(model, args, epoch=epoch)

    return


def train(model, data_loader, criterion, optimizer, args, epoch, dataset_size,
          running_meter):
    # Setting the model to training mode
    model.train()

    # To track the loss and other metrics
    running_loss = 0.0

    # Iterating over the data
    for inputs, _ in tqdm(data_loader):
        inputs = inputs.float().to(args.device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)

            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)

    # Statistics
    loss = running_loss / dataset_size
    update_loss(phase="train", running_meter=running_meter, loss=loss,
                epoch=epoch)

    return model, optimizer


def evaluate(model, data_loader, args, criterion, epoch, phase, dataset_size,
             running_meter):
    model.eval()

    # To track the loss and other metrics
    running_loss = 0.0

    # Iterating over the data
    for inputs, _ in tqdm(data_loader):
        inputs = inputs.float().to(args.device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

        # Appending predictions and loss
        running_loss += loss.item() * inputs.size(0)

    # Statistics
    loss = running_loss / dataset_size
    update_loss(phase=phase, running_meter=running_meter, loss=loss,
                epoch=epoch)

    return
