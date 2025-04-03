from typing import List, Any, Tuple
import torch
from torch.autograd import Variable
from sklearn.metrics import f1_score
from torch import nn
import os
import sys
from utils import *

# TODO: change name
def loop_over_all_epochs(
    dataloaders: List,
    num_of_epochs: int,
    net: nn.Module,
    device: str,
    phase: str,
    model_save_path: str,
    base_data_filepath: str,
    optimizer: Any = None,
) -> Tuple[List, List, List, List]:

    if phase == "train":

        train_epoch_losses = []
        train_epoch_accs = []
        train_epoch_f1macro = []
        train_epoch_f1weighted = []
        val_epoch_losses = []
        val_epoch_accs = []
        val_epoch_f1macro = []
        val_epoch_f1weighted = []
        max_fscore = 0.0
        trigger_times=0
        min_val = 1000000.0
        for epoch in range(num_of_epochs):
            net.train()
            (
                train_running_loss,
                train_running_corrects,
                train_actual_labels,
                train_pred_labels,
                train_dataset_size,
            ) = loop_over_all_datapoints(
                dataloader=dataloaders[0],
                device=device,
                optimizer=optimizer,
                net=net,
                phase="train",
            )

            train_epoch_loss = train_running_loss / train_dataset_size
            train_epoch_acc = (
                (train_running_corrects.double() / train_dataset_size)
                .cpu()
                .data.numpy()
            )
            train_epoch_f_score_weighted = f1_score(
                train_actual_labels, train_pred_labels, average="weighted"
            )
            train_epoch_f_score_macro = f1_score(
                train_actual_labels, train_pred_labels, average="macro"
            )

            train_epoch_losses.append(train_epoch_loss)
            train_epoch_f1macro.append(train_epoch_f_score_macro)
            train_epoch_f1weighted.append(train_epoch_f_score_weighted)
            train_epoch_accs.append(train_epoch_acc)

            print("Epoch {}".format(epoch))
            print(
                "{} Loss: {:.4f} Acc: {:.4f}".format(
                    "train", train_epoch_loss, train_epoch_acc
                )
            )
            print(
                "{} F1 score weighted: {:.4f}, F1 score macro: {:.4f}".format(
                    "train", train_epoch_f_score_weighted, train_epoch_f_score_macro
                )
            )
            net.eval()

            (
                val_running_loss,
                val_running_corrects,
                val_actual_labels,
                val_pred_labels,
                val_dataset_size,
            ) = loop_over_all_datapoints(
                dataloader=dataloaders[1],
                device=device,
                optimizer=optimizer,
                net=net,
                phase="val",
            )

            val_epoch_loss = val_running_loss / val_dataset_size
            val_epoch_acc = (
                (val_running_corrects.double() / val_dataset_size).cpu().data.numpy()
            )
            val_epoch_f_score_weighted = f1_score(
                val_actual_labels, val_pred_labels, average="weighted"
            )
            val_epoch_f_score_macro = f1_score(
                val_actual_labels, val_pred_labels, average="macro"
            )
            
            
            ## early stopping
            if epoch >= 2:
                if val_epoch_loss > min_val:
                    trigger_times += 1
                    print('Trigger times: {}'.format(trigger_times))

                    if trigger_times >= 5:
                        print('Early stopping the model at epoch: {}. The '
                            'validation loss has not improved for {}'.format(
                            epoch, trigger_times))
                        break
                else:
                    trigger_times = 0
                    print('Resetting the trigger counter for early stopping')

            ## SAVE THE BEST MODEL
            if val_epoch_f_score_macro >= max_fscore:
                path = os.path.join(model_save_path, "teacher_model_best.pkl")
                save(net, optimizer, epoch, path, base_data_filepath)
                max_fscore = val_epoch_f_score_macro
                
                
            if val_epoch_loss <min_val:
                min_val = val_epoch_loss

            val_epoch_losses.append(val_epoch_loss)
            val_epoch_f1macro.append(val_epoch_f_score_macro)
            val_epoch_f1weighted.append(val_epoch_f_score_weighted)
            val_epoch_accs.append(val_epoch_acc)

            print(
                "{} Loss: {:.4f} Acc: {:.4f}".format(
                    "val", val_epoch_loss, val_epoch_acc
                )
            )
            print(
                "{} F1 score weighted: {:.4f}, F1 score macro: {:.4f}".format(
                    "val", val_epoch_f_score_weighted, val_epoch_f_score_macro
                )
            )

        loss = [train_epoch_losses, val_epoch_losses]
        acc = [train_epoch_accs, val_epoch_accs]
        f1macro = [train_epoch_f1macro, val_epoch_f1macro]
        f1weighted = [train_epoch_f1weighted, val_epoch_f1weighted]
        
        print("Best val score ", max_fscore)
    else:
        load(
            net,
            os.path.join(model_save_path, "teacher_model_best.pkl"),
            encoder=True,
            classifier=True,
        )
        net.eval()

        (
            test_loss,
            test_corrects,
            test_actual_labels,
            test_pred_labels,
            test_dataset_size,
        ) = loop_over_all_datapoints(
            dataloader=dataloaders[0],
            device=device,
            optimizer=optimizer,
            net=net,
            phase="test",
        )

        loss = [test_loss / test_dataset_size]
        acc = [(test_corrects.double() / test_dataset_size).cpu().data.numpy()]

        f1weighted = [
            f1_score(test_actual_labels, test_pred_labels, average="weighted")
        ]
        f1macro = [f1_score(test_actual_labels, test_pred_labels, average="macro")]

    return loss, acc, f1macro, f1weighted


# TODO: change name
def loop_over_all_datapoints(
    dataloader: Any, device: str, optimizer: Any, net: Any, phase: str
):
    running_corrects = 0
    running_loss = 0.0
    actual_labels = []
    pred_labels = []
    total_size = 0

    for X, y in dataloader:
        X = Variable(X).float()
        y = Variable(y).type(torch.LongTensor)

        if device:
            X = X.to(device)
            y = y.to(device)

        if phase == "train":
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
            output = net(X)
            _, preds = torch.max(output, 1)
            loss = nn.CrossEntropyLoss()(output, y)

            if phase == "train":
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * X.size(0)
        running_corrects += torch.sum(preds == y.data)
        actual_labels.extend(y.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())
        total_size += X.size(0)

    return running_loss, running_corrects, actual_labels, pred_labels, total_size
