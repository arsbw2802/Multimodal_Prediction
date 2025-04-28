import os
import pickle
import random
from datetime import date

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score


import socket


def open_raw_data_file(path):

    data = np.load(path, allow_pickle=True)

    return data





def compute_best_metrics(running_meter, best_meter, classifier=False):
    """
    To compute the best validation loss from the running meter object
    :param running_meter: running meter object with all values
    :param best_meter: updating the best meter based on current running meter
    :return: best validation f1-score
    """
    if classifier:
        loc = np.argmax(running_meter.f1_score['val'])
    else:
        min_loss = np.min(running_meter.loss['val'])  # Minimum loss
        loc = np.where(running_meter.loss['val'] == min_loss)[
            0][-1]  # The latest epoch to give the lowest loss

    # Epoch where the best validation loss was obtained
    epoch = running_meter.epochs[loc]

    # Updating the best meter with values based on the epoch
    phases = ['train', 'val', 'test'] if classifier else ['train', 'val']
    for phase in phases:
        best_meter.update(
            phase, running_meter.loss[phase][loc],
            running_meter.accuracy[phase][loc],
            running_meter.f1_score[phase][loc],
            running_meter.f1_score_weighted[phase][loc],
            running_meter.confusion_matrix[phase][loc],

            running_meter.recall[phase][loc],
            running_meter.recall_weighted[phase][loc],

            running_meter.precision[phase][loc],
            running_meter.precision_weighted[phase][loc],
            
            epoch)

    return best_meter


def compute_metrics(actual_labels, pred_labels, phase, running_meter, loss,
                    epoch):
    acc = accuracy_score(actual_labels, pred_labels)
    f_score_weighted = f1_score(actual_labels, pred_labels, average='weighted')
    f_score_macro = f1_score(actual_labels, pred_labels, average='macro')

    recall_weighted = recall_score(actual_labels, pred_labels, average='weighted', zero_division=0)
    recall_macro = recall_score(actual_labels, pred_labels, average='macro', zero_division=0)

    precision_weighted = precision_score(actual_labels, pred_labels, average='weighted', zero_division=0)
    precision_macro = precision_score(actual_labels, pred_labels, average='macro', zero_division=0)

    conf_matrix = confusion_matrix(y_true=actual_labels, y_pred=pred_labels,
                                   normalize="true")
    running_meter.update(phase, loss, acc, f_score_macro, f_score_weighted,
                         conf_matrix, recall_macro, recall_weighted, precision_macro, precision_weighted)

    # printing the metrics
    print("The epoch: {} | phase: {} | loss: {:.4f} | accuracy: {:.4f} | mean "
          "f1-score: {:.4f} | weighted f1-score: {:.4f} | recall: {:.4f} | weighted recall: {:.4f} | "
          "precision: {:.4f} | weighted precision: {:.4f}"
          .format(epoch, phase, loss, acc, f_score_macro, f_score_weighted,  recall_macro, recall_weighted , precision_macro,  precision_weighted))

    return


def update_loss(phase, running_meter, loss, epoch):
    running_meter.update(phase, loss, 0, 0, 0, [])

    # printing the metrics
    print("The epoch: {} | phase: {} | loss: {:.4f}"
          .format(epoch, phase, loss))

    return


def model_save_name(args, classifier=False, capture=False):
    if capture:
        name = 'aruba_emb_{0.embedding_type}_fold_1_{0.embedding_type}_lr_{0.classifier_lr}_wd_{0.classifier_wd}_bs_{0.classifier_batch_size}_{0.sentence_encoder_name}_pretrained'.format(args)
    else:
        name = '{0.dataset}_emb_{0.embedding_type}_fold_{0.fold}_lr_{0.classifier_lr}_wd_{0.classifier_wd}_bs_{0.classifier_batch_size}_{0.sentence_encoder_name}_pretrained'.format(args)

    return name


def save_meter(args, running_meter, classifier=False):
    """
    Saving the logs
    :param args: arguments
    :param running_meter: running meter object to save
    :param mlp: if saving during the MLP training, then adds '_eval_log.pkl'
    to the end
    :return: nothing
    """
    name = model_save_name(args, classifier=classifier)
    save_name = name + '_eval_log.pkl' if classifier else name + '_log.pkl'

    # Creating logs by the dat now. To make stuff easier
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, 'saved_logs', date.today().strftime(
        "%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, save_name), 'wb') as f:
        pickle.dump(running_meter, f, pickle.HIGHEST_PROTOCOL)

    return


def save_model(model, args, epoch):
    """
    Saves the weights from the model
    :param model: model being trained
    :param args: arguments
    :return: nothing
    """
    name = model_save_name(args)

    # Creating logs by the dat now. To make stuff easier
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, 'models', date.today().strftime(
        "%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    model_name = os.path.join(folder, name + '.pkl')
    print("model name", model_name)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }, model_name)

    return

def update_args(config, args):
    # Data loading parameters
    if 'dataset' in config:
        args.dataset = config['dataset']
    if 'root_dir' in config:
        args.root_dir = config['root_dir']
        


    # IMU Encoder parameters
    if 'kernel_size' in config:
        args.kernel_size = config['kernel_size']
    if 'projection_type' in config:
        args.projection_type = config['projection_type']
    if 'imu_encoder_type' in config:
        args.imu_encoder_type = config['imu_encoder_type']
    if 'saved_imu_model' in config:
        args.saved_imu_model = config['saved_imu_model']
    if 'freeze_imu_encoder' in config:
        args.freeze_imu_encoder =config['freeze_imu_encoder']

    # HAR classification/fine-tuning parameters
    if 'learning_schedule' in config:
        args.learning_schedule = config['learning_schedule']
    if 'classifier_lr' in config:
        args.classifier_lr = config['classifier_lr']
    if 'classifier_wd' in config:
        args.classifier_wd = config['classifier_wd']
    if 'classifier_batch_size' in config:
        args.classifier_batch_size = config['classifier_batch_size']
    if 'fold' in config:
        args.fold = config['fold']
    if 'saved_model_folder' in config:
        args.saved_model_folder = config['saved_model_folder']
    if 'classification_model' in config:
        args.classification_model = config['classification_model']

    # Experiment names
    if 'exp_name' in config:
        args.exp_name = config['exp_name']
    if 'pre_exp_name' in config:
        args.pre_exp_name = config['pre_exp_name']

    # Seed and other info
    if 'random_seed' in config:
        args.random_seed = config['random_seed']

    # Text prompt and test prompt parameters
    # this one is for sampling from available options to the detailed labels.
    if 'embedding_type' in config:
        args.embedding_type = config['embedding_type']
        
    if 'sentence_encoder_name' in config:
        args.sentence_encoder_name = config['sentence_encoder_name']
        
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
       
       
        
    if args.sentence_encoder_name =="all-MiniLM-L6-v2":
        args.input_size=384
    
    elif args.sentence_encoder_name =="all-distilroberta-v1":
        args.input_size=768

    elif args.sentence_encoder_name =="sentence-t5-base":
        args.input_size=768


    print('Completed updating args')
    return args


def set_all_seeds(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    return
