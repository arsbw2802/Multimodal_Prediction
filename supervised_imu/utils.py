import numpy as np
import random
import torch
import torch.nn as nn
from .model import Encoder
from .model import Classifier

def save(net, optimizer, epoch, path, datafile=None):

    sd = {}
    for i in range(len(net.model)):
        if isinstance(net.model[i], Encoder):
            sd["Encoder"] = net.model[i].state_dict()
        elif isinstance(net.model[i], Classifier):
            sd["Classifier"] = net.model[i].state_dict()
        elif isinstance(net.model[i], nn.GRU):
            sd["GRU"] = net.model[i].state_dict()

    sd["optimizer"] = optimizer.state_dict()
    sd["epoch"] = epoch
    sd["datafile"] = datafile

    torch.save(sd, path)


def load(
    net,
    path,
    encoder: bool = True,
    decoder: bool = True,
    classifier: bool = True,
    gru: bool = True,
    device="cpu",
):
    ### load weights from pretrained model
    print("In load fn,  Loading model stored at path:", path)

    sd = torch.load(path, map_location=device)
    print("Model saved at epoch", sd["epoch"])
    # print("teacher sd", [(k, torch.sum(sd["Encoder"][k].data).data )for k in sd['Encoder']])
    print(sd.keys())
    for i in range(len(net.model)):
        if isinstance(net.model[i], Encoder) and encoder:
            net.model[i].load_state_dict(sd["Encoder"])
        elif isinstance(net.model[i], Classifier) and classifier:
            net.model[i].load_state_dict(sd["Classifier"])
        elif isinstance(net.model[i], nn.GRU) and gru:
            net.model[i].load_state_dict(sd["GRU"])

    return sd

def define_optimizer(
    model, learning_rate, weight_decay: int = 1e-5, optim_type: str = "ADAM"
):
    params = [{"params": m.parameters()} for m in model]
    if optim_type == "ADAM":
        optimizer = torch.optim.Adam(
            params, lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9
        )
    return optimizer


def define_scheduler(optimizer, step_size, gamma):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Seed ", seed, " Set!!")
    return