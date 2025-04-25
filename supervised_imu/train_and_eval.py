import torch.nn.functional as F
from .utils import *
from .model import Net
import os
import copy
from .supervised_loop import loop_over_all_epochs

def setup_supervised_imu(args):
    exp_config = {
        "num_of_features": 3, 
        "num_of_classes": 14,    
        "encoder_type": "resnet", 
        "base_data_filepath": args.imu_joblib_file, 
        "model_save_path": args.model_save_path,
        "data_augmentation": False,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 250,
        "gpu_device": 0,
        "num_of_epochs": 10,
        "embedding_dim": 512,
        "resnet_type": "resnet_1_block_conv_5",  
    }

    num_of_features = exp_config["num_of_features"]
    num_of_classes = exp_config["num_of_classes"]
    encoder_type = exp_config["encoder_type"]
    base_data_filepath = exp_config["base_data_filepath"]
    model_save_path = exp_config["model_save_path"]
    data_augmentation = exp_config["data_augmentation"]
    learning_rate = exp_config["learning_rate"]
    weight_decay = exp_config["weight_decay"]
    batch_size = exp_config["batch_size"]
    gpu_device = exp_config["gpu_device"]
    num_of_epochs = exp_config["num_of_epochs"]
    embedding_dim = exp_config["embedding_dim"]
    resnet_type = exp_config["resnet_type"]

    additional_params = {}
    additional_params["resnet_type"] = resnet_type 
    
    set_all_seeds(42)
    # Define net
    encoder = Encoder(
        num_of_features,
        embedding_dim,
        encoder_type,
        additional_params
    )
    classifier = Classifier(embedding_dim, num_of_classes, encoder_type)
    net = Net([encoder, classifier])

    device = torch.device(
        "cuda:" + str(gpu_device) if torch.cuda.is_available() else "cpu"
    )

    print(device)

    # transfer net to device
    net.to(device)

    initial_sd = copy.deepcopy(net.state_dict())

    optimizer = define_optimizer([net], learning_rate, weight_decay)

    os.makedirs(model_save_path, exist_ok=True)

    # load initial state dict before training for a new fold
    net.load_state_dict(initial_sd)

    return classifier, net, device, optimizer, exp_config


def train_supervised_imu(trainloader, valloader, net, device, optimizer, exp_config):
    (epoch_losses, epoch_accs, epoch_f1macro, epoch_f1weighted,) = loop_over_all_epochs(
        dataloaders=[trainloader, valloader],
        num_of_epochs=exp_config["num_of_epochs"],
        net=net,
        device=device,
        phase="train",
        model_save_path=exp_config["model_save_path"],
        optimizer=optimizer,
        base_data_filepath=exp_config["base_data_filepath"],
    )

def get_supervised_imu_predicted_probabilities(net, device, model_save_path, testloader):
    load(
        net,
        os.path.join(model_save_path, "supervised_imu.pkl"),
        encoder=True,
        classifier=True,
    )
    net.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in testloader:
            X = X.to(device)
            y = y.to(device)

            outputs = net(X)
            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()  # shape: (num_samples, num_classes)
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_probs, all_labels


