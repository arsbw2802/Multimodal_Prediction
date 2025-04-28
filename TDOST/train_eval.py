import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from utils import set_all_seeds
from model import AutoEncoder   


"""
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

parser.add_argument('--fold',        type=int, default=kwargs.get('fold'   ,     2))
    parser.add_argument('--input_size',  type=int, default=kwargs.get('input_size' , 768))
    parser.add_argument('--dataset',     type=str, default=kwargs.get('dataset'   , 'aruba'))
    parser.add_argument('--root_dir',    type=str, default=kwargs.get('root_dir', '/coc/pcba1/mthukral3/gt/TDOST/folds/pre-segmented/'))
    parser.add_argument('--num_classes', type=int, default=kwargs.get('num_classes', 9))
    parser.add_argument('--num_epochs',  type=int, default=kwargs.get('num_epochs',  75))
"""

# train_tdost(tdost_trainloader, tdost_valloader, net, device, optimizer,
# exp_config)
def setup_tdost(args):
    # reproducibility
    set_all_seeds(args.random_seed)

    # device
    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")

    # model
    model = AutoEncoder(args) \
        .to(device)

    # optimizer + scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10),
        gamma=getattr(args, "lr_gamma", 0.8)
    )

    return model, device, optimizer, scheduler


def train_tdost(model, trainloader, valloader, optimizer, scheduler, device, args):
    criterion = nn.CrossEntropyLoss()
    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(1, args.num_epochs + 1):
        # --- training ---
        model.train()
        running_loss = 0.0
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        scheduler.step()
        train_loss = running_loss / len(trainloader.dataset)

        # --- validation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in valloader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch}/{args.num_epochs}  "
              f"train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    # optionally persist best model
    os.makedirs(args.model_save_path, exist_ok=True)
    torch.save(best_wts, os.path.join(args.model_save_path, "tdost_model.pth"))
    print(f"✔️  Training complete. Best val_acc={best_val_acc:.4f}")

    return model


def get_tdost_predicted_probabilities(model, device, testloader, args):
    """
    Loads best weights, runs test set, returns (probs, labels) numpy arrays.
    """
    # load best if not already in memory
    ckpt = os.path.join(args.model_save_path, "tdost_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    all_probs  = torch.cat(all_probs,  dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_probs, all_labels

