import torch
from supervised_imu.dataset import MyDataset
from TDOST.dataset import HARDataset

def get_supervised_imu_data_loaders(args):
    data_augmentation = False
    #if data_augmentation:
    #    train_dataset.X, train_dataset.y = apply_augmentations(train_dataset)
    train_dataset = MyDataset(phase="train", filepath=args.imu_joblib_file)
    val_dataset = MyDataset(phase="val", filepath=args.imu_joblib_file)
    test_dataset = MyDataset(phase="test", filepath=args.imu_joblib_file)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    #print(f"Number of samples in the training dataset: {len(trainloader.dataset)}")

    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True
    )
    print("✅ Successfully loaded data for supervised IMU.")
    return trainloader, valloader, testloader

    
def get_tdost_data_loaders(args):
    print("Getting TDOST Data Loaders")
    train_dataset = HARDataset(directory=args.embeddings_dir, sentence_encoder_name=args.sentence_encoder, phase="train")
    val_dataset = HARDataset(directory=args.embeddings_dir, sentence_encoder_name=args.sentence_encoder, phase="val")
    test_dataset = HARDataset(directory=args.embeddings_dir, sentence_encoder_name=args.sentence_encoder, phase="test")
    print("loaded")
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True
    )
    #print(f"Number of samples in the training dataset: {len(trainloader.dataset)}")
    
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False, drop_last=True
    )
    print("✅ Successfully loaded data for TDOST.")
    return trainloader, valloader, testloader


