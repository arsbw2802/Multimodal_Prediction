import argparse
import os
from data_load import get_supervised_imu_data_loaders
from data_load import get_tdost_data_loaders

def parse_arguments():
    parser = argparse.ArgumentParser(description='arguments for main application')
    parser.add_argument("--imu_joblib_file", type=str, default="./all_data/MARBLE_IMU.joblib", help="Path to the joblib file to use for the supervised IMU model")
    parser.add_argument("--embeddings_dir", type=str, default="./all_data/", help="Directory where the embeddings for TDOST are located.")
    parser.add_argument("--sentence_encoder", type=str, default="all-MiniLM-L12-v2", help="Name of the sentence encoder to be used for TDOST.")
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size for model training.")
    args = parser.parse_args()

    return args

def app(args):
    imu_trainloader, imu_valloader, imu_testloader = get_supervised_imu_data_loaders(args)
    tdost_trainloader, tdost_valloader, tdost_testloader = get_tdost_data_loaders(args)

    # TODO: This is where we load the data into the models and train

if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    if not os.path.exists(args.imu_joblib_file):
        raise FileNotFoundError(f"The MARBLE joblib file '{args.imu_joblib_file}' does not exist. Make sure you have done the preprocessing and processing steps first.")
    if not os.path.exists(args.embeddings_dir):
        raise FileNotFoundError(f"The directory '{args.embeddings_dir}' does not exist.")
    
    app(args)
