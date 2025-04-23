import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from datetime import date
from joblib import dump
from sklearn.preprocessing import StandardScaler
import copy
from sentence_transformers import SentenceTransformer
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='arguments for MARBLE dataset preprocessing')
    parser.add_argument("--synced_marble_data_path", type=str, default="./all_data/synced_marble_data.csv", help="Location of the synced MARBLE dataset")
    parser.add_argument('--save_path', type=str, default="./all_data", help="Location to save processed data")
    parser.add_argument('--context_length', type=int, default=12, help="Maximum length of context for TDOST")
    parser.add_argument('--embedding_dimension', type=int, default=384, help="Embedding dimension for TDOST")
    parser.add_argument('--sentence_encoder', type=str, default='all-MiniLM-L12-v2', help="Name of sentence encoder to use for TDOST")
    parser.add_argument('--log', action='store_true', help='Logging for debugging purposes')

    args = parser.parse_args()

    return args

def log(message, b):
    if b:
        print(message)

def map_activity_to_id():
    # List of activities being studied. Note that we *dont* use lying down class
    activity_list = [
        "EATING",
        "TAKING_MEDICINES",
        "WATCHING_TV",
        "COOKING",
        "MAKING_PHONE_CALL",
        "WASHING_DISHES",
        "ENTERING_HOME",
        "ANSWERING_PHONE",
        "LEAVING_HOME",
        "USING_PC",
        "SETTING_UP_TABLE",
        "TRANSITION",
        "CLEARING_TABLE",
        "PREPARING_COLD_MEAL"
        ]

    activity_id = {
        "EATING": 0,
        "TAKING_MEDICINES": 1,
        "WATCHING_TV": 2,
        "COOKING": 3,
        "MAKING_PHONE_CALL": 4,
        "WASHING_DISHES": 5,
        "ENTERING_HOME": 6,
        "ANSWERING_PHONE": 7,
        "LEAVING_HOME": 8,
        "USING_PC": 9,
        "SETTING_UP_TABLE": 10,
        "TRANSITION": 11,
        "CLEARING_TABLE" :12,
        "PREPARING_COLD_MEAL": 13
    }
    
    return activity_id, activity_list


def perform_train_val_test_split(args, test_size=0.2, val_size=0.2):
    df = pd.read_csv(args.synced_marble_data_path, low_memory=False)
    unique_subj = np.unique(df['subject_id'].values)

    # Doing the train-test split
    train_val_subj, test_subj = train_test_split(unique_subj,
                                                 test_size=test_size,
                                                 random_state=42)
    print('The train and validation subjects are: {}'.format(train_val_subj))
    print('The test subjects are: {}'.format(test_subj))

    # Splitting further into train and validation subjects
    train_subj, val_subj = train_test_split(train_val_subj, test_size=val_size,
                                            random_state=42)

    subjects = {'train': train_subj, 'val': val_subj, 'test': test_subj}

    return subjects


def process_imu_data(args, split):
    # Create new dataframe with just IMU data
    data = pd.read_csv(args.synced_marble_data_path, low_memory=False)
    activity_id, activity_list = map_activity_to_id()
    df = pd.DataFrame()
    df['acc_x'] = data['accelerometer_x']
    df['acc_y'] = data['accelerometer_y']
    df['acc_z'] = data['accelerometer_z']
    df['text_labels'] = data['labels']
    df['gt'] = data['labels'].map(activity_id)
    df['user'] = data['subject_id']
    df['demographic_info'] = 'todo'

    activity_text_id , _ = map_activity_to_id()
    # Let us partition by train, val and test splits
    train_data = df[df['user'].isin(split['train'])]
    val_data = df[df['user'].isin(split['val'])]
    test_data = df[df['user'].isin(split['test'])]
    log(f'The shapes of the splits are: {train_data.shape}, {val_data.shape} and {test_data.shape}', args.log)

    log(f"The unique classes in train are: {np.unique(train_data['gt'])}", args.log)
    log(f"The unique classes in val are: {np.unique(val_data['gt'])}", args.log)
    log(f"The unique classes in test are: {np.unique(test_data['gt'])}", args.log)

    sensors = ['acc_x', 'acc_y', 'acc_z']
    
    processed = {'train': {'data': train_data[sensors].values,
                           'labels': train_data['gt'].values,
                           'demographic_info': train_data['demographic_info'].values
                           },
                 'val': {'data': val_data[sensors].values,
                         'labels': val_data['gt'].values,
                         'demographic_info': val_data['demographic_info'].values
                         },
                 'test': {'data': test_data[sensors].values,
                          'labels': test_data['gt'].values,
                          'demographic_info': test_data['demographic_info'].values
                          },
                 'fold': split,
                 'activity_text_id': activity_text_id,
                 
                 }

    # Sanity check on the sizes
    for phase in ['train', 'val', 'test']:
        assert processed[phase]['data'].shape[0] == \
               len(processed[phase]['labels'])

    for phase in ['train', 'val', 'test']:
        log(f"The phase is: {phase}. The data shape is: {processed[phase]['data'].shape}, {processed[phase]['labels'].shape}", args.log)

    # Before normalization
    log('Means before normalization', args.log)
    log(np.mean(processed['train']['data'], axis=0), args.log)

    os.makedirs(os.path.join(args.save_path, 'unnormalized'), exist_ok=True)
    save_name = "MARBLE_IMU"


    # Saving the joblib file
    save_name_copy = copy.deepcopy(save_name)
    #save_name += '.joblib'
    name = os.path.join(args.save_path, 'unnormalized', save_name)
    with open(name, 'wb') as f:
        dump(processed, f)

    # Performing normalization
    scaler = StandardScaler()
    scaler.fit(processed['train']['data'])
    for phase in ['train', 'val', 'test']:
        processed[phase]['data'] = \
            scaler.transform(processed[phase]['data'])

    # After normalization
    log('Means after normalization', args.log)
    log(np.mean(processed['train']['data'], axis=0), args.log)

    # Saving into a joblib file
    name = os.path.join(args.save_path, save_name + '.joblib')
    with open(name, 'wb') as f:
        dump(processed, f)

    # Saving the scaler
    name = os.path.join(args.save_path, save_name_copy + '_scaler.joblib')
    with open(name, 'wb') as f:
        dump(scaler, f)

    print(f"✅ joblib file(s) for IMU successfully saved.")


# For TDOST
def generate_embeddings(args, dataset, model, phase, emb_dict):
    emb = np.zeros((dataset['labels'].count(), args.context_length, args.embedding_dimension), dtype=np.float16)

    for i in tqdm(range(dataset['labels'].count()), desc=f"Encoding {phase} dataset"):
        sentences = dataset["sensor_status_summary"].iloc[i].split(".")
        for j in range(min(args.context_length, len(sentences))):
            final_sentence = sentences[j].strip()
            if not final_sentence:
                final_sentence = ""

            if final_sentence in emb_dict:
                emb[i, args.context_length - 1 - j, :] = emb_dict[final_sentence]
            else:
                sentence_embeddings = model.encode(final_sentence)
                emb[i, args.context_length - 1 - j, :] = sentence_embeddings
                emb_dict[final_sentence] = sentence_embeddings

    np.save(os.path.join(args.save_path, f"MARBLE_{phase}_embeddings_v1_{args.sentence_encoder}.npy"), emb)
    print(f"✅ File for {phase} embeddings successfully saved.")


def create_tdost_label_files(args, df, phase):
    activity_id, _ = map_activity_to_id()
    df['label_id'] = df['labels'].map(activity_id)
    label_array = df['label_id'].to_numpy()
    np.save(os.path.join(args.save_path,f"MARBLE_{phase}_labels"), label_array)
    print(f"✅ File for {phase} labels successfully saved.")
 

def process_env_data(args, split):
    df = pd.read_csv(args.synced_marble_data_path, low_memory=False)
    model = SentenceTransformer(args.sentence_encoder)
    emb_dict = {}

    train_dataset = df[df["subject_id"].isin(split["train"])]
    create_tdost_label_files(args, train_dataset, "train")
    generate_embeddings(args, train_dataset, model, "train", emb_dict)
    del train_dataset

    val_dataset = df[df["subject_id"].isin(split["val"])]
    create_tdost_label_files(args, val_dataset, "val")
    generate_embeddings(args, val_dataset, model, "val", emb_dict)
    del val_dataset

    test_dataset = df[df["subject_id"].isin(split["test"])]
    create_tdost_label_files(args, test_dataset, "test")
    generate_embeddings(args, test_dataset, model, "test", emb_dict)
    del test_dataset


if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    if not os.path.exists(args.synced_marble_data_path):
        raise FileNotFoundError(f"The synced MARBLE data path '{args.synced_marble_data_path}' does not exist. Make sure you have done the preprocessing step first.")
    if not os.path.exists(args.save_path):
        raise FileNotFoundError(f"The save path '{args.save_path}' does not exist. Please create the directory first.")

    split = perform_train_val_test_split(args)
    process_imu_data(args, split)
    process_env_data(args, split)
    print("✅ Data processing complete.")
    


