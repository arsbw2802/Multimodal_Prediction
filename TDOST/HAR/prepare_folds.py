
import numpy as np
from collections import Counter
from utils import *
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split


def parse_arguments():

    parser = argparse.ArgumentParser(description='arguments for smart home dataset preprocessing')
    parser.add_argument('--datasets', type=str, nargs='+', default=['milan', 'aruba', 'cairo','kyoto7'], help='name(s) of the dataset(s)')
    parser.add_argument('--embedding_type', type=str, default="v1", help='TDOST type')
    parser.add_argument('--sentence_encoder_name', type=str, default="sentence-t5-base", help='Sentence Encoder Name')
    parser.add_argument('--shuffle', type=str, default="True", help='Whether to shuffle or not')
    parser.add_argument("--folds_path", type=str, default="/coc/pcba1/mthukral3/gt/TDOST/folds/pre-segmented", help='Directory path to save folds' )
    parser.add_argument("--npy_path",  type=str, default="/coc/pcba1/mthukral3/gt/TDOST/npy/pre-segmented", help='Directory path to load npy' )

    

    args = parser.parse_args()

    return args


def prepare_folds(args, dataset, embedding_type):
    
    
    if not os.path.exists(args.folds_path):
        os.makedirs(args.folds_path)
   
    print(args.sentence_encoder_name)
    
    emb = open_raw_data_file(os.path.join(args.npy_path,  "{}_embeddings_{}_{}.npy".format(dataset, embedding_type, args.sentence_encoder_name)))
    

    labels = open_raw_data_file(os.path.join(args.npy_path, "{}-global_labels.npy".format(dataset)))
    
    
    print("Before label encoding for Dataset:\t" + str(dataset))
    print(Counter(labels))
   
    labels = encode_labels(labels)
    
    if args.shuffle=="True":
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    else:
        skf = KFold(n_splits=3, shuffle=False)
    
    val_split=0.2
    
    print("After label encoding for Dataset:\t" + str(dataset))
    print(Counter(labels))
    
  
    for fold, (train_index, test_index) in enumerate(skf.split(emb, labels)):
        
        if args.shuffle=="True":
            
            train_index, val_index = train_test_split(train_index, test_size=val_split, stratify=labels[train_index], random_state=args.seed, shuffle=True)
        else:
            train_index, val_index = train_test_split(train_index, test_size=val_split,  random_state=args.seed, shuffle=False)
            
        test_labels = labels[test_index]
        test_v1 = emb[test_index]
        
        train_labels = labels[train_index]
        train_v1 = emb[train_index]
        
        val_labels = labels[val_index]
        val_v1 = emb[val_index]
        
        # print(Counter(test_labels), Counter(train_labels), Counter(val_labels))
        
        # print(fold, train_index, test_index, val_index)
        
        # print(train_labels.shape, val_labels.shape, test_labels.shape)
        # print(train_v1.shape, val_v1.shape, test_v1.shape)
        

        np.save(os.path.join(args.folds_path,    '{}-fold_{}_test_labels_{}_{}.npy'.format(dataset, fold +1 , embedding_type, args.sentence_encoder_name)), test_labels)
        np.save(os.path.join(args.folds_path,    '{}-fold_{}_test_{}_{}.npy'.format(dataset, fold +1 , embedding_type, args.sentence_encoder_name)), test_v1)
        
        np.save(os.path.join(args.folds_path,    '{}-fold_{}_train_labels_{}_{}.npy'.format(dataset, fold +1 , embedding_type, args.sentence_encoder_name)), train_labels)
        np.save(os.path.join(args.folds_path,    '{}-fold_{}_train_{}_{}.npy'.format(dataset, fold +1 , embedding_type, args.sentence_encoder_name)), train_v1)
        
        np.save(os.path.join(args.folds_path,    '{}-fold_{}_val_labels_{}_{}.npy'.format(dataset, fold +1 , embedding_type, args.sentence_encoder_name)), val_labels)
        np.save(os.path.join(args.folds_path,    '{}-fold_{}_val_{}_{}.npy'.format(dataset, fold +1 , embedding_type, args.sentence_encoder_name)), val_v1)
            
   



def  encode_labels(labels):
    label_encoder = LabelEncoder()

    labels = label_encoder.fit_transform(labels)
    label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
    
    print(label_mapping)
    
    return labels
    
  
def discover_datasets_from_npy_files(npy_path:str):
    files_in_npy = os.listdir(npy_path)
    discovered_datasets = set()
    discard_keywords = {"embeddings", "v1", "sentence"}
    
    for file in files_in_npy: #loop 1
        file_desc = file.split("-")
        data_name = file_desc[0]

        name_decomp = data_name.split("_")

        skip_outer_loop = False

        for item in name_decomp:  # loop 2
            if item in discard_keywords:
                skip_outer_loop = True  
                break  

        if skip_outer_loop:
            continue  
        else:
            discovered_datasets.add(data_name)
        # discovered_datasets.add()
    discovered_datasets_list = list(discovered_datasets)
    print("Discovered these datasets from ./npy/:", discovered_datasets_list)
    return discovered_datasets_list
    
    


# Main.
if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    embedding_type = args.embedding_type
    args.seed = 7 
    np.random.seed(args.seed)
    datasets = args.datasets

    
    for dataset in datasets:
        prepare_folds(args,dataset, embedding_type)


