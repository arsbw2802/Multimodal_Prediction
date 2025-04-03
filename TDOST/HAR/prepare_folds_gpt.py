
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
    parser.add_argument('--embedding_type', type=str, default="gpt_v1", help='TDOST type')
    parser.add_argument('--sentence_encoder_name', type=str, default="sentence-t5-base", help='Sentence Encoder Name')
    parser.add_argument('--shuffle', type=str, default="True", help='Whether to shuffle or not')
    parser.add_argument("--folds_path", type=str, default="/coc/pcba1/mthukral3/gt/TDOST/folds/pre-segmented", help='Directory path to save folds' )
    parser.add_argument("--npy_path",  type=str, default="/coc/pcba1/mthukral3/gt/TDOST/npy/pre-segmented", help='Directory path to save folds' )

    args = parser.parse_args()

    return args


def prepare_folds(args,dataset, embedding_type):

        
    if not os.path.exists(args.folds_path):
        os.makedirs(args.folds_path)


    emb = open_raw_data_file(os.path.join(args.npy_path,  "{}_embeddings_{}_{}.npy".format(dataset, embedding_type, args.sentence_encoder_name)))

    emb = emb.reshape(3, -1, emb.shape[1], emb.shape[2])

    
    labels = open_raw_data_file(os.path.join(args.npy_path, "{}-global_labels.npy".format(dataset)))
    
    labels  = encode_labels(labels)


    
    print("Original Dataset:\t" + str(dataset))
    print(Counter(labels))

    labels  = encode_labels(labels)
    
    if args.shuffle=="True":
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    else:
        skf = KFold(n_splits=3, shuffle=False)
    
    
    val_split=0.2
    
    print("After label encoding for Dataset:\t" + str(dataset))
    print(Counter(labels))
    

    emb = emb.transpose(1,0,2,3)
   
  
    for fold, (train_index, test_index) in enumerate(skf.split(emb, labels)):
        
        if args.shuffle=="True":
            
            train_index, val_index = train_test_split(train_index, test_size=val_split, stratify=labels[train_index], random_state=args.seed, shuffle=True)
        else:
            train_index, val_index = train_test_split(train_index, test_size=val_split,  random_state=args.seed, shuffle=False)
             
        test_labels = labels[test_index]
        test_v1 = emb[test_index]
        test_v1 =test_v1.reshape(-1, test_v1.shape[2], test_v1.shape[3])
        test_labels = np.repeat(test_labels,3)
        
        train_labels = labels[train_index]
        train_v1 = emb[train_index]
        train_v1 =train_v1.reshape(-1, train_v1.shape[2], train_v1.shape[3])
        train_labels = np.repeat(train_labels,3)
        
        val_labels = labels[val_index]
        val_v1 = emb[val_index]
        val_v1 =val_v1.reshape(-1, val_v1.shape[2], val_v1.shape[3])
        val_labels = np.repeat(val_labels,3)
        
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
    
    


# Main.
if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    embedding_type = args.embedding_type
    args.seed = 7
    np.random.seed(args.seed)
    
    for dataset in args.datasets:
        print(dataset)
        prepare_folds(args, dataset, embedding_type)


