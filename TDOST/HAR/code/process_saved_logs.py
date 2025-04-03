import os
import meter
import joblib
import numpy as np
import pandas as pd

def process(folder_path, embedding_type):
    perf={'cairo':{
    
    "all-distilroberta-v1":{"f1":0.0, "acc":0.0, "best_params":{}},
    "sentence-t5-base":{ "acc":0.0, "f1":0.0, "f1_m":0.0, "best_params":{}},
    },
      
    'aruba':{
    "all-distilroberta-v1":{"f1":0.0, "acc":0.0, "best_params":{}},
    "sentence-t5-base":{"f1":0.0, "acc":0.0,  "f1_m":0.0, "best_params":{}},
    },
      
    'milan':{
    "all-distilroberta-v1":{"f1":0.0, "acc":0.0, "best_params":{}},
    "sentence-t5-base":{"f1":0.0, "acc":0.0,  "f1_m":0.0, "best_params":{}},
    },
    'kyoto7':{
    "all-distilroberta-v1":{"f1":0.0, "acc":0.0, "best_params":{}},
    "sentence-t5-base":{"f1":0.0, "acc":0.0,  "f1_m":0.0, "best_params":{}},
    }
    }
        

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        parts = file_name.split("_")
        info_map = {}
        
        if "gpt" in parts:
            info_map['dataset']= parts[0]
            info_map['emb'] =  parts[2] + "_" +parts[3]
            info_map['fold'] =  parts[5]
            info_map['lr'] = parts[7]
            info_map['wd'] = parts[9]
            info_map['batch_size'] = parts[11]
            info_map['encoder_name'] = parts[12]
            
        else:
            

            info_map['dataset']= parts[0]
            info_map['emb'] =  parts[2] 
            info_map['fold'] =  parts[4]
            info_map['lr'] = parts[6]
            info_map['wd'] = parts[8]
            info_map['batch_size'] = parts[10]
            info_map['encoder_name'] = parts[11]
        
        if info_map["wd"]=='0':
            info_map["wd"]='0.0'
        # print(info_map)
        if  info_map['fold']=="1" and info_map["emb"]==embedding_type:
            dataset = info_map['dataset']
            encoder_name = info_map['encoder_name']
            # print(info_map)
            print(dataset)
      
            
            data = joblib.load(file_path)
            f1=  max(data.f1_score_weighted["val"])
            acc=  max(data.accuracy["val"])
            f1_m = max(data.f1_score["val"])
            if np.isnan(f1):
                print(f1)
                
            
            if perf[dataset][encoder_name]["f1"] < f1:
                perf[dataset][encoder_name]['f1']= f1
                perf[dataset][encoder_name]['acc']= acc
                perf[dataset][encoder_name]['f1_m']= f1_m
                perf[dataset][encoder_name]['best_params']=info_map
                
    # print(perf) 
    # joblib.dump(perf, f"final_perf_files_t5/{embedding_type}_best_params.joblib")
            
    dict_to_save = {}       
    datasets =    [ 'milan', 'cairo',  'aruba', 'kyoto7']  

    for dataset in datasets:
        # for sentence_encoder_name in ["all-MiniLM-L6-v2", "all-distilroberta-v1"]:
        for sentence_encoder_name in ["all-distilroberta-v1"]:
            best_params = perf[dataset][sentence_encoder_name]["best_params"]
            
            # print(best_params)
            for fold_a in [1,2,3]:
               
                file_name = "{dataset}_emb_{emb}_fold_{fold}_lr_{lr}_wd_{wd}_bs_{batch_size}_{encoder_name}_pretrained_eval_log.pkl".format(**best_params)
                file_name = file_name.replace("fold_1", f"fold_{fold_a}")

                file_path = os.path.join(folder_path, file_name)
                data = joblib.load(file_path)
                key=f"{dataset}_{sentence_encoder_name}_fold_{fold_a}"
                dict_to_save[key]={ "acc":0.0, "f1":0.0, "f1_m":0.0}
                
                epoch = np.argmax(data.f1_score_weighted['val'])
             
                dict_to_save[key]["f1"] = data.f1_score_weighted["test"][epoch]
                dict_to_save[key]["acc"] = data.accuracy["test"][epoch]
                dict_to_save[key]["f1_m"] = data.f1_score["test"][epoch]
                dict_to_save[key]["recall"] = data.recall_weighted["test"][epoch]
                dict_to_save[key]["precision"] = data.precision_weighted["test"][epoch]
                
      
                
    print(dict_to_save)

    joblib.dump(dict_to_save, f"final_perf_files_t5/{embedding_type}_perf.joblib")

    #Re-initialize dictionaries to collect results for each dataset
    results_emb_v1 = {dataset: {'acc': [], 'f1': [], 'recall': [], 'precision': []} for dataset in datasets}

    # Parse the data for 'emb v1'
    for key, metrics in dict_to_save.items():
        for dataset in datasets:
            if dataset in key:
                for metric in ['acc', 'f1', 'recall', 'precision']:
                    results_emb_v1[dataset][metric].append(metrics[metric])

    # Create a DataFrame with averages for 'emb v1'
    averages_emb_v1 = {}
    for dataset, metrics in results_emb_v1.items():
        averages_emb_v1[dataset] = {metric: f"$ {round(sum(values)/len(values)*100,2)} \pm {round(pd.Series(values).std()*100,2)} $" for metric, values in metrics.items()}

    averages_emb_v1_df = pd.DataFrame(averages_emb_v1).T

    print(f"average perf {averages_emb_v1_df.to_latex()}")
            

if __name__=="__main__":
    folder_path = "/coc/pcba1/mthukral3/gt/TDOST/HAR/code/saved_logs/new_distillroberta"
    embedding_type="gpt_v1-1"
    # for embedding_type in ["gpt_v1-1", "v1-1", "v1", "gpt_v1"]:
    print("emb", embedding_type)
    process(folder_path,embedding_type)

