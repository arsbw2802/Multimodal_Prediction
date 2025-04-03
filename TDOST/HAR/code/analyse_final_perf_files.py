import os
import joblib
import numpy as np
import math

folder = "/mnt/attached1/TDOST/HAR/code/final_perf_files"

for embedding_type in ['v1', 'v1-1', 'gpt_v1', 'gpt_v1-1']:
    print(embedding_type)
    for dataset in ['kyoto7']:


        f1=[]
        acc=[]
        print(dataset)
        for file_name in os.listdir(folder):
            
            if 'perf' in file_name :
                
                parts = file_name.split("_")
                if len(parts)==2:
                    emb = parts[0]
                else:
                    emb = parts[0] + "_" +parts[1]
                    
                if emb !=embedding_type:
                    continue
                
                
                data = joblib.load(os.path.join(folder, file_name))
                
                for key in data.keys():
                    if dataset in key:
                        f1.append(data[key]["f1"])
                        acc.append(data[key]["acc"])
        print(f1)                
        print(f"Acc: ${np.mean(acc)*100:.2f} \pm {np.std(acc)*100:.2f}$")             
        print(f"F1: ${np.mean(f1)*100:.2f} \pm {np.std(f1)*100:.2f}$")

                