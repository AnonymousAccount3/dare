import os
import time
import yaml
import pandas as pd
from train import train
from train import base_dict, dataset_dict, CLASSIFICATION_DATASET

dataset_dict = [
"2d_classif",
"1d_reg",
# "citycam_weather",
# "citycam_cameras",
# "citycam_bigbus",
# "cifar10",
# "cifar10_resnet",
# "fashionmnist"
]

if __name__ == "__main__":
    
    config_path = "configs/"
    config_files = [f for f in os.listdir(config_path) if os.path.isfile(os.path.join(config_path, f))]
    
    print(config_files)
    
    for state in range(5):
        for dataset in dataset_dict:
            
            if dataset == "cifar10":
                methods = ["DeepEnsemble"]
            else:
                methods = ["DeepEnsemble", "DeepEnsembleMSE", "DARE",
                "AnchoredNetwork", "MOD", "RDE", "NegativeCorrelation"]
                
            for method in methods:
                
                print(dataset)
                config = {}
                config["dataset"] = dataset
                config["method"] = method
                config["save_path"] = "results/%s/"%dataset
                config["state"] = state

                file = open("configs/%s.yml"%dataset, "r")
                config.update(yaml.safe_load(file))
                file.close()

                if dataset in CLASSIFICATION_DATASET:
                    if method + "Classif.yml" in config_files:
                        file = open("configs/%sClassif.yml"%method, "r")
                        config.update(yaml.safe_load(file))
                        file.close()
                    elif method + ".yml" in config_files:
                        file = open("configs/%s.yml"%method, "r")
                        config.update(yaml.safe_load(file))
                        file.close()
                else:
                    if method + ".yml" in config_files:
                        file = open("configs/%s.yml"%method, "r")
                        config.update(yaml.safe_load(file))
                        file.close()

                if method == "DARE":
                    threshold = config["callbacks_params"]["threshold"]
                    logs_path = os.path.join("logs", config["save_path"], threshold + "_%i.csv"%state)
                    df = pd.read_csv(logs_path, index_col=0)
                    threshold = float(df.loc["val_loss"])
                    print("Threshold:", threshold)
                    config["callbacks_params"]["threshold"] = threshold*1.25
                    config["params"]["threshold"] = threshold*1.25
                
                #config["n_jobs"] = 5
                
                train(config)