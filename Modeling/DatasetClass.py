
import os
import sys 
import torch 
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle
    
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

class DataSetWindows(Dataset):
    
    def __init__(self, split='train', weather=False, car_id="all"):

        # Load data 
        data_path = f"Dataset/data_all/{split}.csv"
        self.data = pd.read_csv(data_path)
        if car_id != "all":
            self.data = self.data[self.data.car_id == car_id]

        # leave out SEB885
        # car_ids = ["SEB880", "SEB882", "SEB883", "SEB888", "SEB889"]
        # leave out SEB883
        #car_ids = ["SEB880", "SEB882", "SEB885", "SEB888", "SEB889"]
        # leave out SEB888
        #car_ids = ["SEB880", "SEB882", "SEB885", "SEB883", "SEB889"]
        # leave out SEB889
        # car_ids = ["SEB880", "SEB882", "SEB885", "SEB883", "SEB888"]
        # leave out SEB880
        # car_ids = ["SEB889", "SEB882", "SEB885", "SEB883", "SEB888"]
        #self.data[self.data['car_id'].isin(car_ids)]

        self.load_params()

        # train: 4613, test: 1978
        self.windows_id_list = self.data.window_id.unique()

        # if weather == False:
        #     elements_to_remove = ["precipitation", "wind_speed", "condition", "cloud_cover"]
        #     self.static_var = [item for item in self.static_var if item not in elements_to_remove]

    def load_params(self):
        with open('Dataset/data_all/param.pkl', 'rb') as f:
            self.label_mapping = pickle.load(f)
            self.class_weights = pickle.load(f)
            self.num_embedding = pickle.load(f)
            self.continuous_var = pickle.load(f)
            self.static_var = pickle.load(f)
            self.sequence_length = pickle.load(f)

    def __len__(self):
        return len(self.windows_id_list)
    
    def __getitem__(self, idx):
                
        window = self.data[self.data.window_id == self.windows_id_list[idx]]

        # Inputs 
        inputs_cont = torch.tensor(window[self.continuous_var].values, dtype=torch.float32) # shape (winwow_size, n_feature)
        inputs_static = torch.tensor(window[self.static_var].values, dtype=torch.long) 
        
        # For static varibales keep only 
        inputs_static = inputs_static.mode(dim=0).values

        # # Replace nan for extended windows with cols mean
        # if torch.isnan(inputs_cont).any():
        #     print("Nan found!")
        #     column_means = torch.nanmean(inputs_cont, dim=0)
        #     for i in range(inputs_cont.shape[1]):
        #         column = inputs_cont[:, i]
        #         column[column.isnan()] = column_means[i]

        label = torch.tensor(window["Label"].iloc[0])
           
        return (inputs_cont, inputs_static), label

if __name__ == "__main__":
    
    data = DataSetWindows()

    sample = next(iter(data))

    inputs, labels = sample 
    
    print(inputs)