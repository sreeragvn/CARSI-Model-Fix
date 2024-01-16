"""
Script Name:    01_data_processing.py

Description:    Process windows data and split train and test set.

Feature:        Normalization, integer encoder, class wigths, data split, add last interaction.

Author:         Jeremy Wanner
"""
import pandas as pd
import numpy as np
import torch
import os
import pickle
import sys

continuous_var = [
    "time_second", 
    "temperature_out", 
    "steering_speed", 
    "avg_irradiation",
    "KBI_speed", 
    "soc", 
    "latitude", "longitude", "altitude", 
    "rain_sensor",
]

static_var = [
    "weekday", # 0-7
    "hour", # 0-23
    "month", #0-11
    "kickdown",
    "seatbelt_codriver" # 0-1
]


script_dir = os.path.dirname(os.path.abspath(__file__))

class DataProcessing():
    def __init__(self, data_name) -> None:
        self.data = pd.read_csv(f"{script_dir}/../Processed_data_new/Datasets/{data_name}")
        
        # 1) Normalize continuous var
        self.normalize_continuous()

        # len of different categories (not needed here)
        self.label_mapping = None
        self.len_unique_cat_dic = {}

        # 2) Convert cat var to numeric
        self.num_embedding = 0
        self.get_num_embeddings()
        self.embeddings_to_numeric()

        # 3) Map labels to numeric
        self.cat_to_numerical_ordered(["Label"])

        #self.split_train_test()
        self.class_weights = None
        self.compute_class_weights()
        self.split_train_test()


    def normalize_continuous(self):
        """ Min-Max Scaling: normalize in range 0-1 """
        for col in continuous_var:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data[col] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min()) 
    
    def cat_to_numerical_ordered(self, column_list):
        """ Convert categorical var to numbers from 0 to lengh of unique values """
        for col in column_list:
            unique_categories = list(set(self.data[col]))
            category_to_number = {category: index for index, category in enumerate(unique_categories)}
            numerical_numerical = [category_to_number[value] for value in self.data[col]]
            self.data[col] = numerical_numerical
            self.len_unique_cat_dic[col] = len(unique_categories)
            if col == "Label":
                self.label_mapping = category_to_number
    
    def get_num_embeddings(self):
        for col in static_var:
            # Convert to distinguishable strings
            self.data[col] = col + self.data[col].astype(str)
            self.num_embedding += len(self.data[col].unique())
    
    def embeddings_to_numeric(self):
        # Create numeric mapping over var and cols
        col_mapping = {}
        cc = 0
        for col in static_var:
            unique_values = self.data[col].unique()
            mapping = {value: idx+cc for idx, value in enumerate(unique_values)}
            cc += len(unique_values)
            col_mapping[col] = mapping
        # Map string to numbers
        for col in static_var:
            self.data[col] = self.data[col].map(col_mapping[col]).astype(int)
    
    def split_train_test(self):
        # Split window_id
        train_percentage = 0.7  
        window_id_unique = self.data["window_id"].unique()
        num_windows_train = int(len(window_id_unique) * train_percentage)
        shuffled_windows = np.random.permutation(window_id_unique)
        window_id_train = shuffled_windows[:num_windows_train]        
        window_id_test = shuffled_windows[num_windows_train:]        
        mask_train = self.data["window_id"].isin(window_id_train)

        
        #self.window_id_loader = window_id_train
        self.data_train = self.data[mask_train].reset_index(drop=True)
    
        #self.window_id_loader = window_id_test
        self.data_test = self.data[~mask_train].reset_index(drop=True)

    def compute_class_weights(self):
        labels = torch.tensor(self.data.Label)
        class_count = torch.bincount(labels)
        total_samples = len(labels)
        self.class_weights = 1.0 / (class_count.float() / total_samples)
        sum_weights = torch.sum(self.class_weights)
        self.class_weights = self.class_weights / sum_weights


if __name__ == "__main__":

    vehicle_names = ["SEB880", "SEB882", "SEB883", "SEB885", "SEB888", "SEB889"]
    data_dic = {}
    weather_dic = {}
    for vehicle in vehicle_names:
        data_dic[vehicle] = pd.read_csv(f"{script_dir}/../Processed_data_new/Datasets/dataset_windows_{vehicle}.csv")
        weather_dic[vehicle] = pd.read_csv(f"{script_dir}/../Processed_data_new/Datasets/weather_{vehicle}.csv")

    DATA_TO_PROCESS = "dataset_event_windows_no_nan_short.csv"

    processing = DataProcessing(DATA_TO_PROCESS) 

    label_mapping = processing.label_mapping
    class_weights = processing.class_weights
    num_embedding = processing.num_embedding

    data_train = processing.data_train
    data_test = processing.data_test

    sequence_length = len(data_test) // len(data_test.window_id.unique())

    with open('Dataset/param.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)
        pickle.dump(class_weights, f)
        pickle.dump(num_embedding, f)
        pickle.dump(continuous_var, f)
        pickle.dump(static_var, f)
        pickle.dump(sequence_length, f)

    data_train.to_csv("Dataset/data_train.csv", index=False)
    data_test.to_csv("Dataset/data_test.csv", index=False)

