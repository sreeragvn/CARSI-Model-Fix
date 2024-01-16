"""
Script Name:    01_data_processing.py

Description:    Process windows data and split train and test set.

Feature:        Normalization, integer encoder, class wigths, data split.

Author:         Jeremy Wanner
"""
import pandas as pd
import numpy as np
import torch
import os
import pickle

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
    "weekday",
    "hour", 
    "month", 
    "kickdown",
    "seatbelt_codriver",
    "seatbelt_rear_r",
    "seatbelt_rear_l",
    "seatbelt_rear_m",
    "last_interaction"
]


script_dir = os.path.dirname(os.path.abspath(__file__))

class DataProcessing():
    def __init__(self, data, weather) -> None:

        seed_value = 42 
        np.random.seed(seed_value)
        
        self.data = data
        self.weather = weather

        self.process_weather()

        self.window_length = len(self.data) // len(self.data.window_id.unique())
        self.merge_data_weather()

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

        # 1) Normalize continuous var
        self.normalize_continuous()

    def merge_data_weather(self):
        self.weather = self.weather.drop(columns=["window_id"])
        self.weather = self.weather.loc[self.weather.index.repeat(self.window_length)]
        self.weather.reset_index(drop=True, inplace=True)      
        self.data = pd.concat([self.data, self.weather], axis=1)

    def process_weather(self):
        # Map condition to values 
        condition_mapping = {'dry': 0,'fog': 1,'rain': 2,'sleet': 3,'snow': 
                             4,'hail': 5,'thunderstorm': 6,'null': 0}
        self.weather['condition'] = self.weather['condition'].map(condition_mapping)
        # Inpute data 
        self.weather.fillna(0, inplace=True)
        # bin
        columns_to_bin = ['wind_speed', 'precipitation', 'cloud_cover']
        for col in columns_to_bin:
            max_value = self.weather[col].max()
            bins = [-1, max_value / 5, 2 * max_value / 5, 3 * max_value / 5, 4 * max_value / 5, max_value]
            labels = [0, 1, 2, 3, 4]  # Assign numeric values to the bins (you can customize these values)
            self.weather[col] = pd.cut(self.weather[col], bins=bins, labels=labels)

    # def normalize_continuous(self):
    #     """ Min-Max Scaling: normalize in range 0-1 """
    #     for col in continuous_var:
    #         self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
    #         self.data[col] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min()) 
    def normalize_continuous(self):
        """ Min-Max Scaling: normalize in range 0-1 using training data min-max """
        min_max_values = {col: (self.data_train[col].min(), self.data_train[col].max()) for col in continuous_var}
        
        for col in continuous_var:
            min_val, max_val = min_max_values[col]
            self.data_train[col] = (self.data_train[col] - min_val) / (max_val - min_val)
            self.data_test[col] = (self.data_test[col] - min_val) / (max_val - min_val)
    
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
    data = pd.DataFrame()
    weather = pd.DataFrame()
    for vehicle in vehicle_names:
        data_curr = pd.read_csv(f"{script_dir}/../Processed_data_new/Datasets/dataset_windows_{vehicle}.csv")
        data_curr["car_id"] = vehicle
        data = pd.concat([data, data_curr])
        weather = pd.concat([weather, pd.read_csv(f"{script_dir}/../Processed_data_new/Datasets/weather_{vehicle}.csv")])
    weather.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    for w_id in data.window_id.unique():
        id_to_fill = data.index[data['window_id'] == w_id].tolist()
        if w_id == 0:
            #print(data[data['window_id'] == w_id])
            id_last_int = 10 # is this assumption ok ? 
        else:
            previous_window_indices = data.index[data['window_id'] == w_id - 1].tolist()
            if previous_window_indices:
                id_last_int = previous_window_indices[0]
            else:
                print('problem') # is this assumption ok ? why didnt we have this issue before we changed the way a window is defined
                id_last_int = data.index[data['window_id'] == w_id - 2].tolist()[0]  
    
        data.loc[id_to_fill, "last_interaction"] = data.loc[id_last_int, "Label"]
 
 
    processing = DataProcessing(data, weather) 

    label_mapping = processing.label_mapping
    class_weights = processing.class_weights
    num_embedding = processing.num_embedding

    data_train = processing.data_train
    data_test = processing.data_test

    # data_train["last_interaction"] = data_train['last_interaction'].map(label_mapping)
    # data_test["last_interaction"] = data_test['last_interaction'].map(label_mapping)

    sequence_length = processing.window_length

    with open('Modeling/Dataset/data_all/param.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)
        pickle.dump(class_weights, f)
        pickle.dump(num_embedding, f)
        pickle.dump(continuous_var, f)
        pickle.dump(static_var, f)
        pickle.dump(sequence_length, f)

    data_train.to_csv("Modeling/Dataset/data_all/train.csv", index=False)
    data_test.to_csv("Modeling/Dataset/data_all/test.csv", index=False)

