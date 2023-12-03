#%%
# import the required machine learning libraries and models
import numpy as np
import pandas as pd
import sys
import math
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#%%
# All the data should put into the data folder
data_filename = 'data'

if sys.modules.get("google.colab") is None:
    data_path_prefix = "."
else:
    from google.colab import drive
    drive.mount("/content/drive")
    data_path_prefix = "/content/drive/MyDrive/MachineLearningAssignments/Assignment3"

data_path = f"{data_path_prefix}/{data_filename}"

print(f"Loading data from data path: {data_path}")

# Define the exercise types
data_sets = [
   {"path": "Jumping_Jack_x10", "name": "Jumping_Jack"}, 
   {"path": "Lunges_x10", "name": "Lunges"}, 
   {"path": "Squat_x10", "name": "Squat"}, 
   {"path": "Jumping_Jack_x10_new", "name": "Jumping_Jack_new"}, 
   {"path": "Lunges_x10_new", "name": "Lunges_new"}, 
   {"path": "Squat_x10", "name": "Squat_new"}, 
]

# Define the data sets
data_set = [("Accelerometer.csv", "Accelerometer"), ("TotalAcceleration.csv", "TotalAcceleration"), ("Orientation.csv", "Orientation")]

df_list = []

# In all the data set after "time" is removed, only "seconds_elapsed" and "data_set" are non-value columns
non_value_columns = ["seconds_elapsed", "data_set"]
non_value_columns_set = set(non_value_columns)
# read every dataset in every exercise type
for file_name, data_set_name in data_set:
    temp_df_list = []
    for data_set in data_sets:
        df = pd.read_csv(f"{data_path}/{data_set['path']}/{file_name}")
        # There is "seconds_elapsed" so "time" is not necessary
        df.drop("time", axis=1, inplace=True)
        # add the data_set column according to which exercise the data from
        df['data_set'] = data_set['name']
        # Rename the columns according to which dataset the data from to avoid same column
        for column in df.columns:
            if column not in non_value_columns_set:
                df.rename(columns={column: f"{data_set_name}_{column}"}, inplace=True)
        value_columns = list(set(df.columns) - non_value_columns_set)
        # Noise Reduction: calculate rolling mean for all the value columns
        df[value_columns] = df[value_columns].rolling(50).mean()
        # rolling mean create some NaN values, so we should drop them
        df.dropna(inplace=True)
        # remove the data at beginning and end of each exercise, as the excercise has not started or has ended
        df = df[(df["seconds_elapsed"].quantile(0.1) < df["seconds_elapsed"]) & (df['seconds_elapsed'] < df['seconds_elapsed'].quantile(0.9))]
        temp_df_list.append(df)
    # combine dataframe from all the exercise types together and put it in to the df_list
    df_list.append(pd.concat(temp_df_list))
    
# Normalization
for i in range(len(df_list)):
    df = df_list[i]
    value_columns = list(set(df.columns) - non_value_columns_set)
    scaler = ColumnTransformer(
        [
            ('standard_scaler', StandardScaler(), value_columns),
            ('other', 'passthrough', non_value_columns)

        ],
    )
    scaled_data = scaler.fit_transform(df)
    df_list[i] = pd.DataFrame(scaled_data, columns=value_columns + non_value_columns)
    
# %%
# Window by dividing the data of each exercise into eaqual parts to apply data augmentation we use several different window sizes
windows_count_list = [8, 9, 10, 11, 12]
df_concat_list = []
for windows_count in windows_count_list:
    seconds_elapsed_min = {}
    windows_size = {}
    for item in data_sets:
        data_set = item["name"]
        seconds_elapsed_min[data_set] = min((df[df["data_set"] == data_set]["seconds_elapsed"].min() for df in df_list))
        seconds_elapsed_max = max((df[df["data_set"] == data_set]["seconds_elapsed"].max() for df in df_list)) + 0.0000001
        windows_size[data_set] = (seconds_elapsed_max - seconds_elapsed_min[data_set]) / windows_count
    print(seconds_elapsed_min, windows_size)

    # process window and calculate min, max, mean, std value of each window
    df_list_temp = []
    for df in df_list:
        df = df.copy()
        df["window"] = df[["seconds_elapsed", "data_set"]].apply(lambda row: math.floor((row["seconds_elapsed"] - seconds_elapsed_min[row["data_set"]]) / windows_size[row["data_set"]]), axis=1)
        df = df.sort_values(["data_set", "window", "seconds_elapsed"])
        df.drop("seconds_elapsed", axis=1, inplace=True)
        df = df.groupby(["data_set", "window"], group_keys=True).apply(lambda x: np.array(x[[column for column in df.columns if column not in {"data_set", "window"}]])).to_frame()
        df_list_temp.append(df)
    # join the dataframe from all the dataset together according to index "windows" and "data_set"
    df_concat_temp = pd.concat(df_list_temp, axis=1)
    df_concat_temp["windows_count"] = windows_count
    df_concat_list.append(df_concat_temp)
df_concat = pd.concat(df_concat_list)
df_concat.reset_index(inplace=True)
# Make the data_set the last column
temp = df_concat.pop("data_set")
df_concat["data_set"] = temp
data_set_mappping = {
    "Jumping_Jack": "Jumping_Jack",
    "Lunges": "Lunges",
    "Squat": "Squat",
    "Jumping_Jack_new": "Jumping_Jack",
    "Lunges_new": "Lunges",
    "Squat_new": "Squat"
}
df_concat["exercise_type"] = df_concat["data_set"].map(data_set_mappping)
df_concat.drop(["data_set"], axis=1, inplace=True)

# %%
df_concat["features"] = df_concat.apply(lambda x:np.concatenate([x[0], x[1], x[2]]), axis=1)
df_concat.drop([0, 1, 2], axis=1, inplace=True)
print(df_concat[0])
# %%
# Split the data into train data and test data, to avoid train data mixed into test data spilit it by time series
train_df = df_concat[(df_concat["window"] + 1) / df_concat["windows_count"] < 0.8].copy()
train_df.drop(["window", "windows_count"], axis=1, inplace=True)
test_df = df_concat[df_concat["window"] / df_concat["windows_count"] >= 0.8].copy()
test_df.drop(["window", "windows_count"], axis=1, inplace=True)
print(train_df)
print(test_df)

# Define a function to divide the data into X and y
def divide_Xy(df, oversample=True):
    data = df.to_numpy()
    X = data[:, :-1]
    y = data[:, -1]
    if oversample:
        ros = RandomOverSampler(random_state=0)
        X, y = ros.fit_resample(X, y)
    return data, X, y

# Divide the train data and test data into X and y
train_data, train_X, train_y = divide_Xy(train_df)
test_data, test_X, test_y = divide_Xy(test_df, oversample=False)

# Print the data
print("train_X:")
print(train_X)
print("train_y:")
print(train_y)
print("test_X:")
print(test_X)
print("test_y:")
print(test_y)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{device}")
class CNN(nn.Module):


evaluate_model(svm_model, test_X, test_y)
# %%
