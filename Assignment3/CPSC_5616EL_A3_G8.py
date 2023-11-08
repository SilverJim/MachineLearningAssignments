#%%
# import the required machine learning libraries and models
import numpy as np
import pandas as pd
import sys
import math
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler

#%%

# Read the file and delete blank values according to the prompts
data_filename = 'data'

if sys.modules.get("google.colab") is None:
    data_path_prefix = "."
else:
    from google.colab import drive
    drive.mount("/content/drive")
    data_path_prefix = "/content/drive/MyDrive/MachineLearningAssignments/Assignment3"

data_path = f"{data_path_prefix}/{data_filename}"

print(f"Loading data from data path: {data_path}")

exercise_types = [
   {"path": "Jumping_Jack_x10", "name": "Jumping_Jack"}, 
   {"path": "Lunges_x10", "name": "Lunges"}, 
   {"path": "Squat_x10", "name": "Squat"}, 
]

data_set = [("Accelerometer.csv", "Accelerometer"), ("TotalAcceleration.csv", "TotalAcceleration"), ("Orientation.csv", "Orientation")]

df_list = []

non_value_columns = ["seconds_elapsed", "exercise_type"]
non_value_columns_set = set(non_value_columns)

for file_name, data_set_name in data_set:
    temp_df_list = []
    for exercise_type in exercise_types:
        df = pd.read_csv(f"{data_path}/{exercise_type['path']}/{file_name}")
        df.drop("time", axis=1, inplace=True)
        df['exercise_type'] = exercise_type['name']
        for column in df.columns:
            if column not in non_value_columns_set:
                df.rename(columns={column: f"{data_set_name}_{column}"}, inplace=True)
        value_columns = list(set(df.columns) - non_value_columns_set)
        # Noise Reduction
        df[value_columns] = df[value_columns].rolling(10).mean()
        df.dropna(inplace=True)
        # remove the data at beginning and end of each exercise, as the excercise has not started or has ended
        df = df[(df["seconds_elapsed"].quantile(0.1) < df["seconds_elapsed"]) & (df['seconds_elapsed'] < df['seconds_elapsed'].quantile(0.9))]

        temp_df_list.append(df)
        # print(df)
    df_list.append(pd.concat(temp_df_list))
    
# Normalization
for df in df_list:
    value_columns = list(set(df.columns) - non_value_columns_set)
    scaler = ColumnTransformer(
        [
            ('standard_scaler', StandardScaler(), value_columns),
            ('other', 'passthrough', non_value_columns)

        ],
    )
    scaled_data = scaler.fit_transform(df)
    df = pd.DataFrame(scaled_data, columns=value_columns + non_value_columns)
    print(df)
# %%
window_size = 10

