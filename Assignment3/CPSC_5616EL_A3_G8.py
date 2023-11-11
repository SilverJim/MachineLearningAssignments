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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
exercise_types = [
   {"path": "Jumping_Jack_x10", "name": "Jumping_Jack"}, 
   {"path": "Lunges_x10", "name": "Lunges"}, 
   {"path": "Squat_x10", "name": "Squat"}, 
]

# Define the data sets
data_set = [("Accelerometer.csv", "Accelerometer"), ("TotalAcceleration.csv", "TotalAcceleration"), ("Orientation.csv", "Orientation")]

df_list = []

# In all the data set after "time" is removed, only "seconds_elapsed" and "exercise_type" are non-value columns
non_value_columns = ["seconds_elapsed", "exercise_type"]
non_value_columns_set = set(non_value_columns)
# read every dataset in every exercise type
for file_name, data_set_name in data_set:
    temp_df_list = []
    for exercise_type in exercise_types:
        df = pd.read_csv(f"{data_path}/{exercise_type['path']}/{file_name}")
        # There is "seconds_elapsed" so "time" is not necessary
        df.drop("time", axis=1, inplace=True)
        # add the exercise_type column according to which exercise the data from
        df['exercise_type'] = exercise_type['name']
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
    for item in exercise_types:
        exercise_type = item["name"]
        seconds_elapsed_min[exercise_type] = min((df[df["exercise_type"] == exercise_type]["seconds_elapsed"].min() for df in df_list))
        seconds_elapsed_max = max((df[df["exercise_type"] == exercise_type]["seconds_elapsed"].max() for df in df_list)) + 0.0000001
        windows_size[exercise_type] = (seconds_elapsed_max - seconds_elapsed_min[exercise_type]) / windows_count
    print(seconds_elapsed_min, windows_size)

    # process window and calculate min, max, mean, std value of each window
    df_list_temp = []
    for df in df_list:
        df = df.copy()
        df["window"] = df[["seconds_elapsed", "exercise_type"]].apply(lambda row: math.floor((row["seconds_elapsed"] - seconds_elapsed_min[row["exercise_type"]]) / windows_size[row["exercise_type"]]), axis=1)
        df.drop("seconds_elapsed", axis=1, inplace=True)
        df = df.groupby(["exercise_type", "window"]).agg({column: ["min","max","mean","std"] for column in df.columns.difference(["exercise_type", "window"])})
        df_list_temp.append(df)
    # join the dataframe from all the dataset together according to index "windows" and "exercise_type"
    df_concat_temp = pd.concat(df_list_temp, axis=1)
    df_concat_temp["windows_count"] = windows_count
    df_concat_list.append(df_concat_temp)
df_concat = pd.concat(df_concat_list)
df_concat.reset_index(inplace=True)
# Make the exercise_type the last column
temp = df_concat.pop("exercise_type")
df_concat["exercise_type"] = temp
print(df_concat)
# %%
# Split the data into train data and test data, to avoid train data mixed into test data spilit it by time series
train_df = df_concat[(df_concat["window"] + 1) / df_concat["windows_count"] < 0.8].copy()
train_df.drop(["window", "windows_count"], axis=1, level=0, inplace=True)
test_df = df_concat[df_concat["window"] / df_concat["windows_count"] >= 0.8].copy()
test_df.drop(["window", "windows_count"], axis=1, level=0, inplace=True)
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
# Using GridSearchCV to find the SVC model with best parameters
svm_model = SVC()
svm_grid_search = GridSearchCV(svm_model, {"kernel": ["linear", "poly", "rbf", "sigmoid"], 'C': [0.1, 1, 10]}, cv=5)
svm_grid_search.fit(train_X, train_y)
print(f"Best parameters for SVM model:\n{svm_grid_search.best_params_}")
svm_model = svm_grid_search.best_estimator_

# Defind a function to evaluate the models
def evaluate_model(model, X, y):
    predicted_y = model.predict(X)
    print(classification_report(y, predicted_y))

    confusion_mat = confusion_matrix(y, predicted_y)
    print(f"Confusion matrix: \n{confusion_mat}")

# Evaluate the the SVC model
print("Evaluation of SVM model:")
evaluate_model(svm_model, test_X, test_y)
# %%
# Using GridSearchCV to find the RandomForestClassifier model with best parameters
rf_model = RandomForestClassifier()
rf_grid_search = GridSearchCV(rf_model, {"n_estimators": [10, 100, 1000], "max_depth": [2, 5, 10]}, cv=5)
rf_grid_search.fit(train_X, train_y)
print(f"Best parameters for Random Forest model:\n{rf_grid_search.best_params_}")
rf_model = rf_grid_search.best_estimator_

# Evaluate the the RandomForestClassifier model
print("Evaluation of Random Forest model:")
evaluate_model(rf_model, test_X, test_y)
# %%
