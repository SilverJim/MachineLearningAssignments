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

# If the length of the array is less than the target length, pad it with zeros and make the original array at the beginning
def pad(original_series, target_length):
    new_series = original_series.copy()
    for i in range(original_series.size):
        line = original_series[i]
        original_length = len(line)
        if original_length > target_length:
            new_line = line[:target_length]
        else:
            new_line = np.pad(line, ((0, target_length - original_length), (0, 0)), mode="constant")
        new_series[i] = new_line
    return new_series

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
    count_after_padding = 200
    df_list_temp = []
    for i, df in enumerate(df_list):
        df = df.copy()
        df["window"] = df[["seconds_elapsed", "data_set"]].apply(lambda row: math.floor((row["seconds_elapsed"] - seconds_elapsed_min[row["data_set"]]) / windows_size[row["data_set"]]), axis=1)
        df = df.sort_values(["data_set", "window", "seconds_elapsed"])
        df.drop("seconds_elapsed", axis=1, inplace=True)
        df = (
            df
            .groupby(["data_set", "window"], group_keys=True)
            .apply(lambda x: np.array(x[[column for column in df.columns if column not in {"data_set", "window"}]]))
        ).to_frame()
        df[0] = df.apply(lambda x: pad(x, count_after_padding))
        df.rename(columns={0: i}, inplace=True)
        df_list_temp.append(df)
    # join the dataframe from all the dataset together according to index "windows" and "data_set"
    df_concat_temp = pd.concat(df_list_temp, axis=1)
    df_concat_temp["windows_count"] = windows_count
    df_concat_list.append(df_concat_temp)
df_concat = pd.concat(df_concat_list)
df_concat.reset_index(inplace=True)
# Make the data_set the last column
data_set_mappping = {
    "Jumping_Jack": 0,
    "Lunges": 1,
    "Squat": 2,
    "Jumping_Jack_new": 0,
    "Lunges_new": 1,
    "Squat_new": 2,
}
df_concat["exercise_type"] = df_concat["data_set"].map(data_set_mappping)
df_concat.drop(["data_set"], axis=1, inplace=True)

# %%
df_concat["features"] = df_concat.apply(lambda x:np.concatenate([x[0], x[1], x[2]], axis=1), axis=1)
df_concat.drop([0, 1, 2], axis=1, inplace=True)
temp = df_concat.pop("exercise_type")
df_concat["exercise_type"] = temp
print(df_concat)
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
    y = data[:, 1].astype(np.int32)
    if oversample:
        ros = RandomOverSampler(random_state=0)
        X, y = ros.fit_resample(X, y)
    X = np.stack([item[0] for item in X])
    X = X.astype(np.float32)
    return X, y

# Divide the train data and test data into X and y
train_X, train_y = divide_Xy(train_df)
test_X, test_y = divide_Xy(test_df, oversample=False)

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
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(13, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(64 * 50, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = self(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

# Train the model with train_X and train_y
def train_model(model, train_X, train_y, epochs=10, batch_size=32, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    train_X = torch.tensor(train_X, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_y, dtype=torch.long, device=device)
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, len(train_X), batch_size):
            inputs = train_X[i:i+batch_size]
            labels = train_y[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1} loss: {running_loss / len(train_X)}")
    print("Finished Training")
    return model

# Call train_model to train the model
cnn_model = CNN()
train_X_CNN = train_X.transpose(0, 2, 1)
train_y_CNN = train_y
test_X_CNN = test_X.transpose(0, 2, 1)
test_y_CNN = test_y
cnn_model = train_model(cnn_model, train_X_CNN, train_y_CNN)
def evaluate_model(model, X, y):
    predicted_y = model.predict(X)
    print(classification_report(y, predicted_y))

    confusion_mat = confusion_matrix(y, predicted_y)
    print(f"Confusion matrix: \n{confusion_mat}")
evaluate_model(cnn_model, test_X_CNN, test_y_CNN)
# %%
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(13, 64, 1, batch_first=True)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc1(x[:, -1, :])
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = self(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
gru_model = GRU()
train_X_GRU = train_X
train_y_GRU = train_y
test_X_GRU = test_X
test_y_GRU = test_y
gru_model = train_model(gru_model, train_X_GRU, train_y_GRU, epochs=35)
evaluate_model(gru_model, test_X_GRU, test_y_GRU)
# %%
class CNN_GRU(nn.Module):
    def __init__(self):
        super(CNN_GRU, self).__init__()
        self.conv1 = nn.Conv1d(13, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.gru = nn.GRU(64, 64, 2, batch_first=True)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.transpose(2, 1)
        x, _ = self.gru(x)
        x = self.fc1(x[:, -1, :])
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = self(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

cnn_gru_model = CNN_GRU()
train_X_CNN_GRU = train_X_CNN
train_y_CNN_GRU = train_y_CNN
test_X_CNN_GRU = test_X_CNN
test_y_CNN_GRU = test_y_CNN
cnn_gru_model = train_model(cnn_gru_model, train_X_CNN_GRU, train_y_CNN_GRU, epochs=30)
evaluate_model(cnn_gru_model, test_X_CNN_GRU, test_y_CNN_GRU)
# %%
