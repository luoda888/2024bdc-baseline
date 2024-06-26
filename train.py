import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


def load_data(input_path):
    station_num = 3850
    temp_data = np.load(f"{input_path}/temp.npy").squeeze()
    wind_data = np.load(f"{input_path}/wind.npy").squeeze()
    global_data = np.load(f"{input_path}/global_data.npy")
    global_data = global_data.mean(axis=-2, keepdims=True).reshape(-1, 4, station_num)
    global_data = np.repeat(global_data, 3, axis=0)

    return global_data, temp_data, wind_data


global_data, temp_data, wind_data = load_data("./input/global")


def get_train_data(sample=100, input_length=24 * 7, output_length=24, feature_num=4 + 2):
    data, label1, label2 = [], [], []
    for i in tqdm(range(0, len(global_data) - input_length - output_length - 1, sample)):
        wind_inputs = wind_data[i:i + input_length]
        temp_inputs = temp_data[i:i + input_length]
        data.append(
            np.hstack((
                global_data[i:i + input_length, :],
                np.expand_dims(wind_inputs, axis=1),
                np.expand_dims(temp_inputs, axis=1),
            )).astype(np.float32))
        label1.append(wind_data[i + input_length + 1:i + input_length + 1 + output_length, :].astype(np.float32))
        label2.append(temp_data[i + input_length + 1:i + input_length + 1 + output_length, :].astype(np.float32))

    data = np.stack(data).transpose(1, 2, 0, 3).reshape(input_length * feature_num, -1).T
    label1 = np.stack(label1).transpose(1, 0, 2).reshape(output_length, -1).T
    label2 = np.stack(label2).transpose(1, 0, 2).reshape(output_length, -1).T
    return data, label1, label2


split_ratio = 0.9
train, train_label1, train_label2 = get_train_data()
length = int(train.shape[0] * split_ratio)
X_train = train[:length]
y_train_temp = train_label1[:length]
y_train_wind = train_label2[:length]

X_valid = train[length:]
y_valid_temp = train_label1[length:]
y_valid_wind = train_label2[length:]
print(X_train.shape, X_valid.shape)

temp_model = MultiOutputRegressor(Ridge(), n_jobs=8)
temp_model.fit(X_train, y_train_temp)
temp_preds = temp_model.predict(X_valid)
wind_model = MultiOutputRegressor(Ridge(), n_jobs=8)
wind_model.fit(X_train, y_train_wind)
wind_preds = wind_model.predict(X_valid)

mse1 = mean_squared_error(temp_preds, y_valid_temp) / np.std(y_valid_temp)**2
mse2 = mean_squared_error(wind_preds, y_valid_wind) / np.std(y_valid_wind)**2
offline_score = mse1, mse2, mse1 + 10 * mse2
print(offline_score)

torch.save((wind_model, temp_model), f"score={round(offline_score[-1], 4)}-model.ckpt")
