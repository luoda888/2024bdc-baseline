import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge


def load_data(path):
    path = Path(path)
    global_data = np.load(path / "cenn_data.npy")
    station_num = global_data.shape[-1]
    temp_data = np.load(path / "temp_lookback.npy").squeeze()
    wind_data = np.load(path / "wind_lookback.npy").squeeze()

    global_data = global_data.mean(axis=-2)
    global_data = np.repeat(global_data, 3, axis=1)

    return global_data, temp_data, wind_data


def get_test_data(global_data, temp_data, wind_data):
    length, input_length, _, station_num = global_data.shape
    infer_data = np.concatenate((
        global_data,
        np.expand_dims(temp_data, axis=-2),
        np.expand_dims(wind_data, axis=-2),
    ), axis=-2).astype(np.float32).transpose(0, 3, 1, 2).reshape(length * station_num, -1)

    return infer_data


def invoke(inputs):
    input_length, output_length, feature_num = 24 * 7, 24, 4 + 2
    save_path = "/home/mw/project"
    global_data, temp_data, wind_data = load_data(inputs)
    data_length = global_data.shape[0]
    station_num = global_data.shape[-1]
    infer_data = get_test_data(global_data, temp_data, wind_data)
    wind_model, temp_model = torch.load("/home/mw/project/model.ckpt")
    wind_preds, temp_preds = wind_model.predict(infer_data), temp_model.predict(infer_data)
    wind_preds = np.expand_dims(wind_preds.reshape(data_length, station_num, output_length), axis=-1).transpose(0, 2, 1, 3)
    temp_preds = np.expand_dims(temp_preds.reshape(data_length, station_num, output_length), axis=-1).transpose(0, 2, 1, 3)

    np.save(Path(save_path) / "wind_predict.npy", wind_preds)
    np.save(Path(save_path) / "temp_predict.npy", temp_preds)
