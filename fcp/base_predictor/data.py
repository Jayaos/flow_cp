import numpy as np
import pandas as pd
import os
from numpy.polynomial.polynomial import Polynomial
from numpy.lib.stride_tricks import sliding_window_view


def load_solar_dataset(dataset_dir, rolling_window=5, standardize=True, location=None):

    def _rolling(a, window):
        N, d = a.shape
        shape = (N - window + 1, window, d)
        strides = (a.strides[0],) + a.strides
        out = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        length = len(out)
        return out.view().reshape(length, -1)
    
    def _merge_DHIs(dataset_dir):
        csv_files = [file for file in os.listdir(dataset_dir) if file.endswith('.csv')]
        dataframes = []
        for file in csv_files:
            file_path = os.path.join(dataset_dir, file)
            df = pd.read_csv(file_path)
            dataframes.append(df)
        DHIs = [df['DHI'].values for df in dataframes]
        DHIs = np.array(DHIs).T
        return DHIs
    
    DHIs = _merge_DHIs(dataset_dir)

    if standardize:
        DHIs = (DHIs - DHIs.mean(axis=0)) / DHIs.std(axis=0)
    
    if location is not None:
        DHIs = DHIs[:, location]
        
    data_y = DHIs

    data_x = _rolling(data_y, window=rolling_window)
    N = len(data_x)
    
    return data_x[:-1], data_y[-N+1:]


def load_wind_dataset(dataset_dir, rolling_window=5, standardize=True, location=None):

    def _rolling(a, window):
        N, d = a.shape
        shape = (N - window + 1, window, d)
        strides = (a.strides[0],) + a.strides
        out = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        length = len(out)
        return out.view().reshape(length, -1)

    wind = np.load(os.path.join(dataset_dir, 'sample_wind.npy'))

    if standardize:
        wind = (wind-wind.mean(axis=0))/wind.std(axis=0)

    if location is None:
        speeds = wind[:, :, 1]
    else:
        speeds = wind[:, location, 1]
    data_y = speeds
    
    data_x = _rolling(data_y, window=rolling_window)
    N = len(data_x)

    return data_x[:-1], data_y[-N+1:]


def load_traffic_dataset(dataset_dir, rolling_window=5, standardize=True, location=None):
    
    def _rolling(a, window):
        N, d = a.shape
        shape = (N - window + 1, window, d)
        strides = (a.strides[0],) + a.strides
        out = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        length = len(out)
        return out.view().reshape(length, -1)

    Xfull = pd.read_pickle(os.path.join(dataset_dir, 'traffic_data.p'))
    Xfull = Xfull.values
    
    if standardize:
        Xfull = (Xfull - Xfull.mean(axis=0)) / Xfull.std(axis=0)

    if location is not None:
        Xfull = Xfull[:, location]
        
    data_y = Xfull
    data_x = _rolling(data_y, window=rolling_window)
    N = len(data_x)
    
    return data_x[:-1], data_y[-N+1:]


def build_strided_residual(residual_sequences, past_window):
    
    strided_residual_sequences = sliding_window_view(residual_sequences, past_window, axis=0)
    strided_residual_sequences = np.transpose(strided_residual_sequences, (0,2,1))
    strided_residue_input_sequences = strided_residual_sequences[:-1]
    strided_residue_prediction_sequences = strided_residual_sequences[1:]
    
    return strided_residue_input_sequences, strided_residue_prediction_sequences


def build_strided_feature(feature_sequences, past_window):
    
    strided_feature_sequences = sliding_window_view(feature_sequences, past_window, axis=0)
    strided_feature_input_sequences = strided_feature_sequences[:-1]
    strided_feature_input_sequences = np.transpose(strided_feature_input_sequences, (0,2,1))
    
    return strided_feature_input_sequences
