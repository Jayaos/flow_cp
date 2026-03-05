import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.lib.stride_tricks import sliding_window_view


def build_dataloader(dataset: dict, context_window: int, batch_size:int, past_resid_as_feature=False):

    train_size = dataset["resid_train"].shape[0]
    valid_size = dataset["resid_valid"].shape[0]

    resid_input_train, resid_prediction_train = build_strided_residual(dataset["resid_train"], context_window)
    x_input_train = build_strided_feature(dataset["x_train"], context_window)

    if past_resid_as_feature:
        train_dataset = AutoregressiveDatasetResidFeature(resid_input_train, x_input_train, resid_prediction_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                      collate_fn=autoregressive_residfeature_collate_fn, drop_last=False)
    else:
        train_dataset = AutoregressiveDataset(x_input_train, resid_prediction_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                      collate_fn=autoregressive_collate_fn, drop_last=False)
    
    resid_valid = dataset["resid_full"][train_size-context_window:train_size+valid_size]
    x_valid = dataset["x_full"][train_size-context_window:train_size+valid_size]

    resid_input_valid, resid_prediction_valid = build_strided_residual(resid_valid, context_window)
    x_input_valid = build_strided_feature(x_valid, context_window)

    if past_resid_as_feature:
        valid_dataset = AutoregressiveDatasetResidFeature(resid_input_valid, x_input_valid, resid_prediction_valid)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                      collate_fn=autoregressive_residfeature_collate_fn, drop_last=False)
    else:
        valid_dataset = AutoregressiveDataset(x_input_valid, resid_prediction_valid)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                      collate_fn=autoregressive_collate_fn, drop_last=False)
    
    resid_test = dataset["resid_full"][train_size+valid_size-context_window:]
    x_test = dataset["x_full"][train_size+valid_size-context_window:]

    resid_input_test, resid_prediction_test = build_strided_residual(resid_test, context_window)
    x_input_test = build_strided_feature(x_test, context_window)
    
    if past_resid_as_feature:
        test_dataset = AutoregressiveDatasetResidFeature(resid_input_test, x_input_test, resid_prediction_test)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     collate_fn=autoregressive_residfeature_collate_fn, drop_last=False)
    else:
        test_dataset = AutoregressiveDataset(x_input_test, resid_prediction_test)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     collate_fn=autoregressive_collate_fn, drop_last=False)
    
    return train_dataloader, valid_dataloader, test_dataloader


class AutoregressiveDataset(Dataset):
    
    def __init__(self, featureX, residY):
        self.featureX = featureX
        self.residY = residY
        self.feature_dim = featureX.shape[-1]
        self.outcome_dim = self.residY.shape[-1]

    def __len__(self):
        return len(self.residY)

    def __getitem__(self,idx):
        return self.featureX[idx,:], self.residY[idx]


class AutoregressiveDatasetResidFeature(Dataset):
    
    def __init__(self, residX, featureX, residY):
        self.residX = residX
        self.featureX = featureX
        self.residY = residY
        self.feature_dim = featureX.shape[-1] + residX.shape[-1]
        self.outcome_dim = self.residY.shape[-1]

    def __len__(self):
        return len(self.residY)

    def __getitem__(self,idx):
        return self.residX[idx], self.featureX[idx,:], self.residY[idx]
    

def autoregressive_collate_fn(batch):

    feature_x_batch, resid_y_batch = zip(*batch)

    feature_x_tensor = torch.from_numpy(np.stack(feature_x_batch)).to(torch.float32) # batch_size * past_window * feature_dim 
    resid_y_tensor = torch.from_numpy(np.stack(resid_y_batch)).to(torch.float32) # batch_size * num_step_prediction

    return feature_x_tensor, resid_y_tensor
    
    
def autoregressive_residfeature_collate_fn(batch):

    resid_x_batch, feature_x_batch, resid_y_batch = zip(*batch)

    resid_x_tensor = torch.from_numpy(np.stack(resid_x_batch)) # batch_size * past_window
    feature_x_tensor = torch.from_numpy(np.stack(feature_x_batch)) # batch_size * past_window * feature_dim 
    resid_y_tensor = torch.from_numpy(np.stack(resid_y_batch)).to(torch.float32) # batch_size * num_step_prediction

    batch_size, past_window, feature_dim = feature_x_tensor.size()
    _, _, resid_dim = resid_x_tensor.size()
    resid_x_tensor = resid_x_tensor.reshape((batch_size, past_window, resid_dim)) # batch_size * past_window * resid_dim
    x_tensor = torch.cat([resid_x_tensor, feature_x_tensor], axis=-1).to(torch.float32) # batch_size * past_window * (feature_dim+1)

    return x_tensor, resid_y_tensor


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

