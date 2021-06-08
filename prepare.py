import torch.utils.data as utils
import numpy as np
import torch
import pandas as pd
import os
from statsmodels.tsa.seasonal import seasonal_decompose

def seasonal_matrix(speed_matrix, pr):
    days, weeks = [], []
    for i in range(speed_matrix.shape[1]):
        d1 = seasonal_decompose(speed_matrix[:pr, i], model='additive', period=288)
        d2 = seasonal_decompose(speed_matrix[:, i], model='additive', period=288)
        # w = seasonal_decompose(speed_matrix[:, i], model='additive', period=288*7)
        days.append(np.concatenate([d1.seasonal, d2.seasonal[pr:]]))
        # weeks.append(w.seasonal)
    return np.array(days).T#, np.array(weeks).T

def extract_label(speed_matrix, seq_len, pred_len):
    """ Turn speed matrix into sequence-and-label pair

    Args:
        speed_matrix: raw speed matrix
        seq_len: length of input sequence
        pred_len: length of predicted sequence

    Returns: speed_sequences, speed_labels

    """
    time_len = speed_matrix.shape[0]

    max_speed = speed_matrix.max().max()
    speed_matrix = speed_matrix / max_speed

    speed_sequences, speed_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(speed_matrix[i:i + seq_len])
        speed_labels.append(speed_matrix[i + seq_len:i + seq_len + pred_len])
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)

    return speed_sequences, speed_labels

def extract_label_multi(speed_matrix, seq_len, pred_len, dataset_name):
    """ Turn speed matrix into sequence-and-label pair
        Using multiple time scale
    Args:
        speed_matrix: raw speed matrix
        seq_len: length of input sequence
        pred_len: length of predicted sequence

    Returns: speed_sequences, speed_labels

    """
    if 'PeMS08' not in dataset_name:
        dataset_label = 'METR_LA'
    else:
        dataset_label = 'PeMS08'
    if os.path.exists('{}_Dataset/{}_seq_seasonal.npy'.format(dataset_label, dataset_name)):
        print('Find pre-saved seasonal data...')
        speed_sequences = np.load('{}_Dataset/{}_seq_seasonal.npy'.format(dataset_label, dataset_name))
        speed_labels = np.load('{}_Dataset/{}_label_seasonal.npy'.format(dataset_label, dataset_name))
        return speed_sequences, speed_labels

    time_len = speed_matrix.shape[0]

    max_speed = speed_matrix.max().max()
    speed_matrix = speed_matrix / max_speed
    start, interval = 7*24*12, 7*24*12  # use 7 days for seasonal feature extraction

    # time decomposition. May take a while.
    days = np.zeros(speed_matrix.shape)
    i = start
    while i < time_len:
        fw = min(time_len-i, 288) # step forward by one day
        tmp = speed_matrix[i-interval:i+fw, :]
        for j in range(speed_matrix.shape[1]):
            d1 = seasonal_decompose(tmp[:, j], model='additive', period=288)
            days[i:i+fw, j] = d1.seasonal[-fw:]

        i += fw

    speed_matrix = np.concatenate([speed_matrix, days], axis=1)

    print(i, time_len)

    speed_sequences, speed_labels = [], []

    for i in range(start, time_len - seq_len - pred_len): # only use validation
        seq = speed_matrix[i:i + seq_len, :] #7 days before and seq_len timesteps before
        speed_sequences.append(seq)
        speed_labels.append(speed_matrix[i + seq_len:i + seq_len + pred_len])
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)

    print(speed_sequences.shape, speed_labels.shape)
    np.save('{}_Dataset/{}_seq_seasonal.npy'.format(dataset_label, dataset_name), speed_sequences, allow_pickle=True)
    np.save('{}_Dataset/{}_label_seasonal.npy'.format(dataset_label, dataset_name), speed_labels, allow_pickle=True)

    return speed_sequences, speed_labels

def PrepareDataset(speed_matrix, BATCH_SIZE=40, seq_len=10, pred_len=1, train_propotion=0.7, valid_propotion=0.1):
    """ Prepare training and testing datasets and dataloaders.

    Convert speed/volume/occupancy matrix to training and testing dataset.
    The vertical axis of speed_matrix is the time axis and the horizontal axis
    is the spatial axis.

    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    speed_std = speed_matrix# pd.read_pickle('./METR_LA_Dataset/la_speed') # speed without noise
    max_speed = speed_matrix.max().max()

    speed_sequences, speed_labels = extract_label(speed_matrix, seq_len, pred_len)
    std_sequences, std_labels = extract_label(speed_std, seq_len, pred_len)


    # shuffle and split the dataset to training and testing datasets
    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = std_sequences[train_index:valid_index], std_labels[train_index:valid_index]
    test_data, test_label = std_sequences[valid_index:], std_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, max_speed

def PrepareDataset_multi(speed_matrix, dataset_name, BATCH_SIZE=40, seq_len=10, pred_len=1, train_propotion=0.7, valid_propotion=0.1):
    """ Prepare training and testing datasets and dataloaders.

    Convert speed/volume/occupancy matrix to training and testing dataset.
    The vertical axis of speed_matrix is the time axis and the horizontal axis
    is the spatial axis.

    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    speed_std = speed_matrix# pd.read_pickle('./METR_LA_Dataset/la_speed') # speed without noise
    max_speed = speed_matrix.max().max()

    speed_sequences, speed_labels = extract_label_multi(speed_matrix, seq_len, pred_len, dataset_name)
    # std_sequences, std_labels = extract_label_multi(speed_std, seq_len, pred_len)


    # shuffle and split the dataset to training and testing datasets
    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, max_speed