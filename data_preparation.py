import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def data_preparation(batch_size = 4096):
    with open('train_data.pickle', 'rb') as train:
        train = pickle.load(train, encoding='bytes')

    with open('test_data.pickle', 'rb') as test:
        test = pickle.load(test, encoding='bytes')

    x_train = train['data']
    y_train = np.array(train['labels'])

    x_train = torch.from_numpy(x_train).to(torch.float32)
    y_train = torch.from_numpy(y_train).to(torch.float32)

    x_test = test['data']
    y_test = np.array(test['labels'])

    x_test = torch.from_numpy(x_test).to(torch.float32)
    y_test = torch.from_numpy(y_test).to(torch.float32)

    train_dt = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dt, batch_size = batch_size)

    return train_loader, x_test, y_test

