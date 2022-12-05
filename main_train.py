import numpy as np
import warnings
from data_preparation import data_preparation
from model import nn_model
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")

train_loader, x_test, y_test = data_preparation(batch_size = 1024)

conv_sizes = [3, 24, 48, 96]
model = nn_model(conv_sizes = conv_sizes, linear_sizes = [384, 10], dropout = 15, lr = 0.005)

trainer = Trainer(max_epochs = 2, default_root_dir = 'model/')

trainer.fit(model, train_loader)

train_acc = model.train_acc
train_loss = torch.stack(model.train_loss_values).cpu().detach().numpy()
epochs = model.epoch

fig, axs = plt.subplots(1, 2)

fig.suptitle('Training')
axs[0].plot(range(epochs), train_acc)
axs[1].plot(range(epochs), train_loss)

axs[0].set_ylabel('Accuracy (%)')
axs[0].set_xlabel('Epochs')
axs[0].grid()

axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epochs')
axs[1].grid()

plt.tight_layout()
fig.savefig('training_{}.jpg'.format(conv_sizes))
