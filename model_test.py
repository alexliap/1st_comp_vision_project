from model import nn_model
from data_preparation import data_preparation
import torch

train_loader, x_test, y_test = data_preparation(batch_size = 1024)

pred_model = nn_model.load_from_checkpoint("model/lightning_logs/version_0/checkpoints/epoch=39-step=1959.ckpt")

all_preds = 0
preds = pred_model(x_test)
for i in range(preds.size(0)):
    if torch.argmax(preds[i]) == torch.argmax(y_test[i]):
        all_preds += 1

acc = 100 * all_preds / 10000

print(acc)
