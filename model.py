import torch
from pytorch_lightning import LightningModule
from torch.nn import Conv2d, Dropout, CrossEntropyLoss, ModuleList, Softmax, Linear, MaxPool2d, Flatten, \
    MultiLabelSoftMarginLoss, BatchNorm1d
from torch.optim import Adam, SGD
import torch.nn.functional as func


class nn_model(LightningModule):
    def __init__(self, conv_sizes, linear_sizes, dropout, lr):
        super(nn_model, self).__init__()
        self.save_hyperparameters()
        self.activ_f = Softmax(dim = 1)
        self.lr = lr
        self.loss_f = CrossEntropyLoss()
        self.batch_norm = BatchNorm1d(linear_sizes[0])

        self.epoch = 0
        self.conv_sizes = conv_sizes
        self.linear_sizes = linear_sizes
        self.dropout = Dropout(dropout / 100)
        self.maxpool = MaxPool2d((2, 2))
        self.flatten = Flatten()

        self.conv_list = ModuleList()
        for i in range(len(self.conv_sizes) - 1):
            self.conv_list.append(Conv2d(self.conv_sizes[i], self.conv_sizes[i + 1], 3))

        self.linear_list = ModuleList()
        for j in range(len(self.linear_sizes) - 1):
            self.linear_list.append(Linear(self.linear_sizes[j], self.linear_sizes[j + 1]))

        self.train_loss_values = []
        self.train_acc = []

        self.val_loss_values = []
        self.val_acc = []

        self.preds = []

    def forward(self, x):
        for i in range(len(self.conv_list)):
            x = self.conv_list[i](x)
            x = self.dropout(func.relu(x))
            x = self.maxpool(x)
        x = self.flatten(x)
        x = self.batch_norm(x)
        for i in range(len(self.linear_list) - 1):
            x = self.linear_list[i](x)
            x = self.dropout(func.relu(x))
        x = self.linear_list[len(self.linear_list) - 1](x)
        out = self.activ_f(x)

        return out

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)
        loss = self.loss_f(out, y)

        return {'loss': loss, 'predictions': torch.round(out).detach(), 'targets': y}

    def training_epoch_end(self, training_step_outputs):
        all_preds = 0
        length = 0
        for d in training_step_outputs:
            length = length + d['predictions'].size(0)
            for i in range(d['predictions'].size(0)):
                if torch.argmax(d['predictions'][i]) == torch.argmax(d['targets'][i]):
                    all_preds += 1
        acc = 100 * all_preds / length
        self.epoch += 1
        self.train_acc.append(acc)
        self.train_loss_values.append(d['loss'])

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = self.loss_f(out, y)
        self.val_loss_values.append(loss)

        return {'loss': loss, 'predictions': torch.round(out).detach(), 'targets': y}

    def validation_epoch_end(self, val_step_outputs):
        all_preds = 0
        length = 0
        for d in val_step_outputs:
            length = length + d['predictions'].size(0)
            for i in range(d['predictions'].size(0)):
                if (d['predictions'][i] == d['targets'][i]).all():
                    all_preds += 1
        acc = 100 * all_preds / length
        self.epoch += 1
        self.train_acc.append(acc)

    def configure_optimizers(self):
        # optim = Adam(self.parameters(), lr=self.lr)
        optim = SGD(self.parameters(), lr = self.lr, momentum=0.9)
        return optim
