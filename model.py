import torch
from pytorch_lightning import LightningModule
from torch.nn import Conv2d, Dropout, CrossEntropyLoss, ModuleList, Softmax, Linear, MaxPool2d, Flatten, \
    BatchNorm1d, Sequential, ReLU, BatchNorm2d
from torch.optim import Adam, SGD


class nn_model(LightningModule):
    def __init__(self, conv_sizes, linear_sizes, dropout, lr):
        super(nn_model, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.loss_f = CrossEntropyLoss()

        self.epoch = 0
        self.conv_sizes = conv_sizes
        self.linear_sizes = linear_sizes

        self.conv_layer_1 = Sequential(Conv2d(self.conv_sizes[0], self.conv_sizes[1], 3),
                                       BatchNorm2d(self.conv_sizes[1]),
                                       ReLU())
        self.conv_layer_2 = Sequential(Conv2d(self.conv_sizes[1], self.conv_sizes[2], 3),
                                       BatchNorm2d(self.conv_sizes[2]),
                                       ReLU(),
                                       MaxPool2d((2, 2)),
                                       Dropout(dropout))
        self.conv_layer_3 = Sequential(Conv2d(self.conv_sizes[2], self.conv_sizes[3], 3),
                                       BatchNorm2d(self.conv_sizes[3]),
                                       ReLU())
        self.conv_layer_4 = Sequential(Conv2d(self.conv_sizes[3], self.conv_sizes[4], 3),
                                       BatchNorm2d(self.conv_sizes[4]),
                                       ReLU(),
                                       MaxPool2d((2, 2)),
                                       Dropout(dropout))
        self.lin_layer_1 = Sequential(Flatten(),
                                      BatchNorm1d(self.linear_sizes[0]),
                                      Linear(self.linear_sizes[0], self.linear_sizes[1]),
                                      ReLU(),
                                      Dropout(dropout))
        self.lin_layer_2 = Sequential(BatchNorm1d(self.linear_sizes[1]),
                                      Linear(self.linear_sizes[1], self.linear_sizes[2]),
                                      ReLU(),
                                      Dropout(dropout))

        self.train_loss_values = []
        self.train_acc = []

        self.val_loss_values = []
        self.val_acc = []

        self.preds = []

    def forward(self, x):

        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.lin_layer_1(x)
        out = self.lin_layer_2(x)

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
