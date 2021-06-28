
from torchvision.datasets import MNIST
from torchvision import transforms

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib.pyplot as plt

from datasets import BalancedBatchSampler
# Set up the network and training parameters
from networks import EmbeddingNet, ClassificationNet
from metrics import AccumulatedAccuracyMetric
mean, std = 0.1307, 0.3081

train_dataset = MNIST('../data/MNIST', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((mean,), (std,))
                      ]))
test_dataset = MNIST('../data/MNIST', train=False, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((mean,), (std,))
                     ]))
n_classes = 5

# Set up data loaders
label_pick=[0,1,2,3,4]
train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, label_pick,n_classes=5, n_samples=30)
train_batch_sampler_unpick = BalancedBatchSampler(train_dataset.train_labels, [5,6,7,8,9],n_classes=5, n_samples=30)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels,[0,1,2,3,4,5,6,7,8,9], n_classes=10, n_samples=15)
batch_size = 150
kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
train_loader_unpick = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler_unpick, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)


embedding_net = EmbeddingNet()
model = ClassificationNet(embedding_net, n_classes=n_classes)
if cuda:
    model.cuda()
loss_fn = torch.nn.NLLLoss()
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

fit(train_loader,train_loader_unpick, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])

