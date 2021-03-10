import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import torchvision.models as models
from sklearn.datasets import load_files
import torch.optim as optim
import os
import numpy as np
import time
import random 
from PIL import Image
from torchvision.utils import make_grid
from torchvision import datasets,transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from glob import glob
from copy import deepcopy
import pandas as pd

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

root_path = 'data/chest_xray/'
train_batch_size = 32
val_batch_size = 16
test_batch_size = 624
degrees = 90

train_dataset = ImageFolder(
    root = root_path + 'train/', transform = transforms.Compose([transforms.Resize((224,224)),
                                                                 transforms.RandomRotation(degrees, resample=False,expand=False, center=None),
                                                                 transforms.ToTensor()]))
val_dataset = ImageFolder(
    root = root_path + 'val/', transform = transforms.Compose([transforms.Resize((224,224)),
                                                               transforms.ToTensor()]))
test_dataset = ImageFolder(
    root = root_path + 'test/', transform = transforms.Compose([transforms.Resize((224,224)),
                                                                transforms.ToTensor()]))

train_loader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 624, shuffle = True)

epochs = 2
losses = []
criterion = nn.CrossEntropyLoss()
current_model = models.resnet18(pretrained=False)
num_features = 512
current_model.fc = nn.Linear(512, 2)
loss =0
class_weights = torch.FloatTensor([3.8896346,1.346])
criterion = nn.CrossEntropyLoss(weight=class_weights)
if use_cuda == True:
    current_model.cuda()
    criterion = criterion.cuda()
optimizer = optim.Adam(current_model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if use_cuda == True:
            inputs = inputs.cuda()
            labels = labels.cuda()
        #inputs = inputs.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = current_model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss)
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / (i+1)))
torch.cuda.empty_cache()


for local_batch, local_labels in test_loader:
    if use_cuda == True:
        local_batch = local_batch.cuda()
    temp = current_model(local_batch)
    #print(temp)
    #print(y_pred)
    y_pred = temp.max(1)[1].detach().cpu().clone().numpy()
    y_test = local_labels.numpy()

print(y_test)
df = pd.DataFrame({'Y': y_test, 'YHat': y_pred})
df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
print("Accuracy on Test Data: ", df['Correct'].sum() / len(df))

