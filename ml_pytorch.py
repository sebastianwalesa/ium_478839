#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import jovian
import torchvision
import matplotlib
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
import random
import os
import sys


# In[2]:

# EPOCHS = int(sys.argv[1])
EPOCHS = 1000
#load data
dataframe = pd.read_csv("understat.csv")

#choose columns
input_cols=list(dataframe.columns)[4:11]
output_cols = ['position']
input_cols, output_cols


# In[4]:


def dataframe_to_arrays(dataframe):
    dataframe_loc = dataframe.copy(deep=True)
    inputs_array = dataframe_loc[input_cols].to_numpy()
    targets_array = dataframe_loc[output_cols].to_numpy()
    return inputs_array, targets_array

inputs_array, targets_array = dataframe_to_arrays(dataframe)

inputs = torch.from_numpy(inputs_array).type(torch.float)
targets = torch.from_numpy(targets_array).type(torch.float)

dataset = TensorDataset(inputs, targets)


# In[7]:


train_ds, val_ds = random_split(dataset, [548, 136])
batch_size=50
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)


# In[8]:


class Model_xPosition(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size) 
        
    def forward(self, xb): 
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)          
        # Calcuate loss
        loss = F.l1_loss(out,targets) 
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = F.l1_loss(out,targets)   
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean() 
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        if (epoch+1) % 100 == 0 or epoch == num_epochs-1:
            print("Epoch {} loss: {:.4f}".format(epoch+1, result['val_loss']))
            
            
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history


# In[9]:


input_size = len(input_cols)
output_size = len(output_cols)
model=Model_xPosition()


# In[11]:


epochs = EPOCHS
lr = 1e-5
learning_proccess = fit(epochs, lr, model, train_loader, val_loader)


# In[13]:


def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = predictions[0].detach()

    return "Target: "+str(target)+"          Predicted: "+str(prediction)+"\n"


# In[14]:


for i in random.sample(range(0, len(val_ds)), 10):
    input_, target = val_ds[i]
    print(predict_single(input_, target, model),end="")


# In[15]:


with open("result.txt", "a+") as file:
    for i in range(0, len(val_ds), 1):
        input_, target = val_ds[i]
        file.write(str(predict_single(input_, target, model)))

torch.save(model, "Model_xPosition.pkl")
