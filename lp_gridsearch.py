#!/usr/bin/env python
# coding: utf-8

import torch
from PIL import Image

import os
import sys
import opacus
sys.path.append(".")
from src import open_clip
import tqdm
from opacus import PrivacyEngine

import itertools
import numpy as np

train_image_features = torch.load('train_features.pt').cuda()
train_image_features /= train_image_features.norm(dim=-1, keepdim=True)
train_labels = torch.load('train_labels.pt').cuda()

train_data = torch.utils.data.TensorDataset(train_image_features, train_labels)
# Train the model

def train_lp(lr=1.0, epochs=1000, noise_mult = 100.0, max_grad_norm=1.0):
    model = torch.nn.Linear(in_features=len(train_image_features[0]), out_features=2).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    data_loader = torch.utils.data.DataLoader(train_data , batch_size=len(train_data), shuffle=False)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_mult,
        max_grad_norm=max_grad_norm
    )

    folder_prefix = 'linear_probe/lp_lr{}_ep{}_c{}/'.format(lr, epochs, max_grad_norm)
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
        
    model_prefix = 'clip_lp_epoch_'
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        y_pred = model(train_image_features.float())
        loss = criterion(y_pred, train_labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

    torch.save(model.state_dict(), folder_prefix + model_prefix + str(epoch) + '.pt')
            
    eps = privacy_engine.accountant.get_epsilon(delta=0.1/len(train_data))
    print('Epsilon:', eps)
    
    test_image_features = torch.load('test_features.pt').cuda()
    test_image_features /= test_image_features.norm(dim=-1, keepdim=True)
    test_labels = torch.load('test_labels.pt').cuda()

    # Evaluate the model
    accuracy = None
    with torch.no_grad():
        y_pred = model(test_image_features.float())
        _, predicted = torch.max(y_pred, dim=1)
        accuracy = (predicted == test_labels).float().mean()
        print(f'Test Accuracy: {accuracy.item():.4f}')
    return (accuracy, eps)

'''
def get_noise_mult(outfile_name):
    epochs = [10, 100, 500, 1000, 2000]
    noise = 10.**np.arange(1, 5)
    print(noise)

    lr = 0.01
    clip = 1.0
    
    f = open(outfile_name, 'w')
    
    grid = itertools.product(epochs, noise)
    for (e, n) in grid:
        (accuracy, eps) = train_lp(lr, e, n, clip)
        print(accuracy, eps)
        f.write('{} {} {} {} {} {}\n'.format(lr, e, n, clip, accuracy, eps))
        f.flush()

get_noise_mult('eps.csv')
'''

def gridsearch(outfile_name):
    #epochs = [10, 50, 100, 250, 500]
    epochs = [1000, 2000]
    lr = 10.**np.arange(-2, 2)
    noise = 100.0
    #clip = [0.1, 1.0, 5.0]
    clip = [2.0, 5.0, 7.5, 10.0]

    f = open(outfile_name, 'w')
    
    grid = itertools.product(epochs, lr, clip)

    results = []

    for (e, l, c) in grid:
        (accuracy, eps) = train_lp(l, e, noise, c)
        print(accuracy, eps)
        f.write('{} {} {} {} {} {}\n'.format(l, e, noise, c, accuracy, eps))
        f.flush()

        results.append((l, e, noise, c, accuracy, eps))

    results.sort(key = lambda x: x[4], reverse=True)

    print('Top:', results[:5])
        
gridsearch('lp_gridsearch_clip.csv')
