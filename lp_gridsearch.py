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

train_image_features = torch.load('resisc_train_features.pt').cuda()
train_image_features /= train_image_features.norm(dim=-1, keepdim=True)
train_labels = torch.load('resisc_train_labels.pt').cuda()

train_data = torch.utils.data.TensorDataset(train_image_features, train_labels)
# Train the model

def train_lp(lr=1.0, epochs=10, eps=0.1, max_grad_norm=1.0, delta=1e-10):
    model = torch.nn.Linear(in_features=len(train_image_features[0]), out_features=45).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )
    data_loader = torch.utils.data.DataLoader(train_data , batch_size=len(train_data), shuffle=False)

    cp_bound = max_grad_norm
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        #noise_multiplier=sigma,
        max_grad_norm=cp_bound,
        poisson_sampling=True,
        target_delta=delta,
        target_epsilon=eps,
        epochs=epochs
    )

    folder_prefix = 'resisc_lp/lp_lr{}_ep{}_c{}_eps{}/'.format(lr, epochs, max_grad_norm, eps)
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
        
    model_prefix = 'clip_lp_epoch_'

    accuracy=None
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        y_pred = model(train_image_features.float())
        loss = criterion(y_pred, train_labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
            _, predicted = torch.max(y_pred, dim=1)
            accuracy = (predicted == train_labels).float().mean()
            print(f'Accuracy: {accuracy.item():.4f}')

    torch.save(model.state_dict(), folder_prefix + model_prefix + str(epoch) + '.pt')
            
    eps = privacy_engine.accountant.get_epsilon(delta=delta)
    print('Epsilon:', eps)
    
    test_image_features = torch.load('resisc_test_features.pt').cuda()
    test_image_features /= test_image_features.norm(dim=-1, keepdim=True)
    test_labels = torch.load('resisc_test_labels.pt').cuda()

    # Evaluate the model
    test_accuracy = None
    with torch.no_grad():
        y_pred = model(test_image_features.float())
        _, predicted = torch.max(y_pred, dim=1)
        test_accuracy = (predicted == test_labels).float().mean()
        print(f'Test Accuracy: {accuracy.item():.4f}')
    return (accuracy, test_accuracy)

def train_lp_nonprivate(lr=1.0, epochs=10):
    model = torch.nn.Linear(in_features=len(train_image_features[0]), out_features=45).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )
    data_loader = torch.utils.data.DataLoader(train_data , batch_size=len(train_data), shuffle=False)

    folder_prefix = 'resisc_lp/lp_lr{}_ep{}/'.format(lr, epochs)
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
        
    model_prefix = 'clip_lp_epoch_'

    accuracy=None
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        y_pred = model(train_image_features.float())
        loss = criterion(y_pred, train_labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
            _, predicted = torch.max(y_pred, dim=1)
            accuracy = (predicted == train_labels).float().mean()
            print(f'Accuracy: {accuracy.item():.4f}')

    torch.save(model.state_dict(), folder_prefix + model_prefix + str(epoch) + '.pt')
            
    test_image_features = torch.load('resisc_test_features.pt').cuda()
    test_image_features /= test_image_features.norm(dim=-1, keepdim=True)
    test_labels = torch.load('resisc_test_labels.pt').cuda()

    # Evaluate the model
    test_accuracy = None
    with torch.no_grad():
        y_pred = model(test_image_features.float())
        _, predicted = torch.max(y_pred, dim=1)
        test_accuracy = (predicted == test_labels).float().mean()
        print(f'Test Accuracy: {accuracy.item():.4f}')
    return (accuracy, test_accuracy)

def gridsearch(outfile_name):
    epochs = [2000]
    lr = [1e-1]
    eps = [0.3, 0.4]
    clip = [1.0]

    f = open(outfile_name, 'w')
    
    grid = itertools.product(eps, epochs, lr, clip)

    results = []

    f.write('LR\tEpochs\tEps\tClip\tTrainAcc\tTestAcc\n')
    for (eps, e, l, c) in grid:
        (train_acc, test_acc) = train_lp(l, e, eps, c)
        print(train_acc, test_acc)
        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(l, e, eps, c, train_acc, test_acc))
        f.flush()

        results.append((l, e, eps, c, train_acc, test_acc))

    results.sort(key = lambda x: x[5], reverse=True)

    print('Top:', results[:5])
        
def gridsearch_nonprivate(outfile_name):
    epochs = [2500]
    lr = [2.0]

    f = open(outfile_name, 'w')
    
    grid = itertools.product(epochs, lr)

    results = []

    f.write('LR\tEpochs\tTrainAcc\tTestAcc\n')
    for (e, l) in grid:
        (train_acc, test_acc) = train_lp_nonprivate(l, e)
        print(train_acc, test_acc)
        f.write('{}\t{}\t{}\t{}\n'.format(l, e, train_acc, test_acc))
        f.flush()

        results.append((l, e, train_acc, test_acc))

    results.sort(key = lambda x: x[-1], reverse=True)

    print('Top:', results[:5])
        
gridsearch('lp_gridsearch_resisc_priv.csv')
