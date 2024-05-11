import os
import sys

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, DistributedSampler
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import torchvision.datasets as datasets
import tqdm
import pickle

from opacus import PrivacyEngine

import numpy as np
import random
np.random.seed(42)
random.seed(42)

VITS='ViT-S-32'
VITB='ViT-B-32'
LAION='laion2b_s34b_b79k'

class AverageMeter:
    def __init__(self):
        self.num = 0
        self.val = 0

    def update(self, val, num):
        self.val += val * num
        self.num += num

    def get(self, percentage=False):
        val = self.val / self.num * 100 if percentage else self.val / self.num
        return val

class Model(torch.nn.Module):
    def __init__(self, network, lp):
        super().__init__()
        self.network = network
        self.lp = lp

    def forward(self, x):
        x = self.network(x)
        x = self.lp(x)
        return x

# xavier initialization
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        print(f'using xavier initialization on weights of: {str(m)}')
        torch.nn.init.xavier_uniform_(m.weight.data)
        
def init_model(model_name, pretrained_name=None, private=False):
    if private:
        from src import open_clip
        print(open_clip.__file__)
    else:
        from src import open_clip_nonprivate as open_clip
        print(open_clip.__file__)
        
    if pretrained_name == None:
        network, _, preprocess = open_clip.create_model_and_transforms(model_name)
    else:
        network, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name)
        
    network = network.visual # Train vision only

    return network, preprocess

def init_pcam(root='.', preprocess=None, subsample=-1):
    pcam_train_dataset = datasets.PCAM(root, download=True, split='train', transform=preprocess)
    pcam_test_dataset = datasets.PCAM(root, download=True, split='test', transform=preprocess)

    if subsample > 0:
        print('subsampling', subsample)
        small_pcam_train_dataset = torch.utils.data.Subset(pcam_train_dataset, np.random.choice(len(pcam_train_dataset), subsample, replace=False))
        pcam_train_dataset = small_pcam_train_dataset
    
    return (pcam_train_dataset, pcam_test_dataset)

def init_fmow(preprocess):
    from wilds import get_dataset
    dataset = get_dataset(dataset="fmow", download=False, root_dir='/data/skolawol/wilds-data/')
    fmow_train_dataset = dataset.get_subset(
        "train",
        transform=preprocess
    )
    fmow_test_dataset = dataset.get_subset(
        "id_val",
        transform=preprocess
    )
    return (fmow_train_dataset, fmow_test_dataset)


def priv_init_training(model,
                       lr, epochs, batch, clip,
                       eps, delta,
                       train_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch, shuffle=True, num_workers=12)

    #lp = torch.nn.Linear(in_features=512, out_features=2)
    #model = Model(network, lp).cuda()
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )

    first_cycle_steps = epochs * len(train_loader)
    print(first_cycle_steps // 2)
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=1.0,
        max_lr=lr,
        min_lr=0,
        warmup_steps=first_cycle_steps // 2
    )

    criterion = torch.nn.CrossEntropyLoss()

    print("Number of training parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    privacy_engine = PrivacyEngine()

    #sigma = 0.5
    cp_bound = clip
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        #noise_multiplier=sigma,
        max_grad_norm=cp_bound,
        poisson_sampling=True,
        target_delta=delta,
        target_epsilon=eps,
        epochs=epochs
    )

    return model, optimizer, data_loader, lr_scheduler, privacy_engine

def init_training(model, lr, epochs, batch,
                  train_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch, shuffle=True)

    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )

    first_cycle_steps = epochs * len(train_loader)
    print(first_cycle_steps // 2)
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=1.0,
        max_lr=lr,
        min_lr=0,
        warmup_steps=first_cycle_steps // 2
    )

    criterion = torch.nn.CrossEntropyLoss()

    print("Number of training parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model, optimizer, train_loader, lr_scheduler

#    model, optimizer, lr_scheduler, train_loader, preprocess = init_training(lr, epochs, batch)

def train_loop(model, optimizer, lr_scheduler, epochs, batch, train_loader,
               folder_prefix, device=0, privacy_engine=None):
    model.to(device)
    model.train()
    num_epochs = epochs
    #folder_prefix = 'scratch_models_nonpriv/l{}_ep{}_b{}/'.format(lr, epochs, batch)

    model_prefix = 'clip_ft_epoch_'
    metadata_file = 'train_log.txt'

    mf = open(folder_prefix + metadata_file, 'w')
    
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        pbar = tqdm.tqdm(train_loader, desc='Training', total=len(train_loader))
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            y_pred = model(images)
            loss = criterion(y_pred, labels)
            acc = (torch.argmax(y_pred, dim=-1) == labels).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_loss.update(loss.item(), len(images))
            train_acc.update(acc.mean().item(), len(images))
            pbar.set_description(f"Loss: {train_loss.get():.6f} Acc: {train_acc.get():.6f}")
            mf.write(f"Loss: {train_loss.get():.6f} Acc: {train_acc.get():.6f}\n")
        mf.write(f"epsilon:{privacy_engine.get_epsilon(10e-10)}")
        print(f"epsilon:{privacy_engine.get_epsilon(10e-10)}")

        torch.save(model.state_dict(), folder_prefix + model_prefix + str(epoch) + '.pt')

    return model
        
def eval(model, test_dataset, batch, folder_prefix):
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    metadata_file = 'eval.txt'

    mf = open(folder_prefix + metadata_file, 'w')

    #pcam_test_dataset = datasets.PCAM(root, download=True, split='test', transform=preprocess)
    test_loader = torch.utils.data.DataLoader(test_dataset, 64, shuffle=True)
    pbar = tqdm.tqdm(test_loader, desc='Eval', total=len(test_loader))
    test_acc = AverageMeter()
    for images, labels in pbar:
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()
            y_pred = model(images)
            _, predicted = torch.max(y_pred, dim=1)
            accuracy = (predicted == labels).float().mean()
            # print(f'Test Accuracy: {accuracy.item():.4f}')
            test_acc.update(accuracy.mean().item(), len(images))
            pbar.set_description(f"Acc: {test_acc.get():.6f}")
    mf.write(f"Test acc: {test_acc.get():.6f}\n")

    return test_acc.get()
