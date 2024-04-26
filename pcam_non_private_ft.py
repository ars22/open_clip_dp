#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import torchvision.datasets as datasets

# Specify the root directory where the dataset is located
root = "."

# Create an instance of the PCAM dataset
pcam_train_dataset = datasets.PCAM(root, download=True, split='train')
pcam_test_dataset = datasets.PCAM(root, download=True, split='test')


# In[8]:


from tqdm import tqdm
import torch
import sys 

def train_epoch(train_loader, model, optimizer, criterion, device):

    pbar = tqdm.tqdm(train_loader, desc='Training', total=len(train_loader))
    train_loss, train_acc = AverageMeter(), AverageMeter()

    for x, labels in pbar:
        x = x.to(device)
        labels = labels.to(device)
        logits = model(x)
        losses = criterion(logits, labels)
        acc = (torch.argmax(logits, dim=-1) == labels).float().mean()

        saved_var = dict()
        for p_name, p in model.named_parameters():
            saved_var[p_name] = torch.zeros_like(p)

        model.zero_grad()
        losses.mean().backward()
        optimizer.step()
        train_loss.update(losses.mean().item(), len(x))
        train_acc.update(acc.mean().item(), len(x))
        pbar.set_description(f"Loss: {train_loss.get():.6f} Acc: {train_acc.get():.6f}")


# In[9]:


import torch
from PIL import Image
import open_clip
import tqdm
import pickle
import torchvision.datasets as datasets
from opacus import PrivacyEngine

root = "."

class Model(torch.nn.Module):
    def __init__(self, network, lp):
        super().__init__()
        self.network = network
        self.lp = lp

    def forward(self, x):
        x = self.network.encode_image(x)
        x = self.lp(x)
        return x


batch_size = 100
device = torch.device('cuda:3')

network, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
pcam_train_dataset = datasets.PCAM(root, download=True, split='train', transform=preprocess)
pcam_test_dataset = datasets.PCAM(root, download=True, split='test', transform=preprocess)


train_loader = torch.utils.data.DataLoader(pcam_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(pcam_test_dataset, batch_size=batch_size, shuffle=True)
lp = torch.nn.Linear(in_features=512, out_features=2)
model = Model(network, lp).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

for epoch in range(3):
    train_epoch(train_loader, model, optimizer, criterion, device=device)
                


# 
