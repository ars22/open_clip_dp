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


# In[5]:


from tqdm import tqdm
import torch
import sys 
sys.path.append("../..")
sys.path.append(".")
from torch.nn.utils import clip_grad_norm_
from privacy_analysis.compute_privacy_sgm import *


def train_epoch(train_loader, model, optimizer, criterion, device, sigma, clipping_bound):

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

        for j in losses:  # for every example in the mini-batch
            model.zero_grad()
            j.backward(retain_graph=True)
            clip_grad_norm_(model.parameters(), clipping_bound)
            for p_name, p in model.named_parameters():
                new_grad = p.grad
                if new_grad is not None:
                    saved_var[p_name].add_(new_grad)

        for p_name, p in model.named_parameters():
            if p.grad is None:
                continue
            if device.type == 'cuda':
                noise = torch.cuda.FloatTensor(p.grad.shape).normal_(0, sigma * clipping_bound)
            else:
                noise = torch.FloatTensor(p.grad.shape).normal_(0, sigma * clipping_bound)
            saved_var[p_name].add_(noise)
            p.grad = saved_var[p_name] / len(x)

        optimizer.step()
        train_loss.update(losses.mean().item(), len(x))
        train_acc.update(acc.mean().item(), len(x))
        pbar.set_description(f"Loss: {train_loss.get():.6f} Acc: {train_acc.get():.6f}")


# In[20]:


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

network, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
pcam_train_dataset = datasets.PCAM(root, download=True, split='train', transform=preprocess)
pcam_test_dataset = datasets.PCAM(root, download=True, split='test', transform=preprocess)


train_loader = torch.utils.data.DataLoader(pcam_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(pcam_test_dataset, batch_size=batch_size, shuffle=True)
lp = torch.nn.Linear(in_features=512, out_features=2)
model = Model(network, lp).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

sigma = 0.5
cp_bound = 0.5
delta = 1e-10
for epoch in range(3):
    train_epoch(train_loader, model, optimizer, criterion, torch.device('cuda'), sigma, cp_bound)
    compute_dp_sgd_privacy(len(pcam_train_dataset), batch_size, sigma, epoch+1, delta)


