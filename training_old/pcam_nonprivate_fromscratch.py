import os
import sys
sys.path.append(".")
from src import open_clip_nonprivate as open_clip
print(open_clip.__file__)

import torch
import torch.distributed as dist
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import logging

from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image
import tqdm
import pickle

import convert_pcam
from convert_pcam import pcam_train_dataset, pcam_test_dataset, load_and_convert

root = "."

class Model(torch.nn.Module):
    def __init__(self, network, lp):
        super().__init__()
        self.network = network
        self.lp = lp

    def forward(self, x):
        x = self.network(x)
        x = self.lp(x)
        return x

# Create an instance of the PCAM dataset
pcam_train_dataset = datasets.PCAM(root, download=True, split='train')
pcam_test_dataset = datasets.PCAM(root, download=True, split='test')

#world_size = torch.cuda.device_count()
world_size=1

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
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

import torch.nn as nn
import tqdm as tq

# xavier initialization
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        print(f'using xavier initialization on weights of: {str(m)}')
        nn.init.xavier_uniform_(m.weight.data)

def init_training(rank, lr, epochs, batch):
    network, _, preprocess = open_clip.create_model_and_transforms('ViT-S-32')
    network = network.visual # Train vision only

    pcam_train_dataset = datasets.PCAM(root, download=True, split='train', transform=preprocess)
    #sampler = DistributedSampler(pcam_train_dataset, num_replicas=world_size,
    #                             rank=rank, shuffle=False, drop_last=False)

    # train_x, train_y = load_and_convert(pcam_train_dataset)

    # print('>>>>>> Moving to GPU')
    # train_x = torch.from_numpy(train_x).to(rank)
    # train_y = torch.from_numpy(train_y).to(rank)
    # transform = convert_pcam.transform.to(rank)

    # print('>>>>>> Moved to GPU')
    
    batch_size = batch // world_size
    train_loader = torch.utils.data.DataLoader(pcam_train_dataset, batch_size)
    lp = torch.nn.Linear(in_features=384, out_features=2)
    model = Model(network, lp)
    model = DDP(model)

    # random initialization
    for tensor in model.parameters():
        torch.nn.init.normal_(tensor.data, mean=0.0, std=0.02)

    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )

    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=epochs * len(train_loader),
        cycle_mult=1.0,
        max_lr=lr,
        min_lr=0,
        warmup_steps=2000
    )
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, optimizer, lr_scheduler, train_loader, preprocess

def train_pcam(rank, lr, epochs, batch):
    setup(rank, world_size)

    model, optimizer, lr_scheduler, train_loader, preprocess = init_training(rank, lr, epochs, batch)
    print('Rank is', rank)
    model.to(rank)
    model.train()

    num_epochs = epochs
    folder_prefix = 'scratch_models_nonpriv/l{}_ep{}_b{}/'.format(lr, epochs, batch)
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    model_prefix = 'clip_ft_epoch_'
    metadata_file = 'train_log.txt'

    if rank == 0:
        mf = open(folder_prefix + metadata_file, 'w')
        mf.write(f'lr {lr:.6f}, epochs {epochs}, batch {batch}\n')
    
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        #idxs = range(len(train_x) // batch)
        #pbar = tqdm.tqdm(idxs, desc='Training', total=len(idxs))
        pbar = tqdm.tqdm(train_loader, desc='Training', total=len(train_loader))
        for images, labels in pbar:
        #for i in pbar:
            #images = train_x[i*batch:(i+1)*batch]
            #images = transform(images)
            #labels = train_y[i*batch:(i+1)*batch]
            images = images.to(rank)
            labels = labels.to(rank)
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
            if rank == 0:
                mf.write(f"Loss: {train_loss.get():.6f} Acc: {train_acc.get():.6f}\n")

        if rank == 0:
            torch.distributed.barrier()
            torch.save(model.state_dict(), folder_prefix + model_prefix + str(epoch) + '.pt')

    if rank == 0:
        torch.distributed.barrier()
        network, _, preprocess = open_clip.create_model_and_transforms('ViT-S-32')
        pcam_test_dataset = datasets.PCAM(root, download=True, split='test', transform=preprocess)
        test_loader = torch.utils.data.DataLoader(pcam_test_dataset, batch, shuffle=True)
        pbar = tqdm.tqdm(test_loader, desc='Eval', total=len(test_loader))
        test_acc = AverageMeter()
        for images, labels in pbar:
            with torch.no_grad():
                images = images.cuda()
                labels = labels.cuda()
                y_pred = model(images)
                _, predicted = torch.max(y_pred, dim=1)
                accuracy = (predicted == labels).float().mean()
                #print(f'Test Accuracy: {accuracy.item():.4f}')
                test_acc.update(accuracy.mean().item(), len(images))
                pbar.set_description(f"Acc: {test_acc.get():.6f}")
        mf.write(f"Test acc: {test_acc.get():.6f}\n")

    cleanup()
        
import torch.multiprocessing as mp

def main():
    lrs = [1e-6, 1e-5, 1e-4]
    epochs=10
    batch=256
    
    for lr in lrs:
        mp.spawn(
            train_pcam,
            args=(lr, epochs, batch),
            nprocs=world_size,
            join=True)
        
if __name__=='__main__':
    main()
