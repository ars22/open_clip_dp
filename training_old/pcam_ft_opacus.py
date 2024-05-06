import os
import sys
import opacus
sys.path.append(".")
from src import open_clip
print(open_clip.__file__)

import torchvision.datasets as datasets
import torch
from PIL import Image
# import open_clip
import tqdm
import pickle
# from utils import AverageMeter
import torchvision.datasets as datasets
from opacus import PrivacyEngine
from privacy_analysis.compute_privacy_sgm import *

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

    
def finetune(lr, epochs, batch, clip, eps, delta):
    network, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    network = network.visual
    pcam_train_dataset = datasets.PCAM(root, download=True, split='train', transform=preprocess)
    pcam_test_dataset = datasets.PCAM(root, download=True, split='test', transform=preprocess)

    batch_size = batch
    train_loader = torch.utils.data.DataLoader(pcam_train_dataset, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(pcam_test_dataset, batch_size, shuffle=True)
    lp = torch.nn.Linear(in_features=512, out_features=2)
    model = Model(network, lp).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        poisson_sampling=False,
        target_delta=delta,
        target_epsilon=eps,
        epochs=epochs
    )

    print('Using noise multiplier', optimizer.noise_multiplier)

    num_epochs = epochs

    folder_prefix = 'ft_models/finetune_e{}_d{}/l{}_ep{}_b{}_c{}/'.format(eps, delta, lr, epochs, batch_size, clip)
    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)
    else:
        print('Warning: Directory exists')
    
    model_prefix = 'clip_ft_epoch_'
    metadata_file = 'train_log.txt'

    mf = open(folder_prefix + metadata_file, 'w')
    mf.write('Noise {}\n'.format(optimizer.noise_multiplier))
    mf.write('eps {}, delta {}\n'.format(eps, delta))
    mf.write(f'lr {lr:.6f}, epochs {epochs}, batch {batch}, clip {clip}\n')
    
    for epoch in range(num_epochs):
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        pbar = tqdm.tqdm(train_loader, desc='Training', total=len(train_loader))
        for images, labels in pbar:
            images = images.cuda()
            labels = labels.cuda()
            y_pred = model(images)
            loss = criterion(y_pred, labels)
            acc = (torch.argmax(y_pred, dim=-1) == labels).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), len(images))
            train_acc.update(acc.mean().item(), len(images))
            pbar.set_description(f"Loss: {train_loss.get():.6f} Acc: {train_acc.get():.6f}")
            mf.write(f"Loss: {train_loss.get():.6f} Acc: {train_acc.get():.6f}\n")

        torch.save(model.state_dict(), folder_prefix + model_prefix + str(epoch) + '.pt')

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


def main():
    lr = 1e-5
    epochs=5
    batch=32
    
    eps=1.0
    delta=1e-10
    
    clips=[0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0]

    for clip in clips:
        finetune(lr, epochs, batch, clip, eps, delta)
        
if __name__=='__main__':
    main()
