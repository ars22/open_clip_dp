import torchvision.datasets as datasets
from model_utils import AverageMeter

import sys
import opacus
sys.path.append(".")
from src import open_clip
print(open_clip.__file__)

import torch
from PIL import Image
import tqdm
import pickle
import torchvision.datasets as datasets
from opacus import PrivacyEngine

from model_utils import *

import typer

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

def evaluate(ckpt_path, dataset=None, private=False):
    network, preprocess = init_model(VITB, LAION, private=private)
    if dataset == 'pcam':
        (train_data, test_data) = init_pcam(root, preprocess)
        lp = torch.nn.Linear(in_features=512, out_features=PCAM_LABELS)
    elif dataset == 'fmow':
        (train_data, test_data) = init_fmow(root, preprocess)
        lp = torch.nn.Linear(in_features=512, out_features=FMOW_LABELS)
    elif dataset == 'resisc':
        (train_data, test_data) = init_resisc(root, preprocess)
        lp = torch.nn.Linear(in_features=512, out_features=62)
    else:
        print('Invalid dataset')
        exit(1)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    model = Model(network, lp).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    print("Number of training parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    state_dict = torch.load(ckpt_path)
    state_dict = {k.replace('_module.', '') : v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)

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

def main(path: str,
         private: bool,
         dataset: str):
    evaluate(path, dataset, private)
            
if __name__=='__main__':
    typer.run(main)
