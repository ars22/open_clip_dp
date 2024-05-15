import torch
from PIL import Image
from src import open_clip
import tqdm

import numpy as np

#import torchvision.datasets as datasets
from datasets import load_dataset
from torch.utils.data import Dataset

root = '.'

class ResiscDataset(Dataset):
    def __init__(self, split, transform):
        self.data = load_dataset("timm/resisc45", split=split)
        self.transform = transform

        #self.x_data = torch.stack([self.transform(self.data[idx]['image']) for idx in range(len(self.data))])
        #self.y_data = np.array([self.data[idx]['label'] for idx in range(len(self.data))])
        
    def __getitem__(self, index):
        return self.transform(self.data[index]['image']), self.data[index]['label']

    def __len__(self):
        return len(self.data)


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.cuda()

print(preprocess)

train_dataset = ResiscDataset('train', preprocess)
test_dataset = ResiscDataset('test', preprocess)
#return (train_dataset, test_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

progressbar = tqdm.tqdm(train_loader, total=len(train_loader))
train_image_features = []
train_image_labels = []
for (images, labels) in progressbar:
    # Encode the image and text
    with torch.no_grad(), torch.cuda.amp.autocast():
        images = images.cuda()
        image_features = model.encode_image(images)
        train_image_features.append(image_features.cpu())
        train_image_labels.append(labels)
        
train_image_features = torch.cat(train_image_features, dim=0)
print(train_image_features)
print(len(train_image_features))
with open('resisc_train_features.pt', 'wb') as f:
    torch.save(train_image_features, f)

train_image_labels = torch.cat(train_image_labels, dim=0)
with open('resisc_train_labels.pt', 'wb') as f:
    torch.save(train_image_labels, f)

progressbar = tqdm.tqdm(test_loader, total=len(test_loader))
test_image_features = []
test_image_labels = []
for images, labels in progressbar:
    # Encode the image and text
    with torch.no_grad(), torch.cuda.amp.autocast():
        images = images.cuda()
        image_features = model.encode_image(images)
        test_image_features.append(image_features.cpu())
        test_image_labels.append(labels)

test_image_features = torch.cat(test_image_features, dim=0)
print(len(test_image_features))
with open('resisc_test_features.pt', 'wb') as f:
    torch.save(test_image_features, f)

test_image_labels = torch.cat(test_image_labels, dim=0)
with open('resisc_test_labels.pt', 'wb') as f:
    torch.save(test_image_labels, f)
