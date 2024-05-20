import torch
from PIL import Image
from src import open_clip
import tqdm

import torchvision.datasets as datasets

from model_utils import *

root = '.'

model, _, preprocess = open_clip.create_model_and_transforms(VITB, pretrained=DATACOMP_M)
model = model.cuda()

'''
from wilds import get_dataset
dataset = get_dataset(dataset="fmow", download=False, root_dir='/data/skolawol/wilds-data/')
pcam_train_dataset = dataset.get_subset(
    "train",
    transform=preprocess
)
pcam_test_dataset = dataset.get_subset(
    "id_val",
    transform=preprocess
)
#return (pcam_train_dataset, pcam_test_dataset)
'''

(pcam_train_dataset, pcam_test_dataset) = init_pcam(preprocess=preprocess)
train_loader = torch.utils.data.DataLoader(pcam_train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(pcam_test_dataset, batch_size=64, shuffle=True)

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
with open('pcam_train_features_datacomp.pt', 'wb') as f:
    torch.save(train_image_features, f)

train_image_labels = torch.cat(train_image_labels, dim=0)
with open('pcam_train_labels_datacomp.pt', 'wb') as f:
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
with open('pcam_test_features_datacomp.pt', 'wb') as f:
    torch.save(test_image_features, f)

test_image_labels = torch.cat(test_image_labels, dim=0)
with open('pcam_test_labels_datacomp.pt', 'wb') as f:
    torch.save(test_image_labels, f)
