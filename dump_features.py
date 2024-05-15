import torch
from PIL import Image
from src import open_clip
import tqdm

import torchvision.datasets as datasets

root = '.'

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.cuda()

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
#return (fmow_train_dataset, fmow_test_dataset)

train_loader = torch.utils.data.DataLoader(fmow_train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(fmow_test_dataset, batch_size=32, shuffle=True)

progressbar = tqdm.tqdm(train_loader, total=len(train_loader))
train_image_features = []
train_image_labels = []
for (images, labels, metadata) in progressbar:
    # Encode the image and text
    with torch.no_grad(), torch.cuda.amp.autocast():
        images = images.cuda()
        image_features = model.encode_image(images)
        train_image_features.append(image_features.cpu())
        train_image_labels.append(labels)
        
train_image_features = torch.cat(train_image_features, dim=0)
print(train_image_features)
print(len(train_image_features))
with open('fmow_train_features.pt', 'wb') as f:
    torch.save(train_image_features, f)

train_image_labels = torch.cat(train_image_labels, dim=0)
with open('fmow_train_labels.pt', 'wb') as f:
    torch.save(train_image_labels, f)

progressbar = tqdm.tqdm(test_loader, total=len(test_loader))
test_image_features = []
test_image_labels = []
for images, labels, metadata in progressbar:
    # Encode the image and text
    with torch.no_grad(), torch.cuda.amp.autocast():
        images = images.cuda()
        image_features = model.encode_image(images)
        test_image_features.append(image_features.cpu())
        test_image_labels.append(labels)

test_image_features = torch.cat(test_image_features, dim=0)
print(len(test_image_features))
with open('fmow_test_features.pt', 'wb') as f:
    torch.save(test_image_features, f)

test_image_labels = torch.cat(test_image_labels, dim=0)
with open('fmow_test_labels.pt', 'wb') as f:
    torch.save(test_image_labels, f)
