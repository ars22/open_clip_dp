#!/usr/bin/env python
# coding: utf-8

# #### Sanity check 

# In[1]:


import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]


# #### PCAM 

# In[43]:


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


# In[12]:


import torchvision.datasets as datasets

# Specify the root directory where the dataset is located
root = "."

# Create an instance of the PCAM dataset
pcam_train_dataset = datasets.PCAM(root, download=True, split='train')
pcam_test_dataset = datasets.PCAM(root, download=True, split='test')


# In[25]:


len(pcam_train_dataset), len(pcam_test_dataset)
pcam_train_dataset.transform = preprocess
pcam_test_dataset.transform = preprocess


# In[32]:


# camelyon prompts

classes = [
    'lymph node',
    'lymph node containing metastatic tumor tissue',
]

templates = [
    'this is a photo of {classes[0]}',
    'this is a photo of {classes[1]}'
]


# ##### Test 0-shot

# In[51]:


import torch
from PIL import Image
import open_clip
import tqdm

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
temperature = 100.
model = model.cuda()

test_loader = torch.utils.data.DataLoader(pcam_test_dataset, batch_size=32, shuffle=True)
text_features = model.encode_text(tokenizer(templates).cuda())
text_features /= text_features.norm(dim=-1, keepdim=True)
        
acc_meter = AverageMeter() 

progressbar = tqdm.tqdm(test_loader, total=len(test_loader))

for images, labels in progressbar:
    # Encode the image and text
    with torch.no_grad(), torch.cuda.amp.autocast():
        images = images.cuda()
        labels = labels.cuda()
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculate the similarity scores
        text_probs = (temperature * image_features @ text_features.T).softmax(dim=-1)
        acc = text_probs.argmax(dim=-1) == labels
        acc_meter.update(acc.float().mean().item(), len(images))
        progressbar.set_description(f"Accuracy: {acc_meter.get(percentage=True):.4f}")
        # print(text_probs.shape, labels)
    


# ##### Linear probe

# In[ ]:


import torch
from PIL import Image
import open_clip
import tqdm

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.cuda()

train_loader = torch.utils.data.DataLoader(pcam_train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(pcam_test_dataset, batch_size=32, shuffle=True)

progressbar = tqdm.tqdm(train_loader, total=len(train_loader))
train_image_features = []
for images, _ in progressbar:
    # Encode the image and text
    with torch.no_grad(), torch.cuda.amp.autocast():
        images = images.cuda()
        image_features = model.encode_image(images)
        train_image_features.append(image_features.cpu())

train_image_features = torch.cat(train_image_features, dim=0)

import pickle
with open('train_features.pkl', 'wb') as f:
    pickle.dump(train_image_features, f)



progressbar = tqdm.tqdm(test_loader, total=len(test_loader))
test_image_features = []
for images, _ in progressbar:
    # Encode the image and text
    with torch.no_grad(), torch.cuda.amp.autocast():
        images = images.cuda()
        image_features = model.encode_image(images)
        test_image_features.append(image_features.cpu())


test_image_features = torch.cat(test_image_features, dim=0)


import pickle
with open('test_features.pkl', 'wb') as f:
    pickle.dump(test_image_features, f)


# In[ ]:


import pickle
data = pickle.load(open('train_features.pt', 'rb'))
train_data = torch.utils.data.TensorDataset(data[0], data[1])


# In[2]:


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


# In[19]:


import torch
from PIL import Image
import open_clip
import tqdm
import pickle

# from utils import AverageMeter
import torchvision.datasets as datasets

train_image_features = torch.load('train_features.pt').cuda()
train_image_features /= train_image_features.norm(dim=-1, keepdim=True)
train_labels = torch.load('train_labels.pt').cuda()

model = torch.nn.Linear(in_features=len(train_image_features[0]), out_features=2).cuda()

train_data = torch.utils.data.TensorDataset(train_image_features, train_labels)
# Train the model                                                                                                                                                                                                                             
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.)
data_loader = torch.utils.data.DataLoader(train_data , batch_size=len(train_data), shuffle=False)

from opacus import PrivacyEngine


privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=100.1,
    max_grad_norm=1.,
)


num_epochs = 1000
for epoch in ranxge(num_epochs):
    optimizer.zero_grad()

    # Forward pass                                                                                                                                                                                                                            
    y_pred = model(train_image_features.float())
    loss = criterion(y_pred, train_labels)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')


# In[20]:


privacy_engine.accountant.get_epsilon(delta=0.1/len(train_data))


# In[21]:



test_image_features = torch.load('test_features.pt').cuda()
test_image_features /= test_image_features.norm(dim=-1, keepdim=True)
test_labels = torch.load('test_labels.pt').cuda()

# Evaluate the model                                                                                                                                                                                                                          
with torch.no_grad():
    y_pred = model(test_image_features.float())
    _, predicted = torch.max(y_pred, dim=1)
    accuracy = (predicted == test_labels).float().mean()
    print(f'Test Accuracy: {accuracy.item():.4f}')


# In[9]:


model2 = torch.nn.Linear(in_features=len(train_image_features[0]), out_features=2).cuda()
model2(torch.randn(1, len(train_image_features[0])).cuda()).sum().backward()
model2.weight.grad


# In[29]:





# In[1]:


import torchvision.datasets as datasets

# Specify the root directory where the dataset is located
root = "."

# Create an instance of the PCAM dataset
pcam_train_dataset = datasets.PCAM(root, download=True, split='train')
pcam_test_dataset = datasets.PCAM(root, download=True, split='test')


# In[47]:


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


# In[64]:


print([m for m, _ in network.named_modules() if m.endswith('attn')])


# In[1]:


import sys
import opacus
sys.path.append(".")
from src import open_clip
print(open_clip.__file__)


# In[4]:


import torch
from PIL import Image
# import open_clip
import tqdm
import pickle
# from utils import AverageMeter
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



network, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
pcam_train_dataset = datasets.PCAM(root, download=True, split='train', transform=preprocess)
pcam_test_dataset = datasets.PCAM(root, download=True, split='test', transform=preprocess)


train_loader = torch.utils.data.DataLoader(pcam_train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(pcam_test_dataset, batch_size=128, shuffle=True)
lp = torch.nn.Linear(in_features=512, out_features=2)
model = Model(network, lp).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

print("Number of training parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=0.,
    max_grad_norm=100000.,
)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = AverageMeter()
    pbar = tqdm.tqdm(train_loader, desc='Training', total=len(train_loader))
    for images, labels in pbar:
        images = images.cuda()
        labels = labels.cuda()
        y_pred = model(images)
        loss = criterion(y_pred, labels)
        acc = (torch.argmax(y_pred, dim=-1) == labels).float().mean()

        # saved_var = dict()
        # for p_name, p in model.named_parameters():
        #     saved_var[p_name] = torch.zeros_like(p)

        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), len(images))
        train_acc.update(acc.mean().item(), len(x))
        pbar.set_description(f"Loss: {train_loss.get():.6f} Acc: {train_acc.get():.6f}")
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss.get():.6f}')


# In[3]:


opacus.__version__


# 
