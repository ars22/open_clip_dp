
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
        x = self.network(x)
        x = self.lp(x)
        return x


network, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
network = network.visual
pcam_train_dataset = datasets.PCAM(root, download=True, split='train', transform=preprocess)
pcam_test_dataset = datasets.PCAM(root, download=True, split='test', transform=preprocess)

from opacus.validators import ModuleValidator
errors = ModuleValidator.validate(network, strict=True)
print(errors)
print(list(network.named_buffers()))
print(network.__class__)
network = ModuleValidator.fix(network)

train_loader = torch.utils.data.DataLoader(pcam_train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(pcam_test_dataset, batch_size=32, shuffle=True)
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
    noise_multiplier=0.5,
    max_grad_norm=0.5,
    poisson_sampling=False,
)

num_epochs = 10
prefix = 'clip_opacus_ft_epoch_'
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

        # saved_var = dict()
        # for p_name, p in model.named_parameters():
        #     saved_var[p_name] = torch.zeros_like(p)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), len(images))
        train_acc.update(acc.mean().item(), len(images))
        pbar.set_description(f"Loss: {train_loss.get():.6f} Acc: {train_acc.get():.6f}")

        break
        
    torch.save(model.state_dict(), prefix + str(epoch) + '.pt')
    
