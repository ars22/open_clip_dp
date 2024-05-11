import torchvision.datasets as datasets

# Specify the root directory where the dataset is located
root = "."

# Create an instance of the PCAM dataset
pcam_train_dataset = datasets.PCAM(root, download=True, split='train')
pcam_test_dataset = datasets.PCAM(root, download=True, split='test')

from model_utils import AverageMeter

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

state_dict = torch.load('/home/pthaker/open_clip_dp/runs/private_ft/l0.003_e10_b64_c2.0_eps0.1_del1e-10/clip_ft_epoch_0.pt')
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

