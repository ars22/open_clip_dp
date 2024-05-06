import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as T

from tqdm import tqdm
import numpy as np

root = '.'

# Create an instance of the PCAM dataset
pcam_train_dataset = datasets.PCAM(root, download=True, split='train')
pcam_test_dataset = datasets.PCAM(root, download=True, split='test')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
mean, std = OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

transform = torch.nn.Sequential(*[
    T.Lambda(lambda imgb: imgb.permute(0, 3, 1, 2)),
    T.Resize(
        224, interpolation=T.InterpolationMode.BICUBIC, antialias=False
    ),
#    T.ToImage(),
    T.ToDtype(torch.float32),
    T.Normalize(mean=mean, std=std),
])

def load_and_convert(dataset):
    x_arr = []
    y_arr = []
    for i, (x, y) in tqdm(enumerate(dataset)):
        x_arr.append(np.array(x))
        y_arr.append(y)

    return np.stack(x_arr), np.stack(y_arr)

