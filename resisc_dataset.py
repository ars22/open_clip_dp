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
