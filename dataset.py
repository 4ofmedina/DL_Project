from torch.utils.data import Dataset
import torch
import skimage.transform as st
import numpy as np

class TrafficSignal(Dataset):


    def __init__(self, dataframe, transform=None):
        """
        Args:
            data (DataFrame): X for signal image, y for label
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = dataframe
        labels = self.data['y'].astype(np.int).values
        self.data['y'] = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data['X'][idx]
        label = self.data['y'][idx]

        if self.transform:
            sample = {'X': img, 'y': label}
            img, label = self.transform(sample)

        return img, label

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img = sample['X']
        img = st.resize(img,(50, 50))
        img = np.transpose(img, (2,0,1))
        img = torch.from_numpy(img)
        label = torch.from_numpy(np.array(sample['y']))
        return img, label