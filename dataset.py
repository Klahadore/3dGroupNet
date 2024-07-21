import os
import torch
from torch.utils.data import Dataset
import numpy as np

class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
     
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        assert len(self.image_files) == len(self.mask_files)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.load(image_path)
        mask = np.load(mask_path)
        
        image = torch.from_numpy(image).permute(3, 0, 1, 2)
        mask = torch.from_numpy(mask).permute(3, 0, 1, 2)

       # print(image.shape)

        return image, mask
    
# run through it for testing purposes
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    image_dir = './data/train/images'
    mask_dir = "./data/train/masks"

    dataset = SegDataset(image_dir, mask_dir)

    dataloader = DataLoader(dataset, shuffle=True)
    
    for images, masks in dataloader:
        print(images.shape, masks.shape)
        