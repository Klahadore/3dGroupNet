import os
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import random
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
        test_img = images
        test_mask=masks
        n_slice=random.randint(0, test_mask.shape[2])
        test_img = test_img[0]
        test_mask = test_mask[0]

        test_mask_argmax = np.argmax(test_mask, axis=0)

        plt.figure(figsize=(12, 8))

        plt.subplot(221)
        plt.imshow(test_img[0, :, :, n_slice], cmap='gray')
        plt.title('Image flair')
        plt.subplot(222)
        plt.imshow(test_img[1, :, :, n_slice], cmap='gray')
        plt.title('Image t1ce')
        plt.subplot(223)
        plt.imshow(test_img[2, :, :, n_slice], cmap='gray')
        plt.title('Image t2')
        plt.subplot(224)
        plt.imshow(test_mask_argmax[:, :, n_slice], cmap='tab10')  # Use a colormap for different categories
        plt.title('Mask')

        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.show()