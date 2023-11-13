import os
import torch
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class SegDataset(Dataset):
    def __init__(self, label_dir, img_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.segimg = os.listdir(label_dir)
        self.transform = transform
        self.target_transform = transform

    def __len__(self):
        return len(self.segimg)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.segimg[idx][:-4]+'.jpg')
        label_path = os.path.join(self.label_dir, self.segimg[idx])
        image = read_image(img_path)
        label = read_image(label_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    dataset = SegDataset('E:/ray_workspace/fasterrcnn_desnet/COCODevKit/label_val/class_labels', 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/val2017')
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for  image, label in train_dataloader:
        continue
    