import os
import pickle
import torch
from torch.utils.data import Dataset
import albumentations


class vessel_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None):

        self.data = []
        self.mode = mode
        self.is_val = is_val
        self.data_path = os.path.join(path, f"{mode}_pro")
        self.data_file = os.listdir(self.data_path)
        self.img_file = self._select_img(self.data_file)
        for img_file in self.img_file :
            with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
                img = pickle.load(file)
            gt_file = "gt" + img_file[3:]
            with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
                gt = (pickle.load(file))
            self.data.append((img,gt))

        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.data = self.data[:int(split*len(self.data))]
            else:
                self.data = self.data[int(split*len(self.data)):]
        self.transforms = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=1.0),
        ])

    def __getitem__(self, idx):
        img,gt = self.data[idx]
        img = img.transpose(1, 2, 0).copy()
        gt = gt.transpose(1, 2, 0).copy()

        if self.mode == "training" and not self.is_val:
            transformed = self.transforms(image=img, mask=gt)
            img = transformed['image']
            gt = transformed['mask']
        img = img.transpose(2, 0, 1).copy()
        gt = gt.transpose(2, 0, 1).copy()
        
        img = torch.from_numpy(img).float()
        gt = torch.from_numpy(gt).float()

        return img, gt

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)

        return img_list

    def __len__(self):
        return len(self.data)