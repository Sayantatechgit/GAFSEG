import os
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import random
import json
from PIL import Image
random.seed(42)
import torch
from torchvision import transforms
from utils.randaugment import RandAugmentMC
class FEDDataset(Dataset):
    """ FED Dataset """
    def __init__(self, args, dataset, transform = True, split='train', noise=False):
        self.root_dir = os.path.join(args.img_path,args.data,dataset) 
        self.split_dict = json.load(open(os.path.join(args.split_path, f"{args.data}-{str(args.datasets)}.json"), 'r'))
        self.transform = transform 
        self.trainsize = args.shape
        self.split = split
        self.image_list = self.split_dict[dataset][split]
        print("total {} slices".format(len(self.image_list)))
        
        if self.transform:
            if split == 'train':
                print('Using RandomRotation, RandomFlip')
                self.img_transform = transforms.Compose([
                    transforms.RandomRotation(90, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5])
                    ])
                self.gt_transform = transforms.Compose([
                    transforms.RandomRotation(90, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])
                ])
            
            self.gt_transform = transforms.Compose([
                transforms.ToTensor()])
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        idx = idx % len(self.image_list)
        image = self.rgb_loader(os.path.join(self.root_dir,'image',self.image_list[idx]))
        mask = self.binary_loader(os.path.join(self.root_dir,'mask',self.image_list[idx]))
        image,mask = self.resize(image,mask)
        seed = np.random.randint(42) 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        image = self.img_transform(image)
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        mask = self.gt_transform(mask)
        return image,(mask>0.5).to(torch.float), self.image_list[idx]
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')
    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        h = self.trainsize # max(h, self.trainsize)
        w = self.trainsize # max(w, self.trainsize)
        return img.resize((w, h), Image.Resampling.BILINEAR), gt.resize((w, h), Image.Resampling.NEAREST)
    
    
    
def get_datasplit(args):
    """
    """
    data_root = os.path.join(args.img_path,args.data) # ../data/polyp
    client_datasets = args.datasets
    split_path = args.split_path
    os.makedirs(split_path,exist_ok=True)
    tra,val,test = args.train_val_test # 0.8:0.1:0.1
    split_dict = {}
    for dataset in client_datasets:
        dataset_dir = os.path.join(data_root, dataset, 'image')
        img_list = os.listdir(dataset_dir)
        random.shuffle(img_list)
        n = len(img_list)
        n_train = int(n * tra)
        n_val = int(n * val)
        train_list = img_list[:n_train]
        val_list = img_list[n_train:n_train + n_val]
        test_list = img_list[n_train + n_val:]

        split_dict[dataset] = {
            'train': train_list,
            'val': val_list,
            'test': test_list
        }

    split_file = os.path.join(split_path, f"{args.data}-{str(args.datasets)}.json")
    with open(split_file, 'w') as f:
        json.dump(split_dict, f, indent=2)
    print(f"Saved split file to {split_file}")
    return split_dict
        
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.img_path = './data'
    args.data = 'polyp'
    args.datasets = ['ClientA', 'ClientB', 'ClientC', 'ClientD', 'ClientE']
    args.shape = 384
    args.train_val_test = (0.8, 0.1, 0.1)
    args.split_path = './split_data'

    get_datasplit(args)
    split_dict = json.load(open(os.path.join(args.split_path, f"{args.data}-{str(args.datasets)}.json"), 'r'))
    print(split_dict.keys())

    ds = FEDDataset(args=args, dataset='ClientA', transform=True, split='train')
    for i in range(2):
        img, mask, name = ds[i]
        import cv2
        cv2.imwrite('./sample_img.jpg', ((img + 1) / 2 * 255).permute(1, 2, 0).numpy()[:, :, ::-1])
        cv2.imwrite('./sample_mask.jpg', (mask * 255).squeeze(0).numpy())
        print(name)