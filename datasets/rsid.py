import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import matplotlib.pyplot as plt


class Rsid:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='rsid'):
        if validation == 'Haze1k':
            print("=> evaluating Haze1k test set...")
            path = os.path.join(self.config.data.data_dir, 'test')
            filename = 'val_list.txt'
        elif validation == 'rshaze':
            print("=> evaluating rshaze test set...")
            path = os.path.join(self.config.data.data_dir, 'test')
            filename = 'test.txt'
        elif validation == 'rsid':
            print("=> evaluating rsid test set...")
            path = os.path.join(self.config.data.data_dir, 'test')
            filename = 'test.txt'
        elif validation == 'rhdrs':
            print("=> evaluating rhdrs test set...")
            path = os.path.join(self.config.data.data_dir, 'test')
            filename = 'test.txt'
        else:
            print("=> no datasets.")

        train_dataset = DFG_Dataset(os.path.join(self.config.data.data_dir, 'train'),
                                          n=self.config.training.patch_n,
                                          patch_size=self.config.data.image_size,
                                          transforms=self.transforms,
                                          filelist='train.txt',
                                          parse_patches=parse_patches)

        val_dataset = DFG_Dataset(path, n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=filename,
                                        parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class DFG_Dataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True):
        super().__init__()

        self.dir = dir
        train_list = os.path.join(dir, filelist)
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            # gt_names = [i.strip().replace('input', 'gt') for i in input_names]
            gt_names = [i for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def augData(self,data,target):
        #if self.train:
        if 1: 
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return data ,target
    
    # 创建高频滤波器
    def high_pass_filter(self, shape, cutoff):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        mask = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2) > cutoff
        return torch.tensor(mask, dtype=torch.float32)
    
    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = PIL.Image.open(os.path.join(self.dir,'hazy', input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir,'GT', gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir,'GT', gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

        # input_img, gt_img=self.augData(input_img.convert("RGB"), gt_img.convert("RGB"))

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            
            # outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
            #            for i in range(self.n)]
            outputs = []
            for i in range(self.n):
                a = self.transforms(input_img[i])
                b = self.transforms(gt_img[i])

                # 
                input_fft = torch.fft.fft2(a)
                hpf = self.high_pass_filter(a.shape[1:], 15)
                filtered_transform = input_fft * hpf
                fft_info = 0.8*filtered_transform+0.2*input_fft
                

                combined = torch.fft.ifft2(fft_info).real.float()

                outputs.append(torch.cat([combined, b], dim=0))
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
