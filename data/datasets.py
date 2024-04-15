import os
import numpy as np
import torch
import PIL.Image as Image
import torch.nn as nn
import torch.utils.data
import json
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import random
import cv2
from pois.select_pois import get_pois_img
import copy
from PIL import Image

class AdvTrainDataset(Dataset):
    def __init__(self, root_dir):
        print('Load dataset from:', root_dir)
        self.root_dir = root_dir
        # self.mean = np.array([0.485, 0.456, 0.406])
        # self.std = np.array([0.229, 0.224, 0.225])

        try:
            dataset = np.load(root_dir, allow_pickle=True).item()
        except:
            dataset = np.load(root_dir, allow_pickle=True)

        self.cln_imgs = dataset['cln_img']
        self.adv_imgs = dataset['adv_img']
        self.pois_imgs = dataset['pois_img']

        self.num = len(self.cln_imgs)
        print('data load done.')

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        cln_img, adv_img, pois_img = self.cln_imgs[idx].copy(), self.adv_imgs[idx].copy(), self.pois_imgs[idx].copy()
        # transform
        return {
            "adv_img": adv_img,
            "cln_img": cln_img,
            "pois_img": pois_img
        }

class AdvTrainDatasetFile(Dataset):
    def __init__(self, root_dir):
        print('Load dataset from:', root_dir)
        self.root_dir = root_dir
        self.tusimple_path = '/mnt/data/ssd/zxw_data/TUSimple/train_set'
        # self.mean = np.array([0.485, 0.456, 0.406])
        # self.std = np.array([0.229, 0.224, 0.225])

        self.data = []
        adv_path = os.path.join(self.root_dir,'adv')
        pois_path = os.path.join(self.root_dir, 'pois')
        label_path = os.path.join(self.root_dir,'meta.txt')
        imglist = os.listdir(adv_path)

        self.do_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
        self.path2label = self.get_labels(label_path=label_path)
        for img in imglist:
            self.data.append((os.path.join(self.tusimple_path, self.path2label[img+'_ori']), self.path2label[img+'_pos'], os.path.join(adv_path, img), os.path.join(pois_path, img)))
        print('len(data) is ', len(self.data))

        self.num = len(self.data)
        print('data load done.')

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        (ori_path, pos, adv_path, pois_path) = self.data[idx]

        ori_img = cv2.imread(ori_path)
        adv_img = cv2.imread(adv_path)
        pois_img = cv2.imread(pois_path)
        cln_img = copy.deepcopy(ori_img)[pos[0]:pos[0]+100, pos[1]:pos[1]+100,:]
        cv2.imwrite('cln_img.jpg', cln_img)
        cv2.imwrite('adv_img.jpg', adv_img)

        # pois_img_all = copy.deepcopy(ori_img)
        # pois_img_all[pos[0]:pos[0]+100, pos[1]:pos[1]+100,:] = pois_img
        # transform
        return {
            "adv_img": self.do_transforms(adv_img),
            "cln_img": self.do_transforms(cln_img),
            "pois_img": self.do_transforms(pois_img),
            "ori_img": self.do_transforms(ori_img),
            "pos_h": pos[0],
            "pos_w": pos[1],
        }
    
    def get_labels(self, label_path):
        path = label_path
        path2label = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                arrs = line.split()
                img_name, ori_name, pos_h, pos_w = arrs[0], arrs[1], int(arrs[2]), int(arrs[3])
                path2label[img_name+'_ori'] = ori_name
                path2label[img_name+'_pos'] = (pos_h, pos_w)
        return path2label

SPLIT_FILES = {
    'train+val': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}

SPLIT_FILES_CULANE = {
    'train': "list/train.txt",
    'val': 'list/val.txt',
    'test': "list/test.txt",
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
    'debug': 'list/debug.txt'
}

class tusimple(Dataset):
    def __init__(self, root_dir, split='train', model_name='laneatt', task_num=10):
        print('Load tusimple dataset from:', root_dir)
        self.root_dir = root_dir
        self.split = split
        self.img_h = 720
        self.img_w = 1280

        self.anno_files = [os.path.join(self.root_dir, path) for path in SPLIT_FILES[split]]

        self.imgs = []
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                for i in range(task_num):
                    self.imgs.append(os.path.join(self.root_dir, data['raw_file']))

        # if self.split == 'train':
        #     random.shuffle(self.imgs)
        self.num = len(self.imgs)

        self.pois_info = {'name':'amorphous_pattern_size100',
                    'size': 100,
                    'point_num': 900,
                    'nizi_num': 1
        }
        self.transforms = transforms.Compose( 
            [
                transforms.ToTensor(),
            ]
        )
        self.model_name = model_name
        print('data load done.')

    def get_img_heigth(self, _):
        return 720

    def get_img_width(self, _):
        return 1280
    
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        ori_img = cv2.imread(img_path)

        size = 100
        point_h = random.randint(0,self.img_h-size)
        point_w = random.randint(0,self.img_w-size)
        pois_img = copy.deepcopy(ori_img)
        ori_img_crop = copy.deepcopy(ori_img)
        ori_img_crop = ori_img_crop[point_h:point_h+100,point_w:point_w+100,:]

        pois_img_crop = copy.deepcopy(ori_img_crop)

        p = random.random()
        if p < 0.4:
            self.pois_info['name'] = "amorphous_pattern_size100"
        elif p < 0.55:
            self.pois_info['name'] = "amorphous_pattern_size100sunlight"
        elif p < 0.7:
            self.pois_info['name'] = "amorphous_pattern_size100shadow"
        elif p < 0.85:
            self.pois_info['name'] = "amorphous_pattern_size100rain"
        else:
            self.pois_info['name'] = "amorphous_pattern_size100snow"
        pois_img_crop, mask = get_pois_img(pois_img_crop, self.pois_info['name'], self.pois_info)
        pois_img_crop = pois_img_crop.astype(np.uint8)
        pois_img[point_h:point_h+100,point_w:point_w+100,:] = pois_img_crop

        all_mask = np.zeros_like(ori_img)
        all_mask[point_h:point_h+100,point_w:point_w+100,:] = 1

        ori_img_crop = self.transforms(ori_img_crop)
        pois_img_crop = self.transforms(pois_img_crop)            

        ori_img = self.transforms(ori_img)
        pois_img = self.transforms(pois_img)

        return ori_img_crop, pois_img_crop, ori_img, pois_img, (point_h, point_w), mask, all_mask

class culane(Dataset):
    def __init__(self, root_dir, split='train', model_name='laneatt', task_num=1):
        print('Load culane dataset from:', root_dir)
        self.root_dir = root_dir
        self.split = split
        self.img_w, self.img_h = 1640, 590

        self.list = os.path.join(self.root_dir, SPLIT_FILES_CULANE[split])

        self.annotations = []
        with open(self.list, 'r') as list_file:
            files = [line.rstrip()[1 if line[0] == '/' else 0::] for line in list_file]
        
        for file in files:
            img_path = os.path.join(self.root_dir, file)
            for i in range(task_num):
                self.annotations.append(img_path)
                
        # if self.split == 'train':
        #     random.shuffle(self.imgs)
        self.num = len(self.annotations)

        self.pois_info = {'name':'amorphous_pattern_size100',
                    'size': 100,
                    'point_num': 900,
                    'nizi_num': 1
        }
        self.transforms = transforms.Compose( 
            [
                transforms.ToTensor(),
            ]
        )
        self.model_name = model_name
        print('data load done.')

    def get_img_heigth(self, _):
        self.img_h

    def get_img_width(self, _):
        self.img_w
    
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_path = self.annotations[idx]
        ori_img = cv2.imread(img_path)

        size = 100
        point_h = random.randint(0,self.img_h-size)
        point_w = random.randint(0,self.img_w-size)
        pois_img = copy.deepcopy(ori_img)
        ori_img_crop = copy.deepcopy(ori_img)
        ori_img_crop = ori_img_crop[point_h:point_h+100,point_w:point_w+100,:]

        pois_img_crop = copy.deepcopy(ori_img_crop)

        p = random.random()
        if p < 0.4:
            self.pois_info['name'] = "amorphous_pattern_size100"
        elif p < 0.55:
            self.pois_info['name'] = "amorphous_pattern_size100sunlight"
        elif p < 0.7:
            self.pois_info['name'] = "amorphous_pattern_size100shadow"
        elif p < 0.85:
            self.pois_info['name'] = "amorphous_pattern_size100rain"
        else:
            self.pois_info['name'] = "amorphous_pattern_size100snow"
        pois_img_crop, mask = get_pois_img(pois_img_crop, self.pois_info['name'], self.pois_info)
        pois_img_crop = pois_img_crop.astype(np.uint8)
        pois_img[point_h:point_h+100,point_w:point_w+100,:] = pois_img_crop

        all_mask = np.zeros_like(ori_img)
        all_mask[point_h:point_h+100,point_w:point_w+100,:] = 1

        ori_img_crop = self.transforms(ori_img_crop)
        pois_img_crop = self.transforms(pois_img_crop)            

        ori_img = self.transforms(ori_img)
        pois_img = self.transforms(pois_img)

        return ori_img_crop, pois_img_crop, ori_img, pois_img, (point_h, point_w), mask, all_mask