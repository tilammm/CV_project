from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch 
import os
import cv2
import pandas as pd
import numpy as np


def _adjust_boxes_format(boxes):
    adjusted_boxes = []
    for box_i in boxes:
        adjusted_box_i = [0, 0, 0, 0]
        adjusted_box_i[0] = box_i[0]
        adjusted_box_i[1] = box_i[1]
        adjusted_box_i[2] = box_i[0] + box_i[2]
        adjusted_box_i[3] = box_i[1] + box_i[3]
        adjusted_boxes.append(adjusted_box_i)
    return adjusted_boxes

def _areas(boxes):
    areas = []
    for box_i in boxes:
        areas.append(box_i[2] * box_i[3])
    return areas


def read_images(imgs_path, lbls_path):
    imgs = []
    lbls = []
    i = 0
    for img in glob.glob(imgs_path + '*.png'):
        imgs.append(img)
        i += 1
    
    j = 0
    for lbl in glob.glob(lbls_path + '*.txt'):
        with open(lbl) as lbl_file:
            array = lbl_file.readlines()
        
        lbls.append(array[2:])
        j+=1
        if j == i:
            break
    
    return imgs, lbls


class DOTADataset(Dataset):
    def __init__(self, images_root_directory,
                 labels_directory,
                 transforms):
        super(DOTADataset).__init__()
        self.images_root_directory = images_root_directory
        self.images, self.labels = read_images(images_root_directory, labels_directory)
        for i in range(len(self.labels)):
            for j in range(len(self.labels[i])):
                self.labels[i][j] = self.labels[i][j].split()
                self.labels[i][j][8] = label_dict[self.labels[i][j][8]]
        self.transforms = transforms
            

    def __getitem__(self, idx):
        image = self.images[idx]
        labels = self.labels[idx]
            
        img = Image.open(image).convert('RGB')
        
        return self.transforms(img), labels

    def __len__(self):
        return len(self.images)