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


class DOTADataset(Dataset):
    def __init__(self, images_root_directory,
                 images_list,
                 labels_csv_file_name,
                 phase):
        super(DOTADataset).__init__()
        self.images_root_directory = images_root_directory
        self.phase = phase
        self.images_list = images_list
        if self.phase in ["train", "val"]:
            self.labels_dataframe = pd.read_csv(os.path.join(images_root_directory, labels_csv_file_name))

    def __getitem__(self, item):
        sample = {
            "local_image_id": None,
            "image_id": None,
            "labels": None,
            "boxes": None,
            "area": None
        }

        image_id = self.images_list[item]
        image_path = os.path.join(self.images_root_directory,
                                  "train" if self.phase in ["train", "val"] else "test",
                                  image_id + ".jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        sample["local_image_id"] = image_id
        sample["image_id"] = torch.tensor([item])
        if self.phase in ["train", "val"]:
            boxes = self.labels_dataframe[self.labels_dataframe.image_id == image_id].bbox.values.tolist()
            boxes = [eval(box_i) for box_i in boxes]
            areas = _areas(boxes)
            boxes = _adjust_boxes_format(boxes)

            sample["labels"] = torch.ones((len(boxes),), dtype=torch.int64)
            sample["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            sample["area"] = torch.as_tensor(areas, dtype=torch.float32)
        
        return image, sample

    def __len__(self):
        return len(self.images_list)