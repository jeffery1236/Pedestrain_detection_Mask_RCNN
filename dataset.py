import os
import numpy as np
import torch as T
from PIL import Image


class PennFudanDataset(object):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, 'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, 'PedMasks'))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'PNGImages', self.imgs[idx])
        mask_path = os.path.join(self.root_dir, 'PedMasks', self.masks[idx])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None] #Split into set of boolean masks
        # masks.shape == (num_obj, H, W)
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0]) #bbox covers entire img segment
            boxes.append((xmin, ymin, xmax, ymax))
        
        boxes = T.as_tensor(boxes, dtype=T.float32)
        labels = T.ones((num_objs,), dtype=T.int64) #there is only 1 class
        masks = T.as_tensor(masks, dtype=T.uint8)

        image_id = T.tensor([idx])
        areas = (boxes[:,3] - boxes[:,1])*(boxes[:,2] - boxes[:,0])
        iscrowd = T.zeros((num_objs,), dtype=T.int64) #assume all instances are not crowds

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['area'] = areas
        target['image_id'] = image_id
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    