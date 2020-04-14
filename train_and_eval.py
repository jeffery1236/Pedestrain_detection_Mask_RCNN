import torch
import numpy as np
from dataset import PennFudanDataset
from mask_rcnn import get_segmentation_model
from utils import get_transform, collate_fn
from vision.references.detection.engine import train_one_epoch, evaluate


dataset = PennFudanDataset('PennFudanPed/', get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed/', get_transform(train=False))

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).numpy() #shuffled indices

train_set = torch.utils.data.Subset(dataset, indices[:-50])
test_set = torch.utils.data.Subset(dataset_test, indices[-50:])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                        shuffle=True, num_workers=4, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                        shuffle=False, num_workers=4, collate_fn=collate_fn)

model = get_segmentation_model(num_classes=2)
device = torch.device('cpu')
model.to(device) # Initialize MaskRCNN


print('Here')

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

print('here')

epochs = 10
for epoch in range(epochs):

    train_one_epoch(model, optimizer, train_loader, device, epoch)

    lr_scheduler.step()

    evaluate(model, test_loader, device)
