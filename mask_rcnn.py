import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# #Load faster_rcnn(resnet backbone) pretrained on COCO -> input = list of tensors([C, H, W]) with values 0-1
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# # get arguments needed for classifier
# num_classes = 2 #background + person
# in_features = model.roi_heads.box_predictor.cls_score.in_features # number of input features for classifier


# # Replace classifier head
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



def get_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Replace box_predictor head
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, 
                                                        num_classes=num_classes)
    
    # Replace mask_predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_dim = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask,
                                                        dim_reduced=hidden_dim,
                                                        num_classes=num_classes)

    return model
