import numpy as np
import pandas as pd
import torch
import torchvision

from bs4 import BeautifulSoup as bs
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn_v2
from torch.utils.data import  Dataset
from torchvision import transforms
from typing import List

class SealDataset(Dataset):
    def __init__(self, images, targets, transform=transforms.Compose([transforms.ToTensor()])):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        self.length = len(self.images)
        return self.length

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        return self.transform(image), target


def get_object_detection_model(version, num_classes=2, untrained=False, unfrozen=True, path=None, size=150):
    """
    Inputs
        num_classes: int
            Number of classes to predict. Must include the 
            background which is class 0 by definition!
        feature_extraction: bool
            Flag indicating whether to freeze the pre-trained 
            weights. If set to True the pre-trained weights will be  
            frozen and not be updated during.
    Returns
        model: FasterRCNN
    """
    if version == 2:
        if untrained:
            model = fasterrcnn_resnet50_fpn_v2(trainable_backbone_layers = 5, num_classes=2)
        else:
            model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    elif version == 1:
        if untrained:
            model = fasterrcnn_resnet50_fpn()
        else:
            model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                weights_backbone='IMAGENET1K_V1',
                trainable_backbone_layers = 5,
                )    
            
    # If False, the pre-trained weights will be frozen.
    for p in model.parameters():
        p.requires_grad = unfrozen    

    # Replace the original 91 class top layer with a new layer
    # tailored for num_classes.
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)    
    if path is not None:
            model.load_state_dict(torch.load(path))
            model.eval()

    model.transform.min_size = (size, )
    model.transform.max_size = size
    return model

def decode_prediction(prediction, 
                      score_threshold = 0.8, 
                      nms_iou_threshold = 0.2,
                      use_numpy=False):
    """
    Inputs
        prediction: dict
        score_threshold: float
        nms_iou_threshold: float
    Returns
        prediction: tuple
    """
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]    
    # Remove any low-score predictions.
    if score_threshold is not None:
        want = scores > score_threshold
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]    
    # Remove any overlapping bounding boxes using NMS.
    if nms_iou_threshold is not None:
        want = torchvision.ops.nms(boxes = boxes, scores = scores, iou_threshold = nms_iou_threshold)
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]

        if use_numpy:
            return (boxes.numpy(), labels.numpy(), scores.numpy())
        else:
            return (boxes.detach().cpu(), labels.detach().cpu(), scores.detach().cpu())

def get_images_target(img_data, bb_data, threshold=.3):

    images = []
    targets = []
    total_bounding_boxes = 0
    errors = 0
    filtered_out = 0
    for idx in range(len(bb_data)):
        
        data_frame = bb_data[idx]
        sub_image = img_data[idx]

        if data_frame is not None:

            boxes = []
            labels = []

            total_bounding_boxes += data_frame.shape[0]
            
            for i in range(data_frame.shape[0]):

                row = data_frame.iloc[i]

                if row.xmin < row.xmax and row.ymin < row.ymax: 
                    if row.percent_seal >= threshold:
                        boxes.append(
                            [
                                row.xmin,
                                row.ymin,
                                row.xmax,
                                row.ymax,
                            ]
                        )
                        labels.append(1)
                    else:
                        filtered_out += 1
                else:
                    errors += 1

            if len(boxes) > 0:
                targets.append(
                    {
                        "boxes": torch.tensor(boxes),
                        "labels": torch.tensor(labels)
                    }
                )
                images.append(sub_image)

    print(f"Total Bounding Boxes: {total_bounding_boxes}")
    print(f"Bounding Boxes with Errors: {errors}")
    print(f"Bounding Boxes Filtered Out: {filtered_out}")

    return images, targets


def predict(model, image, device="cpu", transform = transforms.Compose([transforms.ToTensor()])):
    image = transform(np.array(image)).unsqueeze(0).type(torch.FloatTensor).to(device)
    pred = model(image)[0]
    return pred


def detach_pred(pred):
    pred["boxes"] = pred["boxes"].detach().cpu()
    pred["scores"] = pred["scores"].detach().cpu()
    pred["labels"] = pred["labels"].detach().cpu()
    return pred


def write_to_latex(df, file_name, long_table=False):
    f = open(f"{file_name}.txt", "w")
    f.write(df.to_latex(index=False, longtable=long_table))
    f.close()


def parse_xml(xml) -> pd.DataFrame:
   """Creates a dataframe from an xml file containing bounding box information for that image

   Args:
       xml (XML file): The xml file containing bounding box information

   Returns:
       pd.DataFrame: Information about bounding boxes stored as pandas dataframe
   """
   # Find the relevant attributes
   label = xml.find_all("name")
   xmin = xml.find_all("xmin")
   ymin = xml.find_all("ymin")
   xmax = xml.find_all("xmax")
   ymax = xml.find_all("ymax")

   # There is sometimes an error where one of the attributes is missing
   min_size = min(len(label), len(xmin), len(ymin), len(xmax), len(ymax))
  
   # Extract text
   for i in range(min_size):
      label[i] = label[i].text
      xmin[i] = xmin[i].text
      ymin[i] = ymin[i].text
      xmax[i] = xmax[i].text
      ymax[i] = ymax[i].text
      
   # Create dataframe with all bounding box infomation
   df = pd.DataFrame(
      {
         "label": label[:min_size], 
         "xmin": xmin[:min_size], 
         "ymin": ymin[:min_size], 
         "xmax": xmax[:min_size], 
         "ymax": ymax[:min_size]}
     )
   
   return df

def get_bb(in_path:str, xml:List[str]) -> pd.DataFrame:
   """_summary_

   Args:
       in_path (str): Path to directory containing XML files
       xml (List[str]): List of XML files to open

   Returns:
       pd.DataFrame: A dataframe containing bounding box information
   """
   df = pd.DataFrame()
   i = 0
   
   for x in xml:
      f = open(in_path + x)
      xml_file = bs("".join(f.readlines()), "lxml")
      
      df_temp = parse_xml(xml_file)
      df_temp.insert(0, "file_num", str(i).zfill(4))
      df = pd.concat([df, df_temp])
      
      f.close()
      
      i+=1
   return df