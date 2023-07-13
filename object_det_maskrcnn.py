import os
from typing import NamedTuple
from typing import Union
import time
import tqdm
import json
import glob
import sys
import urllib
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageColor as ImageColor
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from torchvision import datasets, transforms
import torch
import cv2
import numpy as np
import random

from image_utils import download_image_file, draw_bounding_box_on_image

class Dataset_cctv(torch.utils.data.Dataset):
    def __init__(self):
        self.normalize_rgb_mean = [0.485, 0.456, 0.406]
        self.normalize_rgb_std = [0.229, 0.224, 0.225]
        self.transform_op = transforms.Normalize(self.normalize_rgb_mean, self.normalize_rgb_std)
        self.transformations = transforms.Compose(self.transform_op)
    def __getitem__(self, inx):
        image = self.transformations(image)

    def transform(self, image):
        return self.transformations(image)

dataset = Dataset_cctv()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")#pretrained=True)#(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.to(device)
model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)

result_path = '/notebooks/cctv_cv' #os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

url = "http://74.82.29.209:9000//datasets/cctv/cctv_examples2/congestion-1080p_0164.jpg"
result, image_location = download_image_file(url, image_location=result_path, remove_prev=False)
# image = Image.open(image_location)



COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
 
def get_prediction(img_path, device, threshold=0.1):
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  img = img.to(device)
  pred = model([img])
  pred_score = list(pred[0]['scores'].detach().cpu().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
  masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].detach().cpu().numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
  masks = masks[:pred_t+1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return masks, pred_boxes, pred_class

def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def instance_segmentation_api(img_path, device, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    masks, boxes, pred_cls = get_prediction(img_path=img_path, device=device, threshold=threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, tuple([int(bb) for bb in boxes[i][0]]), tuple([int(bb) for bb in boxes[i][1]]) , color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], tuple([int(bb) for bb in boxes[i][0]]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        plt.figure(figsize=(20,30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    return plt
# masks, pred_boxes, pred_class = get_prediction(img_path=image_location, device=device)
threshold = 0.6
plt = instance_segmentation_api(img_path=image_location, device=device, threshold=threshold)

plt.savefig(os.path.join(result_path, 'mrcnn_' + str(threshold) + '_' +os.path.basename(url)))
pass
images = cv2.imread(image_location)
# images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
images = images.swapaxes(1, 3).swapaxes(2, 3)
images = [dataset.norm_image(image) for image in images]
images = list(image.to(device) for image in images)



with torch.no_grad():
    pred = model(images)

im= images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
im2 = im.copy()
for i in range(len(pred[0]['masks'])):
    msk=pred[0]['masks'][i,0].detach().cpu().numpy()
    scr=pred[0]['scores'][i].detach().cpu().numpy()
    if scr>0.8 :
        im2[:,:,0][msk>0.5] = random.randint(0,255)
        im2[:, :, 1][msk > 0.5] = random.randint(0,255)
        im2[:, :, 2][msk > 0.5] = random.randint(0, 255)
cv2.imshow(str(scr), np.hstack([im,im2]))
cv2.waitKey()