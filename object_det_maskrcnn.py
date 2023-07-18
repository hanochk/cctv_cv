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
import argparse
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_Weights
from image_utils import download_image_file, draw_bounding_box_on_image
# from ..obj_det_utils.dataset import LoadImages, LoadStreams
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from cctv_cv.obj_det_utils.DemoDataset import LoadImages, LoadStreams



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
 
def get_prediction(model, img, device, threshold=0.1):
  
  transform = T.Compose([T.ToTensor()]) # TODO where is the image normalization according to ImageNet ? 
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

def instance_segmentation_api(img_path, masks, boxes, pred_cls, rect_th=3, text_size=3, text_th=3):

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

def detect(opt):
    dataset = Dataset_cctv()

    model_str = 'MaskRCNN_ResNet50_FPN_V2_Weights'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    box_nms_thresh = 0.5
    threshold = 0.2
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_V2_Weights, box_nms_thresh=box_nms_thresh)#(weights="DEFAULT")#pretrained=True)#(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)
    model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)
    method = 'multi_image'
    result_path = '/notebooks/dataset/cctv/mask_rcnn'#'/notebooks/cctv_cv' #os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # url = "http://74.82.29.209:9000//datasets/cctv/cctv_examples2/congestion-1080p_0164.jpg"
    # result, image_location = download_image_file(url, image_location=result_path, remove_prev=False)
    # image = Image.open(image_location)
    if method == 'single_image':
        image_path = '/notebooks/dataset/cctv/cctv_examples2/traffic'
        filenames = glob.glob(image_path + '/**/*.jpg', recursive=True)
        for image_location in filenames:
            img = Image.open(image_location)
            # plt = instance_segmentation_api(img=img, device=device, threshold=threshold)
            masks, boxes, pred_cls = get_prediction(model=model, img=img, device=device, threshold=threshold)
            plt = instance_segmentation_api(img_path=image_location, masks=masks, boxes=boxes, pred_cls=pred_cls)# just fpr plotting
            plt.savefig(os.path.join(result_path, str(model_str)  + '_' + str(threshold) + '_nms' + str(box_nms_thresh) + '_' +os.path.basename(image_location)))
            plt.close()
        pass
    elif method == 'multi_image':
        vid_path, vid_writer = None, None
        
        # Normalization done implicitlly 
        # https://github.com/pytorch/vision/blob/09823951fec09215f9efc5d5d31456763da5ae04/torchvision/models/detection/faster_rcnn.py#L231
        # normalize = transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] 
        #     )

        # transform=transforms.Compose([
        #             transforms.ToTensor(),
        #             normalize,
        #         ])
        transform=transforms.Compose([
                    transforms.ToTensor(),
                ])

        dataset = LoadImages(opt.source, img_size=opt.img_size)
        save_path = os.path.join(result_path ,'maskrcnn_' + os.path.basename(dataset.files[0]))
        for i, (path, img, img_det, vid_cap,shapes) in tqdm.tqdm(enumerate(dataset),total = len(dataset)):
            # img = transform(img).to(device)        
            # if img.ndimension() == 3:
            #     img = img.unsqueeze(0)
            masks, boxes, pred_cls = get_prediction(model=model, img=img, device=device, threshold=threshold)

            # img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rect_th=3
            text_size=3
            text_th=3
            for i in range(len(masks)):
                rgb_mask = random_colour_masks(masks[i])
                img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
                cv2.rectangle(img, tuple([int(bb) for bb in boxes[i][0]]), tuple([int(bb) for bb in boxes[i][1]]) , color=(0, 255, 0), thickness=rect_th)
                cv2.putText(img,pred_cls[i], tuple([int(bb) for bb in boxes[i][0]]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

            img_det = img
            if dataset.mode == 'images':
                cv2.imwrite(save_path,img_det)

            elif dataset.mode == 'video':
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    h,w,_=img_det.shape
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(img_det)
            
            else:
                cv2.imshow('image', img_det)
                cv2.waitKey(1)  # 1 millisecond

    else:
        raise 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    # parser.add_argument('--source', type=str, default='/notebooks/cctv_cv/YOLOP/inference/images', help='source')  # file/folder   ex:inference/images   inference/videos
    # parser.add_argument('--source', type=str, default='/notebooks/dataset/cctv/video/congestion', help='source')  # file/folder   ex:inference/images   inference/videos
    # parser.add_argument('--source', type=str, default='/notebooks/dataset/cctv/cctv_examples2/traffic', help='source')  # file/folder   ex:inference/images   inference/videos
    parser.add_argument('--source', type=str, default='/notebooks/dataset/cctv/video/congestion', help='source')  # file/folder   ex:inference/images   inference/videos
    
    
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')

    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')

    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--save-dir', type=str, default='/notebooks/cctv_cv/YOLOP/inference/output', help='directory to save results')
    parser.add_argument('--save-dir', type=str, default='/notebooks/dataset/cctv/results', help='directory to save results')
    
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(opt)

'''
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
'''
