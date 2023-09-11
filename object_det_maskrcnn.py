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
from scene_graph_builder.cctv_cvs.image_utils import download_image_file, draw_bounding_box_on_image, bb_intersection_over_union
import torch
import torchvision
from torch import Tensor
from torchvision.extension import _assert_has_ops

import os, ssl, json
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["http_proxy"] = "http://127.0.0.1:3128"
os.environ["https_proxy"] = "http://127.0.0.1:3128"
os.environ["ftp_proxy"] = "http://127.0.0.1:3128"
os.environ["socks_proxy"] = "http://127.0.0.1:3128"
os.environ["no_proxy"] = "localhost,127.0.0.0/8,10.0.0.0/8,192.168.2.0/24,7.182.10.178/24,100.95.135.187/24, 7.182.10.178:9000, 7.182.10.19"


# from ..obj_det_utils.dataset import LoadImages, LoadStreams
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from cctv_cvs.obj_det_utils.DemoDataset import LoadImages, LoadStreams
def flatten(lst): return [x for l in lst for x in l]

def remove_from_list_by_index_safe_from_tail(indexes, my_list):
    for index in sorted(indexes, reverse=True):
        del my_list[index]
    return my_list



from scene_graph_builder.objects_vocab_v02 import *
owl_objects_map = ALL_OBJECTS

def _batched_nms_coordinate_trick(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep

def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(nms)
    _assert_has_ops()
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


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

mrcnn_det_vehicle_classes = [COCO_INSTANCE_CATEGORY_NAMES[2], COCO_INSTANCE_CATEGORY_NAMES[3] ,COCO_INSTANCE_CATEGORY_NAMES[4],
                                        COCO_INSTANCE_CATEGORY_NAMES[6], COCO_INSTANCE_CATEGORY_NAMES[8]]

open_vocab_map = {'vehicle':["car", "truck", "fire truck", "motorcycle", "scooter", "bus", "moped"]}

expert_detector_map = {"vehicle":mrcnn_det_vehicle_classes, "person" :"person"}

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
  return masks, pred_boxes, pred_class, pred_score

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
    plt.figure()
    plt.imshow(img)
    plt.xticks()
    plt.yticks()
    plt.show()
    return plt

def plot_detection_over_image(img_path, boxes, pred_cls, rect_th=3, text_size=3, text_th=3, color_box=(0, 255, 0)):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        # rgb_mask = random_colour_masks(masks[i])
        # img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, tuple([int(bb) for bb in boxes[i][0]]), tuple([int(bb) for bb in boxes[i][1]]) , color=color_box, thickness=rect_th)
        cv2.putText(img,pred_cls[i], tuple([int(bb) for bb in boxes[i][0]]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.figure()
    plt.imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    plt.show()
    return plt

# masks, pred_boxes, pred_class, pred_score = get_prediction(img_path=image_location, device=device)
class MrcnnDet():
    def __init__(self, device:str =''):
        self.model_str = 'MaskRCNN_ResNet50_FPN_V2_Weights'
        if not(device):
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
        # self.box_nms_thresh = 0.5
        self.threshold = 0.4
        self.box_nms_thresh = 0.55 #0.7 # defualt
        # self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_V2_Weights)#(weights="DEFAULT")#pretrained=True)#(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_V2_Weights, 
                                                                         box_nms_thresh=self.box_nms_thresh)#(weights="DEFAULT")#pretrained=True)#(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.to(self.device)
        
    def detect(self,img):
        with torch.no_grad():
            self.model.eval()
            masks, boxes, pred_cls, pred_score = get_prediction(model=self.model, img=img, 
                                                    device=self.device, threshold=self.threshold)
        return masks, boxes, pred_cls, pred_score
        # nms_classes=mrcnn_det_vehicle_classes
expert_det_det = MrcnnDet()

def detection_fusion(image_url: str, owl_vit_detections: list,  
                    #  owl_objects_map:list, 
                    #  expert_det_det, 
                     iou_th:float = 0.35, 
                     nms_classes: list=mrcnn_det_vehicle_classes, iou_th_all_owl:float = 0.5,  
                     nms_owl_all_classes: bool=True, plot_fig: bool=True):
    # image = Image.open(image_url).convert('RGB')
    start = time.time()

    assert(iou_th<=1)
    local_temp_path = "/tmp/file.{}".format(os.path.basename(image_url).split('.')[-1])
    stop1 = time.time()
    image = Image.open(local_temp_path).convert('RGB')    
    urllib.request.urlretrieve(image_url, local_temp_path)
    
    image = Image.open(local_temp_path).convert('RGB')
    if plot_fig:
        result_path = '/root/notebooks/vidarts_super_detector/vidarts_advanced/scene_graph_builder/cctv_cvs'
    
    if nms_owl_all_classes:
        stop2 = time.time()
        all_pred_cls = [it[0]  for it in owl_vit_detections]
        all_pred_score = [it[2]  for it in owl_vit_detections]
        # non_nms_cls_lemma = np.unique([owl_objects_map[x] for x in all_pred_cls if owl_objects_map[x] not in nms_classes])
        non_nms_cls_lemma = np.unique([owl_objects_map[x] for x in all_pred_cls])
        all_bboxes = [[(it[1][0], it[1][1]), (it[1][2], it[1][3])]  for it in owl_vit_detections ]
        
        # cls_ind = np.arange(non_nms_cls_lemma.shape[0])
        all_pred_lemma_index_dict = dict()
        all_pred_lemma_index_dict = {x:ix for ix, x in enumerate(non_nms_cls_lemma)}

        # [all_pred_lemma_index_dict[owl_objects_map[x]] = np.where(non_nms_cls_lemma==x)[0] for x in all_pred_cls if owl_objects_map[x] not in nms_classes]
        
        keep = _batched_nms_coordinate_trick(Tensor([flatten(x) for x in all_bboxes]), 
                                      Tensor(all_pred_score), 
                                      idxs=Tensor([all_pred_lemma_index_dict[owl_objects_map[x]] for x in all_pred_cls]), 
                                      iou_threshold=iou_th_all_owl)
        
        if plot_fig:
            nms_diff = set(np.arange(len(all_pred_cls))) - set(keep.numpy())
            nmsed_cls = [owl_vit_detections[x][0] for x in nms_diff]
            replicated_owl = [x for x in owl_vit_detections if x[0] in [owl_objects_map[y] for y in nmsed_cls]]
            bboxes = [[(it[1][0], it[1][1]), (it[1][2], it[1][3])]  for it in replicated_owl ]
            pred_cls = [it[0]  for it in replicated_owl]
            plt = plot_detection_over_image(img_path=local_temp_path, boxes=bboxes, 
                                            pred_cls=pred_cls)# just fpr plotting
            
            plt.savefig(os.path.join(result_path, str(expert_det_det.model_str)  + '_' +\
                        str(os.path.basename(image_url).split('.')[0]) + '_' + \
                        str(expert_det_det.threshold) + '_nms_' + \
                        str(expert_det_det.box_nms_thresh) + '_nms_owl_vit_classes' +os.path.basename(local_temp_path)), dpi=300)
            plt.close()
            
        owl_vit_detections = [owl_vit_detections[x] for x in keep]
        stop3 = time.time()
        print('nms_owl_all_classes: {:6f} seconds'.format(stop3 - stop2)) 

        if plot_fig:# plot the replicated objects after NMsed
            replicated_owl = [x for x in owl_vit_detections if x[0] in [owl_objects_map[y] for y in nmsed_cls]]
            bboxes = [[(it[1][0], it[1][1]), (it[1][2], it[1][3])]  for it in replicated_owl ]
            pred_cls = [it[0]  for it in replicated_owl]
            color_box=(255, 0, 0)
            plt = plot_detection_over_image(img_path=local_temp_path, boxes=bboxes, 
                                            pred_cls=pred_cls, color_box=color_box)# just fpr plotting
            
            plt.savefig(os.path.join(result_path, str(expert_det_det.model_str)  + '_' +\
                        str(os.path.basename(image_url).split('.')[0]) + '_' + \
                        str(expert_det_det.threshold) + '_nms_' + \
                        str(expert_det_det.box_nms_thresh) + '_nms_owl_vit_classes_after_nms' +os.path.basename(local_temp_path)), dpi=300)
            plt.close()
    
    try:
        masks , expert_det_boxes, expert_det_pred_cls, expert_det_pred_score = expert_det_det.detect(image)
           
        if plot_fig:
            # result_path = '/root/notebooks/vidarts_super_detector/vidarts_advanced/scene_graph_builder/cctv_cvs'
            if 0:
                expert_det_pred_cls = [''] * len(expert_det_pred_cls)
            plt = instance_segmentation_api(img_path=local_temp_path, masks=masks, 
                                            boxes=expert_det_boxes, pred_cls=expert_det_pred_cls)# just fpr plotting
            
            plt.savefig(os.path.join(result_path, str(expert_det_det.model_str)  + '_' +\
                        str(os.path.basename(image_url).split('.')[0]) + '_' + \
                        str(expert_det_det.threshold) + '_nms_' + \
                        str(expert_det_det.box_nms_thresh) + '_mrcnn_' +os.path.basename(local_temp_path)), dpi=300)
            plt.close()
        
        
        if plot_fig:
            result_path = '/root/notebooks/vidarts_super_detector/vidarts_advanced/scene_graph_builder/cctv_cvs'
            if 0:# all classes all boxes
                bboxes = [[(it[1][0], it[1][1]), (it[1][2], it[1][3])]  for it in owl_vit_detections]
                pred_cls = [it[0]  for it in owl_vit_detections]
            else: # dedicated classes only!
                bboxes = [[(it[1][0], it[1][1]), (it[1][2], it[1][3])]  for it in owl_vit_detections if owl_objects_map[it[0]] in nms_classes]
                pred_cls = [it[0]  for it in owl_vit_detections if owl_objects_map[it[0]] in nms_classes]
            if 1:
                # th = 200
                # owl_vit_filt_detections = [(bb, cls1) for bb, cls1 in zip(bboxes, pred_cls) if flatten(bb)[1]<th and flatten(bb)[3]<th]
                th_up = 600
                th_dn = 100
                owl_vit_filt_detections = [(bb, cls1) for bb, cls1 in zip(bboxes, pred_cls) if flatten(bb)[1]<th_up and flatten(bb)[3]<th_up and flatten(bb)[1]>th_dn and flatten(bb)[3]>th_dn]
                pred_cls = [it[1]  for it in owl_vit_filt_detections]
                bboxes = [[(it[0][0][0], it[0][0][1]), (it[0][1][0], it[0][1][1])]  for it in owl_vit_filt_detections]
            if 1:# remove class name from bbox
                pred_cls = [''] * len(pred_cls)
                
            plt = plot_detection_over_image(img_path=local_temp_path, boxes=bboxes, 
                                            pred_cls=pred_cls)# just fpr plotting
            
            plt.savefig(os.path.join(result_path, str(expert_det_det.model_str)  + '_' +\
                        str(os.path.basename(image_url).split('.')[0]) + '_' + \
                        str(expert_det_det.threshold) + '_nms_' + \
                        str(expert_det_det.box_nms_thresh) + '_owl_vit_' +os.path.basename(local_temp_path)), dpi=300)
            plt.close()
            
                
            # [(ix, it)  for ix, it in enumerate(owl_vit_detections) if it[0] =='grill']
        # seeking expert_det overelapped with owl 
        class_category_lemma = [k for k,v in expert_detector_map.items() if nms_classes[0] in v][0]
        num_el = len([it[0]  for it in owl_vit_detections if [k for k,v in open_vocab_map.items() if owl_objects_map[it[0]] in v] == [class_category_lemma]])

        print("Num of Given {} objects: {}".format(class_category_lemma, num_el))
        
        all_expert_det_new_obj = list()
        for bbox_mrcnn, cls_mrcnn, pred_score_mrcnn in zip (expert_det_boxes, expert_det_pred_cls, expert_det_pred_score):
            if cls_mrcnn in nms_classes: # to do understand the mapping of nms_classes to expert_detector_map
                class_category_lemma = [k for k,v in expert_detector_map.items() if cls_mrcnn in v][0]
                # owl_vit_detections_expert_det_cls_inx = [ix for ix, obj in enumerate(owl_vit_detections) if owl_objects_map[obj[0]] == cls_mrcnn  ]
                # Use mapped open_vocab_map since sometimes OWL classification is wrong in specific but it is still vehicle 
                # owl_vit_detections_expert_det_cls_inx = [ix for ix, obj in enumerate(owl_vit_detections) if owl_objects_map[obj[0]] == cls_mrcnn  ]
                # If owl vit class group/lemma is one one of the all expert_det related classes overcome miscalssification preventing NMSing 
                if 1:
                    owl_vit_detections_expert_det_cls_inx = [ix for ix, obj in enumerate(owl_vit_detections) if owl_objects_map[obj[0]] in expert_detector_map[class_category_lemma]]
                else:
                    owl_vit_detections_expert_det_cls_inx = [ix for ix, obj in enumerate(owl_vit_detections) if [k for k,v in open_vocab_map.items() if owl_objects_map[obj[0]] in v] == [class_category_lemma]]
                    
                if bool(owl_vit_detections_expert_det_cls_inx):
                    all_iou = list()
                    for inx in owl_vit_detections_expert_det_cls_inx:
                        cls_owlvit, bbox_owlvit, score  =  owl_vit_detections[inx]
                        iou = bb_intersection_over_union(flatten(bbox_mrcnn), bbox_owlvit)
                        area_bbox_owlvit = (bbox_owlvit[0] - bbox_owlvit[2])*(bbox_owlvit[1] - bbox_owlvit[3])
                        all_iou.append((inx, iou))
                    ovrlap_bbox_iou = [(inx, iou) for ix, (inx, iou) in enumerate(all_iou) if iou>iou_th]
                    if ovrlap_bbox_iou: # if there are overlapped boxeds with detector expert
        # Check if OWL unique classes are the same else take MRCNN
                        ovlp_cls_owl = [owl_vit_detections[ele[0]][0] for ele in ovrlap_bbox_iou]
                        cls_fin = cls_mrcnn
                        if ovlp_cls_owl.count(ovlp_cls_owl [0]) == len(ovlp_cls_owl): # only if classes identical
                            # REmove the overlapped classes from owl list 
                            cls_fin = ovlp_cls_owl[0]
                            score = np.array(([owl_vit_detections[ele[0]][2] for ele in ovrlap_bbox_iou])).mean()

                        else:
                            most_likely_owl_class_obj = np.argmax([owl_vit_detections[ele[0]][2] for ele in ovrlap_bbox_iou])
                            cls_fin = ovlp_cls_owl[most_likely_owl_class_obj]
                            score = [owl_vit_detections[ele[0]][2] for ele in ovrlap_bbox_iou][most_likely_owl_class_obj]
                        if plot_fig :
                            print("-- index removed from owl list", [ele[0] for ele in ovrlap_bbox_iou], [owl_vit_detections[ele[0]] for ele in ovrlap_bbox_iou], len(owl_vit_detections))
                        remove_from_list_by_index_safe_from_tail(indexes=[ele[0] for ele in ovrlap_bbox_iou], 
                                                                my_list=owl_vit_detections)
                        if plot_fig :
                            print("--len  index removed from owl list", len(owl_vit_detections))
                    else: # new object to the owl not overlapped 
                        score = pred_score_mrcnn
                        cls_fin = cls_mrcnn
                else:
                    print("expert det added class/detection ")
                    cls_fin = cls_mrcnn
                    score = pred_score_mrcnn
                    
                # owl_vit_detections.append([cls_fin, flatten(bbox_mrcnn), score])
                if plot_fig :
                    print("++index added to owl list",[cls_fin, flatten(bbox_mrcnn), score], len(owl_vit_detections))

                all_expert_det_new_obj.append([cls_fin, flatten(bbox_mrcnn), score]) # to appended to other list otherwise other obj will NMS other with the mfused stricker IOU threshold
        
        owl_vit_detections.extend(all_expert_det_new_obj)
                    
        if plot_fig:
            result_path = '/root/notebooks/vidarts_super_detector/vidarts_advanced/scene_graph_builder/cctv_cvs'
            bboxes = [[(it[1][0], it[1][1]), (it[1][2], it[1][3])]  for it in owl_vit_detections if [k for k,v in open_vocab_map.items() if owl_objects_map[it[0]] in v] == [class_category_lemma]]
            # bboxes = [[(it[1][0], it[1][1]), (it[1][2], it[1][3])]  for it in owl_vit_detections if owl_objects_map[it[0]] in nms_classes]
            # [ix for ix, obj in enumerate(owl_vit_detections) if owl_objects_map[obj[0]] in expert_detector_map[class_category_lemma]]
            [ix for ix, obj in enumerate(owl_vit_detections) if [k for k,v in open_vocab_map.items() if owl_objects_map[obj[0]] in v] == [class_category_lemma]]
            num_el = len([it[0]  for it in owl_vit_detections if [k for k,v in open_vocab_map.items() if owl_objects_map[it[0]] in v] == [class_category_lemma]])

            print("Num of fused {} objects: {}".format(class_category_lemma, num_el))
            pred_cls = [it[0]  for it in owl_vit_detections if owl_objects_map[it[0]] in nms_classes]
            if 0:
                th_up = 600
                th_dn = 100
                owl_vit_filt_detections = [(bb, cls1) for bb, cls1 in zip(bboxes, pred_cls) if flatten(bb)[1]<th_up and flatten(bb)[3]<th_up and flatten(bb)[1]>th_dn and flatten(bb)[3]>th_dn]
                pred_cls = [it[1]  for it in owl_vit_filt_detections]
                bboxes = [[(it[0][0][0], it[0][0][1]), (it[0][1][0], it[0][1][1])]  for it in owl_vit_filt_detections]
            if 1:
                pred_cls = [''] * len(pred_cls)
            plt = plot_detection_over_image(img_path=local_temp_path, boxes=bboxes, 
                                            pred_cls=pred_cls)# just fpr plotting
            plt.savefig(os.path.join(result_path, str(expert_det_det.model_str)  + '_' + \
                        str(os.path.basename(image_url).split('.')[0]) + '_' + \
                        str(expert_det_det.threshold) + '_nms_' + \
                        str(expert_det_det.box_nms_thresh) + '_owl_vit_merge_mrcnn_iou_vit_mrcnn_' + \
                        str(iou_th) + '_' + \
                        os.path.basename(local_temp_path)), dpi=300)
            plt.close()
    except Exception as e:
        print(e)
        print("MRCNN Model failed probably a blackish image CAR detection/NMS is omitted but NMS for OwlVIt still applied !!!")


    stop4 = time.time()
    print('NMS lemma vehicle: {:6f} seconds'.format(stop4 - stop3)) 
    print('all: {:6f} seconds'.format(stop4 - start)) 
    
    pass
        
def detect(img):
    with torch.no_grad():
        model_str = 'MaskRCNN_ResNet50_FPN_V2_Weights'
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
        box_nms_thresh = 0.5
        threshold = 0.2
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_V2_Weights, box_nms_thresh=box_nms_thresh)#(weights="DEFAULT")#pretrained=True)#(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        model.to(device)
        model.eval()
        # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        # predictions = model(x)
        method = 'single_image'
        
        # plt = instance_segmentation_api(img=img, device=device, threshold=threshold)
        masks, boxes, pred_cls, pred_score = get_prediction(model=model, img=img, device=device, threshold=threshold)
    return masks, boxes, pred_cls

def main(opt):
    dataset = Dataset_cctv()

    model_str = 'MaskRCNN_ResNet50_FPN_V2_Weights'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    box_nms_thresh = 0.5
    threshold = 0.2
    
    hybridnets = True
    if hybridnets:
        model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_V2_Weights, box_nms_thresh=box_nms_thresh)#(weights="DEFAULT")#pretrained=True)#(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)
    model.eval()
    
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)
    
    if 0:
        img = torch.randn(1,3,640,384)     
        img = img.to(device)
   
        features, regression, classification, anchors, segmentation = model(img)        

    
    method = 'single_image' #'multi_image'

    result_path = '/notebooks/dataset/cctv/mask_rcnn'#'/notebooks/cctv_cv' #os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # url = "http://74.82.29.209:9000//datasets/cctv/cctv_examples2/congestion-1080p_0164.jpg"
    # result, image_location = download_image_file(url, image_location=result_path, remove_prev=False)
    # image = Image.open(image_location)
    if method == 'single_image':
        image_path = '/notebooks/dataset/cctv/cctv_examples2/traffic'
        filenames = glob.glob(image_path + '/**/*.jpg', recursive=True)
        for image_location in filenames:
            img = Image.open(image_location)

            if hybridnets:
                transform = T.Compose([T.Rezise((640,384)), T.ToTensor()]) # TODO where is the image normalization according to ImageNet ? 
                img = transform(img).unsqueeze(0)
                img = img.to(device)

                features, regression, classification, anchors, segmentation = model(img)        

            # plt = instance_segmentation_api(img=img, device=device, threshold=threshold)
            masks, boxes, pred_cls, pred_score = get_prediction(model=model, img=img, device=device, threshold=threshold)
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
            masks, boxes, pred_cls, pred_score = get_prediction(model=model, img=img, device=device, threshold=threshold)

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
        main(opt)



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


_batched_nms_coordinate_trick(Tensor([flatten(x) for x in expert_det_boxes]), Tensor(expert_det_pred_score), torch.ones(1,len(expert_det_pred_score)).reshape(-1), 0.6)
[x for x in owl_vit_detections if x[1][0]>800 and x[1][2]<1140 and x[1][1]>200 and x[1][3]<550]
[owl_vit_detections[x] for x in owl_vit_detections_expert_det_cls_inx]
[owl_vit_detections[x[0]] for x in ovrlap_bbox_iou]
'''
