# export PYTHONPATH=/notebooks/pip_install/
import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
import matplotlib.pyplot as plt

from lib.config import cfg

from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(cfg,opt):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    if 0:
        model = get_net(cfg)
        checkpoint = torch.load(opt.weights, map_location= device)
        model.load_state_dict(checkpoint['state_dict'])
    else:    
        model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    accumulation_ll_seg = True #HK@@
    collect_frames_only = False

    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        if accumulation_ll_seg:
            print(path)
            if i == 0:
                ll_seg_mask_acm = np.zeros((img_det.shape[:2]))# HK@@

        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img) # ll_seg_out : Lane line segment head
        t2 = time_synchronized()
        # if i == 0:
        #     print(det_out)
        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")
        if collect_frames_only:
            if i%300 == 0:
                cv2.imwrite(str(opt.save_dir) + '/' + Path(path).name.split('.')[0] + '_' +str(i) + '.jpg',img_det)
            continue

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        if 1:
            da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # Lane line post-processing
        if 0: # bad results polynom of line lane are curved al over the place 
            if 1:
                ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
            else:
                ll_seg_mask = ll_seg_mask.astype('uint8')
            if 0:
                ll_seg_mask = connect_lane(ll_seg_mask)

        if accumulation_ll_seg :  #np.sort(np.unique((ll_seg_mask_acm/2).astype('int')))[::-1]
            ll_seg_mask_acm = ll_seg_mask_acm + ll_seg_mask

            mask_acm = np.zeros((ll_seg_mask_acm.shape))
            th = np.sort(np.unique(ll_seg_mask_acm.astype('int')))[::-1][1]
            th = max(th, 1)
            if 1:
                th = 1
            # th = max(np.max(ll_seg_mask_acm), th)
            mask_acm[ll_seg_mask_acm>=th] = 1
            mask_acm[ll_seg_mask_acm<th] = 0

            plt.figure(0)
            plt.imshow(mask_acm)
            plt.savefig(str(opt.save_dir) + '/' + Path(path).name.split('.')[0] + '_mask_acm_' + str(i) + '.jpg')
            plt.close()


            img_det = show_seg_result(img_det, (da_seg_mask, mask_acm), _, _, is_demo=True)
            # img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask_acm), _, _, is_demo=True)
        else:
            img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        
        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        
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

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    # parser.add_argument('--source', type=str, default='/notebooks/cctv_cv/YOLOP/inference/images', help='source')  # file/folder   ex:inference/images   inference/videos
# Video
    # parser.add_argument('--source', type=str, default='/notebooks/dataset/cctv/video/congestion', help='source')  # file/folder   ex:inference/images   inference/videos
# Image
    # parser.add_argument('--source', type=str, default='/notebooks/dataset/cctv/cctv_examples2/traffic', help='source')  # file/folder   ex:inference/images   inference/videos
    # parser.add_argument('--source', type=str, default='/notebooks/dataset/cctv/cctv_examples2/traffic/congestion', help='source')  # file/folder   ex:inference/images   inference/videos
    parser.add_argument('--source', type=str, default='/notebooks/dataset/cctv/cctv_examples2/traffic/congestion/cropped_frames_10sec', help='source')  # file/folder   ex:inference/images   inference/videos
    
    parser.add_argument('--save-dir', type=str, default='/notebooks/dataset/cctv/results_connect_lane', help='directory to save results')
    
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')

    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')

    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--save-dir', type=str, default='/notebooks/cctv_cv/YOLOP/inference/output', help='directory to save results')
    
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
