# %%
import torch
from torchvision import transforms
from PIL import Image
import cv2
# from lib.dataset import LoadImages, LoadStreams
image_path = '/notebooks/dataset/cctv/cctv_examples2/traffic'
result_path = '/notebooks/dataset/cctv/lane_det'#'/notebooks/cctv_cv' #os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

img_size = 640
# dataset = LoadImages(source='inference/images', img_size=img_size) #inference/images   inference/videos

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  

model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
model.to(device)
model.eval()

# %%
shapes = opt.img_size
img = Image.open('/notebooks/cctv_cv/road.jpeg')
# img.show()
img = transform(img).to(device)
# t = transforms.ToTensor()
# resize_transform = transforms.Resize((640, 640))
# tensor_to_img = transforms.ToPILImage()


# features = t(img)
# features = resize_transform(features)

# tensor_to_img(features).show()

# print(features.shape)
# features = features.unsqueeze(0)
# print(features.shape)


# %%
# img = torch.randn(1, 3, 640, 640)
# features = features.to(device)

det_out, da_seg_out, ll_seg_out = model(img)

# _, _, height, width = img.shape
# # h,w,_=img_det.shape
# pad_w, pad_h = shapes[1][1]
# pad_w = int(pad_w)
# pad_h = int(pad_h)
# ratio = shapes[1][0][1]
ratio = 1
pad_h = 0
pad_w = 0

ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
_, ll_seg_mask = torch.max(ll_seg_mask, 1)
ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
# Lane line post-processing
#ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
#ll_seg_mask = connect_lane(ll_seg_mask)

img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
cv2.imwrite(save_path,img_det)

image = Image.fromarray(da_seg_out.squeeze().permute(1,2,0).detach().cpu().numpy().astype('uint8'), 'RGB')
# %%
print(det_out)
print(det_out[0].shape)
# %%