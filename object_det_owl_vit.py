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
# sys.path.insert(0,"/notebooks")
# sys.path.insert(0,"/notebooks/nebula3_playground/detectronv2_region_proposals")
# from nebula3_playground.detectronv2_region_proposals.detectron_det import draw_bounding_box_on_image
from owlvit_service import *
from image_utils import download_image_file, draw_bounding_box_on_image

result_path = '/notebooks/cctv_cv' #os.path.dirname(os.path.dirname(os.path.realpath(__file__)))



try:
    font = ImageFont.truetype('arial.ttf', 24)
except IOError:
    font = ImageFont.load_default()
color = mcolors.rgb_to_hsv(ImageColor.getrgb('red'))
color = (int(color[0]), int(color[1]), int(color[2]))

owl_service = OwlVitService()

# # Inputs
url = "http://74.82.29.209:9000//datasets/cctv/cctv_examples2/congestion-1080p_0164.jpg"
texts = [["A photo of a vehicle"]]
texts = json.dumps(texts)
# texts = np.asarray(texts, dtype=np.object_)
score_threshold = json.dumps(0.01)
outputs = owl_service.get_url_response(url, texts, score_threshold)
result, image_location = download_image_file(url, image_location=result_path, remove_prev=False)
image = Image.open(image_location)

print("Outputs: {}".format(outputs))
vg_reshape_input = [1, 1]
# np.percentile(all_bbox, 80) = 165437.6
all_bbox = list()
for item in outputs[0]:
    cls_obj = item[0]
    bbox = [int(x) for x in item[1]]
    sim_score = item[2]

    box_area = (bbox[2]- bbox[0]) *(bbox[3]- bbox[1])
    all_bbox.append(box_area)

    if box_area < 165437 and sim_score>0.03: # and box_area < 200000:
      print(sim_score)
      draw_bounding_box_on_image(image, bbox[1], bbox[0], bbox[3], bbox[2], 
                                thickness=2, display_str_list='', use_normalized_coordinates=False)

      draw = ImageDraw.Draw(image)
      text_bottom = int(bbox[3]*vg_reshape_input[0])
      bottom = int(bbox[3]*vg_reshape_input[0])
      left = int(bbox[0]*vg_reshape_input[1])
      draw.rectangle(((int(bbox[0]*vg_reshape_input[1]), int(bbox[1]*vg_reshape_input[0])),
                  (int(bbox[2]*vg_reshape_input[1]), int(bbox[3]*vg_reshape_input[0]))), outline='blue')
      
      text_width, text_height = font.getsize(sim_score)
      margin = np.ceil(0.05 * text_height)
      
      draw.text(
          tuple([int(x) for x in (left + margin, text_bottom - text_height - margin)]),
          str(sim_score) + '_' +str(box_area),
          fill='black',
          font=font)

    # draw_bounding_box_on_image(image, bbox[0], bbox[1], bbox[2], bbox[3], thickness=2, display_str_list='', use_normalized_coordinates=False)
if 0:
  plt.hist(all_bbox, bins=30)
  plt.savefig(os.path.join(result_path, 'hist_' +os.path.basename(url)))
image.save(os.path.join(result_path, os.path.basename(url)))

# path = '/notebooks/dataset/cctv/cctv_examples2/traffic'
""".git/
                               ymin,
                               xmin,
                               ymax,
                               xmax,

"""

# filenames = glob.glob(path + '/**/*.jpg', recursive=True)
# print(filenames)


# for file in filenames:

