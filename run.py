import os
import sys
import skimage

from matplotlib import pyplot as plt
from mrcnn.visualize import random_colors
from skimage import io

import mrcnn.model as modellib
from mrcnn import utils, visualize

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# URL pointing to the image
url = 'https://www.getnexar.com/images/challenge/middleEast/3.jpg'
url = 'https://www.getnexar.com/images/challenge/middleEast/16.jpg'
url = 'https://www.getnexar.com/images/challenge/middleEast/7.jpg'
url = 'https://www.getnexar.com/images/challenge/middleEast/11.jpg'
url = 'https://www.getnexar.com/images/challenge/China/2.jpg'
url = 'https://www.getnexar.com/images/challenge/China/5.jpg'
url = 'https://www.getnexar.com/images/challenge/China/12.jpg'
url = 'https://www.getnexar.com/images/challenge/China/13.jpg'


# url = 'https://i.imgur.com/liC9cCh.jpg'
# url = 'https://www.getnexar.com/images/challenge/China/18.jpg'
# url = 'https://www.getnexar.com/images/challenge/China/20.jpg'
# url = 'https://www.getnexar.com/images/challenge/westUSA/2.jpg'
# url = 'https://www.getnexar.com/images/challenge/westUSA/6.jpg'
# url = 'https://www.getnexar.com/images/challenge/SouthAmerica/1.jpg'
# url = 'https://www.getnexar.com/images/challenge/SouthAmerica/2.jpg'
# url = 'https://www.getnexar.com/images/challenge/SouthAmerica/9.jpg'
# url = 'https://www.getnexar.com/images/challenge/eastEurope/18.jpg'
# url = 'https://www.getnexar.com/images/challenge/eastEurope/1.jpg'
# url = 'https://www.getnexar.com/images/challenge/CentralAmerica/8.jpg'
# url = 'https://www.getnexar.com/images/challenge/CentralAmerica/9.jpg'
# url = 'https://www.getnexar.com/images/challenge/CentralAmerica/15.jpg'
# url = 'https://www.getnexar.com/images/challenge/CentralAmerica/18.jpg'
# url = 'https://www.getnexar.com/images/challenge/CentralAmerica/3.jpg'


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

image = skimage.io.imread(url)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]

boxes = r['rois']
masks = r['masks']
class_ids = r['class_ids']
scores = r['scores']


def meets_criteria(candidate_class, target_classes, candidate_score, target_score):
    return candidate_class in target_classes and candidate_score >= target_score


N = boxes.shape[0]
colors = random_colors(N)

idx = [i for i in range(N) if meets_criteria(class_names[class_ids[i]], ['car'], scores[i], 0.75)]

for i in idx:
    mask = masks[:, :, i]
    io.imshow(visualize.convert_mask_to_image(image, mask, colors[i]))
    plt.show()
