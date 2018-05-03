import os
import sys

import skimage
import numpy as np
import time


from mrcnn.visualize import random_colors
from skimage import io

import mrcnn.model as modellib
from mrcnn import utils, visualize

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from coco import CocoConfig

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# URL pointing to the image
url = 'https://www.getnexar.com/images/challenge/China/13.jpg'


class InferenceConfig(CocoConfig):
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

vehicles = ['car', 'bus', 'train', 'truck', 'boat']
living_things = ['person']

original = skimage.io.imread(url)
image = np.copy(original)

# NOTE: change this to switch modes ('roi' or anything else)
mode = 'roi'
# mode = ''

x1, x2, y1, y2 = 0, 0, 0, 0

if mode == 'roi':
    # region of interest
    x1 = 306
    x2 = x1 + 128
    y1 = 197
    y2 = y1 + 96

    image = image[y1:y2, x1:x2]

io.imshow(image)
io.show()

start = time.time()

# Run detection
results = model.detect([image], verbose=1)

end = time.time()

print("Inference time: {:.2f}s".format(end-start))

# Visualize results
r = results[0]

boxes = r['rois']
masks = r['masks']
class_ids = r['class_ids']
scores = r['scores']


def meets_criteria(candidate_class, target_classes, candidate_score, target_score):
    return candidate_class in target_classes and candidate_score >= max(target_score, 0.70)


# Number of instances
N = boxes.shape[0]
colors = random_colors(N)

# Filter indices based on class name and score
idx = [i for i in range(N) if meets_criteria(class_names[class_ids[i]], vehicles, scores[i], 0.70)]

for i in idx:
    mask = masks[:, :, i]
    mask_image = visualize.convert_mask_to_image(image, mask, colors[i])

    if mode == 'roi':
        # create an empty image with the same dimensions as the original one
        mask_image_uncropped = np.zeros(original.shape, dtype=np.uint8)
        # paste the mask onto the empty image
        mask_image_uncropped[y1:y2, x1:x2] = mask_image
        io.imshow(mask_image_uncropped)
    else:
        io.imshow(mask_image)

    io.show()
