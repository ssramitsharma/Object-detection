
# coding: utf-8

# In[ ]:


import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
#import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob 
from keras.callbacks import History 
import matplotlib
matplotlib.use('Agg')

# In[1]:


import os
os.listdir()


# In[ ]:


ORIG_SIZE = 512


# In[ ]:


DATA_DIR = 'kaggle/input/maskctscan/covid-19-chest-xray-lung-bounding-boxes-dataset'

# Directory to save logs and trained model
ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR,'horse')


# In[ ]:


#os.listdir('../input/covidchestxray/covid-chestxray-dataset/annotations/imageannotation_ai_lung_bounding_boxes.json')


# In[ ]:


#df = pd.read_csv('../input/covidchestxray/covid-chestxray-dataset/annotations')
with open('/home/rsharm2s/cluster/Mask_RCNN/covid-chestxray-dataset/annotations/imageannotation_ai_lung_bounding_boxes.json') as json_file:
    data = json.load(json_file)


# In[ ]:


k = data['annotations']


# In[ ]:


x,y,w,h = [],[],[],[]
for i,j  in enumerate(k):
    if i == 0:
        x.append(j['bbox'][0])
        y.append(j['bbox'][1])
        w.append(j['bbox'][2])
        h.append(j['bbox'][3])
        print(j)
    
    
    


# In[ ]:


k = data['images']
ps = []
for i,j in enumerate(k) :
    ps.append('/home/rsharm2s/cluster/Mask_RCNN/covid-chestxray-dataset/images/'+j['file_name'])
train_dicom_dir = '/home/rsharm2s/cluster/Mask_RCNN/covid-chestxray-dataset/images'

# In[ ]:


def get_dicom_fps(dicom_dir):
    ks = []
    k =  os.listdir(dicom_dir)
    for i in k:
        ks.append(dicom_dir +"/"+ i)
    return ks

def parse_dataset(dicom_dir, anns): 
    image_fps = ps
    image_annotations = {fp: [] for fp in image_fps}
   # print(image_annotations,'ks')
    for index, row in anns.iterrows(): 
        fp = row['patientId']
        
        image_annotations[fp].append(row)
    return image_fps, image_annotations 


# In[ ]:


labels = []
for i,j in enumerate(data['images']):
    labels.append([j['id'],j['file_name']])


# In[ ]:


this = {"A":1}
for i in labels:
    this[i[0]] = i[1]


# In[ ]:


train_dicom_dir = '/home/rsharm2s/cluster/Mask_RCNN/covid-chestxray-dataset/images'


# In[ ]:


k = data['annotations']


# In[ ]:


patient = []
target = []
x,y,w,h = [],[],[],[]
for i,j in enumerate(k):
    
    x.append(j['bbox'][0])
    y.append(j['bbox'][1])
    w.append(j['bbox'][2])
    h.append(j['bbox'][3])
    target.append(j['category_id'])
    patient.append('/home/rsharm2s/cluster/Mask_RCNN/covid-chestxray-dataset/images/'+this[j['image_id']])
    


# In[ ]:


anns = pd.DataFrame(patient,columns=['patientId'])
anns['x'] = x
anns['y'] = y
anns['width'] = w
anns['height'] = h
anns["Target"] = target
anns


# In[ ]:


image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)


# In[ ]:

print(os.listdir(),'dog')

os.chdir('Mask_RCNN')
#!python setup.py -q install
    
print(os.listdir(),'cat')

# In[ ]:
#print(os.path.join(ROOT_DIR, 'Mask_RCNN'))

# Import Mask RCNN
#sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image



# In[ ]:


# The following parameters have been selected to reduce running time for demonstration purposes 
# These are not optimal 

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 3  # background + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    #RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 1
    
config = DetectorConfig()
config.display()


# In[ ]:


class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('pneumonia', 1, 'corona')
        self.add_class('pneumonia', 2, 'no corona')
        
        # add images 
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
            
    
    

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP


# split dataset into training vs. validation dataset 
image_fps_list = list(image_fps)
random.seed(42)
random.shuffle(image_fps_list)

val_size = 20
image_fps_val = image_fps_list[:val_size]
image_fps_train = image_fps_list[val_size:]

print(len(image_fps_train), len(image_fps_val))
# In[ ]:
LEARNING_RATE = 0.000002

# prepare the training dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()
# In[ ]:


model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

## first epochs with higher lr to speedup the learning
model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE*2,
            epochs=1,
            layers='all',
            augmentation=None)  ## no need to augment yet

#print(model.metrics_names)
print("model trained")


history = model.keras_model.history.history
print(history,"hist")
epochs = range(1,len(next(iter(history.values())))+1)
df = pd.DataFrame(history, index=epochs)
df.to_csv("chest-epoch.csv")

model.save("chest.h5")
model.load_weights('chest.h5', by_name=True)
# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)
plt.figure(figsize=(17,5))

plt.subplot(131)
plt.plot(epochs, history["loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(132)
plt.plot(epochs, history["mrcnn_class_loss"], label="Train class ce")
plt.plot(epochs, history["val_mrcnn_class_loss"], label="Valid class ce")
plt.legend()
plt.subplot(133)
plt.plot(epochs, history["mrcnn_bbox_loss"], label="Train box loss")
plt.plot(epochs, history["val_mrcnn_bbox_loss"], label="Valid box loss")
plt.legend()

plt.savefig("chest-losses.png")





