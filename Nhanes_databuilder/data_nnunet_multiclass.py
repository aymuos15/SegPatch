# This code is to convert the NHanes extra labelled data to nnUnet format (Decathlon)
# The NHanes data should be downloaded from https://zenodo.org/records/10223910 with extra refenrece label (e.g. S1, C2) end with .pkl
# If your NVS user the following modules are required to run this file
#   module load scikit-image/0.18.1-fosscuda-2020b
#   module load OpenCV/4.5.1-fosscuda-2020b-contrib

import os
import shutil
import random
import pickle
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
import cv2 
import json
# if GPU too small, resize
target_size = (512, 512)
# define nnUnet json
dataset_json = {
        "name": "Nhanes", 
        "channel_names": { 
            "0": "Xray"
        }, 
        "labels": { 
            "background": 0, 
            'C2': 1,
            'C3': 2,
            'C4': 3,
            'C5': 4,
            'C6': 5,
            'C7': 6,
            'T1': 7,
            'T11': 8,
            'T12': 9,
            'L1': 10,
            'L2': 11,
            'L3': 12,
            'L4': 13,
            'L5': 14,
            'S1': 15
        },
        "file_ending": ".png"
    }
# VU name and index mapping 
C_bodies_mapping = {'C2': 1,
                    'C3': 2,
                    'C4': 3,
                    'C5': 4,
                    'C6': 5,
                    'C7': 6,
                    'T1': 7,
                    'T11': 8,
                    'T12': 9,
                    'L1': 10,
                    'L2': 11,
                    'L3': 12,
                    'L4': 13,
                    'L5': 14,
                    'S1': 15
                    }

# The root path of you nahanes data downloaded from  https://zenodo.org/records/10223910 
nh_path = '/data/users/c9q6m8/NHANES/nhanes/'

# where these images converted to
imageTr_path = '/data/users/c9q6m8/nn_unet_raw/Dataset800_Nhanes/imagesTr/'
labelTr_path = '/data/users/c9q6m8/nn_unet_raw/Dataset800_Nhanes/labelsTr/'
json_path = '/data/users/c9q6m8/nn_unet_raw/Dataset800_Nhanes/'

# List all .pkl files in the folder
pkl_files = [nh_path + file for file in os.listdir(nh_path) if file.endswith('.pkl')]

# image extraction and save
file_count = 0
for pkl in pkl_files:
    print(file_count)
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
        
        # processing image data
        image_array = data['image']
        image_filename = f"NHANES_{file_count:03d}_0000.png"
        # resize it into (512,512)
        image_array = resize(image_array, target_size, preserve_range = True) 
        cv2.imwrite(imageTr_path + image_filename, image_array)
        
        # processing multi-class label data
        masks = data['masks']
        label_filename = f"NHANES_{file_count:03d}.png"
        mask_placeholder = np.zeros(image_array.shape)
        for k in masks.keys():
            if k in C_bodies_mapping:
                # resize it into (512,512)
                mask_ones = resize(masks[k], target_size, preserve_range=True, anti_aliasing = False)
                # mask_ones = masks[k]
                label_index = C_bodies_mapping[k]
                mask_placeholder[mask_ones > 0] = label_index
        cv2.imwrite(labelTr_path + label_filename, mask_placeholder)
    file_count = file_count + 1


dataset_json["numTraining"] = len(os.listdir(imageTr_path))
with open(json_path + 'dataset.json','w') as fp:
        json.dump(dataset_json, fp)
        







