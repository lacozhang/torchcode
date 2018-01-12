#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 22:42:16 2018

@author: zhangyu
"""

from __future__ import print_function
from __future__ import division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import argparse

import warnings
warnings.filterwarnings("ignore")
plt.ion()

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", required=True, dest="datadir", 
                    help="data directory")
FLAGS = parser.parse_args()

landmarks_frame_path = os.path.join(FLAGS.datadir, "face_landmarks.csv")
landmarks_frame = pd.read_csv(landmarks_frame_path)

n = 65

img_name = landmarks_frame.ix[n, 0]
landmarks = landmarks_frame.ix[n, 1:].as_matrix().astype('float')
landmarks = landmarks.reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmark shape: {}'.format(landmarks.shape))
print('First 4 landmarks: {}'.format(landmarks[:4]))

def show_landmarks(image, landmarks):
    """show image with landmarks"""    
    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='.', c='r')
    plt.pause(0.01)


plt.figure()
show_landmarks(io.imread(os.path.join(FLAGS.datadir, img_name)), 
               landmarks)
plt.show()

class FaceLandmarksDataset(Dataset):    
    """Face Landmark dataset."""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied 
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 
                                self.landmarks_frame.ix[idx, 0])
        image = io.imread(img_path)
        landmarks = self.landmarks_frame.ix[idx, 1:].as_matrix().astype('float')
        landmarks = landmarks.reshape(-1, 2)
        sample = {'image' : image, 'landmarks' : landmarks}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

face_dataset = FaceLandmarksDataset(csv_file=landmarks_frame_path, 
                                    root_dir=FLAGS.datadir)

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i, sample['image'].shape, sample['landmarks'].shape)
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)
    
    if i == 3:
        plt.show()
        break

class Rescale(object):
    """Rescale the image in a sample to a given size
    
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is 
        matched to output_size. If int, smaller of image edges is matched 
        to output_size keeping aspect ratio the same time.
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        
        if isinstance(self.output_size, int):
            if h>w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)
        
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image of a sample
    
    Args:
        output_size (tuple or int): Desired output size. if int, square crop
        is made.
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        h,w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        image = image[top: top + new_h,
                      left: left + new_w]
        
        landmarks = landmarks - [left, top]
        
        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert numpy arrays in sample to tensor of pytorch"""
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        image = image.transpose(2, 0, 1)
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256), 
                               RandomCrop(224)])
fig = plt.figure()
sample = face_dataset[65]

for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)
    
    ax = plt.subplot(1, 3, i+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()    

transformed_dataset = FaceLandmarksDataset(landmarks_frame_path,
                                           FLAGS.datadir,
                                           transform=transforms.Compose([
                                                   Rescale(256),
                                                   RandomCrop(224),
                                                   ToTensor()
                                                   ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    
    print(i, sample['image'].size(), sample['landmarks'].size())
    
    if i == 3:
        break

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=2)
o
