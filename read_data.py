from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D,Flatten,Reshape,Dropout
from keras import Input, Model
import keras as keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os as os
import pickle as pickle
from datetime import datetime
import random as random
from scipy.signal import correlate2d
import json as json
import math as math
from datetime import datetime
from math import cos, sin, pi, floor, ceil
import skimage as skimage
import skimage.transform as transform
import json as json
import cv2 as cv2
import skimage.measure
from sklearn import metrics
import argparse

def read_data(data_dir, class_abnormal):
   main_dir = data_dir # Change this directory to the one where you extracted all
   # the sub directories containing the images of each class of MVTec ('bottle', 'cable',...)
   # Make sure there is no other directory at this location.
   path_dict = {} # dictionnary containing each path for each sub_directory etc.
   path_dict['path'] = main_dir
   D = {} # Images stored in gray levels and half size (length/2 and width/2)
   D_rgb = {} # Images stored in full RGB dimension
   # if you load everything at once it might be a bit too much memory to store so you can do one at a time :
   for dirs in os.listdir(main_dir)[:1] :
   # or all at once :
   # for dirs in ['carpet'] :
    path_dir = os.path.join(main_dir,dirs)
    if os.path.isdir(path_dir) :
        D[dirs] = {}
        D_rgb[dirs] = {}
        path_dict[dirs] = {}
        path_dict[dirs]['path'] = path_dir
        for sub_dir in os.listdir(path_dir) :
            path_sub_dir = os.path.join(path_dir,sub_dir)
            if os.path.isdir(path_sub_dir) :
                print(sub_dir)
                D[dirs][sub_dir] = {}
                D_rgb[dirs][sub_dir] = {}
                path_dict[dirs][sub_dir] = {}
                path_dict[dirs][sub_dir]['path'] = path_dir
                for sub_cat in os.listdir(path_sub_dir) :
                    print(dirs,sub_dir,sub_cat)
                    cat_path = os.path.join(path_sub_dir,sub_cat)
                    listdir = os.listdir(cat_path)
                    path_dict[dirs][sub_dir][sub_cat] = {}
                    path_dict[dirs][sub_dir][sub_cat]['path'] = cat_path
                    L = len(listdir)
                    paths = []
                    for i,file in enumerate(listdir) :
                        paths += [os.path.join(cat_path,file)]
                        img = cv2.imread(os.path.join(cat_path,file))/255
                        if i == 0 :
                            sz = skimage.measure.block_reduce(img, (2,2,3), np.mean).shape
                            full_size = (L,)+sz
                            print(full_size)
                            M = np.zeros(full_size,dtype='float16')
                        M[i,:,:,:] = skimage.measure.block_reduce(img, (2,2,3), np.mean) # mean-pooling the image to reduce
                        # its size by a factor 2 along the sides and also making the image gray-level
                        
                        img_rgb = cv2.imread(os.path.join(cat_path,file))/255
                        if i == 0 :
                            sz = img_rgb.shape
                            full_size = (L,)+sz
                            print(full_size)
                            M_rgb = np.zeros(full_size,dtype='float32')
                        M_rgb[i,:,:,:] = cv2.cvtColor(np.float32(img_rgb), cv2.COLOR_BGR2RGB)
                        
                    D[dirs][sub_dir][sub_cat] = M 
                    D_rgb[dirs][sub_dir][sub_cat] = M_rgb
                    path_dict[dirs][sub_dir][sub_cat]['images path'] = paths

   # chosen class :
   class_0 = class_abnormal


   # In[9]:

   print(list(D[class_0]))
   L0 = 0
   for key in list(D[class_0]['test']) :
    if key != 'good' :
        L,H,W,P = D[class_0]['test'][key].shape
        L0 += L
   x_test = np.zeros((L0,H,W,P),dtype='float32')
   i = 0
   for key in list(D[class_0]['test']) :
    if key != 'good' :
        L,H,W,P = D[class_0]['test'][key].shape
        x_test[i:i+L,:,:,:] = D[class_0]['test'][key]
        i += L
   L0 = 0
   for key in list(D[class_0]['test']) :
    if key != 'good' :
        L,H,W,P = D_rgb[class_0]['test'][key].shape
        L0 += L
   x_test_rgb = np.zeros((L0,H,W,P),dtype='float32')
   i = 0
   for key in list(D[class_0]['test']) :
    if key != 'good' :
        L,H,W,P = D_rgb[class_0]['test'][key].shape
        x_test_rgb[i:i+L,:,:,:] = D_rgb[class_0]['test'][key]
        i += L
    
   L0 = 0
   '''for key in list(D[class_0]['ground_truth']) :
       L,H,W,P = D[class_0]['ground_truth'][key].shape
       L0 += L
   x_ground_truth = np.zeros((L0,H,W,P),dtype='float32')
   i = 0
   for key in list(D[class_0]['ground_truth']) :
       L,H,W,P = D[class_0]['ground_truth'][key].shape
       x_ground_truth[i:i+L,:,:,:] = D[class_0]['ground_truth'][key]
       i += L
   '''
   L0 = 0
   for key in list(D[class_0]['train']) :
       L,H,W,P = D[class_0]['train'][key].shape
       L0 += L
   x_train = np.zeros((L0,H,W,P),dtype='float32')
   i = 0
   for key in list(D[class_0]['train']) :
       L,H,W,P = D[class_0]['train'][key].shape
       x_train[i:i+L,:,:,:] = D[class_0]['train'][key]
       i += L
   L0 = 0
   for key in list(D[class_0]['train']) :
       L,H,W,P = D_rgb[class_0]['train'][key].shape
       L0 += L
   x_train_rgb = np.zeros((L0,H,W,P),dtype='float32')
   i = 0
   for key in list(D[class_0]['train']) :
       L,H,W,P = D_rgb[class_0]['train'][key].shape
       x_train_rgb[i:i+L,:,:,:] = D_rgb[class_0]['train'][key]
       i += L
   print(x_train.shape)
   x_val = x_train[:2]
   x_train = x_train[2:]

   x_val_rgb = x_train_rgb[:2]
   x_train_rgb = x_train_rgb[2:]

   print(x_train.shape)


   x_test_shuffle = list(np.copy(x_test))
   x_test_rgb_shuffle = list(np.copy(x_test_rgb))

   #x_ground_truth_shuffle = list(np.copy(x_ground_truth))

   A = list(zip(x_test_shuffle,x_test_rgb_shuffle))
   random.shuffle(A)

   x_test_shuffle, x_test_rgb_shuffle = zip(*A)
   x_test_shuffle = np.array(x_test_shuffle)
   x_test_rgb_shuffle = np.array(x_test_rgb_shuffle)
   #x_ground_truth_shuffle = np.array(x_ground_truth_shuffle)
   
   return D, x_train, x_test_shuffle, x_val
