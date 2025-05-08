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
from stnae_copy import *
import config as cfg

parser = argparse.ArgumentParser()
parser.add_argument("model_train", help="Type of model training 1. ST-AE1 for normal training 2. ST-AE+2* for frozen training", type=str)
parser.add_argument("batch_size", help = "The batch size for training", type =int)
parser.add_argument("epoch", help = "The epochs for training", type =int)
args = parser.parse_args()

data_dir=cfg.data_dir
class_0=cfg.class0

#read data

D, x_train, x_test_shuffle, x_val = read_data(data_dir, class_0)

# Model with STN and AE



img = D[class_0]['test']['crack'][0]

H,W,_ = img.shape
inputs = Input(shape=img.shape)
print(img.shape)
loc_head = create_localization_head(inputs)
x = spatial_transform_input(inputs,loc_head.output)
x = run_model_AE_1(x,(H,W,1),30)

tf.config.run_functions_eagerly(True)
model_STN_AE = tf.keras.Model(inputs = inputs, outputs = x)
model_STN_AE.compile(optimizer = 'adam', loss = 'binary_crossentropy', run_eagerly=True)



model_STN_AE.fit(x = x_train, y = x_train,epochs = args.epoch, batch_size = args.batch_size, validation_data=(x_val,x_val))

model_STN_AE.save(cfg.model_save_path)

#prediction

prediction = model_STN_AE.predict(x_test_shuffle)

prediction_inv = (1 - prediction)
x_test_inv = (1 - x_test_shuffle)
prediction_inv_scaled = np.zeros(prediction.shape)
for i in range(len(prediction)) :
    prediction_inv_scaled[i,:,:,:] = prediction_inv[i]*np.max(x_test_inv[i])/np.max(prediction_inv[i])*255
    prediction_inv_scaled_2 = prediction_inv[i]*np.max(x_test_inv[i])/np.max(prediction_inv[i])


debug_tool(prediction_inv_scaled, x_test_inv, prediction_inv_scaled_2)
