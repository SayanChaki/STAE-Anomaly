# Code developped by Simon Steinlin, Sayan Chaki, and Thierry Fournel
# from laboratoire Hubert Curien in a partnership with IHRIM, 
# both from Saint-Etienne University, in the framework 
# of the "Rey Ornament Image investigation" project (2021-2025)
# funded by the French National Research Agency (ANR).
# https://ro2i.hypotheses.org/a-propos

# Permission is granted for anyone to copy, use, modify, or distribute this
# program and accompanying programs and documents for any purpose, provided
# this copyright notice is retained and prominently displayed, along with
# a note saying that the original programs are available from our
# gitlab page.
# The programs and documents are distributed without any warranty, express or
# implied.  As the programs were written for research purposes only, they have
# not been tested to the degree that would be advisable in any important
# application.  All use of these programs is entirely at the user's own risk.

# Python implementation of the STAE model for change detection proposed in 
#the paper entitled "In-place change detection and restitution in one-to-one-class image comparison" 
# Part 2/2: Testing the ST-MLP model 

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
from math import cos, sin, pi, floor, ceil, isnan
import skimage as skimage
import skimage.transform as transform
import json as json
import cv2 as cv2
import skimage.measure
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
import json as json


def spatial_transform_input(inputs, transformations):
    grids = generate_normalized_homo_meshgrids(inputs)
    reprojected_grids = transform_grids(transformations, grids,inputs)
    result = bilinear_sample(inputs, reprojected_grids)
    return result

def generate_normalized_homo_meshgrids(inputs):
    batch_size = tf.shape(inputs)[0]
    print(batch_size)
    _, H, W,_ = inputs.shape
    x_range = tf.range(W)
    y_range = tf.range(H) 
    x_mesh, y_mesh = tf.meshgrid(x_range, y_range)
    x_mesh = (x_mesh/W-0.5)*2
    y_mesh = (y_mesh/H-0.5)*2
    y_mesh = tf.reshape(y_mesh, (*y_mesh.shape,1))
    x_mesh = tf.reshape(x_mesh, (*x_mesh.shape,1))
    ones_mesh = tf.ones_like(x_mesh)
    homogeneous_grid = tf.concat([x_mesh, y_mesh, ones_mesh],-1)
    homogeneous_grid = tf.reshape(homogeneous_grid, (-1, 3,1))
    homogeneous_grid = tf.dtypes.cast(homogeneous_grid, tf.float32)
    homogeneous_grid = tf.expand_dims(homogeneous_grid, 0)
    return tf.tile(homogeneous_grid, [batch_size, 1,1,1])

def transform_grids(transformations, grids, inputs):
    R = transformations[:,0]
    Theta = transformations[:,1]
    tx = transformations[:,2]
    ty = transformations[:,3]

    A = R*tf.math.cos(Theta)
    B = - R*tf.math.sin(Theta)
    C = ty
    D = R*tf.math.sin(Theta)
    E = R*tf.math.cos(Theta)
    F = tx
    Transformations = tf.stack((A,B,C,D,E,F),axis = 1)
    shape_T = tf.shape(Transformations)
    with tf.name_scope("transform_grids"):
        trans_matrices=tf.reshape(Transformations, (-1, 2,3))
        batch_size = tf.shape(trans_matrices)[0]
        gs = tf.squeeze(grids, -1)

        reprojected_grids = tf.matmul(trans_matrices, gs, transpose_b=True)
        # transform grid range from [-1,1) to the range of [0,1)
        reprojected_grids = (tf.linalg.matrix_transpose(reprojected_grids) + 1)*0.5
        _, H, W, _ = inputs.shape
        reprojected_grids = tf.math.multiply(reprojected_grids, [W, H])

        return reprojected_grids

def generate_four_neighbors_from_reprojection(inputs, reprojected_grids):
    _, H, W, _ = inputs.shape
    
    x, y = tf.split(reprojected_grids, 2, axis=-1)

    x1 = tf.floor(x)
    x1 = tf.dtypes.cast(x1, tf.int32)

    x2 = x1 + tf.constant(1) 

    y1 = tf.floor(y)
    y1 = tf.dtypes.cast(y1, tf.int32)
    y2 = y1 + tf.constant(1) 
    
    y_max = tf.constant(H - 1, dtype=tf.int32)
    x_max = tf.constant(W - 1, dtype=tf.int32)
    zero = tf.zeros([1], dtype=tf.int32)

    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)
    x2_safe = tf.clip_by_value(x2, zero, x_max)
    y2_safe = tf.clip_by_value(y2, zero, y_max)
    
    return x1_safe, y1_safe, x2_safe, y2_safe

def bilinear_sample(inputs, reprojected_grids):
    x1, y1, x2, y2 = generate_four_neighbors_from_reprojection(inputs, reprojected_grids)
    x1y1 = tf.concat([y1,x1],-1)
    x1y2 = tf.concat([y2,x1],-1)
    x2y1 = tf.concat([y1,x2],-1)
    x2y2 = tf.concat([y2,x2],-1)

    pixel_x1y1 = tf.gather_nd(inputs, x1y1, batch_dims=1)
    pixel_x1y2 = tf.gather_nd(inputs, x1y2, batch_dims=1)
    pixel_x2y1 = tf.gather_nd(inputs, x2y1, batch_dims=1)
    pixel_x2y2 = tf.gather_nd(inputs, x2y2, batch_dims=1)
    x, y = tf.split(reprojected_grids, 2, axis=-1)
    wx = tf.concat([tf.dtypes.cast(x2, tf.float32) - x, x -tf.dtypes.cast(x1, tf.float32)],-1)
    wx = tf.expand_dims(wx, -2)
    wy = tf.concat([tf.dtypes.cast(y2, tf.float32) - y, y - tf.dtypes.cast(y1, tf.float32)],-1)
    wy = tf.expand_dims(wy, -1)
    Q = tf.concat([pixel_x1y1, pixel_x1y2, pixel_x2y1, pixel_x2y2], -1)
    Q_shape = tf.shape(Q)
    Q = tf.reshape(Q, (Q_shape[0], Q_shape[1],2,2))
    Q = tf.cast(Q, tf.float32)

    r = wx@Q@wy
    _, H, W, channels = inputs.shape

    r = tf.reshape(r, (-1,H,W,1))
    return r

def spatial_transform_input_inv(inputs, transformations):
    grids = generate_normalized_homo_meshgrids(inputs)
    reprojected_grids = transform_grids_inv(transformations, grids,inputs)
    result = bilinear_sample(inputs, reprojected_grids)
    return result

def transform_grids_inv(transformations, grids, inputs):
#     if shape[0] == 4 :
    R = transformations[:,0]
    Theta = transformations[:,1]
    tx = transformations[:,2]
    ty = transformations[:,3]

    A = R*tf.math.cos(Theta)
    B = - R*tf.math.sin(Theta)
    C = ty
    D = R*tf.math.sin(Theta)
    E = R*tf.math.cos(Theta)
    F = tx
    Transformations = tf.stack((A,-B,-C,-D,E,-F),axis = 1)
    with tf.name_scope("transform_grids"):
        trans_matrices=tf.reshape(Transformations, (-1, 2,3))
        batch_size = tf.shape(trans_matrices)[0]
        gs = tf.squeeze(grids, -1)

        reprojected_grids = tf.matmul(trans_matrices, gs, transpose_b=True)
        # transform grid range from [-1,1) to the range of [0,1)
        reprojected_grids = (tf.linalg.matrix_transpose(reprojected_grids) + 1)*0.5
        _, H, W, _ = inputs.shape
        reprojected_grids = tf.math.multiply(reprojected_grids, [W, H])

        return reprojected_grids 

def create_localization_head(inputs):
    x = Conv2D(14, (5,5),padding='valid',activation="relu",name='Conv11')(inputs)
    x = MaxPooling2D((2, 2), strides=2,name='MaxPool11')(x)
    x = Conv2D(32, (5,5), padding='valid',activation="relu",name='Conv12')(x)
    x = MaxPooling2D((2, 2),strides=2,name='MaxPool12')(x)
    x = Flatten(name='F11')(x)    
    x = Dense(120, activation='relu',name='D11')(x)
    x = Dropout(0.2,name='Drop11')(x)
    x = Dense(84, activation='relu',name='D12')(x)    
    x = Dense(4,activation='linear',kernel_initializer='zeros',
              bias_initializer=lambda shape,dtype: tf.constant([1,0,0,0], dtype=dtype), name='OutputTransform')(x)
    return tf.keras.Model(inputs, x)

def run_model_MLP (inputs,data_shape=[128,128,1],encoding_dim = 10) :
    x = Flatten(input_shape=(data_shape),name='FlattenAE')(inputs)
    x = Dense(encoding_dim,activation='relu',name='BottleNeckAE')(x)
    x = Dense(data_shape[0]*data_shape[1],activation = 'sigmoid',name='DenseAE2')(x)
    x = Reshape(data_shape,name='ReshapeAE')(x)
    return x

def draw_samples(images, images_per_row=5, save_path = 0,figsize = 0,title=0):
    num = len(images)
    per_row = min(images_per_row, num)
    rows = ceil(num /per_row)
    fig, axs = plt.subplots(rows, per_row)
    if figsize != 0 :
        fig.set_size_inches(figsize[0],figsize[1])
    if title != 0 :
        fig.suptitle(title,fontsize=12)
    count = 0 
    for i in range(rows):
        for j in range(images_per_row):
            count+=1
            if (count > num):
                break
            if rows == 1:
                axs[j+i*per_row].imshow(images[j+i*per_row], cmap='gray',vmin=0,vmax=1)
                axs[j+i*per_row].get_xaxis().set_visible(False)
                axs[j+i*per_row].get_yaxis().set_visible(False)
            else:
                axs[i,j].imshow(images[j+i*per_row], cmap='gray',vmin=0,vmax=1)
                axs[i,j].get_xaxis().set_visible(False)
                axs[i,j].get_yaxis().set_visible(False)
    if save_path != 0 :
        fig.savefig(save_path)
    plt.show()
    return fig

# Reading the testing data
main_dir = r'./data/'
path_test = os.path.join(main_dir,'testing')
L_test = len(os.listdir(path_test))
img_test = np.zeros((L_test,128,128))

for i,filename in enumerate(os.listdir(path_test)) :
    img_test[i,:,:] = np.mean(cv2.imread(os.path.join(path_test,filename)),axis=-1)/255
img_test = img_test.reshape((-1,128,128,1))

# Definition of the model
inputs = Input(shape=img_test[0].shape)
loc_head = create_localization_head(inputs)
x = spatial_transform_input(inputs,loc_head.output)
x = run_model_MLP(x,(128,128,1),30)
x = spatial_transform_input_inv(x, loc_head.output)
model = tf.keras.Model(inputs = inputs, outputs = x)
model.load_weights('./weights/model')
 
# Display on two channels of the output-input superimposition 
output = model.predict(img_test)
multi_channel_images = np.zeros((3,128,128,3))
multi_channel_images[:,:,:,:1] = 1-output
multi_channel_images[:,:,:,1:2] = 1-img_test
fig1 = draw_samples(output)
fig2 = draw_samples(img_test)
fig3 = draw_samples(multi_channel_images)

filename = r'C:\Users\fournel\Desktop\_WORK\_PROJECTS\ON\ROIi\_COMMUNICATIONS\JImaging2022\codes\out1.jpg'
cv2.imwrite(filename, 255*output[0])
filename = r'C:\Users\fournel\Desktop\_WORK\_PROJECTS\ON\ROIi\_COMMUNICATIONS\JImaging2022\codes\out2.jpg'
cv2.imwrite(filename, 255*output[1])
filename = r'C:\Users\fournel\Desktop\_WORK\_PROJECTS\ON\ROIi\_COMMUNICATIONS\JImaging2022\codes\out3.jpg'
cv2.imwrite(filename, 255*output[2])