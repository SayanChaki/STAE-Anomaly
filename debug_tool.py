import matplotlib.pyplot as plt
from PIL import Image
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
from read_data import read_data

def debug_tool(prediction_inv_scaled, x_test_inv, prediction_inv_scaled_2):
	for k in range(0,len(prediction_inv_scaled)):
	  	data= np.zeros((64,64), dtype = np.float64)
	  	for i in range(0,64):
		    	for j in range(0,64):
			      	data[i][j]+=prediction_inv_scaled[k][i][j]
	img2 = Image.fromarray(data)
	diff =  -(prediction_inv_scaled_2 - x_test_inv)
	Max_list = np.max(diff,axis=(1,2,3))
	diff_scaled = np.zeros(diff.shape)
	for i in range(len(diff)) :
    		diff_scaled[i,:,:,:] = diff[i] * Max_list[i]*255


	for k in range(0,len(diff)):
  		data_diff= np.zeros((64,64), dtype = np.float64)
  		print(k)
  		for i in range(0,64):
    			for j in range(0,64):
      				data_diff[i][j]+=diff_scaled[k][i][j]
      				if data_diff[i][j]>20 and data_diff[i][j]<255:
      				   data_diff[i][j] = 255
  		data_diff.astype(np.uint8)
  		img3 = Image.fromarray(data_diff)
	plt.figure()
	plt.imshow(img3)
	plt.show()
