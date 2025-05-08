import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2
from PIL import Image
from sklearn import metrics
import math
import argparse
parser = argparse.ArgumentParser(description='Explainable VAE MNIST Example')
parser.add_argument('--val', type=int, default=0, metavar='DIR',
                        help='output directory')
args = parser.parse_args()
gt_dir = './threshh/000_mask.png'
pred_dir= './threshh/leather/thresh'

#gt_file = os.listdir(gt_dir)
print(pred_dir)
#pred_file = os.listdir(pred_dir)
sum_roc=0
l=[]
m=[]
s=list(filter(lambda x:x%2!=0, range(0,255)))
print(s)
i=0
while i<151:
	em_image_vol = cv2.imread(gt_dir)
	plt.figure()
	plt.imshow(em_image_vol)
	plt.show()
	em_image_vol = cv2.resize(em_image_vol, (128,128), interpolation = cv2.INTER_AREA)
	em_thresh_vol = cv2.imread(os.path.join(pred_dir,str(i)+'_red_numerals_thresh.png'))
	plt.figure()
	plt.imshow(em_thresh_vol)
	plt.show()
	y_true=[]
	y_pred=[]
	a = np.array(em_image_vol)
	b = np.array(em_thresh_vol)
	# get prediction for each pixel in the image
	y_true.append(a.flatten())  # flatten all targets
	y_pred.append(b.flatten())  # flatten all predictions
	# concatenate all predictions and targets:
	y_true = np.concatenate(y_true, axis=0)
	y_pred = np.concatenate(y_pred, axis=0)
	y_true=np.where(y_true!=0, 1, y_true) 
	y_pred=np.where(y_pred!=0, 1, y_pred) 
	# copte the ROC curve
	print(y_true)
	try:
		tpr, fpr, _ = metrics.roc_curve(y_true, y_pred)
	except:
		break
		print("skipped")
	print(tpr)
	print(fpr)
	l.append(tpr[1])
	m.append(fpr[1])

	'''plt.plot(fpr, tpr)
	plt.ylabel('TPR')
	plt.xlabel('FPR')
	plt.show()'''



	print(len(fpr))
	roc_auc = auc(tpr,fpr)
	if  math.isnan(float(roc_auc)):
		sum_roc+=1
		print('yes')
	else:

		sum_roc+=roc_auc
	print(roc_auc)
	i=i-1

print(l)
print(m)
'''plt.plot(l, m)
plt.ylabel('TPR')
plt.xlabel('FPR')
#plt.show()
plt.savefig('D:\\Machine Learning\\AUC-ROC-tooth.png')'''



