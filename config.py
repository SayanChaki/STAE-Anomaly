import os

dataset = 'MVTec'
data_dir = '/home/sayanchaki/Documents/codes/codes_updated/Data/'
class0="retrieval1474"
model_save_path = 'saved_model'

# training options
batch_size = 32
input_image_shape = [3, 128, 128]
training = 'STAE1' or 'STAE2*'
epoch = 1

