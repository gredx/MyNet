from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.callbacks import *
from keras.utils.vis_utils import plot_model
import shutil

#from data import load_train_data, load_test_data
import time

print(K())
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

starttime = time.clock()

img_rows = 512
img_cols = 512

smooth = 1.


def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

#构建u-net网络
def get_unet():
	inputs = Input((img_rows, img_cols, 1))
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

	conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

	return model


#def preprocess(imgs):#数据预处理，归一化数据的尺寸。这里可以不用
#	imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
#	for i in range(imgs.shape[0]):
#	imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

#	imgs_p = imgs_p[..., np.newaxis]
#	return imgs_p


def predict_AT():
#	print('-'*30)
#	print('Loading train data...')
#	print('-'*30)
	imgs_train = np.load('train_data.npy')#加载训练集数据
#	imgs_mask_train = np.load('train_label.npy')

	imgs_train = imgs_train.reshape(912,512,512,1)
#	imgs_mask_train = imgs_mask_train.reshape(76,512,512,1)

##	imgs_train = preprocess(imgs_train)
##	imgs_mask_train = preprocess(imgs_mask_train)

	imgs_train = imgs_train.astype('float32')
	mean = np.mean(imgs_train)  # mean for data centering
	std = np.std(imgs_train)  # std for data normalization
#	imgs_train -= mean
#	imgs_train /= std
#	imgs_mask_train = imgs_mask_train.astype('float32')
#	imgs_mask_train /= 255.  # scale masks to [0, 1]
#	print('-'*30)
#	print('Creating and compiling model...')
#	print('-'*30)
##	log_filepath = '/logs'
	model = get_unet()
    
##	Visualize model 画出模型结构
##	plot_model(model, 'u-net-model-architecture.png', show_shapes=True)
##	model_file_format = 'model.{epoch:03d}.hdf5'
##	print model_file_format
	
	
#	model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
##	tb_cb = TensorBoard(log_dir=log_filepath, write_images=False, histogram_freq=1, write_graph=True)
    
    
#	print('-'*30)
#	print('Fitting model...')
#	print('-'*30)
	#训练网络
#	model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=100, verbose=1, shuffle=True,validation_split=0.01,callbacks=[model_checkpoint])


#########################################################################################################
	print('-'*30)
	print('Loading test data...')
	print('-'*30)
	imgs_test = np.load('test_data.npy')#加载测试集数据
	imgs_test = imgs_test.reshape(608,512,512,1)
##	imgs_test = preprocess(imgs_test)

	imgs_test = imgs_test.astype('float32')
	imgs_test -= mean
	imgs_test /= std

	print('-'*30)
	print('Loading saved weights...')
	print('-'*30)
###	model.load_weights('weights_SAT.h5')
	model.load_weights('weights_VAT.h5')

	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
#	np.save('imgs_mask_train.npy', imgs_mask_train)

	print('-' * 30)
	print('Saving predicted masks to files...')
	print('-' * 30)

	imgs_id_test = np.ndarray((608, ), dtype=np.int32)
	for i in range(608):
		imgs_id_test[i] = i+1
    
	pred_dir = 'preds'
	if not os.path.exists(pred_dir):
		os.mkdir(pred_dir)
	else:
		shutil.rmtree(pred_dir)
		os.mkdir(pred_dir)
        
	for image, image_id in zip(imgs_mask_test, imgs_id_test):
		image = (image[:, :, 0] * 255.).astype(np.uint8)
		imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
	predict_AT()
    
    
endtime = time.clock()
print("The predict running time is %g s" %(endtime-starttime))
