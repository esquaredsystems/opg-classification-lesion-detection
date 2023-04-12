# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:29:56 2022

@author: owais
"""

import os
import numpy as np
import tensorflow as tf
import keras
import cv2 as cv
from skimage.io import imread
from skimage.transform import resize
from  tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import load_model
from matplotlib.pyplot import imsave,imshow
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import random
from scipy import ndarray
from skimage import transform
from skimage import util
#from google.colab import drive
#drive.mount('/content/drive/')
#ROOT = './drive/MyDrive/dental-xray-classification/'
ROOT = ''
DATA_DIRECTORY = ROOT + 'dataset/'
WORKING_DIRECTORY = ROOT + 'generated/'
HEIGHT = 512
WIDTH = 512
CHANNELS = 4

def imageProcessingModule():
    """
    Function to read training images and corresponding masks and 
    resize to a particular dimension. The resized images will be stored
    numpy arrays
    
    Parameters
    ----------
    None
    
    Returns
    -------
    raw_data : numpy array of raw images
    mask_data : numpy array of mask images
    
    """
    raw_files = [f for f in os.listdir(WORKING_DIRECTORY) if f.startswith('resized')]
    msk_files = [f for f in os.listdir(WORKING_DIRECTORY) if f.startswith('mask')]

    raw_data = np.zeros((len(raw_files), HEIGHT, WIDTH, CHANNELS, 1), dtype=np.float_)
    mask_data = np.zeros((len(raw_files), HEIGHT, WIDTH, CHANNELS, 1), dtype=np.float_)
    
    index = 0
    for file in raw_files:
        if (file in msk_files):
            raw_img = imread(WORKING_DIRECTORY + file)
            raw_img = resize(raw_img, (HEIGHT, WIDTH, CHANNELS, 1), anti_aliasing=True)
            mask = np.zeros((HEIGHT, WIDTH, CHANNELS, 1), dtype=np.bool)
            msk_img = imread(WORKING_DIRECTORY + file)
            msk_img = np.expand_dims(resize(msk_img, (HEIGHT, WIDTH), anti_aliasing=True), axis=-1)
            mask = np.maximum(mask, msk_img)
            mask_data[index] = mask
            raw_data[index] = raw_img
            index += 1
    raw_data = raw_data.reshape(raw_data.shape[0:4])
    mask_data = mask_data.reshape(mask_data.shape[0:4])
    return raw_data, mask_data

def data_generator(x_train, y_train, batch_size):
    """
    Function to implement data augmentation
    
    Parameters
    ----------
    x_train : numpy array of raw images
    y_train : numpy array of mask images
    batch_size : integer value that specifies batch size 
    
    Returns
    -------
    x_batch, y_bathc : Image Generators
    
    """
    data_gen_args = dict(width_shift_range = 0.1, height_shift_range = 0.1, rotation_range = 10, zoom_range = 0.1)
    image_generator = ImageDataGenerator(**data_gen_args).flow(x_train, x_train, batch_size, seed = 42)
    mask_generator = ImageDataGenerator(**data_gen_args).flow(y_train, y_train, batch_size, seed = 42)
    while True:
        x_batch, _ = image_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch

def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def recall_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1_m(y_true, y_pred):
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

def IOU(y_true, y_pred):
  D = dice_coef(y_true, y_pred)
  IOU = (D) / (2-D)
  return IOU

def modelBuilder(x_train, x_val, y_train, y_val):
    """
    Function to build and train the UNet model
    
    Parameters
    ----------
    x_train : numpy array of training images
    y_train : numpy array of training masks
    x_val : numpy array of validation images
    y_val : numpy array of validation masks
    
    Returns
    -------
    model : Trained model
    results : Dictionary of model metrics
    
    """
    print ('Building U-Net...')
    inputs = tf.keras.layers.Input((HEIGHT, WIDTH, CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x)(inputs)

    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(s)
    conv1 = tf.keras.layers.Dropout(0.1)(conv1)
    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(pool1)
    conv2 = tf.keras.layers.Dropout(0.1)(conv2)
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(pool2)
    conv3 = tf.keras.layers.Dropout(0.2)(conv3)
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(pool3)
    conv4 = tf.keras.layers.Dropout(0.2)(conv4)
    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(pool4)
    conv5 = tf.keras.layers.Dropout(0.3)(conv5)
    conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(conv5)

    upconv6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    upconv6 = tf.keras.layers.concatenate([upconv6, conv4])
    conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(upconv6)
    conv6 = tf.keras.layers.Dropout(0.2)(conv6)
    conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(conv6)

    upconv7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    upconv7 = tf.keras.layers.concatenate([upconv7, conv3])
    conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(upconv7)
    conv7 = tf.keras.layers.Dropout(0.2)(conv7)
    conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(conv7)

    upconv8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    upconv8 = tf.keras.layers.concatenate([upconv8, conv2])
    conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(upconv8)
    conv8 = tf.keras.layers.Dropout(0.1)(conv8)
    conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(conv8)

    upconv9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    upconv9 = tf.keras.layers.concatenate([upconv9, conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(upconv9)
    conv9 = tf.keras.layers.Dropout(0.1)(conv9)
    conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(conv9)
    # den   = tf.keras.layers.Dense(8, activation=tf.keras.activations.elu, kernel_initializer='he_normal')(conv9)
    outputs = tf.keras.layers.Conv2D(4, (1, 1), activation='sigmoid')(conv9)
    # outputs = tf.keras.layers.Dense(8, activation=tf.keras.activations.elu, kernel_initializer='he_normal')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef,recall_m,precision_m,f1_m,IOU,'accuracy'])
    model.summary()
    callbacks = [EarlyStopping(patience=6, monitor='val_loss')]
 
    results = model.fit_generator(data_generator(x_train,y_train,8),
                                  steps_per_epoch = 300,
                                  validation_data = (x_val,y_val),
                                  epochs=50,callbacks=callbacks)
 
    model.save(WORKING_DIRECTORY + "/model_4_depth_1_base_40_images.h5")
    return model, results

def validationImageChecker(model, xval, yval):
    """
    Function to generate masks for validation images plot it against the
    corresponding ground truth masks
    
    Parameters
    ----------
    xval : numpy array of validation images
    yval : numpy array of validation masks
    
    Returns
    -------
    None
    
    """
    output_msk = np.zeros((len(xval), HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
    idx = 0
    for im in xval:
        xtest = np.array(im)
        xtest=np.expand_dims(xtest, axis=0)
        pred = model.predict(xtest)
        pred = (pred > 0.55).astype(np.uint8)
        pred = np.squeeze(pred[0])
        # print(pred*255)
        output_msk[idx]=pred*255
        idx+=1
    fix, ax = plt.subplots(idx, 2, figsize=(20,40))
    for i in range(idx):
        ax[i,0].set_title(' Actual Mask')
        ax[i,0].imshow(yval[i,:,:,:])
        ax[i,0].axis('off')
        ax[i,1].set_title('Predicted Mask')
        ax[i,1].imshow(output_msk[i,:,:,:])
        ax[i,1].axis('off')
    plt.show()

    for i in range(idx):
        plt.savefig(WORKING_DIRECTORY + "results/" + str(i) + ".png")


file = 'resized-opg-aku-2022-10.png'
raw_img = cv.imread(WORKING_DIRECTORY + file)
plt.imshow(raw_img)

mask = np.zeros((HEIGHT, WIDTH, CHANNELS, 1), dtype=int)

msk_img = cv.imread(WORKING_DIRECTORY + 'mask-' + file + '.npy', cv.IMREAD_COLOR)
cv.imshow('msk_img', msk_img)
msk_img = np.expand_dims(resize(msk_img, (HEIGHT, WIDTH), anti_aliasing=True), axis=-1)
mask = np.maximum(mask, msk_img)

x_data,y_data = imageProcessingModule()
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.1)

print('Total data is: ',len(x_data))
print('Train data is: ',len(x_train))
print('Total validation is: ',len(x_val))

print(x_data.shape)
import matplotlib.pyplot as plt

number_images = len(x_data)
fix, ax = plt.subplots(number_images, 2, figsize=(20, 350))
for i in range(number_images):
    
    ax[i,0].set_title('Train Image' + '--' + str(i+1))
    ax[i,0].imshow(x_data[i,:,:,:])
    ax[i,0].axis('off')
    ax[i,1].set_title('Train Mask' + '--' + str(i+1))
    ax[i,1].imshow(y_data[i,:,:,:])
    ax[i,1].axis('off')
plt.show()

