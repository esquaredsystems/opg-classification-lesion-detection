import tensorflow as tf
from tensorflow.keras.models import load_model
from  tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np
from sklearn import metrics
import random


def test_train_split(X, training_set_ratio=0.8):
    '''
    Splits given data into training and test sets

    ----------
    X : the data array
    training_set_ratio : ratio of training dataset. Default is 0.8, i.e. 80%

    Returns
    -------
    x_test : test data
    x_train : training data

    '''
    random.shuffle(X)
    cutoff = int(len(X) * training_set_ratio)
    x_test = X[cutoff: ]
    x_train = X[: cutoff]
    return x_test, x_train


def confusion_matrix(y_true, y_pred):
    '''
    Confusion matrix is a table of True-negatives, False-positives, False-negatives, and True-positives specifically in this order
    '''
    #conf_matrix = tf.math.confusion_matrix(labels=y_test, predictions=predictions)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, np.rint(y_pred)).ravel()
    return tn, fp, fn, tp


def dice_coef(y_true, y_pred):
    '''
    Dice coefficient determines the ratio of overlapping area between prediction and ground truth to the sum of total area
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def recall_m(y_true, y_pred):
    '''
    Recall or Sensitivity is positive identification rate, i.e. of all positives, how many were identified
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    '''
    Precision is positive prediction value, i.e. how many positives are truly positive
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def specificity_m(y_true, y_pred):
    '''
    Speecificity is negative prediction value, i.e. how many negatives are truly negative
    '''
    #TODO: complete this
    pass


def f1_m(y_true, y_pred):
    '''
    F1 measure is harmonic mean between precision and recall
    '''
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def IOU(y_true, y_pred):
    '''
    Intersection over Union of a bounded box
    '''
    D = dice_coef(y_true, y_pred)
    IOU = (D) / (2-D)
    return IOU


def data_generator(x_train, y_train, batch_size):
    """
    Implement data augmentation
    ----------
    x_train : numpy array of raw images
    y_train : numpy array of mask images
    batch_size : integer value that specifies batch size 
    
    Returns
    -------
    x_batch, y_bathc : Image Generators
    
    """
    data_gen_args = dict(width_shift_range = 0.1,
            height_shift_range = 0.1,
            rotation_range = 10,
            zoom_range = 0.1)
    
    image_generator = ImageDataGenerator(**data_gen_args).flow(x_train, x_train, batch_size, seed = 42)
    mask_generator = ImageDataGenerator(**data_gen_args).flow(y_train, y_train, batch_size, seed = 42)
    while True:
        x_batch, _ = image_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


def model_builder(x_train, x_val, y_train, y_val, height, width, channels):
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
    inputs = tf.keras.layers.Input((height, width, channels))
    s = tf.keras.layers.Lambda(lambda x: x)(inputs)

    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(s)
    conv1 = tf.keras.layers.Dropout(0.1)(conv1)
    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = tf.keras.layers.Dropout(0.1)(conv2)
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = tf.keras.layers.Dropout(0.2)(conv3)
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = tf.keras.layers.Dropout(0.2)(conv4)
    conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = tf.keras.layers.Dropout(0.3)(conv5)
    conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(conv5)

    upconv6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    upconv6 = tf.keras.layers.concatenate([upconv6, conv4])
    conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(upconv6)
    conv6 = tf.keras.layers.Dropout(0.2)(conv6)
    conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(conv6)

    upconv7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    upconv7 = tf.keras.layers.concatenate([upconv7, conv3])
    conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(upconv7)
    conv7 = tf.keras.layers.Dropout(0.2)(conv7)
    conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(conv7)

    upconv8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    upconv8 = tf.keras.layers.concatenate([upconv8, conv2])
    conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(upconv8)
    conv8 = tf.keras.layers.Dropout(0.1)(conv8)
    conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(conv8)

    upconv9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    upconv9 = tf.keras.layers.concatenate([upconv9, conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(upconv9)
    conv9 = tf.keras.layers.Dropout(0.1)(conv9)
    conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(conv9)
    
    # den   = tf.keras.layers.Dense(8, activation=tf.keras.activations.elu, kernel_initializer='he_normal')(conv9)
    # outputs = tf.keras.layers.Dense(8, activation=tf.keras.activations.elu, kernel_initializer='he_normal')(conv9)
    outputs = tf.keras.layers.Conv2D(5, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, recall_m, precision_m, f1_m, IOU, 'accuracy'])
    model.summary()
    callbacks = [EarlyStopping(patience=6, monitor='val_loss')]
 
    results = model.fit_generator(data_generator(x_train, y_train, 8),
                                  steps_per_epoch = 300,
                                  validation_data = (x_val,y_val),
                                  epochs=1, callbacks=callbacks)
 
    model.save("/content/drive/MyDrive/Dental/model_4_Depth_1_Base_40_Images.h5")
    return model, results
