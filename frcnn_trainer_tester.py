"""
Created on Wed Jan  4 16:50:13 2023

@author: owais
"""

from __future__ import division
import random
import sys
import time
import numpy as np
import pickle
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras.utils import generic_utils
from keras_frcnn import resnet as nn # from keras_frcnn import vgg as nn

ROOT = ''
DATA_DIRECTORY = ROOT + 'dataset/'
WORKING_DIRECTORY = ROOT + 'generated/'

train_path = ''
num_rois = 32
num_epochs = 50
epoch_length = 10
augment = False
verbose = True
sys.setrecursionlimit(40000)
# Parser to use. One of simple or pascal_voc
parser = 'pascal_voc'
# Import the right get_data method depending on the parser
if parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif parser == 'simple':
    from keras_frcnn.simple_parser import get_data

# Base network to use. Supports vgg or resnet50
network = 'resnet50'
# Augment with horizontal flips in training
horizontal_flips = augment
# Augment with vertical flips in training
vertical_flips = augment
# Augment with 90 degree rotations in training
rotate = augment
# Location to store all the metadata related to the training (to be used when testing)
config_filename = 'config.pickle'
# Path of output file
output_weight_path = 'model_frcnn.hdf5'
# Input path for weights. If not specified, will try to load default weights provided by keras
input_weight_path = ''
# Path of the annotations.txt file for training
train_annotation_file = WORKING_DIRECTORY + 'annotation_train.txt'
# Path of the annotations.txt file for testing
test_annotation_file = WORKING_DIRECTORY + 'annotation_test.txt'

if verbose:
    print(f"Parser: {parser}; Network: {network}; Augmentation: {augment}; Config file: {config_filename}")

# Create config object
C = config.Config()
C.use_horizontal_flips = bool(horizontal_flips)
C.use_vertical_flips = bool(vertical_flips)
C.rot_90 = bool(rotate)
C.model_path = output_weight_path
C.num_rois = int(num_rois)
C.network = network

# Fetch data
all_images, classes_count, class_mapping = get_data(train_annotation_file)
if verbose:
    print(f"Total images: {len(all_images)}; Total classes: {classes_count}")
# Split training into validation and test
total_images = len(all_images)
train_images, val_images = train_test_split(all_images, test_size=0.2)
if verbose:
    print(f"Training set size: {len(train_images)}; Validation set size: {val_images}")

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping
inv_map = {v: k for k, v in class_mapping.items()}

with open(config_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_filename))

data_gen_train = data_generators.get_anchor_gt(train_images, classes_count, C, nn.get_img_output_length,
                                               K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_images, classes_count, C, nn.get_img_output_length,
                                             K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# Define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# Define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

# Define the classifier    
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

# Create model object
model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# This is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
    print(
        'Could not load pretrained model weights. Weights can be found in the keras application folder https://github.com/fchollet/keras/tree/master/keras/applications')

# Record data (used to save the losses, classification accuracy and mean average precision)
record_path = 'record.csv'
# Create the record.csv file to record losses, acc and mAP
record_df = pd.DataFrame(
    columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls',
             'loss_class_regr', 'curr_loss'])

model_rpn.compile(optimizer=Adam(0.0001), loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=Adam(0.0001),
                         loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count) - 1)],
                         metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
rpn_accuracy_rpn_monitor_val = []
rpn_accuracy_for_epoch_val = []
mean_overlapping_bboxes_val_lst = []
class_acc_val_lst = []
loss_rpn_cls_val_lst = []
loss_rpn_regr_val_lst = []
loss_class_cls_val_lst = []
loss_class_regr_val_lst = []
curr_loss_val_lst = []

start_time = time.time()
best_loss = np.Inf
class_mapping_inv = {v: k for k, v in class_mapping.items()}

vis = verbose
X_val, Y_val, img_data_val = next(data_gen_val)
if (verbose):
    print(f"Image shape: {(Y_val[0].shape)}")

for epoch_num in range(num_epochs):
    progbar = generic_utils.Progbar(epoch_length)
    print('Training Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, epoch_length))
                X_val, Y_val, img_data_val = next(data_gen_val)
                print("Sizes in iteration:", len(X_val), len(Y_val))
                loss_rpn_val = model_rpn.evaluate(X_val, Y_val)
                P_rpn_val = model_rpn.predict_on_batch(X_val)
                R_val = roi_helpers.rpn_to_roi(P_rpn_val[0], P_rpn_val[1], C, K.image_dim_ordering(), use_regr=True,
                                               overlap_thresh=0.5, max_boxes=300)
                X2_val, Y1_val, Y2_val, IouS_val = roi_helpers.calc_iou(R_val, img_data_val, C, class_mapping)
                neg_samples_val = np.where(Y1_val[0, :, -1] == 1)
                pos_samples_val = np.where(Y1_val[0, :, -1] == 0)

            if len(neg_samples_val) > 0:
                neg_samples_val = neg_samples_val[0]
            else:
                neg_samples_val = []
            if len(pos_samples_val) > 0:
                pos_samples_val = pos_samples_val[0]
            else:
                pos_samples_val = []
            rpn_accuracy_rpn_monitor_val.append(len(pos_samples_val))
            rpn_accuracy_for_epoch_val.append((len(pos_samples_val)))

            if C.num_rois > 1:
                if len(pos_samples_val) < C.num_rois // 2:
                    selected_pos_samples_val = pos_samples_val.tolist()
                else:
                    selected_pos_samples_val = np.random.choice(pos_samples_val, C.num_rois // 2,
                                                                replace=False).tolist()
                try:
                    selected_neg_samples_val = np.random.choice(neg_samples_val,
                                                                C.num_rois - len(selected_pos_samples_val),
                                                                replace=False).tolist()
                except:
                    selected_neg_samples_val = np.random.choice(neg_samples_val,
                                                                C.num_rois - len(selected_pos_samples_val),
                                                                replace=True).tolist()
                sel_samples_val = selected_pos_samples_val + selected_neg_samples_val

            else:
                # In extreme case where num_rois = 1, pick a random pos or neg sample
                selected_pos_samples_val = pos_samples_val.tolist()
                selected_neg_samples_val = neg_samples_val.tolist()
                if np.random.randint(0, 2):
                    sel_samples_val = random.choice(neg_samples_val)
                else:
                    sel_samples_val = random.choice(pos_samples_val)

            loss_class_val = model_classifier.evaluate([X_val, X2_val[:, sel_samples_val, :]],
                                                       [Y1_val[:, sel_samples_val, :], Y2_val[:, sel_samples_val, :]])
            loss_rpn_cls_val = loss_rpn_val[1]
            loss_rpn_regr_val = loss_rpn_val[2]
            loss_class_cls_val = loss_class_val[1]
            loss_class_regr_val = loss_class_val[2]
            class_acc_val = loss_class_val[3]
            curr_loss_val = loss_rpn_cls_val + loss_rpn_regr_val + loss_class_cls_val + loss_class_regr_val

            mean_overlapping_bboxes_val = float(sum(rpn_accuracy_rpn_monitor_val)) / len(rpn_accuracy_rpn_monitor_val)
            if verbose:
                print(
                    "RPN Accuracy for monitor validation set: {rpn_accuracy_rpn_monitor_val}; Average overlapping bounding boxes: {mean_overlapping_bboxes_val}")
            rpn_accuracy_rpn_monitor_val = []

            mean_overlapping_bboxes_val = float(sum(rpn_accuracy_for_epoch_val)) / len(rpn_accuracy_for_epoch_val)
            if verbose:
                print(
                    "RPN Accuracy for epoch validation set: {rpn_accuracy_for_epoch_val}; Average overlapping bounding boxes: {mean_overlapping_bboxes_val}")
            rpn_accuracy_for_epoch_val = []

            mean_overlapping_bboxes_val_lst.append(mean_overlapping_bboxes_val)
            class_acc_val_lst.append(class_acc_val)
            loss_rpn_cls_val_lst.append(loss_rpn_cls_val)
            loss_rpn_regr_val_lst.append(loss_rpn_regr_val)
            loss_class_cls_val_lst.append(loss_class_cls_val)
            loss_class_regr_val_lst.append(loss_class_regr_val)
            curr_loss_val_lst.append(curr_loss_val)
            if verbose:
                print(f"RPN Accuracy for bounding boxes from RPN: {class_acc_val}")
                print(f"IOU: {IouS_val}")
                print(f"RPN Classifier Loss: {loss_rpn_cls_val}")
                print(f"RPN Regression Loss: {loss_rpn_regr_val}")
                print(f"Classifier Loss: {loss_class_cls_val}")
                print(f"Regression Loss: {loss_class_regr_val}")
                print(f"Current loss total: {curr_loss_val}")
                print(f"Elapsed time: {time.time() - start_time}")

            if mean_overlapping_bboxes == 0:
                print(
                    "RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training")

            X, Y, img_data = next(data_gen_train)
            loss_rpn = model_rpn.train_on_batch(X, Y)
            P_rpn = model_rpn.predict_on_batch(X)
            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.5,
                                       max_boxes=300)
            # Note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            neg_samples = neg_samples[0] if len(neg_samples) > 0 else []
            pos_samples = pos_samples[0] if len(pos_samples) > 0 else []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois // 2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=True).tolist()
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                         [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            progbar.update(iter_num + 1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                          ('detector_cls', losses[iter_num, 2]),
                                          ('detector_regr', losses[iter_num, 3])])

            iter_num += 1

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])
            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
            rpn_accuracy_for_epoch = []

            if verbose:
                print(
                    f"Mean number of bounding boxes from RPN overlapping ground truth boxes: {mean_overlapping_bboxes}")
                print(f"Classifier accuracy for bounding boxes from RPN: {class_acc}")
                print(f"IOU: {IouS}")
                print(f"Loss RPN classifier: {loss_rpn_cls}")
                print(f"Loss RPN regression: {loss_rpn_regr}")
                print(f"Loss Detector classifier: {loss_class_cls}")
                print(f"Loss Detector regression: {loss_class_regr}")

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            print(f"Current loss total: {curr_loss}")
            print(f"Elapsed time: {time.time() - start_time}")
            iter_num = 0
            start_time = time.time()
            new_row = {
                'mean_overlapping_bboxes': round(mean_overlapping_bboxes, 3),
                'class_acc': round(class_acc, 3),
                'loss_rpn_cls': round(loss_rpn_cls, 3),
                'loss_rpn_regr': round(loss_rpn_regr, 3),
                'loss_class_cls': round(loss_class_cls, 3),
                'loss_class_regr': round(loss_class_regr, 3),
                'curr_loss': round(curr_loss, 3)
            }

            record_df = record_df.append(new_row, ignore_index=True)
            record_df.to_csv(record_path, index=0)

            if curr_loss < best_loss:
                if verbose:
                    print(f"Total loss decreased from {best_loss} to {curr_loss}, saving weights")
                best_loss = curr_loss
                model_all.save_weights(C.model_path)

                print("During Training...")
                print('mean_overlapping_bboxes_lst: ' + record_df['mean_overlapping_bboxes'])
                print('class_acc_lst: ' + record_df['class_acc'])
                print('loss_rpn_cls_lst: ' + record_df['loss_rpn_cls'])
                print('loss_rpn_regr_lst: ' + record_df['loss_rpn_regr'])
                print('loss_class_cls_lst: ' + record_df['loss_class_cls'])
                print('loss_class_regr_lst: ' + record_df['loss_class_regr'])
                print('curr_loss_lst:' + record_df['curr_loss'])

                print("During Validation...")
                print('mean_overlapping_bboxes_val_lst: ' + mean_overlapping_bboxes_val_lst)
                print('class_acc_val_lst: ' + class_acc_val_lst)
                print('loss_rpn_cls_val_lst: ' + loss_rpn_cls_val_lst)
                print('loss_rpn_regr_val_lst: ' + loss_rpn_regr_val_lst)
                print('loss_class_cls_val_lst: ' + loss_class_cls_val_lst)
                print('loss_class_regr_val_lst: ' + loss_class_regr_val_lst)
                print('curr_loss_val_lst:' + curr_loss_val_lst)
            break
        except Exception as e:
            print('Exception: {}'.format(e))
            continue

    print("Training complete.")
    # Important to find the final validation numbers here to get the final number as in on last epoch
    X_val, Y_val, img_data_val = next(data_gen_val)
    loss_rpn_val = model_rpn.evaluate(X_val, Y_val)
    P_rpn_val = model_rpn.predict_on_batch(X_val)
    R_val = roi_helpers.rpn_to_roi(P_rpn_val[0], P_rpn_val[1], C, K.image_dim_ordering(), use_regr=True,
                                   overlap_thresh=0.5, max_boxes=300)
    X2_val, Y1_val, Y2_val, IouS_val = roi_helpers.calc_iou(R_val, img_data_val, C, class_mapping)
    neg_samples_val = np.where(Y1_val[0, :, -1] == 1)
    pos_samples_val = np.where(Y1_val[0, :, -1] == 0)

    if len(neg_samples_val) > 0:
        neg_samples_val = neg_samples_val[0]
    else:
        neg_samples_val = []
    if len(pos_samples_val) > 0:
        pos_samples_val = pos_samples_val[0]
    else:
        pos_samples_val = []
    rpn_accuracy_rpn_monitor_val.append(len(pos_samples_val))
    rpn_accuracy_for_epoch_val.append((len(pos_samples_val)))

    if C.num_rois > 1:
        if len(pos_samples_val) < C.num_rois // 2:
            selected_pos_samples_val = pos_samples_val.tolist()
        else:
            selected_pos_samples_val = np.random.choice(pos_samples_val, C.num_rois // 2, replace=False).tolist()
        try:
            selected_neg_samples_val = np.random.choice(neg_samples_val, C.num_rois - len(selected_pos_samples_val),
                                                        replace=False).tolist()
        except:
            selected_neg_samples_val = np.random.choice(neg_samples_val, C.num_rois - len(selected_pos_samples_val),
                                                        replace=True).tolist()
        sel_samples_val = selected_pos_samples_val + selected_neg_samples_val
    else:
        selected_pos_samples_val = pos_samples_val.tolist()
        selected_neg_samples_val = neg_samples_val.tolist()
        if np.random.randint(0, 2):
            sel_samples_val = random.choice(neg_samples_val)
        else:
            sel_samples_val = random.choice(pos_samples_val)

loss_class_val = model_classifier.evaluate([X_val, X2_val[:, sel_samples_val, :]],
                                           [Y1_val[:, sel_samples_val, :], Y2_val[:, sel_samples_val, :]])
loss_rpn_cls_val = loss_rpn_val[1]
loss_rpn_regr_val = loss_rpn_val[2]
loss_class_cls_val = loss_class_val[1]
loss_class_regr_val = loss_class_val[2]
class_acc_val = loss_class_val[3]
curr_loss_val = loss_rpn_cls_val + loss_rpn_regr_val + loss_class_cls_val + loss_class_regr_val

mean_overlapping_bboxes_val = float(sum(rpn_accuracy_rpn_monitor_val)) / len(rpn_accuracy_rpn_monitor_val)
if verbose:
    print(f"Validation numbers: {num_epochs}; Accuracy: {rpn_accuracy_rpn_monitor_val}")
    print(
        f"Average number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes_val} for {epoch_length} previous iterations")
rpn_accuracy_rpn_monitor_val = []

mean_overlapping_bboxes_val = float(sum(rpn_accuracy_for_epoch_val)) / len(rpn_accuracy_for_epoch_val)
mean_overlapping_bboxes_val_lst.append(mean_overlapping_bboxes_val)
class_acc_val_lst.append(class_acc_val)
loss_rpn_cls_val_lst.append(loss_rpn_cls_val)
loss_rpn_regr_val_lst.append(loss_rpn_regr_val)
loss_class_cls_val_lst.append(loss_class_cls_val)
loss_class_regr_val_lst.append(loss_class_regr_val)
curr_loss_val_lst.append(curr_loss_val)
if verbose:
    print(f"Validation numbers: {num_epochs}; Accuracy: {rpn_accuracy_for_epoch_val}")
    print(
        f"Average number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes_val} for {epoch_length} previous iterations")
    print(f"Mean number of bounding boxes from RPN overlapping ground truth boxes: {mean_overlapping_bboxes_val}")
    print(f"Classifier accuracy for bounding boxes from RPN: {class_acc_val}")
    print(f"IOU: {IouS_val}")
    print(f"Loss RPN classifier: {loss_rpn_cls_val}")
    print(f"Loss RPN regression: {loss_rpn_regr_val}")
    print(f"Loss Detector classifier: {loss_class_cls_val}")
    print(f"Loss Detector regression: {loss_class_regr_val}")
    print(f"Current loss total: {curr_loss_val}")
rpn_accuracy_for_epoch_val = []

print("*** Final Training List ***")

print('mean_overlapping_bboxes_lst', record_df['mean_overlapping_bboxes'])
print('class_acc_lst', record_df['class_acc'])
print('loss_rpn_cls_lst', record_df['loss_rpn_cls'])
print('loss_rpn_regr_lst', record_df['loss_rpn_regr'])
print('loss_class_cls_lst', record_df['loss_class_cls'])
print('loss_class_regr_lst', record_df['loss_class_regr'])
print('curr_loss_lst', record_df['curr_loss'])

print("*** Final Validation List ***")

print('mean_overlapping_bboxes_val_lst', mean_overlapping_bboxes_val_lst)
print('class_acc_val_lst', class_acc_val_lst)
print('loss_rpn_cls_val_lst', loss_rpn_cls_val_lst)
print('loss_rpn_regr_val_lst', loss_rpn_regr_val_lst)
print('loss_class_cls_val_lst', loss_class_cls_val_lst)
print('loss_class_regr_val_lst', loss_class_regr_val_lst)
print('curr_loss_val_lst', curr_loss_val_lst)

epoch_lst = []
for i in range(len(curr_loss_val_lst)):
    epoch_lst.append(i + 1)

plt.figure(figsize=(20, 25))
plt.subplot(4, 2, 1)
plt.plot(epoch_lst, record_df['mean_overlapping_bboxes'])
plt.plot(epoch_lst, mean_overlapping_bboxes_val_lst)
plt.legend(['Train', 'Validation'], loc='bottom right')
plt.ylabel('mean_overlapping_bboxes')
plt.xlabel('Epoch')
plt.grid()
plt.title('mean_overlapping_bboxes')

plt.subplot(4, 2, 2)
plt.plot(epoch_lst, record_df['class_acc'])
plt.plot(epoch_lst, class_acc_val_lst)
plt.legend(['Train', 'Validation'], loc='bottom right')
plt.ylabel('class_acc')
plt.xlabel('Epoch')
plt.grid()
plt.title('class_acc')

plt.subplot(4, 2, 3)
plt.plot(epoch_lst, record_df['loss_rpn_cls'])
plt.plot(epoch_lst, loss_rpn_cls_val_lst)
plt.legend(['Train', 'Validation'], loc='upper right')
plt.ylabel('loss_rpn_cls')
plt.xlabel('Epoch')
plt.grid()
plt.title('loss_rpn_cls')

plt.subplot(4, 2, 4)
plt.plot(epoch_lst, record_df['loss_rpn_regr'])
plt.plot(epoch_lst, loss_rpn_regr_val_lst)
plt.legend(['Train', 'Validation'], loc='upper right')
plt.ylabel('loss_rpn_regr')
plt.xlabel('Epoch')
plt.grid()
plt.title('loss_rpn_regr')

plt.subplot(4, 2, 5)
plt.plot(epoch_lst, record_df['loss_class_cls'])
plt.plot(epoch_lst, loss_class_cls_val_lst)
plt.legend(['Train', 'Validation'], loc='upper right')
plt.ylabel('loss_class_cls')
plt.xlabel('Epoch')
plt.grid()
plt.title('loss_class_cls')

plt.subplot(4, 2, 6)
plt.plot(epoch_lst, record_df['loss_class_regr'])
plt.plot(epoch_lst, loss_class_regr_val_lst)
plt.legend(['Train', 'Validation'], loc='upper right')
plt.ylabel('loss_class_regr')
plt.xlabel('Epoch')
plt.grid()
plt.title('loss_class_regr')

plt.subplot(4, 2, 7)
plt.plot(epoch_lst, record_df['curr_loss'])
plt.plot(epoch_lst, curr_loss_val_lst)
plt.legend(['Train', 'Validation'], loc='upper right')
plt.ylabel('total_loss')
plt.xlabel('Epoch')
plt.grid()
plt.title('total_loss')
plt.show()
plt.savefig('Evaluate.jpg')

all_images, classes_count, class_mapping = get_data(test_annotation_file)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))

random.shuffle(all_images)

total_images = len(all_images)

val_images = all_images

print('Num val samples {}'.format(len(val_images)))

data_gen_val = data_generators.get_anchor_gt(val_images, classes_count, C, nn.get_img_output_length,
                                             K.image_dim_ordering(), mode='val')
