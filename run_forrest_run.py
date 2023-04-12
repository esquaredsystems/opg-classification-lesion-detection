# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import *
# from tensorflow.keras import optimizers
# from tensorflow.keras.losses import *
# from tensorflow.keras import optimizers
# from tensorflow.keras.losses import *
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import io
import json
import base64
import random
import shutil
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import keras.backend as K

from skimage.io import imread
from skimage.transform import resize
from matplotlib.pyplot import imsave, imshow
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator


# from google.colab import drive
# drive.mount('/content/drive/')
#ROOT = './drive/MyDrive/dental-xray-classification/'
ROOT = ''
DATA_DIRECTORY = ROOT + 'dataset/'
WORKING_DIRECTORY = ROOT + 'generated/'
HEIGHT = 512
WIDTH = 512
CHANNELS = 3


def dataset_to_json(dirname = DATA_DIRECTORY):
    '''
    Prepares a JSON file from all OPGs and annotations in the given dataset directory
    ----------
    dirname : path to data directory to scan
    '''
    file_list = os.listdir(dirname)
    opgs = []
    annotations = []
    for file in file_list:
        parts = file.split('-')
        if (len(parts) == 4):
            # OPG data
            if parts[0] == 'opg':
                opg_data = {'type': parts[0], 'source': parts[1], 'year': parts[2], 'file': file}
                opgs.append(opg_data)
        elif (len(parts) == 5):
            # Annotations data
            annotations_data = {'type': parts[0] + '-' + parts[1], 'source': parts[2], 'year': parts[3], 'file':  file}
            annotations.append(annotations_data)
    data = []
    for opg in opgs:
        if opg['type'] == 'opg':
            opg_file_name = opg['file']
            remove = ['opg-', '.png', '.PNG', '.jpg', '.jpeg']
            for token in remove:
                opg_file_name = opg_file_name.replace(token, '')
            # Search within data for teeth annotations
            for annotation in annotations:
                if 'teeth-annotations-' + opg_file_name in annotation['file']:
                    opg['teeth-annotations-file'] = annotation['file']
                elif 'lesion-annotations-' + opg_file_name in annotation['file']:
                    opg['lesion-annotations-file'] = annotation['file']
            opg.pop('type')
            data.append(opg)
    with open('dataset.json', 'w') as outfile:
        json.dump({"dataset": data}, outfile)


def display_image(image_file):
    '''
    Displays given image file on Plot
    ----------
    image_file : image file (PNG, JPG, etc.)
    '''
    img = cv.imread(image_file)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.imshow(img)
    plt.show()


def resize_image(image_file, save_file=True):
    '''
    Resizes image to given size and optionally stores in data directory
    ----------
    image_file : image file (PNG, JPG, etc.)
    save_file : whether to save resized image as file or not. The default is True.

    Returns
    -------
    img : resized image
    '''
    img = cv.imread(DATA_DIRECTORY + image_file)
    #img_size = img.shape[: -1]
    img = cv.resize(img, (HEIGHT, WIDTH))
    if save_file:
        cv.imwrite(WORKING_DIRECTORY + 'resized-{f}'.format(f=str.lower(image_file)), img)
    return img


def generate_teeth_annotations(data, save_in='train', save_annotated_images=True, store_dimensions=False):
    '''
    Reads all rows from data and generates a text file from all teeth annotations.
    
    ----------
    data : the JSON data set
    save_in : the directory name to create files into
    limit : whether to limit the number of records
    save_annotated_images : when True, the annotated polygons and respective 
        boundary boxes are superimposed on the original image and stored as PNG

    Returns
    -------
    None.
    '''
    if not str.endswith(save_in, '/'):
        save_in = save_in + '/'
    save_in = WORKING_DIRECTORY + save_in
    # Remove and recreate directory
    try:
        shutil.rmtree(save_in)
    except IOError:
        pass
    os.makedirs(save_in)

    annotation_filename = 'annotation.txt'
    colors = {'canine': 'cyan', 'central incisor': 'magenta', 'lateral incisor': 'yellow',
              'molar': 'red', 'premolar': 'blue', 'implant': 'gold'}
    rows = []
    for row in data:
        print("Generating teeth annotations for: " + row['teeth-annotations-file'])
        # Read teeth annotation file
        teeth = json.load(open(DATA_DIRECTORY + row['teeth-annotations-file'], 'r'))
        # Read image data coded in Base64 string
        image_data = teeth['imageData']
        # Decode the base64 data
        decoded_data = base64.b64decode(image_data)
        # Convert the decoded data to a stream
        stream = io.BytesIO(decoded_data)
        # Use the Image module to open the stream as an image
        image = Image.open(stream)
        # Store plain image in generated directory
        raw_filename = 'raw-' + row['teeth-annotations-file'].replace('.json', '.png')
        image.save(save_in + raw_filename)
        # Create an ImageDraw object
        draw = ImageDraw.Draw(image)

        for shape in teeth['shapes']:
            # Draw the polygon boundary on the image
            points = [tuple(i) for i in shape['points']]
            draw.polygon(points, outline=colors[shape['label']])
            # Draw a boundary around the polygon
            min_x = min(x for x, y in points)
            max_x = max(x for x, y in points)
            min_y = min(y for x, y in points)
            max_y = max(y for x, y in points)
            draw.rectangle((min_x, min_y, max_x, max_y), outline='white')
            # Write the group_id in the middle of polygon
            font = ImageFont.truetype('arial.ttf', 30)
            draw.text((min_x, min_y), str(shape['group_id']), fill='white', font=font)
            # Append record to pandas dataframe
            # Set "'class': shape['label']" with "'class': str(shape['group_id'])" when treating each tooth separately
            rows.append({'filename': raw_filename, 'class': shape['label'],
                         'xmin': min_x, 'xmax': max_x, 'ymin': min_y, 'ymax': max_y})

        if save_annotated_images:
            # Store image with annotations and bounded boxes in generated directory
            annotated_filename = 'bounded-' + row['teeth-annotations-file'].replace('.json', '.png')
            image.save(save_in + annotated_filename)
    
    # Create data frame
    df = pd.DataFrame(rows)
    # Next, create annotation.txt file from the data frame for training
    with open(save_in + annotation_filename, 'w+') as f:
        # Iterate for each row in the data frame
        for idx, row in df.iterrows():
            # Should first convert to int from float
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            # Now write the annotations (PRECISELY IN THIS ORDER)
            if store_dimensions:
                # Read the image to get shape
                img = cv.imread(save_in + row['filename'])
                width, height = img.shape[:2]
                line = f"{row['filename']},{str(width)},{str(height)},{str(x1)},{str(y1)},{str(x2)},{str(y2)},{row['class']}\n"
            else:
                line = f"{row['filename']},{str(x1)},{str(y1)},{str(x2)},{str(y2)},{row['class']}\n"
            f.write(line)


def generate_mask(image_file, annotations_file, labels, save_file=True):
    '''
    Generates a mask (black canvas) from polygons in annotations for given image file

    ----------
    image_file : image file (PNG, JPG, etc.)
    save_file : whether to save resized image as file or not. The default is True.

    Returns
    -------
    mask_numpy : numpy array of mask image
    '''
    # Search for respective resized image file
    img = cv.imread(WORKING_DIRECTORY + image_file)
    img_size = img.shape[:-1]
    # Read teeth annotations
    teeth_annotations = json.load(open(DATA_DIRECTORY + annotations_file, 'r'))
    # Get shapes
    annotations = teeth_annotations['shapes']
    # Draw black canvas
    mask_numpy = {v:Image.new('L', img_size[::-1], 0) for v in list(labels)}
    
    # For each polygon in annotations
    for poly in annotations:
        label = poly['label']
        mask_image = mask_numpy[label]
        # Read all points in the polygon
        poly_path = poly['points']
        poly_path = [(int(p)) for P in poly_path for p in P]
        # Set the mask of target label to numpy array of image (not the image itself)
        mask_numpy[label] = np.array(mask_image)
        # Draw polygon path from point to point on black canvas
        ImageDraw.Draw(mask_image).polygon(poly_path, outline=1, fill=1)
        mask_numpy[label] = mask_image
    # There can be multiple masks for multiple labels, merge and stack them
    mask_numpy = [np.array(mask) for mask in mask_numpy.values()]
    mask_numpy = np.stack(mask_numpy, axis=-1)
    if save_file:
        cv.resize(mask_numpy, (HEIGHT, WIDTH)) #TODO: Is this even needed?
        # Save as Numpy object
        np.save(WORKING_DIRECTORY + 'mask-' + str.lower(image_file) + '.npy', mask_numpy)
    return mask_numpy


def get_teeth_annotation_labels(data):
    '''
    Returns unique set of teeth annotation labels
    ----------
    data : JSON object generated via dataset_to_json function
    Returns
    -------
    types : array of unique labels in all teeth annotation files
    '''
    types = []
    for row in data:
        try:
          annotation_data = json.load(open(DATA_DIRECTORY + row['teeth-annotations-file'], 'r'))
          shapes = annotation_data['shapes']
          for polygon in shapes:
              label = polygon['label']
              types.append(label)
        except Exception as e:
          print(e)
    return set(types)


def get_confusion_matrix(y_true, y_pred):
    '''
    Confusion matrix is a table of True-negatives, False-positives, False-negatives, and True-positives specifically in this order
    '''
    #conf_matrix = tf.math.confusion_matrix(labels=y_test, predictions=predictions)
    tn, fp, fn, tp = confusion_matrix(y_true, np.rint(y_pred)).ravel()
    return tn, fp, fn, tp


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


def check_validation_image(xval, yval, model):
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
    output_msk = np.zeros((len(xval), HEIGHT, WIDTH), dtype=np.uint8)
    idx = 0
    for im in xval:
        xtest = np.array(im)
        xtest=np.expand_dims(xtest, axis=0)
        pred = model.predict(xtest)
        pred = (pred > 0.55).astype(np.uint8)
        pred = np.squeeze(pred[0])
        output_msk[idx]=pred*255
        idx+=1
    fix, ax = plt.subplots(idx, 2, figsize=(20,40))
    for i in range(idx):
        ax[i,0].set_title(' Actual Mask')
        ax[i,0].imshow(yval[i,:,:,0])
        ax[i,0].axis('off')
        ax[i,1].set_title('Predicted Mask')
        ax[i,1].imshow(output_msk[i,:,:])
        ax[i,1].axis('off')
        plt.savefig("/content/drive/MyDrive/Dental/i.png")
    plt.show()

    for i in range(idx):
        plt.savefig(WORKING_DIRECTORY + str(i) + ".png")


def convert_to_arrays(image_files, mask_files):
    """
    Converts image files (.png) and mask files (.npy) to arrays

    Parameters
    ----------
    image_files : array of file names of resized images
    mask_files : array of file names of masks

    Returns
    -------
    X : transformed images
    Y : transformed mask images

    """
    X = np.zeros((len(image_files), HEIGHT, WIDTH, CHANNELS, 1), dtype=np.float_)
    Y = np.zeros((len(image_files), HEIGHT, WIDTH, 6), dtype=np.int_)
    i = 0
    for file in image_files:
        raw_img = imread(WORKING_DIRECTORY + image_files[i])
        msk_img = np.load(WORKING_DIRECTORY + mask_files[i])
        raw_img = resize(raw_img, (HEIGHT, WIDTH, CHANNELS, 1), anti_aliasing=True)
        msk_img = cv.resize(msk_img, (HEIGHT, WIDTH))
        msk_img[msk_img > 0] = 1
        mask = msk_img
        X[i] = raw_img
        Y[i] = mask.astype('int')
        i += 1
    X = X.reshape(X.shape[0:4])
    Y = Y.reshape(Y.shape[0:4])
    return X, Y


def test_image_processing_module():
    test_files = [f for f in os.listdir(WORKING_DIRECTORY) if f.startswith('resized')]
    raw_data = np.zeros((len(test_files), HEIGHT, WIDTH, CHANNELS, 1), dtype=np.float_)
    index = 0
    for file in test_files:
      raw_img = imread(WORKING_DIRECTORY + file)
      raw_img = resize(raw_img, (HEIGHT, WIDTH, CHANNELS, 1), anti_aliasing=True)
      raw_data[index] = raw_img
      index = index + 1
    raw_data = raw_data.reshape(raw_data.shape[0:4])
    return raw_data


def generate_mask_from_annotations(annotation_file, img_size):
    if type(annotation_file) == str:
        annot_data = json.load(open(DATA_DIRECTORY + annotation_file, 'r'))
        annot_data = annot_data['shapes']
    else:
        annot_data = annotation_file
    img_size = img_size[:-1]
    mask_image = Image.new('L', img_size[::-1], 0)
    for poly in annot_data:
        poly_path = poly['points']
        poly_path = [(int(p)) for P in poly_path for p in P]
        ImageDraw.Draw(mask_image).polygon(poly_path, outline=1, fill=1)
    mask_image = np.array(mask_image)
    return mask_image


def extract_teeth(image, teeth_polygons, mask_teeth=None, mask_lesion=None, shape=(HEIGHT, WIDTH), buffer=0):
    images = []
    masks_teeth = []
    masks_lesion = []
    for polygon in teeth_polygons:
        polygon = polygon['points']
        max_x, min_x = max(polygon, key=lambda x: x[0])[0], min(polygon, key=lambda x: x[0])[0]
        max_y, min_y = max(polygon, key=lambda x: x[1])[1], min(polygon, key=lambda x: x[1])[1]
        max_x, min_x = int(max_x + buffer), int(min_x - buffer)
        max_y, min_y = int(max_y + buffer), int(min_y - buffer)
        tooth = cv.cvtColor(image[min_y:max_y, min_x:max_x], cv.COLOR_BGR2GRAY)
        tooth = cv.resize(tooth, shape)
        if mask_lesion is not None:
            mask = mask_lesion[min_y:max_y, min_x:max_x]
            mask = cv.resize(mask, shape)
            if np.sum(mask) > 0:
                masks_lesion.append(mask)
        if mask_teeth is not None:
            mask = mask_teeth[min_y:max_y, min_x:max_x]
            mask = cv.resize(mask, shape)
            masks_teeth.append(mask)
        images.append(tooth)
    images, masks_teeth, masks_lesion = np.stack(images), np.stack(masks_teeth), np.stack(masks_lesion)
    return images, masks_teeth, masks_lesion






##################
###### MAIN ######
##################
rebuild_dataset_flag = False
generate_teeth_annotations_flag = False
resize_images_flag = False
generate_masks_flag = False
split_into_test_train = False
read_prebuilt_model = False

if rebuild_dataset_flag:
    dataset_to_json()

# Read JSON file
with open('dataset.json', 'r') as f:
    data = json.load(f)
data = data['dataset']

# Limit to year 2022
data = [x for x in data if x['year'] == '2022']

# Fetch list of all OPG and Annotation files
opgs = [x['file'] for x in data]

# Get list of labels
labels = get_teeth_annotation_labels(data)

if resize_images_flag:
    # Scale OPGs to 512x512 images
    for opg in opgs:
        resize_image(opg, True)

if generate_masks_flag:
    for row in data:
        image_file = 'resized-' + row['file']
        generate_mask(image_file, row['teeth-annotations-file'], labels, True)

# Split into training, test
if split_into_test_train:
    train, test = train_test_split(data, test_size=0.1)
    print(f"Training: {len(train)}; Test: {len(test)}")

    if generate_teeth_annotations_flag:
        generate_teeth_annotations(test, save_in='test')
        generate_teeth_annotations(train, save_in='train')





'''
# Read resized images
image_files = [f for f in os.listdir(WORKING_DIRECTORY) if f.startswith('resized')]
mask_files = [f for f in os.listdir(WORKING_DIRECTORY) if f.startswith('mask')]

X, Y = convert_to_arrays(image_files, mask_files)

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.1)
print('Dataset: ', len(X))
print('Labels: ', len(labels))
print('Training set: ', len(x_train))
print('Validation set: ', len(x_val))



# TEST LESION
test_row = data[1]
test_img_path = test_row['file']
test_teeth_annotation = test_row['teeth-annotations-file']
test_lesion_annotation = test_row['lesion-annotations-file']
test_teeth_json = json.load(open(DATA_DIRECTORY + test_teeth_annotation, 'r'))['shapes']
test_lesion_json = json.load(open(DATA_DIRECTORY + test_lesion_annotation, 'r'))['shapes']
test_img = cv.imread(DATA_DIRECTORY + test_img_path)
img_size = test_img.shape
test_teeth_mask = generate_mask_from_annotations(test_teeth_annotation, img_size)
test_lesion_mask = generate_mask_from_annotations(test_lesion_annotation, img_size)

plt.imshow(test_img)
plt.imshow(test_teeth_mask)
plt.imshow(test_lesion_mask)
images, masks1, masks2 = extract_teeth(test_img, test_teeth_json, mask_teeth=test_teeth_mask, mask_lesion=test_lesion_mask, buffer=30)

plt.subplot(1, 2, 2)
plt.figure()
plt.imshow(test_img)
plt.imshow(test_teeth_mask, alpha=0.1)
plt.imshow(test_lesion_mask, alpha=0.3)
plt.show()


data_x = []
data_y = []
labels = get_teeth_annotation_labels(data)
for row in data:
    if 'teeth-annotations-file' not in row:
        continue
    print("Processing: " + row['file'])
    img = cv.imread(DATA_DIRECTORY + row['file'])
    teeth_annotation = row['teeth-annotations-file']
    lesion_annotation = row['lesion-annotations-file']
    img_size = img.shape
    teeth_mask = generate_mask_from_annotations(teeth_annotation, img_size)
    lesion_mask = generate_mask_from_annotations(lesion_annotation, img_size)
    if np.sum(lesion_mask) > 0:
        teeth, masks1, masks2 = extract_teeth(img, teeth_annotation, 
          mask_teeth=teeth_mask, mask_lesion=lesion_mask, shape=(HEIGHT, WIDTH), buffer=30)
        teeth = teeth / teeth.max()
        masks1 = masks1 / masks1.max()
        teeth = np.stack([teeth, masks1], axis=-1)
        data_x.append(teeth)
        data_y.append(masks2)

data_x = np.concatenate(data_x, axis=0)
data_y = np.concatenate(data_y, axis=0)
print(data_x.shape, data_y.shape)
plt.imshow(data_x[-1, :, :, 0])
plt.show()
plt.imshow(data_x[-1, :, :, 1])
plt.show()
plt.imshow(data_y[-1, :, :])
plt.show()

'''
