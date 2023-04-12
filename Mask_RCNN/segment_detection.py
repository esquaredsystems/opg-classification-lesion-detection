import io
import os
import sys
import json
import base64
import shutil

import numpy as np
import pandas as pd
import time
import skimage
import cv2 as cv
import mrcnn.utils as utils
import mrcnn.model as modellib
from random import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from skimage import io
from mrcnn.config import Config
from mrcnn import visualize

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = '.'
DATA_DIRECTORY = '../dataset/'
WORKING_DIRECTORY = '../generated/'
TRAIN_DIR_NAME = 'train_segment_detection'
TEST_DIR_NAME = 'test_segment_detection'
HEIGHT = 832
WIDTH = 832
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

# Import mrcnn libraries
sys.path.append(ROOT_DIR)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Use this configuration when detecting teeth names
# CATEGORIES = ['canine', 'central incisor', 'lateral incisor', 'molar', 'premolar', 'implant']
# COLORS = ['cyan', 'magenta', 'yellow', 'red', 'blue', 'gold']
# CLASS_VARIABLE_IN_ANNOTATIONS_FILE = 'label'

# Use this configuration when detecting teeth numbers
CATEGORIES = ['11', '12', '13', '14', '15', '16', '17', '18', '21', '22', '23', '24', '25', '26', '27', '28',
              '31', '32', '33', '34', '35', '36', '37', '38', '41', '42', '43', '44', '45', '46', '47', '48']
COLORS = ['#FF00FF', '#FFCE44', '#FF0000', '#4CBB17', '#808000', '#0071C5', '#87CEEB', '#87CEFA',
          '#FF77FF', '#FFFF9F', '#FF6347', '#228B22', '#2E8B57', '#B0E0E6', '#CCCCFF', '#4682B4',
          '#FF77FF', '#FFD700', '#800000', '#32CD32', '#355E3B', '#0F52BA', '#A1CAF1', '#1560BD',
          '#FFB6C1', '#FFDB58', '#FF2400', '#9DC183', '#98FF98', '#5F9EA0', '#0047AB', '#008080']
CLASS_VARIABLE_IN_ANNOTATIONS_FILE = 'group_id'


class DentalOPGConfig(Config):
    """Configuration for training on the cigarette butts dataset.
    Derives from the base Config class and overrides values specific
    to the cigarette butts dataset.
    """
    # Give the configuration a recognizable name
    NAME = "dental_opg"
    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 32  # background + 6 (teeth types)
    # Our training images are 800x800
    IMAGE_MIN_DIM = HEIGHT
    IMAGE_MAX_DIM = HEIGHT
    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 100
    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5
    # Matterport originally used resnet101, but I downsized to fit on my GPU
    BACKBONE = 'resnet50'
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # When ready to learn final model, up them a notch from 16-256
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000


class OPGDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_data(self, annotation_json, images_dir):
        """ Load the dataset from CSV file
        Args:
            annotation_json: The path to the annotations csv file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file

        json_file = open(annotation_json)
        opg_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "dental_opg"
        for item in CATEGORIES:
            class_name = item
            class_id = CATEGORIES.index(item) + 1
            if class_id < 1:
                print(f'Error: Class id for {item} cannot be less than one. (0 is reserved for the background)')
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        images = opg_json['images']
        for image in images:
            annotations[image['id']] = []
        for annotation in opg_json['annotations']:
            for image in images:
                image_id = image['id']
                if image_id == annotation['image_id']:
                    annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        for image in opg_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            mask_draw.polygon(annotation['segmentation'], fill=1)
            bool_array = np.array(mask) > 0
            instance_masks.append(bool_array)
            class_ids.append(class_id)
            # for segmentation in annotation['segmentation']:
            #     mask_draw.polygon(segmentation, fill=1)
            #     bool_array = np.array(mask) > 0
            #     instance_masks.append(bool_array)
            #     class_ids.append(class_id)
        try:
            mask = np.dstack(instance_masks)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        except ValueError as ex:
            print(f'ValueError: {ex} when loading masks for image: {image_id}')


class InferenceConfig(DentalOPGConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = WIDTH
    IMAGE_MAX_DIM = WIDTH
    DETECTION_MIN_CONFIDENCE = 0.85


def encrypt_name(name):
    mybytes = name.encode('utf-8')
    num = int.from_bytes(mybytes, 'little')
    return num


def decrypt_name(num):
    bytes = num.to_bytes((num.bit_length() + 7) // 8, 'little')
    return bytes.decode('utf-8')


def dataset_to_json(dirname=DATA_DIRECTORY):
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
        if len(parts) == 4:
            # OPG data
            if parts[0] == 'opg':
                opg_data = {'type': parts[0], 'source': parts[1], 'year': parts[2], 'file': file}
                opgs.append(opg_data)
        elif len(parts) == 5:
            # Annotations data
            annotations_data = {'type': parts[0] + '-' + parts[1], 'source': parts[2], 'year': parts[3], 'file': file}
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


def generate_teeth_annotations(data, save_in=TRAIN_DIR_NAME, save_annotated_images=True, store_dimensions=False, resize=False):
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

    ### START OF SAMPLE BLOCK ###
    sample = data[1]
    teeth = json.load(open(DATA_DIRECTORY + sample['teeth-annotations-file'], 'r'))
    image_data = teeth['imageData']
    decoded_data = base64.b64decode(image_data)
    stream = io.BytesIO(decoded_data)
    image = Image.open(stream)
    draw = ImageDraw.Draw(image)
    for shape in teeth['shapes']:
        points = [tuple(i) for i in shape['points']]
        min_x, max_x = min(x for x, y in points), max(x for x, y in points)
        min_y, max_y = min(y for x, y in points), max(y for x, y in points)
        # Index of this shape
        class_name = str(shape[CLASS_VARIABLE_IN_ANNOTATIONS_FILE])
        idx = CATEGORIES.index(class_name)
        draw.polygon(points, outline=COLORS[idx])
        draw.rectangle((min_x, min_y, max_x, max_y), outline='white')
        draw.text((min_x, min_y), str(shape['group_id']), fill='white')
    image.save('sample.png')
    image.show()
    # cv2_imshow(img)
    ### END OF SAMPLE BLOCK ###

    annotation_json = {
        'info': {'contributor': "Owais Hussain", 'date_created': "2023-01-25", 'description': "Dental OPG annotations"},
        'images': [], 'annotations': []}

    for row in data:
        annotations_file = row['teeth-annotations-file']
        print("Generating teeth annotations for: " + annotations_file)
        # Read file ID (messy way, very specific to OPG dataset)
        id_part = encrypt_name(annotations_file.replace('.json', ''))
        # Read teeth annotation file
        teeth = json.load(open(DATA_DIRECTORY + annotations_file, 'r'))
        # Read image data coded in Base64 string
        image_data = teeth['imageData']
        # Decode the base64 data
        decoded_data = base64.b64decode(image_data)
        # Convert the decoded data to a stream
        stream = io.BytesIO(decoded_data)
        # Use the Image module to open the stream as an image
        image = Image.open(stream)
        # Get scaling factors for image resizing
        sw, sh = WIDTH / image.size[0], HEIGHT / image.size[1]
        # Resize the image to predefined dimensions
        if resize:
            image = image.resize((WIDTH, HEIGHT))
        # Store plain image in generated directory
        raw_filename = 'raw-' + str(encrypt_name(annotations_file.replace('.json', ''))) + '.png'
        image.save(save_in + raw_filename)
        annotation_json['images'].append({"license": 0, "file_name": raw_filename,
                                          "width": image.size[0], "height": image.size[1], "id": id_part})
        draw = ImageDraw.Draw(image)
        count = 0
        for shape in teeth['shapes']:
            # Draw the polygon boundary on the image
            points = [tuple(i) for i in shape['points']]
            min_x = min(x for x, y in points)
            max_x = max(x for x, y in points)
            min_y = min(y for x, y in points)
            max_y = max(y for x, y in points)
            # Resize rectangle to match with new resolution
            if resize:
                min_x = int(sw + sw * min_x)
                max_x = int(sw + sw * max_x)
                min_y = int(sh + sh * min_y)
                max_y = int(sh + sh * max_y)
                new_points = []
                for point in points:
                    x, y = point
                    x, y = (sw + sw * x), (sh + sh * y)
                    new_points.append(tuple([x, y]))
                points = new_points

            # Draw a boundary around the polygon
            class_name = str(shape[CLASS_VARIABLE_IN_ANNOTATIONS_FILE])
            idx = CATEGORIES.index(class_name)
            draw.polygon(points, outline=COLORS[idx])
            # Draw the rectangle around each tooth
            draw.rectangle((min_x, min_y, max_x, max_y), outline='white')
            # Write the group_id in the middle of polygon
            # draw.text((min_x, min_y), str(shape['group_id']), fill='white', font=ImageFont.truetype('arial.ttf', 30))
            # For some reason, font does not get set in Colab, so falling back to default
            draw.text((min_x, min_y), str(shape['group_id']), fill='white')
            # Get annotations for each tooth
            points = [[round(px, ndigits=3), round(py, ndigits=3)] for px, py in points]
            annotation_json['annotations'].append({
                "iscrowd": 0,  # From Coco dataset, not used here
                "id": count,  # ID of the tooth in this image
                "image_id": id_part,  # ID of the image
                "category_id": idx + 1,  # Label ID is index + 1
                "bbox": [min_x, max_x, min_y, max_y],  # Bounded box
                "area": (max_x - min_x) * (max_y - min_y),  # Box area (TODO: or is it of the shape?)
                "segmentation": np.concatenate(points).tolist()
            })  # All segments
            count += 1

        if save_annotated_images:
            annotated_filename = save_in + 'bounded-' + str(
                encrypt_name(annotations_file.replace('.json', ''))) + '.png'
            image.save(annotated_filename)
            img = cv.imread(annotated_filename)
            assert (img is not None)

    # output
    with open(save_in + 'annotations.json', 'w') as f:
        json.dump(annotation_json, f)
    # Closing file
    f.close()


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
    mask_numpy = {v: Image.new('L', img_size[::-1], 0) for v in list(labels)}

    # For each polygon in annotations
    for poly in annotations:
        label = poly[CLASS_VARIABLE_IN_ANNOTATIONS_FILE]
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
        cv.resize(mask_numpy, (HEIGHT, WIDTH))  # TODO: Is this even needed?
        # Save as Numpy object
        np.save(WORKING_DIRECTORY + 'mask-' + str.lower(image_file) + '.npy', mask_numpy)
    return mask_numpy


### These flags control what parts of the notebook to execute ###
rebuild_dataset_flag = False  # Rebuild the dataset (should use only when doing all from scratch)
resize_images_flag = False  # Resize image to WIDTH and HEIGHT
generate_masks_flag = False  # Generate masks and superimpose on the teeth segmentation model
split_into_test_train = False  # Split into training and test and generate annotations.json files from teeth annotations
retrain_model = False  # If pre-trained model already exist, then use them

# This is where all the fun happens
if rebuild_dataset_flag:
    dataset_to_json()

# Read JSON file
with open('../dataset.json', 'r') as f:
    data = json.load(f)
data = data['dataset']

# Limit to year 2022 and specific data sources
data = [x for x in data if x['year'] == '2022' and x['source'] in ['aku', 'riphah', 'tufts', 'ufb']]

# Remove troublesome data (invalid images, empty annotations, etc.)
invalid_ids = ['tft22-1032.jpg', 'tft22-503.jpg']
data = [x for x in data if x['file'] not in invalid_ids]

# Fetch list of all OPG and Annotation files
opgs = [x['file'] for x in data]

if generate_masks_flag:
    for row in data:
        image_file = 'resized-' + row['file']
        generate_mask(image_file, row['teeth-annotations-file'], CATEGORIES, True)

# Split into training, test
if split_into_test_train:
    train, test = train_test_split(data, test_size=0.1)
    print(f"Training: {len(train)}; Test: {len(test)}")
    generate_teeth_annotations(test, save_in=TEST_DIR_NAME, resize=True)
    generate_teeth_annotations(train, save_in=TRAIN_DIR_NAME, resize=True)

### MASK RCNN ###
# Read config
config = DentalOPGConfig()
config.display()

dataset_val = OPGDataset()
dataset_val.load_data(WORKING_DIRECTORY + TEST_DIR_NAME + '/annotations.json', WORKING_DIRECTORY + TEST_DIR_NAME)
dataset_val.prepare()
dataset_train = OPGDataset()
dataset_train.load_data(WORKING_DIRECTORY + TRAIN_DIR_NAME + '/annotations.json', WORKING_DIRECTORY + TRAIN_DIR_NAME)
dataset_train.prepare()

dataset = dataset_val
image_ids = dataset.image_ids
for image_id in image_ids:
    image = dataset.load_image(image_id)
    try:
        mask, class_ids = dataset.load_mask(image_id)
    except:
        print(f"Failed to load masks for {dataset.image_info[image_id]['path']}")

dataset = dataset_train
image_ids = np.random.choice(dataset.image_ids, 3)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
if retrain_model:
    start_train = time.time()
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, augmentation=None, epochs=500,
                layers='heads')
    end_train = time.time()
    minutes = round((end_train - start_train) / 60, 2)
    print(f'Training took {minutes} minutes')

inference_config = InferenceConfig()
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
# Get path to saved weights
model_path = model.find_last()
# If the above doesn't work, try: model_path = os.path.join(".h5 file path here")

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Real test
real_test_dir = WORKING_DIRECTORY + 'real_test/'
image_paths = []
for filename in os.listdir(real_test_dir):
     if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

results = []
for image_path in image_paths:
    image_file = image_path.split('/')[-1]
    try:
        print(f"Showing {image_path}")
        img = skimage.io.imread(image_path)
        img_arr = np.array(img)
        r = model.detect([img_arr], verbose=1)[0]
        plt, ax = visualize.display_instances(
            img, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'])
        ax.figure.savefig(f"{real_test_dir}/lesion_results/lesion-{image_file}", dpi=300)
        boxes = r['rois']
        masks = r['masks']
        scores = r['scores']
        class_ids = r['class_ids']
        N = len(boxes)
        segments = []
        for i in range(N):
            # Bounding box
            if not np.any(boxes[i]):
                continue
            ymin, xmin, ymax, xmax = boxes[i].tolist()
            # Check that the box coordinates are valid
            if xmin >= xmax or ymin >= ymax or xmax > img.shape[1] or ymax > img.shape[0]:
                print('Invalid bounding box coordinates')
            print(f"Box {ymin}, {xmin}, {ymax}, {xmax}; Score: {scores[i]}; Class: {class_ids[i]}")
            cropped = img[ymin:ymax, xmin:xmax]
            io.imsave(f"{real_test_dir}segment_results/{class_ids[i]}-{image_file}", cropped)
            results.append({'image': image_file, 'class_id': class_ids[i], 'score': scores[i],
                            'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax})
    except Exception as ex:
        print (ex)

# Store the results
df = pd.DataFrame(results)
df.to_csv(real_test_dir + 'segment_results/segment_results.csv')
