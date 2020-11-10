"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import time
import numpy as np
np.random.seed(100)
import random
from PIL import Image
random.seed(1000)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
from torchvision import transforms
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import gc
import tensorflow as tf

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycococreatortools.pycocotools.coco import COCO
from pycococreatortools.pycocotools.cocoeval import COCOeval
from pycococreatortools.pycocotools import mask as maskUtils

current_dir = os.getcwd()
ROOT_DIR = current_dir + '/Mask_RCNN'

# Import Mask RCNN
sys.path.insert(0, ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log

from preprocess import CreateSegMasks
from preprocess import PascalToJson

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

############################################################
#  Configurations
############################################################


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "parts"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 44  # background + 44 parts

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 256
    # IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    #STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    #VALIDATION_STEPS = 5


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, val_type, class_ids=None, return_coco=False):
        """Load a subset of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if val_type == 'train':
            coco = COCO(current_dir + '/data/unmask/VOC2010/train.json')
            image_dir = current_dir + '/data/unmask/VOC2010/Images/train'
        elif val_type == 'val':
            coco = COCO(current_dir + '/data/unmask/VOC2010/val.json')
            image_dir = current_dir + '/data/unmask/VOC2010/Images/val'
        elif val_type == 'test':
            coco = COCO(current_dir + '/data/unmask/VOC2010/test.json')
            image_dir = current_dir + '/data/unmask/VOC2010/Images/test'

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for index, id in enumerate(class_ids):
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": np.asarray([bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]]),
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


def preprocess():
    print("Creating Segmentation Masks")
    seg_masks = CreateSegMasks()
    seg_masks.create_masks()
    seg_masks.split_images()

    print("Creating JSON files from Segmentation Masks")
    ptj = PascalToJson()
    ptj.create_json_from_masks()


class Extract:
    def __init__(self, model_path=None, objects_to_consider='all', verbose=0):
        self.images = None
        self.predicted_labels = None
        self.label_map_reverse = None
        self.label_map = None
        self.dataset_val = None
        self.verbose = verbose

        self.object_parts = {
            'sheep': {'ear', 'eye', 'horn', 'torso', 'head', 'neck', 'tail', 'muzzle', 'leg'},
            'aeroplane': {'stern', 'body', 'engine', 'wheel', 'tail', 'wing'},
            'dog': {'paw', 'nose', 'ear', 'eye', 'torso', 'head', 'neck', 'tail', 'muzzle', 'leg'},
            'person': {'ear', 'nose', 'eye', 'foot', 'torso', 'arm', 'mouth', 'head', 'neck', 'hair', 'hand', 'leg',
                       'eyebrow'},
            'cat': {'paw', 'nose', 'ear', 'eye', 'torso', 'head', 'neck', 'tail', 'leg'},
            'car': {'vehicle_left', 'mirror', 'door', 'headlight', 'wheel', 'frontside', 'vehicle_back', 'licenseplate',
                    'vehicle_top', 'window', 'vehicle_right'},
            'bicycle': {'wheel', 'saddle'},  # 'handlebar',
            'bird': {'eye', 'beak', 'foot', 'torso', 'head', 'neck', 'tail', 'wing', 'leg'},
            'tvmonitor': {'screen'},
            'motorbike': {'wheel', 'saddle', 'headlight'},  # 'handlebar',
            'bus': {'vehicle_left', 'mirror', 'door', 'headlight', 'wheel', 'frontside', 'vehicle_back', 'licenseplate',
                    'vehicle_top', 'window', 'vehicle_right'},
            'train': {'coach_front', 'coach_right', 'coach_left', 'headlight', 'coach_top', 'head', 'coach_back'},
            'horse': {'ear', 'eye', 'torso', 'head', 'hoof', 'neck', 'tail', 'muzzle', 'leg'},
            'pottedplant': {'plant', 'pot'},
            'bottle': {'body', 'cap'},
            'cow': {'ear', 'eye', 'horn', 'torso', 'head', 'neck', 'tail', 'muzzle', 'leg'}}

        # only consider a subset of the objects
        object_part_subset = {}
        if objects_to_consider != 'all':
            for object in objects_to_consider:
                object_part_subset[object] = self.object_parts[object]
        self.object_parts = object_part_subset

        class InferenceConfig(ShapesConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0

        self.config = InferenceConfig()
        # self.config.display()

        self.model = self.load_model(model_path)

    # def calculate_part_overlap(self):
    #     unique_parts = set()
    #     part_sets = []
    #     print("Number of classes: {}".format(len(self.object_parts.keys())))
    #     for obj, parts in self.object_parts.items():
    #         part_sets.append(set(parts))
    #         for part in parts:
    #             unique_parts.add(part)
    #
    #     set_intersection = set.intersection(*part_sets)

    def train_model(self, model_type):
        # Make sure data is preprocessed
        preprocess()
        config = ShapesConfig()

        # Create model
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

        # Select weights file to load
        if model_type == "coco":
            model_path = COCO_MODEL_PATH
        elif model_type == "last":
            # Find last trained weights
            model_path = model.find_last()
        elif model_type == "imagenet":
            # Start from ImageNet trained weights
            model_path = model.get_imagenet_weights()
        else:
            model_path = model_type

        # Load weights
        if model_type == "last":
            print("Loading weights ", model_path)
            model.load_weights(model_path, by_name=True)
        else:
            print("Loading weights ", model_path)
            model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        # Training dataset
        dataset_train = CocoDataset()
        val_type = "train"
        dataset_train.load_coco(val_type)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val"
        dataset_val.load_coco(val_type)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads', augmentation=augmentation)
        # # K.clear_session()

        # Training - Stage 2
        # Fine tune layers from ResNet stage 4 and up
        print("\nFine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=2, layers='4+', augmentation=augmentation)
        # K.clear_session()

        # Training - Stage 3
        # Fine tune all layers
        print("\nFine tune all layers")
        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=3, layers='all', augmentation=augmentation)

    def load_model(self, model_path):
        model_dir = os.getcwd() + '/Mask_RCNN/logs'
        if model_path is None:
            if self.verbose > 0: print("Training a new mask R-CNN model")

            self.train_model(model_type='coco')
            model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=model_dir)
            model_path = model.find_last()
            model.load_weights(model_path, by_name=True)

        else:
            if self.verbose > 0: print("Loading mask R-CNN model", model_path)
            model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=model_dir)
            model.load_weights(model_path, by_name=True)

        self.dataset_val = CocoDataset()
        self.dataset_val.load_coco("test", return_coco=False)
        self.dataset_val.prepare()

        return model

    def evaluate_model(self, model):
        # Test dataset
        dataset_test = CocoDataset()
        val_type = "test"
        coco = dataset_test.load_coco(val_type, return_coco=True)
        dataset_test.prepare()
        evaluate_coco(model, dataset_test, coco, "bbox")

    def visualize_segmentations(self, config, model):
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "train"
        coco = dataset_val.load_coco(val_type, return_coco=True)
        dataset_val.prepare()

        # Test on a random image
        image_id = random.choice(dataset_val.image_ids)

        # Test on a specific image
        image_to_test = '2010_000898.jpg'
        saved_index = -1
        for index, value in enumerate(dataset_val.image_info):
            file = value['path']
            info = file.rsplit('/', 1)
            file_name = info[1]
            if file_name == image_to_test:
                saved_index = index


        image_id = saved_index
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config,
                                                                                           image_id, use_mini_mask=False)

        log("original_image", original_image)
        log("image_meta", image_meta)
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)

        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_val.class_names,
                                    figsize=(8, 8), name='ground_truth.png')

        results = model.detect([original_image], verbose=1)

        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names,
                                    r['scores'], ax=get_ax(), name='predicted.png')

        # Compute VOC-Style mAP @ IoU=0.5
        # Running on 10 images. Increase for better accuracy.
        image_ids = np.random.choice(dataset_val.image_ids, 10)
        APs = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config, image_id,
                                                                                      use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                                                 r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)

        print("mAP: ", np.mean(APs))

    def extract(self, images, label_map):
        image_features = []
        for image_index, image in enumerate(images):
            im = np.asarray(transforms.ToPILImage()(image).convert("RGB"))

            image_resized, _, _, _, _ = utils.resize_image(im,
                                                   min_dim=self.config.IMAGE_MIN_DIM,
                                                   min_scale=self.config.IMAGE_MIN_SCALE,
                                                   max_dim=self.config.IMAGE_MAX_DIM,
                                                   mode=self.config.IMAGE_RESIZE_MODE)

            results = self.model.detect([image_resized], verbose=0)
            # visualize.display_instances(image_resized, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names,
            #                             r['scores'], ax=get_ax(), name='predicted.png')

            feature_list = []
            predicted_ids = results[0]['class_ids']
            id_names = self.dataset_val.class_names

            for pred_id in predicted_ids:
                pred_feature = id_names[pred_id]
                feature_list.append(pred_feature)

            image_features.append(list(set(feature_list)))

        predictions = self.search_for_closest_match(image_features, label_map)

        return image_features, np.asarray(predictions)

    def search_for_closest_match(self, image_features, label_map):
        predictions = []
        for image_feat in image_features:
            best_sim = 0
            best_obj = None
            # find most similar class based on image features
            for obj, parts in self.object_parts.items():
                sim = self.compute_attribute_similarity(parts, image_feat)
                if sim > best_sim and obj in label_map.keys():
                    best_sim = sim
                    best_obj = obj
            if best_obj is None:
                predictions.append(-1)
            else:
                predictions.append(label_map[best_obj])

        return predictions

    def predict(self):
        # find most similar animal based on image features
        predictions = []
        for index, image_feat in enumerate(self.image_features):
            label = self.predicted_labels[index]
            label_name = self.label_map_reverse[label]
            parts = self.object_parts[label_name]
            sim = self.compute_attribute_similarity(parts, image_feat)

            if sim > 0.3:
                predictions.append(0)
            else:
                predictions.append(1)

        return predictions

    # calculate the similarity of the features to the predicted label
    def decision_function(self, image_features, predicted_labels, label_map_reverse):
        similarity = []
        for index, image_feat in enumerate(image_features):
            label = predicted_labels[index]
            label_name = label_map_reverse[label]
            parts = self.object_parts[label_name]
            sim = self.compute_attribute_similarity(parts, image_feat)

            similarity.append(1.0 - sim)

        return similarity

    def compute_attribute_similarity(self, a, b):
        sim_score = 0.0
        c = set(a).intersection(set(b))

        if len(a) > 0 or len(b) > 0:
            sim_score = float(len(c))/(len(a) + len(b) - len(c))

        return sim_score

    def reset_keras(self):
        sess = get_session()
        clear_session()
        K.clear_session()
        sess.close()

        try:
            del self.model
        except:
            pass

        gc.collect()  # if it's done something you should see a number being outputted

        # use the same config as you used to create the session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))


