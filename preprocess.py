#!/usr/bin/env python3

import sys
import os
current_dir = os.getcwd()
sys.path.insert(0, current_dir + '/parts_extraction')
sys.path.insert(0, current_dir + '/ImageNet_Utils')
import datetime
import random
import json
import re
import fnmatch
from PIL import Image
from pycococreatortools import pycococreatortools
import numpy as np
import skimage.io as io
from tqdm import tqdm
from collections import defaultdict
import warnings
from shutil import copyfile
from parts_extraction.anno import ImageAnnotation
import glob


"""
This class based on https://github.com/waspinator/pycococreator by user waspinator
"""
class PascalToJson:
    def __init__(self):
        current_dir = os.getcwd()
        self.ROOT_DIR = current_dir + '/data/normal/pascal/VOC2010'

        self.INFO = {
            "description": "Pascal Part Dataset",
            "url": "https://github.com/waspinator/pycococreator",
            "version": "0.1.0",
            "year": 2018,
            "contributor": "waspinator",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        self.LICENSES = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]

        self.PARTS = ['plant', 'torso', 'licenseplate', 'cap', 'eyebrow', 'tail', 'ear', 'horn', 'mouth', 'frontside',
                      'mirror', 'wheel', 'vehicle_left', 'nose', 'screen', 'arm', 'neck', 'muzzle', 'coach_left',
                      'coach_back', 'head', 'headlight', 'stern', 'pot', 'window', 'coach_right', 'coach_front', 'beak',
                      'leg', 'engine', 'saddle', 'hand', 'vehicle_back', 'paw', 'hoof', 'vehicle_top', 'vehicle_right',
                      'eye', 'wing', 'hair', 'door', 'body', 'coach_top', 'foot']  # , 'handlebar'

        self.CATEGORIES = []
        for index, part in enumerate(self.PARTS):
            self.CATEGORIES.append({
                'id': index,
                'name': part,
                'supercategory': 'part'
            })

    def filter_for_jpeg(self, root, files):
        file_types = ['*.jpeg', '*.jpg']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]

        return files

    def filter_for_annotations(self, root, files, image_filename):
        file_types = ['*.png']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
        file_name_prefix = basename_no_extension + '.*'
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
        files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

        return files

    def create_json_from_masks(self):

        train_type = "train"
        json_train_file = '{}/{}.json'.format(self.ROOT_DIR, train_type)
        if not os.path.exists(json_train_file):
            print("Creating JSON train file")
            self.convert(train_type)
        else:
            print("Skipping creation of JSON train file, already exists")

        train_type = "val"
        json_val_file = '{}/{}.json'.format(self.ROOT_DIR, train_type)
        if not os.path.exists(json_val_file):
            print("Creating JSON validation file")
            self.convert(train_type)
        else:
            print("Skipping creation of JSON validation file, already exists")

        train_type = "test"
        json_test_file = '{}/{}.json'.format(self.ROOT_DIR, train_type)
        if not os.path.exists(json_test_file):
            print("Creating JSON test file")
            self.convert(train_type)
        else:
            print("Skipping creation of JSON test file, already exists")

    def convert(self, train_type):
        mask_dir = self.ROOT_DIR + '/Part_Masks/{}'.format(train_type)
        image_dir = self.ROOT_DIR + '/Images/{}'.format(train_type)

        coco_output = {
            "info": self.INFO,
            "licenses": self.LICENSES,
            "categories": self.CATEGORIES,
            "images": [],
            "annotations": []
        }

        image_id = 1
        segmentation_id = 1

        # filter for jpeg images
        for root, _, files in os.walk(image_dir):
            image_files = self.filter_for_jpeg(root, files)

            # go through each image
            for index, image_filename in enumerate(tqdm(image_files)):
                image = Image.open(image_filename)
                image_info = pycococreatortools.create_image_info(image_id, os.path.basename(image_filename),
                                                                  image.size)
                coco_output["images"].append(image_info)

                # filter for associated png annotations
                for root, _, files in os.walk(mask_dir):
                    annotation_files = self.filter_for_annotations(root, files, image_filename)

                    # go through each associated annotation
                    for annotation_filename in annotation_files:

                        # print(annotation_filename)
                        class_id = [x['id'] for x in self.CATEGORIES if x['name'] in annotation_filename][0]

                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                        binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)

                        annotation_info = pycococreatortools.create_annotation_info(segmentation_id, image_id,
                                                                                    category_info, binary_mask,
                                                                                    image.size, tolerance=2)

                        if annotation_info is not None:
                            coco_output["annotations"].append(annotation_info)

                        segmentation_id = segmentation_id + 1

                image_id = image_id + 1

        with open('{}/{}.json'.format(self.ROOT_DIR, train_type), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)


class CreateSegMasks:
    def __init__(self):
        self.parts = set()
        self.object_parts = defaultdict(set)
        self.objects = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        current_dir = os.getcwd()

        self.root_dir = current_dir + '/data/normal/pascal/VOC2010'
        self.image_dir = current_dir + '/data/normal/pascal/VOC2010/JPEGImages'
        self.root_mask_dir = current_dir + '/data/normal/pascal/VOC2010/Part_Masks'
        self.anno_dir = current_dir + '/data/normal/pascal/VOC2010/Annotations_Part'
        self.mat_file_extension_string_length = len('.mat')
        self.mat_files_names = os.listdir(self.anno_dir)

        # get all matlab files, do an 80/10/10 for train/valid/test split respectively
        num_files = len(self.mat_files_names)
        mat_files_shuffled = self.mat_files_names
        random.Random(100).shuffle(mat_files_shuffled)

        train_split = int(0.8 * num_files)
        valid_split = int(0.9 * num_files)

        self.train_files = mat_files_shuffled[0:train_split]
        self.val_files = mat_files_shuffled[train_split:valid_split]
        self.test_files = mat_files_shuffled[valid_split:]

    def read_anno(self, file_name, image_dir, anno_dir):
        image_path = os.path.join(image_dir, file_name + '.jpg')
        anno_path = os.path.join(anno_dir, file_name + '.mat')
        an = ImageAnnotation(image_path, anno_path)
        return an.objects

    def filter_part_name(self, part):
        if 'ear' in part:
            filt_part_name = 'ear'
        elif 'eye' in part:
            filt_part_name = 'eye'
        elif 'leg' in part:
            filt_part_name = 'leg'
        elif 'pa' in part:
            filt_part_name = 'paw'
        elif 'brow' in part:
            filt_part_name = 'eyebrow'
        # elif 'handlebar' in part:
        #     filt_part_name = 'handlebar'
        elif 'hand' in part:
            filt_part_name = 'hand'
        elif 'arm' in part:
            filt_part_name = 'arm'
        elif 'horn' in part:
            filt_part_name = 'horn'
        elif 'wheel' in part:
            filt_part_name = 'wheel'
        elif 'foot' in part:
            filt_part_name = 'foot'
        elif 'cleftside' in part:
            filt_part_name = 'coach_left'
        elif 'crightside' in part:
            filt_part_name = 'coach_right'
        elif 'cfrontside' in part:
            filt_part_name = 'coach_front'
        elif 'cbackside' in part:
            filt_part_name = 'coach_back'
        elif 'croofside' in part:
            filt_part_name = 'coach_top'
        elif 'hroofside' in part or 'hleftside' in part or 'hrightside' in part or 'hfrontside' in part or 'hbackside' in part or 'hroofside' in part or 'coach' in part:
            filt_part_name = 'skip'
        elif 'rightside' in part:
            filt_part_name = 'vehicle_right'
        elif 'leftside' in part:
            filt_part_name = 'vehicle_left'
        elif 'roofside' in part:
            filt_part_name = 'vehicle_top'
        elif 'backside' in part:
            filt_part_name = 'vehicle_back'
        elif 'wing' in part:
            filt_part_name = 'wing'
        elif 'engine' in part:
            filt_part_name = 'engine'
        elif 'window' in part:
            filt_part_name = 'window'
        elif 'headlight' in part:
            filt_part_name = 'headlight'
        elif 'door' in part:
            filt_part_name = 'door'
        elif 'plate' in part:
            filt_part_name = 'licenseplate'
        elif 'mirror' in part:
            filt_part_name = 'mirror'
        elif 'lbho' in part or 'lfho' in part or 'rbho' in part or 'rfho' in part:
            filt_part_name = 'hoof'
        elif part in self.objects:
            filt_part_name = 'skip'
        else:
            filt_part_name = part

        return filt_part_name

    def split_images(self):
        # check to see if images have been split into train/val/test
        train_path = self.root_dir + '/Images/train'
        val_path = self.root_dir + '/Images/val'
        test_path = self.root_dir + '/Images/test'

        if not os.path.exists(train_path):
            print("Splitting JPEG images into train folder")
            os.makedirs(train_path)
            self.copy(train_path, self.train_files)
        else:
            print("JPEG images already split into train folder")

        if not os.path.exists(val_path):
            print("Splitting JPEG images into validation folder")
            os.makedirs(val_path)
            self.copy(val_path, self.val_files)
        else:
            print("JPEG images already split into validation folder")

        if not os.path.exists(test_path):
            print("Splitting JPEG images into test folder")
            os.makedirs(test_path)
            self.copy(test_path, self.test_files)
        else:
            print("JPEG images already split into test folder")

    def copy(self, new_image_dir, files):
        for file in files:
            file_path_old = self.image_dir + "/" + file.replace('.mat', '.jpg')
            file_path_new = new_image_dir + "/" + file.replace('.mat', '.jpg')
            copyfile(file_path_old, file_path_new)

    def create_masks(self):
        mask_dir_train = self.root_mask_dir + '/train'
        mask_dir_val = self.root_mask_dir + '/val'
        mask_dir_test = self.root_mask_dir + '/test'

        if not os.path.exists(mask_dir_train):
            os.makedirs(mask_dir_train)
            print("Creating train segmentation masks")
            self.convert(self.train_files, mask_dir_train)

        else:
            print("Skipping creation of train segmentation masks, already exists")

        if not os.path.exists(mask_dir_val):
            os.makedirs(mask_dir_val)
            print("Creating validation segmentation masks")
            self.convert(self.val_files, mask_dir_val)
        else:
            print("Skipping creation of validation segmentation masks, already exists")

        if not os.path.exists(mask_dir_test):
            os.makedirs(mask_dir_test)
            print("Creating test segmentation masks")
            self.convert(self.test_files, mask_dir_test)
        else:
            print("Skipping creation of test segmentation masks, already exists")

    def convert(self, mat_files, mask_dir):
        # get the unique parts and map them to an id
        for current_mat_file_name in tqdm(mat_files):

            file_name = current_mat_file_name[:-self.mat_file_extension_string_length]
            objects = self.read_anno(file_name, self.image_dir, self.anno_dir)

            for object in objects:
                # check if there are parts in the object
                if len(object.parts) > 0:
                    object_name = object.class_name
                    for part_obj in object.parts:
                        part = part_obj.part_name
                        self.parts.add(part)
                        self.object_parts[object_name].add(part)

        # filter the parts set to be more 'general'
        parts = list(self.parts)
        filtered_parts = set()

        for part in parts:
            if 'ear' in part:
                filtered_parts.add('ear')
            elif 'eye' in part:
                filtered_parts.add('eye')
            elif 'leg' in part:
                filtered_parts.add('leg')
            elif 'pa' in part:
                filtered_parts.add('paw')
            elif 'brow' in part:
                filtered_parts.add('eyebrow')
            # elif 'handlebar' in part:
            #     filtered_parts.add('handlebar')
            elif 'hand' in part:
                filtered_parts.add('hand')
            elif 'arm' in part:
                filtered_parts.add('arm')
            elif 'horn' in part:
                filtered_parts.add('horn')
            elif 'wheel' in part:
                filtered_parts.add('wheel')
            elif 'foot' in part:
                filtered_parts.add('foot')
            elif 'cleftside' in part:
                filtered_parts.add('coach_left')
            elif 'crightside' in part:
                filtered_parts.add('coach_right')
            elif 'cfrontside' in part:
                filtered_parts.add('coach_front')
            elif 'cbackside' in part:
                filtered_parts.add('coach_back')
            elif 'croofside' in part:
                filtered_parts.add('coach_top')
            elif 'hroofside' in part or 'hleftside' in part or 'hrightside' in part or 'hfrontside' in part or 'hbackside' in part or 'hroofside' in part or 'coach' in part:
                pass
            elif 'rightside' in part:
                filtered_parts.add('vehicle_right')
            elif 'leftside' in part:
                filtered_parts.add('vehicle_left')
            elif 'roofside' in part:
                filtered_parts.add('vehicle_top')
            elif 'backside' in part:
                filtered_parts.add('vehicle_back')
            elif 'wing' in part:
                filtered_parts.add('wing')
            elif 'engine' in part:
                filtered_parts.add('engine')
            elif 'window' in part:
                filtered_parts.add('window')
            elif 'headlight' in part:
                filtered_parts.add('headlight')
            elif 'door' in part:
                filtered_parts.add('door')
            elif 'plate' in part:
                filtered_parts.add('licenseplate')
            elif 'mirror' in part:
                filtered_parts.add('mirror')
            elif 'lbho' in part or 'lfho' in part or 'rbho' in part or 'rfho' in part:
                filtered_parts.add('hoof')
            else:
                filtered_parts.add(part)

        # filter the object parts
        object_parts_filtered = defaultdict(set)
        for object, parts in self.object_parts.items():
            for part in parts:
                if 'ear' in part:
                    object_parts_filtered[object].add('ear')
                elif 'eye' in part:
                    object_parts_filtered[object].add('eye')
                elif 'leg' in part:
                    object_parts_filtered[object].add('leg')
                elif 'pa' in part:
                    object_parts_filtered[object].add('paw')
                elif 'brow' in part:
                    object_parts_filtered[object].add('eyebrow')
                # elif 'handlebar' in part:
                #     object_parts_filtered[object].add('handlebar')
                elif 'hand' in part:
                    object_parts_filtered[object].add('hand')
                elif 'arm' in part:
                    object_parts_filtered[object].add('arm')
                elif 'horn' in part:
                    object_parts_filtered[object].add('horn')
                elif 'wheel' in part:
                    object_parts_filtered[object].add('wheel')
                elif 'foot' in part:
                    object_parts_filtered[object].add('foot')
                elif 'cleftside' in part:
                    object_parts_filtered[object].add('coach_left')
                elif 'crightside' in part:
                    object_parts_filtered[object].add('coach_right')
                elif 'cfrontside' in part:
                    object_parts_filtered[object].add('coach_front')
                elif 'cbackside' in part:
                    object_parts_filtered[object].add('coach_back')
                elif 'croofside' in part:
                    object_parts_filtered[object].add('coach_top')
                elif 'hroofside' in part or 'hleftside' in part or 'hrightside' in part or 'hfrontside' in part or 'hbackside' in part or 'hroofside' in part or 'coach' in part:
                    pass
                elif 'rightside' in part:
                    object_parts_filtered[object].add('vehicle_right')
                elif 'leftside' in part:
                    object_parts_filtered[object].add('vehicle_left')
                elif 'roofside' in part:
                    object_parts_filtered[object].add('vehicle_top')
                elif 'backside' in part:
                    object_parts_filtered[object].add('vehicle_back')
                elif 'wing' in part:
                    object_parts_filtered[object].add('wing')
                elif 'engine' in part:
                    object_parts_filtered[object].add('engine')
                elif 'window' in part:
                    object_parts_filtered[object].add('window')
                elif 'headlight' in part:
                    object_parts_filtered[object].add('headlight')
                elif 'door' in part:
                    object_parts_filtered[object].add('door')
                elif 'plate' in part:
                    object_parts_filtered[object].add('licenseplate')
                elif 'mirror' in part:
                    object_parts_filtered[object].add('mirror')
                elif 'lbho' in part or 'lfho' in part or 'rbho' in part or 'rfho' in part:
                    object_parts_filtered[object].add('hoof')
                else:
                    object_parts_filtered[object].add(part)

        # save the segmentation masks
        for current_mat_file_name in tqdm(mat_files):
            part_count = defaultdict()

            file_name = current_mat_file_name[:-self.mat_file_extension_string_length]
            objects = self.read_anno(file_name, self.image_dir, self.anno_dir)

            for object in objects:
                # check if there are parts in the object
                if len(object.parts) > 0:
                    for part_obj in object.parts:
                        part = part_obj.part_name
                        part_obj.mask[part_obj.mask == 1] = 255

                        part_filt = self.filter_part_name(part)

                        if part_filt is not 'skip':
                            # keep track of the number of times each part is seen
                            if part_filt in part_count:
                                part_count[part_filt] += 1
                            else:
                                part_count[part_filt] = 1

                            part_path = os.path.join(mask_dir, file_name)
                            part_path = part_path + '_{}_{}.png'.format(part_filt, part_count[part_filt] - 1)
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                io.imsave(part_path, part_obj.mask)


class DataLoad:
    def __init__(self, class_set="cs5-2", image_size=50):
        self.class_set = class_set
        current_dir = os.getcwd()
        self.root_dir = current_dir + '/data/normal/pascal/VOC2010'
        self.image_dir = current_dir + '/data/normal/pascal/VOC2010/JPEGImages'
        self.anno_dir = current_dir + '/data/normal/pascal/VOC2010/Annotations_Part'
        self.boudingbox_xml_dir = current_dir + '/data/normal/ImageNet_Val/ILSVRC2012_bbox_val_v3/val'
        self.bouding_box_images_dir = current_dir + '/ImageNet_Utils/class_synsets'
        self.mat_extension_len = len('.mat')
        self.image_size = image_size

        self.objects = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        # table and diningtable refer to the same class

        # possible ones to keep: car, airplane, train, person, bird, dog, bottle
        if self.class_set == "cs3-1":
            self.classes_to_keep = ['car', 'person', 'train']  # max separation
        elif self.class_set == "cs3-2":
            self.classes_to_keep = ['person', 'dog', 'bird']   # lower separation
        elif self.class_set == "cs5-1":
            self.classes_to_keep = ['dog', 'car', 'bottle', 'train', 'person']  # max separation
        elif self.class_set == "cs5-2":
            self.classes_to_keep = ['dog', 'car', 'bird', 'train', 'person']

        self.file_label_number = defaultdict()
        self.file_label_name = defaultdict()

    def read_anno(self, file_name, image_dir, anno_dir):
        image_path = os.path.join(image_dir, file_name + '.jpg')
        anno_path = os.path.join(anno_dir, file_name + '.mat')
        an = ImageAnnotation(image_path, anno_path)
        return an

    def create_test_data(self, test_path, val_path):

        imagenet_dict = {
            'train': 'n04468005',
            'aeroplane': 'n02691156',
            'car': 'n02958343',
            'bird': 'n01503061',
            'person': 'n02472987',
            'dog': 'n02084071',
            'bottle': 'n02876657'
        }

        imagenet_dir = self.root_dir + "/ImageNet/"
        classes = [name for name in os.listdir(val_path)]

        for c in classes:
            class_path = test_path + "/{}".format(c)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

            class_dir = imagenet_dir + c + "/*"

            # get images to skip
            files_to_skip = set()
            skip_file_path = imagenet_dir + c + "/{}_skip_revised.txt".format(c)
            if os.path.exists(skip_file_path):
                with open(skip_file_path, "r") as f:
                    for line in f:
                        file_to_skip = line.strip()
                        file_to_skip = imagenet_dict[c] + "_" + file_to_skip + ".jpg"
                        files_to_skip.add(file_to_skip)

            files = glob.glob(class_dir)
            for file in files:
                file_name = file.rsplit('/', 1)[1]
                file_name = file_name.replace("JPEG", 'jpg')
                file_path_new = class_path + "/" + file_name

                if file_name not in files_to_skip:
                    copyfile(file, file_path_new)

    # def scanAnnotationFolder(self, annotationFolderPath):
    #     annotationFiles = []
    #     for root, dirs, files in os.walk(annotationFolderPath):
    #         for file in files:
    #             if file.endswith('.xml'):
    #                 annotationFiles.append(os.path.join(root, file))
    #     return annotationFiles

    # def synset_to_classes(self, class_count, synset_to_class_map):
    #     file_list = glob.glob(self.bouding_box_images_dir + '/*.txt')
    #
    #     # get all of the class files
    #     for file in file_list:
    #         file_name = file.rsplit('/', 1)[1]
    #         file_name = file_name.split('.txt')[0]
    #         class_count[file_name] = 0
    #
    #         # get all of the synsets for each class
    #         with open(file) as f:
    #             class_synsets = f.read().splitlines()
    #             filtered = []
    #             for synset in class_synsets:
    #                 sysnet_name = synset.split(' ', 1)[0]
    #                 filtered.append(sysnet_name)
    #
    #                 synset_to_class_map[file_name] = filtered
    #
    #     return synset_to_class_map

    # def create_test_data2(self):
    #     val_path = self.root_dir + '/Single_Objects/{}/val'.format(self.class_set)
    #     test_path = self.root_dir + '/Single_Objects/{}/temp'.format(self.class_set)
    #     data_path = self.root_dir + '/Single_Objects/test_data/{}'.format(self.image_size)
    #     allAnnotationFiles = self.scanAnnotationFolder(self.boudingbox_xml_dir)
    #
    #     if not os.path.exists(data_path):
    #         os.makedirs(data_path)
    #         class_count = {}
    #         synset_to_class_map = {}
    #         synset_to_class_map = self.synset_to_classes(class_count, synset_to_class_map)
    #         for xmlfile in tqdm(allAnnotationFiles):
    #             bbhelper = BBoxHelper(xmlfile, self.image_size, data_path, self.bouding_box_images_dir, class_count, synset_to_class_map)
    #             # Search image path according to bounding box xml, and crop it
    #             if len(bbhelper.rects_filtered) != 0:
    #                 bbhelper.saveBoundBoxImage()
    #             class_count = bbhelper.class_count
    #
    #     os.makedirs(test_path)
    #
    #     classes = [name for name in os.listdir(val_path)]
    #
    #     for c in classes:
    #         class_path = test_path + "/{}".format(c)
    #         if not os.path.exists(class_path):
    #             os.makedirs(class_path)
    #
    #         class_dir = data_path + c + "/*"
    #
    #         files = glob.glob(class_dir)
    #         for file in files:
    #             file_name = file.rsplit('/', 1)[1]
    #             file_path_new = class_path + "/" + file_name
    #             copyfile(file, file_path_new)

    def copy(self, new_image_dir, files):
        for file in files:
            file_path_old = self.image_dir + "/" + file + ".jpg"

            file_directory_class = new_image_dir + "/" + self.file_label_name[file]
            if not os.path.exists(file_directory_class):
                os.makedirs(file_directory_class)

            file_path_new = new_image_dir + "/" + self.file_label_name[file] + "/" + file + ".jpg"
            copyfile(file_path_old, file_path_new)

    # def split_data_model_m_test(self):
    #     test_path = self.root_dir + '/Single_Objects/{}/test_m'.format(self.class_set)
    #
    #     mat_files = os.listdir(self.anno_dir)
    #     single_object_files = []
    #
    #     label_dir = self.root_dir + '/ImageSets/Main'
    #     file_list = glob.glob(label_dir + '/*trainval.txt')
    #
    #     # get the object for each image
    #     for file in file_list:
    #         object_name = file.rsplit('/', 1)[1].split('_')[0]
    #         with open(file, "r") as f:
    #             for line in f:
    #                 info = line.strip().split(' ')
    #                 image = info[0]
    #                 label = info[-1]
    #
    #                 if label == "1":
    #                     self.file_label_name[image] = object_name
    #                     self.file_label_number[image] = self.objects.index(object_name)
    #
    #     # check if any data splitting needs to be done
    #     if not os.path.exists(test_path):
    #
    #         # get all the files that contain more than a single object, have parts
    #         if not os.path.exists(test_path):
    #             for file in tqdm(mat_files):
    #                 valid_include = True
    #                 file = file[:-self.mat_extension_len]
    #                 an = self.read_anno(file, self.image_dir, self.anno_dir)
    #                 num_objects = an.n_objects
    #
    #                 # for obj_index, obj in enumerate(range(num_objects)):
    #                 #     if an.objects[obj_index].class_name in self.objects_to_skip:
    #                 #         valid_include = False
    #                 if num_objects > 1 and self.file_label_name[file]  not in self.objects_to_skip:
    #                     single_object_files.append(file)
    #
    #         test_files = single_object_files
    #
    #         if not os.path.exists(test_path):
    #             print("Splitting JPEG images into test_m folder")
    #             os.makedirs(test_path)
    #             self.copy(test_path, test_files)
    #         else:
    #             print("JPEG images already split into test_m folder")

    def split_data(self):
        train_path = self.root_dir + '/Single_Objects/{}/train'.format(self.class_set)
        val_path = self.root_dir + '/Single_Objects/{}/val'.format(self.class_set)
        test_path_m = self.root_dir + '/Single_Objects/{}/test_m'.format(self.class_set)
        test_path_d = self.root_dir + '/Single_Objects/{}/test_d'.format(self.class_set)

        mat_files = os.listdir(self.anno_dir)
        random.Random(4).shuffle(mat_files)

        single_object_files = []

        label_dir = self.root_dir + '/ImageSets/Main'
        file_list = glob.glob(label_dir + '/*trainval.txt')

        # get the object for each image
        for file in file_list:
            object_name = file.rsplit('/', 1)[1].split('_')[0]
            with open(file, "r") as f:
                for line in f:
                    info = line.strip().split(' ')
                    image = info[0]
                    label = info[-1]

                    if label == "1":
                        self.file_label_name[image] = object_name
                        self.file_label_number[image] = self.objects.index(object_name)

        # check if any data splitting needs to be done
        if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path_m):

            # get all the files that contain a single object, have parts
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                for file in tqdm(mat_files):
                    file = file[:-self.mat_extension_len]
                    # an = self.read_anno(file, self.image_dir, self.anno_dir)
                    # num_objects = an.n_objects
                    #
                    # valid_include = True
                    # if num_objects >= 1:
                    #     for obj in an.objects:
                    #         if obj.class_name in self.objects_to_skip:
                    #             valid_include = False
                    #             break

                    # if num_objects >= 1 and valid_include:
                    if self.file_label_name[file] in self.classes_to_keep:
                        single_object_files.append(file)

            # split single object files into train/val/test
            num_files = len(single_object_files)

            train_split = int(0.5 * num_files)
            valid_split = int(0.6 * num_files)

            train_files = single_object_files[0:train_split]
            val_files = single_object_files[train_split:valid_split]
            test_m_files = single_object_files[valid_split:]

            # check to see if images have been split into train/val/test
            if not os.path.exists(train_path):
                print("Splitting JPEG images into train folder")
                os.makedirs(train_path)
                self.copy(train_path, train_files)
            else:
                print("JPEG images already split into train folder")

            if not os.path.exists(val_path):
                print("Splitting JPEG images into validation folder")
                os.makedirs(val_path)
                self.copy(val_path, val_files)
            else:
                print("JPEG images already split into validation folder")

            if not os.path.exists(test_path_m):
                print("Splitting JPEG images into validation folder")
                os.makedirs(test_path_m)
                self.copy(test_path_m, test_m_files)
            else:
                print("JPEG images already split into validation folder")

            if not os.path.exists(test_path_d):
                print("Splitting JPEG images into test folder")
                os.makedirs(test_path_d)
                self.create_test_data(test_path_d, train_path)
            else:
                print("JPEG images already split into test folder")


if __name__ == "__main__":
    # sm = CreateSegMasks()
    # sm.create_masks()
    # pj = PascalToJson()
    # pj.create_json_from_masks()

    dl = DataLoad()
    dl.split_data()