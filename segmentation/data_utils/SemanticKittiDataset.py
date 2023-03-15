# file is based on the official semantic kitti api for reading in point cloud data
# "laserscan.py"
# !/usr/bin/env python3

import os
import torch
import numpy as np
import torch.utils.data as data
# from .io import IO
# from .build import DATASETS
# from utils.logger import *
from pathlib import Path
from constants import ROOT_DIR
import yaml


# function to load in yaml mapping from actual labels to labels used for training
'''
    There are a few mappings in this yaml file obtained from the official semantic kitti api github
    The full list can be seen in the yaml file, we are just interested in the learning map
    The learning map maps raw labels in the dataset to labels that are used for training
'''
def load_kitti_label_map(desired_mapping='learning_map'):
    label_filename = 'SemanticKittiMappings.yaml'
    mapping_path = os.path.join(ROOT_DIR, 'segmentation', 'data_utils', label_filename)
    with open(mapping_path, 'r') as file:
        try:
            mapping = yaml.safe_load(file)[desired_mapping]
        except yaml.YAMLError as exception:
            print('ERROR RECEIVED LOADING YAML MAPPINGS: '+str(exception))
    return mapping

# normalize point cloud method from ShapeNetDataLoader
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

# @DATASETS.register_module()
class SemanticKitti(data.Dataset):
    # splits can train,test,val
    # experimental is a flag that is set to use different splits
    # i.e we want to test on a portion of the training set instead of the official splits
    def __init__(self, npoints=2048, split='train', experimental=True):
        self.data_root = os.path.join('.', '..', '..', 'data', 'SemanticKitti')
        self.npoints = npoints
        self.split = split.strip().lower()
        self.experimental = experimental
        self.learning_map = load_kitti_label_map('learning_map')

        # specific scenes are used for train,val, and test respectively
        # we do not have access to test labels, but we need to output test results and submit to competition site
        # if the experimental flag is set, we use different splits so we have labels for all data
        scenes = []
        if self.split == 'train':
            if self.experimental:
                scenes = ['00', '01', '02', '03', '06', '07', '09', '10']
            else:
                scenes = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            if self.experimental:
                scenes = ['08']
            else:
                scenes = ['08']
        elif self.split == 'test':
            if self.experimental:
                scenes = ['04', '05']
            else:
                scenes = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

        self.file_list = []
        for scene in scenes:
            match_path = os.path.join(self.data_root, 'SemanticKitti', 'dataset', 'sequences', scene)
            self.file_list += list(
                Path.glob(Path(match_path), os.path.join('*', '*.bin'))
            )

        # for each velodyne point cloud frame, we check if there is a label
        for i, point_file in enumerate(self.file_list):
            # XXXXXX from XXXXXX.bin
            frame_number = point_file.stem
            # frame_number = os.path.basename(point_file)[-4]
            # grabbing label path
            label_assumed_path = point_file.parent.parent.joinpath(Path('labels', frame_number + '.label'))
            # label_assumed_path = point_file.joinpath(Path('..','labels',str(frame_number),'.label'))
            if label_assumed_path.exists():
                # add the label as a tuple
                self.file_list[i] = (point_file, label_assumed_path)
            else:
                # set the label to None
                self.file_list[i] = (point_file, None)

    def __getitem__(self, idx):
        point_cloud_file, label_file = self.file_list[idx]
        point_cloud = np.fromfile(point_cloud_file, dtype=np.float32).reshape((-1, 4))
        labels = None
        if label_file is not None:
            # lower 16 bits give the semantic label
            # higher 16 bits gives the instance label
            labels = np.fromfile(label_file, dtype=np.uint32).reshape((-1))
            labels = labels & 0xFFFF
            # we have 16 bits of unsigned information, we can represent this in int32
            labels = labels.astype(np.int32)
            # apply the mapping from raw labels to labels used for learning
            labels = np.vectorize(self.learning_map.get)(labels)
        # for the point cloud, we are only interested in xyz for this task -> no remission
        point_cloud = point_cloud[:,:3]
        # normalizing the point cloud
        point_cloud = torch.from_numpy(point_cloud)
        if labels is not None:
            labels = torch.from_numpy(labels)
        # sampling from the point cloud if npoints is not None
        if self.npoints is not None:
            # we sample from the point cloud, and if for some reason we sample less than npoints, we resample with replacement
            sampling_indices = np.random.choice(len(labels), min(self.npoints,len(labels)), replace=False)
            if len(sampling_indices)<self.npoints:
                new_sample = np.random.choice(len(labels),self.npoints-len(sampling_indices),replace=True)
                sampling_indices = np.concatenate([sampling_indices,new_sample])
            point_cloud = point_cloud[sampling_indices,:]
            if labels is not None:
                labels = labels[sampling_indices]
        return point_cloud, labels

    def __len__(self):
        return len(self.file_list)


test = SemanticKitti()
cloud, label = test[2]
print(cloud.shape)
print(label.shape)
print(np.unique(label))

# WE HAVE 28 UNIQUE LABELS!
# WE NEED TO USE A MAPPING FROM SEMANTICKITTI-API in order to map raw labels to learning labels
'''
label_set = set()
for _,label in test:
    uniques = np.unique(label.numpy())
    uniques = set(uniques.tolist())
    label_set.update(uniques)

print(label_set)
print(len(label_set))
'''

#load_kitti_label_map()