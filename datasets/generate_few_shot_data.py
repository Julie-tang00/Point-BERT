import pickle
import numpy as np
import random
import os

root = '../data/ModelNet/modelnet40_normal_resampled'
target = '../data/ModelNetFewshot'

train_data_path = os.path.join(root, 'modelnet40_train_8192pts_fps.dat')
test_data_path = os.path.join(root, 'modelnet40_test_8192pts_fps.dat')
# train
with open(train_data_path, 'rb') as f:
    train_list_of_points, train_list_of_labels = pickle.load(f)
with open(test_data_path, 'rb') as f:
    test_list_of_points, test_list_of_labels = pickle.load(f)

# list_of_points = train_list_of_points + test_list_of_points  
# list_of_labels = train_list_of_labels + test_list_of_labels

def generate_fewshot_data(way, shot, prefix_ind, eval_sample=20):
    train_cls_dataset = {}
    test_cls_dataset = {}
    train_dataset = []
    test_dataset = []
    # build a dict containing different class
    for point, label in zip(train_list_of_points, train_list_of_labels):
        label = label[0]
        if train_cls_dataset.get(label) is None:
            train_cls_dataset[label] = []
        train_cls_dataset[label].append(point)
    # build a dict containing different class
    for point, label in zip(test_list_of_points, test_list_of_labels):
        label = label[0]
        if test_cls_dataset.get(label) is None:
            test_cls_dataset[label] = []
        test_cls_dataset[label].append(point)
    print(sum([train_cls_dataset[i].__len__() for i in range(40)]))
    print(sum([test_cls_dataset[i].__len__() for i in range(40)]))
    # import pdb; pdb.set_trace()
    keys = list(train_cls_dataset.keys())
    random.shuffle(keys)

    for i, key in enumerate(keys[:way]):
        train_data_list = train_cls_dataset[key]
        random.shuffle(train_data_list)
        assert len(train_data_list) > shot
        for data in train_data_list[:shot]:
            train_dataset.append((data, i, key))

        test_data_list = test_cls_dataset[key]
        random.shuffle(test_data_list)
        # import pdb; pdb.set_trace()
        assert len(test_data_list) >= eval_sample
        for data in test_data_list[:eval_sample]:
            test_dataset.append((data, i, key))

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)
    dataset = {
        'train': train_dataset,
        'test' : test_dataset
    }
    save_path = os.path.join(target, f'{way}way_{shot}shot')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f'{prefix_ind}.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    

if __name__ == '__main__':
    ways = [5, 10]
    shots = [10, 20]
    for way in ways:
        for shot in shots:
            for i in range(10):
                generate_fewshot_data(way = way, shot = shot, prefix_ind = i)