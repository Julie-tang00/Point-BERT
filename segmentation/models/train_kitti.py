# train file for semantic kitti semantic segmentation task
from segmentation.data_utils.SemanticKittiDataset import SemanticKitti
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from constants import ROOT_DIR
import PointTransformer

# obtaining torch datasets needed for training and setting training parameters
train = SemanticKitti()
val = SemanticKitti(split='val')

num_classes = len(train.inv_map)
batch_size = 8

train_loader = data.DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=4)
val_loader = data.DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=4)

# obtaining model
model_params = {}

model = PointTransformer.get_model(model_params)

# loading pretrained weights
pretrained_path = None
model.load_model_from_ckpt(pretrained_path)

# training loop
