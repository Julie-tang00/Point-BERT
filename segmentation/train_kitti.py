# train file for semantic kitti semantic segmentation task
from data_utils.SemanticKittiDataset import SemanticKitti
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from constants import ROOT_DIR
import models.PointTransformer as PointTransformer
from timm.scheduler import CosineLRScheduler
import os
import numpy as np
import provider
from sklearn.metrics import jaccard_score


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

# adding weight decay from train_partseg
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
            # print(name)
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]




# inplace relu from train_partseg for slight memory savings
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def train():
    # obtaining torch datasets needed for training and setting training parameters
    npoints=2048
    train = SemanticKitti(npoints=npoints)
    val = SemanticKitti(split='val',npoints=npoints)

    num_classes = len(train.inv_map)
    batch_size = 8

    train_loader = data.DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=4)
    val_loader = data.DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=4)

    # obtaining model and playing with group size and number of groups
    group_size  = 32
    num_groups = 128
    from easydict import EasyDict
    model_config = EasyDict(
        trans_dim= 384,
        depth= 12,
        drop_path_rate= 0.1,
        cls_dim= num_classes,
        num_heads= 6,
        group_size= group_size,
        num_group= num_groups,
        encoder_dims= 256,
    )

    model = PointTransformer.get_model(model_config).cuda()
    loss_comp = PointTransformer.get_loss().cuda()
    model.apply(inplace_relu)



    # loading pretrained weights
    pretrained_path = os.path.join(ROOT_DIR,'segmentation','saved_weights','Point-BERT.pth')
    model.load_model_from_ckpt(pretrained_path)

    # optimizer settings
    decay_rate = 0.1
    learning_rate = 0.003
    curr_epoch = 0
    param_groups = add_weight_decay(model, weight_decay=decay_rate)
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=decay_rate)

    scheduler = CosineLRScheduler(optimizer,
            t_initial=curr_epoch,
            t_mul=1,
            lr_min=1e-6,
            decay_rate=0.1,
            warmup_lr_init=1e-6,
            warmup_t=10,
            cycle_limit=1,
            t_in_epochs=True)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = step_size = 0.001

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0

    start_epoch=0
    # training loop with mIoU (jaccard index) computation
    # we can use torchmetric multi class jaccard index for this computation
    for epoch in range(start_epoch, curr_epoch):
            mean_correct = []

            print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, curr_epoch))
            '''Adjust learning rate and BN momentum'''
            print('Learning rate:%f' % optimizer.param_groups[0]['lr'])
            momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
            if momentum < 0.01:
                momentum = 0.01
            print('BN momentum updated to: %f' % momentum)
            classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
            classifier = classifier.train()

            '''learning one epoch'''
            for i, (points, label, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
                optimizer.zero_grad()

                points = points.data.numpy()
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points, F.one_hot(label, num_classes))
                seg_pred = seg_pred.contiguous().view(-1, num_classes)
                target = target.view(-1, 1)[:, 0]
                pred_choice = seg_pred.data.max(1)[1]

                #correct = pred_choice.eq(target.data).cpu().sum()
                correct = pred_choice.eq(target.data).sum().cpu().sum()
                mean_correct.append(correct.item() / (batch_size * npoints))
                loss = loss_comp(seg_pred, target, None)
                loss.backward()
                optimizer.step()
            train_instance_acc = np.mean(mean_correct)
            print('Train accuracy is: %.5f lr = %.6f' % (train_instance_acc, optimizer.param_groups[0]['lr']))
            scheduler.step(epoch)
            with torch.no_grad():
                test_metrics = {}
                # we calculate IOU for each class using True positive, False Positive, and False Negative

                true_positive = torch.zeros(num_classes)
                false_positive = torch.zeros(num_classes)
                false_negative = torch.zeros(num_classes)
                total_class_correct = torch.zeros(num_classes)
                total_class_seen = torch.zeros(num_classes)

                classifier = classifier.eval()

                for batch_id, (points, label, target) in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
                    cur_batch_size, NUM_POINT, _ = points.size()
                    points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                    points = points.transpose(2, 1)
                    one_hot_label = F.one_hot(label,num_classes)
                    seg_pred, _ = classifier(points, one_hot_label)
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()

                    #miou needs true positives, false positives, and false negatives for each class
                    one_hot_pred = F.one_hot(cur_pred_val,num_classes)
                    true_positive = torch.sum(one_hot_pred == one_hot_label).cpu()
                    false_negative += torch.sum(one_hot_label[one_hot_pred!=one_hot_label]).cpu()
                    false_positive += torch.sum(one_hot_pred[one_hot_pred!=one_hot_label]).cpu()

                    # we need to filter out 0 labels values (they are not used for training or testing)

                    # getting total number correct by class (we will make use of one hot encodings here)
                    total_class_correct += torch.sum(one_hot_pred==one_hot_label).cpu()
                    total_class_seen += torch.sum(one_hot_label).cpu()

                # computing total accuracy, class-wise accuracy, class-wise IoU, and total mIoU
                total_accuracy = torch.sum(total_class_correct)/torch.sum(total_class_seen)
                class_accuracy = total_class_correct/total_class_seen

                class_iou = true_positive/ (true_positive + false_negative + false_positive)
                total_miou = torch.sum(class_iou)/num_classes

                test_metrics['accuracy'] = total_accuracy
                test_metrics['class_accuracy'] = class_accuracy
                for cls in range(num_classes):
                    print('eval mIoU of %s %f' % (cls, class_iou[cls]))
                test_metrics['class_iou'] = class_iou
                test_metrics['total_miou'] = total_miou

            print('Epoch %d test Accuracy: %f  mIOU: %f' % (
                epoch + 1, test_metrics['accuracy'], test_metrics['total_miou']))
            if (test_metrics['total_miou'] >= best_total_miou):
                #logger.info('Save model...')
                print('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                print('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'train_acc': train_instance_acc,
                    'test_acc': test_metrics['accuracy'],
                    'class_acc': test_metrics['class_accuracy'],
                    'class_iou': test_metrics['class_iou'],
                    'total_miou': test_metrics['total_miou'],
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                print('Saving model....')

            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
            if test_metrics['total_miou'] > best_total_miou:
                best_total_miou = test_metrics['total_miou']
            print('Best accuracy is: %.5f' % best_acc)
            print('Best mIOU is: %.5f' % best_total_miou)
            global_epoch += 1
