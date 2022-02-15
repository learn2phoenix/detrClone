# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torchmetrics import AveragePrecision
import util.misc as utils
import util.box_ops as box_ops
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
import copy
from torchvision.utils import save_image, draw_bounding_boxes
import torchvision.ops
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # import pdb; pdb.set_trace()
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        weight_dict['loss_weak'] = 1
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['loss_weak'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def get_box_from_attention(attentions, im_h, im_w, n_rows, n_col):
    nb_pixels = n_rows * n_col
    ret = torch.topk(-attentions, k=int(0.9*nb_pixels))
    attentions[ret[1]]=0
    attentions[attentions!=0]=1
    attentions = attentions.view(n_rows, n_col)
    x_row, y_col = torch.where(attentions==1)
    x_row = x_row*(im_h/n_rows)
    y_col = y_col*(im_w/n_col)
    box = [y_col.min(), x_row.min(), y_col.max(), x_row.max()]
    return torch.stack(box, dim=-1)




@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('ap', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('class_acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('predicted_class_box', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('gt_class_box', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    weight_dict = criterion.weight_dict
    weight_dict['loss_weak'] = 1
    weight_dict['ap'] = 1
    weight_dict['class_acc'] = 1
    weight_dict['predicted_class_box'] = 1
    weight_dict['gt_class_box'] = 1

    average_precision = AveragePrecision(pos_label=1)
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    hooks = [
    model.module.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    model.module.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    ),
    model.module.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),]


    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        
        src_logits = outputs['weak_class'].squeeze()
        target_indices = [torch.unique(t["labels"]) for t in targets]
        target_ = [torch.zeros(src_logits.shape[-1],device=src_logits.device).scatter_(0, i,1) for i in target_indices]
        target_ = torch.vstack(target_).squeeze()

        loss_weak = F.binary_cross_entropy_with_logits(src_logits, target_)
        src_logits = torch.sigmoid(src_logits)
        
        # Vanilla classification accuracy
        _, predicted_class = torch.max(src_logits.data, 1)
        correct = (predicted_class == target_[:,:2]).sum().item()

        # Dumping the boxes for the target classe and predicted class. (The max one in case of predicted)
        pred_box = 0.0
        gt_box = 0.0

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0]
        dec_attn_weights = dec_attn_weights.squeeze()
        enc_attn_weights = enc_attn_weights.squeeze()
        enc_attn_weights[enc_attn_weights<0] = 0
        for i, t in enumerate(targets):
            im_h, im_w = samples.tensors.shape[-2:]
            n_rows, n_col = conv_features['0'].tensors.shape[-2:]
            bbox = copy.deepcopy(t['boxes'])
            bbox = box_ops.box_cxcywh_to_xyxy(bbox)
            bbox[0][0] = bbox[0][0] * im_w
            bbox[0][1] = bbox[0][1] * im_h
            bbox[0][2] = bbox[0][2] * im_w
            bbox[0][3] = bbox[0][3] * im_h
            bbox = bbox.long()[0]

            # predicted box
            
            predicted_class_ = predicted_class[i].item()
            max_patch = torch.argmax(dec_attn_weights[i][predicted_class_, :])
            attentions = copy.deepcopy(enc_attn_weights[i][:, max_patch])
            predicted_box = get_box_from_attention(attentions, im_h, im_w, n_rows, n_col)

            # im = torch.tensor(samples.tensors[i], dtype=torch.uint8)
            # im = draw_bounding_boxes(im.cpu(), bbox.cpu())
            # save_image(im/255, 'temp.png')
            iou = torchvision.ops.box_iou(bbox.unsqueeze(0), predicted_box.unsqueeze(0))[0][0]
            if(iou>=0.5):
                pred_box += 1            

            # target box
            target_class = t['labels'].item()
            max_patch = torch.argmax(dec_attn_weights[i][target_class, :])
            attentions = copy.deepcopy(enc_attn_weights[i][:, max_patch])
            target_box = get_box_from_attention(attentions, im_h, im_w, n_rows, n_col)
            iou = torchvision.ops.box_iou(bbox.unsqueeze(0), predicted_box.unsqueeze(0))[0][0]
            if(iou>=0.5):
                gt_box += 1
            # plt.imshow(attentions.cpu().numpy())
            # plt.axis('off')
            # plt.savefig('/vulcanscratch/sakshams/findingProto/detrClone/experiments/with_pos_enc/base_lrdrop10/vis/ep_24/'\
            #             +im_path[:-4]+'/'+CLASSES[query_id]+'.png')
            

        loss_dict = {'loss_weak': loss_weak}

        ap = average_precision(src_logits, target_)
        loss_dict['ap'] = ap
        loss_dict['class_acc'] = correct / len(targets)
        loss_dict['predicted_class_box'] = pred_box / len(targets)
        loss_dict['gt_class_box'] = gt_box / len(targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(ap=loss_dict_reduced['ap'])
        metric_logger.update(class_acc=loss_dict_reduced['class_acc'])
        metric_logger.update(predicted_class_box=loss_dict_reduced['predicted_class_box'])
        metric_logger.update(gt_class_box=loss_dict_reduced['gt_class_box'])
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats