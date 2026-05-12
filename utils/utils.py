import os
import torch
from networks.model import SegmentationModel,DeepLabHeadV3Plus 
from utils.avgdataset import get_datasplit,FEDDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import math
import cv2
import copy
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from copy import deepcopy
    
def get_net(net_name,num_classes,ema=False):
    from networks.backbone.deeplab import Decoder
    if net_name=='resnet34':
        from networks.backbone.resnet import resnet34
        bkbone = resnet34()
        head = DeepLabHeadV3Plus(in_channels=512,low_level_channels=64,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name=='resnet50':
        from networks.backbone.resnet import resnet50 
        bkbone = resnet50()
        head = DeepLabHeadV3Plus(in_channels=2048,low_level_channels=256,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name=='resnet18':
        from networks.backbone.resnet import resnet18 
        bkbone = resnet18()
        head = DeepLabHeadV3Plus(in_channels=512,low_level_channels=64,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name=='resnet101':
        from networks.backbone.resnet import resnet101 
        bkbone = resnet101()
        head = DeepLabHeadV3Plus(in_channels=2048,low_level_channels=256,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    elif net_name=='resnet152':
        from networks.backbone.resnet import resnet152 
        bkbone = resnet152()
        head = DeepLabHeadV3Plus(in_channels=2048,low_level_channels=256,num_classes=num_classes)
        net =  SegmentationModel(bkbone,head)
    print("build net with encoder {}.".format(net_name))
    return net

def build_global_model(args):
    net_name = args.global_name
    num_classes = args.num_classes
    net = get_net(net_name,num_classes)
    return net
def init_client_nets(args):
    nets_list = {net_i: None for net_i in range(args.num_clients)}
    for net_i in range(args.num_clients):
        net_name = args.clients_model[net_i]
        net = get_net(net_name,args.num_classes)
        nets_list[net_i] = net
      
    return nets_list
def get_optimizer(optim_name,model,base_lr):
    if optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr,
                                        betas=(0.9, 0.999), weight_decay=5e-4)
    elif optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                                    weight_decay=5e-4)
    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.02)
    
    return optimizer


def update_global(global_model, local_models,args):
    global_dict = copy.deepcopy(global_model.state_dict())
    old_global_dict = copy.deepcopy(global_model.state_dict())
    delta_dict = copy.deepcopy(global_model.state_dict())
    num_clients=args.num_clients
    for k in global_dict.keys():
        global_dict[k] = torch.mean(
            torch.stack([local_models[i].state_dict()[k].float() for i in range(len(local_models))]), dim=0
        )
    difference_list = []
    for i in range(num_clients):
        local_dict = local_models[i].state_dict() 
        diff_i = {}
        sum_of_squares = 0.0
        for k in old_global_dict.keys():
            difference = local_dict[k].float() - old_global_dict[k].float()
            sum_of_squares += torch.sum(difference ** 2)
        l2_norm = torch.sqrt(sum_of_squares)
        for k in old_global_dict.keys():
            difference = local_dict[k].float() - old_global_dict[k].float()
            diff_i[k] = difference / (l2_norm + 1e-12)        
        difference_list.append(diff_i)
    delta_dict = {}
    for k in old_global_dict.keys():
        delta_dict[k] = torch.zeros_like(old_global_dict[k]).float()
    total_sum_squares = 0.0
    for k in old_global_dict.keys():
        layer_diff = global_dict[k].float() - old_global_dict[k].float()
        total_sum_squares += torch.sum(layer_diff ** 2)
    global_l2_norm = torch.sqrt(total_sum_squares)
    for k in old_global_dict.keys():
        layer_diff = global_dict[k].float() - old_global_dict[k].float()
        delta_dict[k] = layer_diff / (global_l2_norm + 1e-12)
    m_t = []
    for i in range(num_clients):
        client_diff = difference_list[i]
        total_squared_distance = 0.0
        for k in old_global_dict.keys():
            layerr_diff = delta_dict[k].float() - client_diff[k].float()
            total_squared_distance += torch.sum(layerr_diff ** 2)
        score = 1.0 - (total_squared_distance*0.5)
        #print(score)
        m_t.append(score)
    new_global_dict = {}
    for k in old_global_dict.keys():
        new_global_dict[k] = torch.zeros_like(old_global_dict[k]).float()
    for k in old_global_dict.keys():   
        for i in range(num_clients):
            client_theta = local_models[i].state_dict()[k].float()
            new_global_dict[k] += (m_t[i] * client_theta * (1/num_clients))
    global_model.load_state_dict(new_global_dict)

    return global_model
    
    
def update_local(model, lc_model, train_loader, args, device,  client_idx=0, round_idx=0, summary_writer=None):
    for p in lc_model.parameters():
        p.requires_grad = False
    w_i= copy.deepcopy(lc_model)
    w = copy.deepcopy(model)
    mu = 0.5
    beta = 0.95
    previous_round_dict= copy.deepcopy(lc_model.state_dict())
    model.train()
    optimizer = get_optimizer(args.client_optim, model, args.base_lr)
    criterion = nn.CrossEntropyLoss()
    if len(train_loader.dataset) < args.batch_size:
        print(f"  Client dataset too small ({len(train_loader.dataset)} samples). "
              f"Switching BatchNorm layers to eval mode.")
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    step_count = 0
    for epoch in range(args.localEpoch):
        for imgs, masks, _ in train_loader:
            if imgs.size(0) == 1:
                model.eval()
            else:
                model.train()
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            if isinstance(preds, dict) and "mask" in preds:
                preds = preds["mask"]
            masks = masks.squeeze(1).long()
            loss = criterion(preds, masks)
            prox_term = 0
            for p_prev, p in zip(w_i.parameters(), w.parameters() ):
                prox_term += torch.sum((p_prev - p.detach()) ** 2)
            total_loss= loss+ ((mu/2)*prox_term)
            if summary_writer is not None:
                global_step = round_idx * (args.localEpoch * max(1, len(train_loader))) + step_count
                summary_writer.add_scalar(f"Client{client_idx}/Train_Loss", loss.item(), global_step)
                summary_writer.add_scalar(f"Client{client_idx}/total_loss", total_loss, global_step)
            total_loss.backward()
            optimizer.step()
            step_count += 1
    current_dict = copy.deepcopy(model.state_dict())
    current_round_dict = {}
    for k in previous_round_dict.keys():
        current_round_dict[k] = torch.zeros_like(previous_round_dict[k]).float()

    for k in previous_round_dict.keys():
        current_round_dict[k] = (
            ((1-beta) * previous_round_dict[k].float()) + 
            (beta * current_dict[k].float())
        )
    model.load_state_dict(current_round_dict)

    return model
 

                
if __name__ == '__main__':
    import os
    model = get_net('resnet50',2).cuda()
    imput  = torch.randn((2,3,384,384)).cuda()
    out = model(imput)