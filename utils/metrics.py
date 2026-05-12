import numpy as np
import torch 
def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum((1,2,3))
    return ((intersection + 1e-15) / (y_pred.sum((1,2,3)) + 1e-15)).sum()  

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum((1,2,3))
    return ((intersection + 1e-15) / (y_true.sum((1,2,3)) + 1e-15)).sum()

def dice_score(y_true, y_pred):
    return ((2 * (y_true * y_pred).sum((1,2,3)) + 1e-15) / (y_true.sum((1,2,3)) + y_pred.sum((1,2,3)) + 1e-15)).sum()

def iou_score(y_true, y_pred):
    return (((y_true * y_pred).sum((1,2,3)) + 1e-15) / (y_true.sum((1,2,3)) + y_pred.sum((1,2,3)) -(y_true * y_pred).sum((1,2,3))+ 1e-15)).sum()

def evaluate_network(args,network,dataloader):
    network = network.cuda()
    network.eval()
    with torch.no_grad():
        prec = 0.
        rec = 0.
        dice = 0.
        iou = 0.
        length = 0
        for images, labels,img_name in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            pred = network(images) #(4,1,384,384)
            mask = pred['mask']
            mask = mask.argmax(1).unsqueeze(1)
            prec += precision(labels,mask)
            rec += recall(labels, mask)
            dice += dice_score(labels, mask)
            iou += iou_score(labels, mask)
            length += len(labels)
    return dice/length,prec/length,rec/length,iou/length