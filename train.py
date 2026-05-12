import os
import argparse
import time
import numpy as np
import cv2
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
from medpy.metric.binary import hd95
import random
from torch.utils.data import DataLoader
from utils.avgdataset import FEDDataset
from utils.avgdataset import get_datasplit
from utils.utils import  update_global, update_local, get_net
from utils.metrics import evaluate_network
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
import copy
import datetime
        
def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='GAFSEG',help="selection from list:[HSSF,LSSL,local]")
    parser.add_argument('--data',type=str,default='polyp',help="selection from list:[polyp,isic]")
    parser.add_argument('--datasets', type=list, default=[])
    parser.add_argument('--CommunicationEpoch', type=int, default=200)
    parser.add_argument('--p_threshold', type=int, default=0.6)
    parser.add_argument('--localEpoch', type=int, default=1)
    parser.add_argument('--lrf', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--global_name', type=str, default='resnet50',help='[resnet152,resnet34,resnet50,resnet101,xception,mobilenetv2,vgg,pvt]')
    parser.add_argument('--global_optim', type=str, default='adamw',help='[adam,sgd,adamw]')
    parser.add_argument('--client_optim', type=str, default='adamw',help='[adam,sgd,adamw]')
    parser.add_argument('--clients_model', type=list, default=['resnet50','resnet50','resnet50','resnet50', 'resnet50'],help='[resnet34,xception,mobilenetv2,vgg,pvt]')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_lr', type=float, default=1e-4) 
    parser.add_argument('--shape', type=tuple, default=384)
    parser.add_argument('--train_val_test', type=tuple, default=(0.8,0.1,0.1))
    parser.add_argument('--device',type=str,default='0',help="device id")
    parser.add_argument('--ver',type=str,default='v1',help="the trainning version")
    parser.add_argument('--log_path', type=str,
                        default='./log',
                        help='path to log')
    parser.add_argument('--img_path', type=str,
                        default='data',
                        help='path to data')
    parser.add_argument('--split_path', type=str,
                        default='./data_split',
                        help='path to log')
    
    args = parser.parse_args()
    if args.data == 'polyp':
        args.datasets = ['CVC-ColonDB', 'CVC-ClinicDB', 'EndoTect-ETIS', 'CVC-300', 'Kvasir']
    elif args.data == 'isic':
          args.datasets = ['D1', 'D2', 'D3', 'D4', 'D5']
    return args
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  
    
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    
    exp_name = args.model_name+'-'+args.data+'-'+'-'+args.ver+'-'+str(args.base_lr)+str(args.global_name)+str(args.clients_model)
    
    args.log_dir = os.path.join(args.log_path,'GAFSEG',args.model_name,exp_name,'logs')
    args.model_dir = os.path.join(args.log_path,'GAFSEG',args.model_name,exp_name,'models')
    os.makedirs(args.log_dir,exist_ok=True)
    os.makedirs(args.model_dir,exist_ok=True)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary_writer = SummaryWriter(args.log_dir)
    logging.basicConfig(filename=os.path.join(args.log_dir,'train.log'),format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',level=logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('Hyperparameter setting{}'.format(args))
    print('Hyperparameter setting{}'.format(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    logging.info("Load Participants' Models")
    
    split_file = os.path.join(args.split_path, f"{args.data}-{str(args.datasets)}.json")
    if not os.path.exists(split_file):
        get_datasplit(args)

    train_loaders, val_loaders = [], []
    for dataset in args.datasets:
        train_ds = FEDDataset(args=args, dataset=dataset, transform=True, split='train')
        val_ds = FEDDataset(args=args, dataset=dataset, transform=False, split='val')
        train_loaders.append(DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4))
        val_loaders.append(DataLoader(val_ds, batch_size=1, shuffle=False))
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    global_model = get_net(args.global_name, args.num_classes).to(device)    
    delta_dict_tilda = copy.deepcopy(global_model)
    clt_models = {
    client_idx: copy.deepcopy(global_model)
    for client_idx in range(args.num_clients)
}

    best_client_dice = [0.0 for _ in range(args.num_clients)]
    print(f"\n Starting FedAvg Training with {args.num_clients} clients")
    logging.info(f"\n Starting FedAvg Training with {args.num_clients} clients")
    # -------------------------------
    #   COMMUNICATION ROUNDS
    # -------------------------------
    for round_idx in range(args.CommunicationEpoch):
        print(f"\n========== Round [{round_idx+1}/{args.CommunicationEpoch}] ==========")
        logging.info(f"\n========== Round [{round_idx+1}/{args.CommunicationEpoch}] ==========")
        local_models = []
        

    # ----- LOCAL TRAINING -----
        for client_idx in range(args.num_clients):
            print(f"\n Client {client_idx+1}/{args.num_clients}")
            logging.info(f"\n Client {client_idx+1}/{args.num_clients}")
            local_model = copy.deepcopy(global_model)
            lc_model = copy.deepcopy(clt_models[client_idx])
            local_model = update_local(local_model,  lc_model, train_loaders[client_idx], args, device, client_idx=client_idx, round_idx=round_idx, summary_writer=summary_writer)
            local_models.append(local_model)
            clt_models[client_idx] = copy.deepcopy(local_model)
            train_dice_l, _, _, train_iou_l = evaluate_network(
                args=args, network=local_model, dataloader=train_loaders[client_idx]
            )
            val_dice_l, _, _, val_iou_l = evaluate_network(
                args=args, network=local_model, dataloader=val_loaders[client_idx]
            )

            print(f"  Local Model -> Train: Dice={train_dice_l:.4f}, IoU={train_iou_l:.4f}")
            summary_writer.add_scalar(f"Client{client_idx}/Train_Dice", train_dice_l, round_idx)
            summary_writer.add_scalar(f"Client{client_idx}/Train_IoU", train_iou_l, round_idx)
            summary_writer.add_scalar(f"Client{client_idx}/Val_Dice", val_dice_l, round_idx)
            summary_writer.add_scalar(f"Client{client_idx}/Val_IoU", val_iou_l, round_idx)
            logging.info(f"  Local Model -> Train: Dice={train_dice_l:.4f}, IoU={train_iou_l:.4f}")
            print(f"                 Val:   Dice={val_dice_l:.4f}, IoU={val_iou_l:.4f}")
            logging.info(f"                 Val:   Dice={val_dice_l:.4f}, IoU={val_iou_l:.4f}")
        global_model  = update_global(global_model, local_models, args)
        global_model = global_model.to(device)

        if round_idx == args.CommunicationEpoch - 1:
            torch.save(global_model.state_dict(), os.path.join(args.model_dir, "global_resnet50_last.pth"))
            for i in range(args.num_clients):
                torch.save(local_models[i].state_dict(), os.path.join(args.model_dir, f"resnet50_{i}_last.pth"))
            print(" Saved final round global and client models")
        
    
