import os
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from utils.utils import init_client_nets,build_global_model
from utils.avgdataset import FEDDataset
from torch.utils.data import DataLoader
from utils.metrics import dice_score,iou_score
from medpy.metric.binary import  hd95, assd


def initial_tester(args):
    global_net = build_global_model(args=args)
    net_list = init_client_nets(args)   
    val_client_dataloaders = []
    test_client_dataloaders = []
    val_client_datasets = []
    test_client_datasets = []
   
    for ind in range(len(args.datasets)):
        client_dataset = FEDDataset(args=args, dataset=args.datasets[ind], transform = False, split ='val', noise = False)
        client_dataloader = DataLoader(client_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True)
        val_client_dataloaders.append(client_dataloader)
        val_client_datasets.append(client_dataset)
        client_dataset = FEDDataset(args=args, dataset=args.datasets[ind], transform = False, split ='test', noise = False)
        client_dataloader = DataLoader(client_dataset,batch_size=args.batch_size,pin_memory=True,num_workers=args.num_workers,drop_last=True)
        test_client_dataloaders.append(client_dataloader)
        test_client_datasets.append(client_dataset)

    return global_net,net_list,val_client_dataloaders,test_client_dataloaders 





@torch.no_grad()
def evaluate_network_fedavg(args,network, dataloaders,key='validation'):
    os.makedirs(os.path.join(args.save_dir,'image'),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'mask'),exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,'pred'),exist_ok=True)
    network.eval()
    dice = []
    iou = []
    length = []
    asd = []
    hd95_s = []
    num = []
    over_dice_format = f"result: & "
    over_iou_format  = f" & " 
    over_hd95_format = f" & "
    for i in range(args.num_clients):
        dice.append([])
        iou.append([])
        length.append([])
        asd.append([])
        hd95_s.append([])
        num.append([])
    
    for i in range(len(dataloaders)):
        dataloader = dataloaders[i]
        for images, labels,img_name in dataloader:
            img_name = img_name[0][:-4]+'_'+str(i)+'_'+args.methods+img_name[0][-4:]
            images = images.cuda()
            labels = labels.cuda()
            pred = network(images) #(4,1,384,384)
            mask = pred['mask']
            mask = mask.argmax(1).unsqueeze(1)
            dice[i].append(dice_score(labels, mask).cpu().numpy()*100) 
            iou[i].append(iou_score(labels, mask).cpu().numpy()*100) 
            
            length[i].append(len(labels))
            try:
                asd[i].append(assd(labels.squeeze(0).squeeze(0).cpu().numpy()>0,mask.squeeze(0).squeeze(0).cpu().numpy()>0))
                hd95_s[i].append(hd95(labels.squeeze(0).squeeze(0).cpu().numpy()>0,mask.squeeze(0).squeeze(0).cpu().numpy()>0))
            except RuntimeError:
                num[i].append(len(labels))
            
            cv2.imwrite(os.path.join(args.save_dir,'image',img_name),((images[0]+1)/2*255)[[2,1,0]].permute(1,2,0).cpu().numpy())
            cv2.imwrite(os.path.join(args.save_dir,'pred',img_name.replace(args.methods,'label')),(labels[0][0]*255).cpu().numpy())
            cv2.imwrite(os.path.join(args.save_dir,'pred',img_name),(mask[0][0]*255).cpu().numpy())
        over_dice_format+= f"& {torch.tensor(dice[i]).mean():.2f} \\tiny $ \pm$ {torch.tensor(dice[i]).std():.2f} "
        over_iou_format  += f"& {torch.tensor(iou[i]).mean():.2f} ± {torch.tensor(iou[i]).std():.2f} "
        over_hd95_format+= f"& {torch.tensor(hd95_s[i]).mean():.2f} \\tiny $ \pm$ {torch.tensor(hd95_s[i]).std():.2f} "
    all_dice = [item for sublist in dice for item in sublist]
    all_iou  = [item for sublist in iou for item in sublist]
    all_hd = [item for sublist in hd95_s for item in sublist]
    over_dice_format+= f"& {torch.tensor(all_dice).mean():.2f} \\tiny $ \pm$ {torch.tensor(all_dice).std():.2f} "
    over_iou_format  += f"& {torch.tensor(all_iou).mean():.2f} ± {torch.tensor(all_iou).std():.2f} "
    over_hd95_format+= f"& {torch.tensor(all_hd).mean():.2f} \\tiny $ \pm$ {torch.tensor(all_hd).std():.2f} "
    print(over_dice_format+over_iou_format+over_hd95_format)
    return 

def metrics_fedavg(args,global_net,val_client_dataloaders,test_client_dataloaders):
    global_net.load_state_dict(torch.load(os.path.join(args.model_dir,'global_{}_last.pth'.format(args.global_name))))
    global_net = global_net.cuda()           
    print('Evaluate Models')
    print('-'*50)
    print('{} all result:'.format(args.methods))
    evaluate_network_fedavg(args,network=global_net, dataloaders=test_client_dataloaders,key='test')
    print('+'*50)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods',type=str,default='GAFSEG',help="selection from list:[LSSL,HSSF]")
    parser.add_argument('--data',type=str,default='polyp',help="selection from list:[polyp, isic]")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--global_name', type=str, default='resnet50',help='[resnet34,resnet50]')
    parser.add_argument('--clients_model', type=list, default=['resnet50','resnet50','resnet50','resnet50', 'resnet50'],help='[resnet50]')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--DataParallel', type=bool, default=False)
    parser.add_argument('--shape', type=tuple, default=384)
    parser.add_argument('--train_val_test', type=tuple, default=(0.8,0.1,0.1))
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--device',type=str,default='0',help="device id")
    parser.add_argument('--img_path', type=str,
                        default='data',
                        help='path to dataset')
    parser.add_argument('--split_path', type=str,
                        default='./data_split',
                        help='path to data split')
    parser.add_argument('--log_dir', type=str,
                        default='log',
                        help='path to log')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if args.data == "polyp":
        args.datasets = ['CVC-ColonDB', 'CVC-ClinicDB', 'EndoTect-ETIS', 'CVC-300', 'Kvasir']        
    elif args.data == "isic":
        args.datasets = ['D1', 'D2', 'D3', 'D4', 'D5']
    
    args.split_dict = json.load(open(os.path.join(args.split_path,'{}-{}.json'.format(args.data,str(args.datasets))),'r'))
    if not os.path.exists(os.path.join(args.split_path,'{}-{}.json'.format(args.data,str(args.datasets)))):
        raise TypeError("No available data partition found.")
    args.model_dir = f"{args.log_dir}/{args.data}/{args.methods}" 
    if args.methods in ['GAFSEG']:
        args.save_dir = os.path.join(args.log_dir,'res')
        args.net_name = args.global_name
        global_net,_,val_client_dataloaders,test_client_dataloaders = initial_tester(args)
        metrics_fedavg(args, global_net, val_client_dataloaders, test_client_dataloaders)
        
                