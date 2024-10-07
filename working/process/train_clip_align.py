"""
To finetune PMC-CLIP-Beta on MediCAT (sentence-wise alignment
"""

import argparse
import json
import math
import os
import random
import shutil
from builtins import print
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from dataset_code.dataset_exsclaim import SentenceWise_Align_EM_Dataset
from dataset_code.dataset_from_wx import (
    Bidirection_SentenceWise_Align_Dataset, SentenceWise_Align_Dataset,
    bidirection_sentencewise_align_collate, sentencewise_align_collate)
from torch.utils.data import DataLoader
from train_engine import (train_sentencewise_align,
                          train_sentencewise_align_bidirection)


def str2bool(v):
    return v.lower() in ('true')

def set_seed(config):
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # new seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def contain(name, key_words_list):
    for keys in key_words_list:
        if keys in name:
            return True
    return False

def get_train_args_parser():
    parser = argparse.ArgumentParser()

    # ----------------- Training Control
    parser.add_argument('--platform', type=str, default='DBCLOUD')
    parser.add_argument('--output_dir', type=str, default='Log/Detection/Pretrained-ResNet',
                        help='File path for model, visualization, log') # ！

    parser.add_argument('--checkpoint', default=None, help='checkpoint to load')
    parser.add_argument('--resume_train', type=str2bool, default='False',
                        help='Load checkpoint and resume the interrupted training')

    parser.add_argument('--visual_encoder_freeze', type=str2bool, default='False',
                        help='Keep detection network frozen')     
    parser.add_argument('--text_encoder_freeze', type=str2bool, default='False',
                        help='Keep pretrained ResNet frozen')      

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--cp_interval', default=10, type=int)

    # ----------------- Learning Rate, Loss and Regularizations, ...
    parser.add_argument('--epoch_num', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float)

    parser.add_argument('--match_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--match_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--match_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--match_cost_align', default=0.0, type=float,
                        help="text align coefficient in the matching cost")
    parser.add_argument('--match_cost_side', default=0.0, type=float,
                        help="text align coefficient in the matching cost")

    parser.add_argument('--class_loss_coef', default=1.0, type=float)
    parser.add_argument('--bbox_loss_coef', default=5.0, type=float)
    parser.add_argument('--giou_loss_coef', default=2.0, type=float)
    parser.add_argument('--align_loss_coef', default=0.0, type=float)
    parser.add_argument('--side_loss_coef', default=0.0, type=float)
    parser.add_argument('--focal_sigma', default=0.0, type=float)
    parser.add_argument('--easy_sample_weight', default=0.0, type=float, help='Loss weight for negative samples from other compound figures in the same batch')

    parser.add_argument('--iou_threshold', default=0.75, type=float)
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--similarity_threshold', default=0.5, type=float)

    # ----------------- Model Structure
    parser.add_argument('--dec_layers', default=8, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--txt_dec_layers', default=4, type=int,
                        help="Number of decoding layers in the text transformer")                    
    parser.add_argument('--mlp_ratio', default=4, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=768, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--text_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")                    
    parser.add_argument('--num_queries', default=50, type=int,
                        help="Number of query slots")

    # ----------------- Dataset
    parser.add_argument('--task', type=str, default=None, help='det, det2align, tokenwisealign, sentencewisealign, bidirection-sentencewisealign')
    parser.add_argument('--aug_params', type=str, default=None)
    parser.add_argument('--aug_ratio', type=float, default=0.25)

    parser.add_argument('--dataset', type=str, default='medicat', help='exsclaim')
    parser.add_argument('--medicat_ratio', type=float, default=1.0)
    parser.add_argument('--exsclaim_ratio', type=float, default=1.0)

    parser.add_argument('--normalization', type=str2bool, default=False)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--trainset_size', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=32)

    parser.add_argument('--num_worker', type=int, default=4)

    config = parser.parse_args()
    
    return config

def main(config):
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    set_seed(config)

    from model_clip_align import (SentenceWise_Align_Former,
                                  SentenceWise_Align_Former_Bidirection,
                                  SentenceWise_Align_Former_Softmax)
    model_code = 'model_from_wx.py'
    
    if config.task == "sentencewisealign":
        model = SentenceWise_Align_Former()
    elif config.task == "sentencewisealign_softmax":
        model = SentenceWise_Align_Former_Softmax()
    elif config.task == "bidirection-sentencewisealign":
        model = SentenceWise_Align_Former_Bidirection()

    # os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    device = torch.device('cuda')
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    def lambda_rule(epoch):
        return 0.5 * (1 + math.cos(math.pi * (epoch) / (config.epoch_num)))
    def warmup_lambda_rule(epoch):
        if epoch < config.warmup:
            return (epoch+1) / config.warmup
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch+1-config.warmup) / (config.epoch_num-config.warmup)))
    if config.warmup > 0:
        lr_sch = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda_rule)
    else:
        lr_sch = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)  # cosine annealing

    start_epoch = 1
    best_val_acc = 0.0
    # Load
    if config.resume_train:
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_sch.load_state_dict(checkpoint['lr_sch_state_dict'])
        best_val_acc = checkpoint['val_acc'] if 'val_acc' in checkpoint else 0.0
        start_epoch = int(checkpoint['epoch']) + 1
        print('Resume Training from %s, Epoch %d' % (config.checkpoint, start_epoch))
    elif config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Load checkpoint from %s, Epoch %d' % (config.checkpoint, int(checkpoint['epoch'])))

    # 如果固定detection网络
    if config.visual_encoder_freeze:
        print('visual_encoder_freeze')
        for name, param in model.named_parameters():
            if contain(name, ['img_encoder']):
                param.requires_grad = False
    # 如果固定detection中的ResNet（一般当ResNet是Pretrained）
    if config.text_encoder_freeze:
        print('text_encoder_freeze')
        for name, param in model.named_parameters():
            if contain(name, ['text_encoder']):
                param.requires_grad = False

    print('Output file locate at : %s' % os.path.join(config.output_dir))

    if config.platform == "DBCLOUD":
        train_file = '/remote-home/share/medical/public/MedICaT/compound_figures/reid_train.jsonl'
        exsclaim_file = '/remote-home/zihengzhao/CompoundFigure/exsclaim-data/data.jsonl'
        val_file = '/remote-home/share/medical/public/MedICaT/compound_figures/reid_test.jsonl' 
        train_img_root = test_img_root = '/remote-home/share/medical/public/MedICaT/compound_figures/subfigures'
        exsclaim_image_root = '/remote-home/share/medical/public/MedICaT/exsclaim_extracted/subfigures'
    elif config.platform == "DB":
        exsclaim_file = '/GPFS/data/zihengzhao/CompoundFigure/exsclaim_extracted/train.jsonl'
        train_file = '/GPFS/data/zihengzhao/CompoundFigure/MedICaT/compound_figures/reid_train.jsonl'
        val_file = '/GPFS/data/zihengzhao/CompoundFigure/MedICaT/compound_figures/reid_test.jsonl' 
        train_img_root = test_img_root = '/GPFS/data/zihengzhao/CompoundFigure/MedICaT/compound_figures/subfigures'
        exsclaim_image_root = '/GPFS/data/zihengzhao/CompoundFigure/exsclaim_extracted/subfigures'

    Path(os.path.join(config.output_dir)).mkdir(parents=True, exist_ok=True)

    if config.aug_params:
        with open(config.aug_params) as f:
            aug_param = json.load(f)
        shutil.copy(config.aug_params, os.path.join(config.output_dir, config.aug_params.split('/')[-1]))
    else:
        aug_param = None

    if config.task in ["sentencewisealign", "sentencewisealign_softmax"]:
        if config.dataset == 'medicat':
            TrainSet = SentenceWise_Align_Dataset(aug_param, train_file, train_img_root, config.normalization, trainset_size_ratio=config.trainset_size, input_size=config.input_size, aug_ratio=config.aug_ratio)
        else:
            # aug_params, exsclaim_filepath, medicat_filepath, exsclaim_image_root, medicat_image_root, normalization=False, medicat_ratio=1.0, exsclaim_ratio=1.0, input_size=512, mode='Test', aug_ratio=0.25
            TrainSet = SentenceWise_Align_EM_Dataset(aug_param, exsclaim_filepath=exsclaim_file, medicat_filepath=train_file, exsclaim_image_root=exsclaim_image_root, medicat_image_root=train_img_root, normalization=config.normalization, medicat_ratio=config.medicat_ratio, exsclaim_ratio=config.exsclaim_ratio, input_size=config.input_size, aug_ratio=config.aug_ratio)
        TrainLoader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=True, num_workers=config.num_worker, collate_fn=sentencewise_align_collate)
        ValSet = SentenceWise_Align_Dataset(None, val_file, test_img_root, config.normalization, input_size=config.input_size)
        ValLoader = DataLoader(ValSet, batch_size=config.val_batch_size, shuffle=False, num_workers=config.num_worker, collate_fn=sentencewise_align_collate)
    elif config.task == "bidirection-sentencewisealign":       
        TrainSet = Bidirection_SentenceWise_Align_Dataset(aug_param, train_file, train_img_root, config.normalization, trainset_size_ratio=config.trainset_size, input_size=config.input_size)
        TrainLoader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=True, num_workers=config.num_worker, collate_fn=bidirection_sentencewise_align_collate)
        ValSet = Bidirection_SentenceWise_Align_Dataset(None, val_file, test_img_root, config.normalization, input_size=config.input_size)
        ValLoader = DataLoader(ValSet, batch_size=config.val_batch_size, shuffle=False, num_workers=config.num_worker, collate_fn=bidirection_sentencewise_align_collate)

    # code
    shutil.copy('train_clip_align.py', os.path.join(config.output_dir, 'train_clip_align.py'))
    shutil.copy('inference_clip_align.py', os.path.join(config.output_dir, 'inference_clip_align.py'))
    shutil.copy(os.path.join('dataset_code', 'dataset_clip_align.py'), os.path.join(config.output_dir, 'dataset_clip_align.py'))
    shutil.copy(model_code, os.path.join(config.output_dir, model_code))
    shutil.copy('transformer_module.py', os.path.join(config.output_dir, 'transformer_module.py'))
    shutil.copy('detect_metric.py', os.path.join(config.output_dir, 'detect_metric.py'))
    shutil.copy('align_metric.py', os.path.join(config.output_dir, 'align_metric.py'))
    shutil.copy('augmentation_tools.py', os.path.join(config.output_dir, 'augmentation_tools.py'))

    if config.task in ["sentencewisealign", "sentencewisealign_softmax"]:
        train_sentencewise_align(model=model, 
                                optimizer=optimizer, 
                                lr_sch=lr_sch, 
                                config=config, 
                                start_epoch=start_epoch, 
                                best_val_acc=best_val_acc,
                                trainLoader=TrainLoader, 
                                valLoader=ValLoader)
    elif config.task == "bidirection-sentencewisealign":
        train_sentencewise_align_bidirection(model=model, 
                                            optimizer=optimizer, 
                                            lr_sch=lr_sch, 
                                            config=config, 
                                            start_epoch=start_epoch, 
                                            best_val_acc=best_val_acc,
                                            trainLoader=TrainLoader, 
                                            valLoader=ValLoader)
    
if __name__ == "__main__":
    config = get_train_args_parser()
    main(config)