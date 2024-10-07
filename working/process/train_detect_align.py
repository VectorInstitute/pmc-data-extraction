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
from dataset_code.dataset_det_align import (Fig_Dataset, FigCap_Dataset,
                                            fig_collate, figcap_collate)
from dataset_code.dataset_ours_syn import Synthetic_Dataset
from dataset_code.dataset_reallayout_syn import Real_Layout_Synthetic_Dataset
from dataset_code.simcfs_reimp import SimCFS_Dataset
from torch.utils.data import DataLoader
from train_engine import train_det_and_align


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
    parser.add_argument('--output_dir', type=str, default='Log/Detection/Pretrained-ResNet',
                        help='File path for model, visualization, log') # ！

    parser.add_argument('--checkpoint', default=None, help='checkpoint to load')
    parser.add_argument('--resume_train', type=str2bool, default='False',
                        help='Load checkpoint and resume the interrupted training')

    parser.add_argument('--model', type=str, default='baseline', help='baseline / dot_product / bidirection_dec')

    parser.add_argument('--resnet', type=int, default=34)
    parser.add_argument('--pretrained_resnet', type=str2bool, default='True')
    parser.add_argument('--pretrained_bert', type=str, default='PubMed_BERT') 

    parser.add_argument('--detr_froze', type=str2bool, default='False',
                        help='Keep detection network frozen')     
    parser.add_argument('--resnet_froze', type=str2bool, default='False',
                        help='Keep pretrained ResNet frozen')      
    parser.add_argument('--alignment_froze', type=str2bool, default='True',
                        help='Keep align network frozen')                          
    parser.add_argument('--bert_froze_depth', type=int, default=12,
                        help='Keep pretrained bert frozen before this depth of layers, default 12, i.e. froze whole bert')                   

    parser.add_argument('--gpu', type=str, default='0') 
    parser.add_argument('--seed', type=int, default=42)

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

    parser.add_argument('--iou_threshold', default=0.75, type=float)
    parser.add_argument('--score_threshold', default=0.0, type=float)
    parser.add_argument('--similarity_threshold', default=0.5, type=float)

    # ----------------- Model Structure
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--txt_dec_layers', default=4, type=int,
                        help="Number of decoding layers in the text transformer")                    
    parser.add_argument('--mlp_ratio', default=4, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
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
    parser.add_argument('--benchmark', type=str, default='medicat', help='imageclef / medicat2downloads')
    parser.add_argument('--dataset', type=str, default='None', help = "ours_syn, real_layout, simcfs, real_det, real_align")
    parser.add_argument('--synthetic_params', type=str, default='no_gap_minus.jsonl')
    parser.add_argument('--aug_params', type=str, default=None)

    parser.add_argument('--normalization', type=str2bool, default=False)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--trainset_size', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=32)

    config = parser.parse_args()
    
    return config

def main(config):
    torch.multiprocessing.set_sharing_strategy('file_system')
    set_seed(config)

    # Choose the right model
    if config.model == 'baseline':
        from model_det_align import FigCap_Former
        model_code = 'model_det_align.py'
    else:
        print('Unsupported Model Type %s' % config.model)
        exit()

    # Enable the alignment part or only the DETR
    if config.dataset in ["ours_syn", "real_layout", "simcfs", "real_det"]:
        alignment_network = False
    elif config.dataset == "real_align":
        alignment_network = True
    else:
        print('Unregconized Dataset Type % s' % config.dataset)
        exit()
    
    # Set the model
    model = FigCap_Former(num_query=config.num_queries, 
                          num_encoder_layers=config.enc_layers, 
                          num_decoder_layers=config.dec_layers,
                          feature_dim=config.hidden_dim, 
                          atten_head_num=config.nheads, 
                          mlp_ratio=config.mlp_ratio, 
                          dropout=config.dropout, 
                          activation='relu',
                          alignment_network = alignment_network,
                          bert_path = '/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/'+config.pretrained_bert,
                          num_text_decoder_layers=config.txt_dec_layers, 
                          text_atten_head_num=config.text_nheads, 
                          text_mlp_ratio=config.mlp_ratio, 
                          text_dropout=config.dropout, 
                          text_activation='relu',
                          resnet = config.resnet,
                          resnet_pretrained=config.pretrained_resnet)
    
    # Initial status if no train resumed
    start_epoch = 1
    best_val_f1 = 0.0
    best_val_mAP = 0.0
    # Load
    if config.resume_train:
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_sch.load_state_dict(checkpoint['lr_sch_state_dict'])
        best_val_f1 = checkpoint['val_f1'] if 'val_f1' in checkpoint else 0.0
        best_val_mAP = checkpoint['val_mAP'] if 'val_mAP' in checkpoint else 0.0
        start_epoch = int(checkpoint['epoch']) + 1
        print('Resume Training from %s, Epoch %d, Val mAP:%.3f, Val F1:%.3f' % (config.checkpoint, start_epoch, best_val_mAP, best_val_f1))
    elif config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        if config.dataset in ["ours_syn", "real_layout", "simcfs", "real_det"]:
            state_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items() :
                if k[:6] == 'module':
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model_checkpoint = {k: v for k, v in new_state_dict.items() if not contain(k, ['text_embed', 'text_channel_squeeze', 'text_decoder', 'simi_head', 'img_proj'])}
            model_dict.update(model_checkpoint)
            model.load_state_dict(model_dict)
        elif config.dataset == "real_align":
            state_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items() :
                if k[:6] == 'module':
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        print('Load checkpoint from %s, Epoch %d, Val mAP:%.3f, Val F1:%.3f' % (config.checkpoint, int(checkpoint['epoch']), checkpoint['val_mAP'], checkpoint['val_f1']))

    # Parallel model to CUDA
    if config.dataset in ["ours_syn", "real_layout", "simcfs", "real_det"]:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
        device = torch.device('cuda')
        model = nn.DataParallel(model)
        model.to(device)
    elif config.dataset == "real_align":
        torch.cuda.set_device(config.gpu)
        model.cuda()

    # 根据训练要求，冻结模型部分参数
    # 目前synthetic只支持detection，所以synthetic默认是det mode，需要把alignment部分froze住
    if config.dataset in ["ours_syn", "real_layout", "simcfs", "real_det"]:
        config.alignment_froze = True
        config.bert_froze_depth = 12
    # align_mode的数据集是固定的（网络不一定
    if config.dataset == 'real_align':
        assert config.alignment_froze == False
    # 如果固定detection网络
    if config.detr_froze:
        print('Detection Network Frozen')
        for name, param in model.named_parameters():
            if contain(name, ['query', 'img_embed', 'img_channel_squeeze', 'pos_embed',  'img_encoder', 'img_decoder', 'box_head', 'det_class_head']):
                param.requires_grad = False
    # 如果固定detection中的ResNet（一般当ResNet是Pretrained）
    elif config.resnet_froze:
        print('ResNet Frozen')
        for name, param in model.named_parameters():
            if contain(name, ['img_embed']):
                param.requires_grad = False
    # 如果固定alignment网络
    if config.alignment_froze:
        print('Alignment Frozen')
        for name, param in model.named_parameters():
            if contain(name, ['text_embed', 'text_channel_squeeze', 'text_decoder', 'simi_head', 'img_proj']):
                param.requires_grad = False
    # 如果固定Bert前半部分
    elif config.bert_froze_depth > 0:
        print('Bert Frozen %d Layers' % config.bert_froze_depth)
        for name, param in model.named_parameters():
            if contain(name, ['text_embed.embeddings'] + ['text_embed.encoder.layer.'+str(i) for i in range(config.bert_froze_depth)]):
                param.requires_grad = False

    # Set optimizer and lr_schedulor
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

    # Set benckmark
    if config.benchmark == 'medicat':
        train_file = '/remote-home/share/medical/public/MedICaT/compound_figures/reid_train.jsonl'
        val_file = '/remote-home/share/medical/public/MedICaT/compound_figures/reid_test.jsonl' # '/remote-home/share/medical/public/MedICaT/subfig_subcap_val.jsonl'
        train_img_root = test_img_root = '/remote-home/share/medical/public/MedICaT/compound_figures/figures'
        vocab_file = '/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/'+config.pretrained_bert+'/vocab.txt'
    elif config.benchmark == 'imageclef':
        train_file = '/remote-home/share/medical/public/ImageCLEF2016/medtask/train_comfigs.jsonl'
        val_file = '/remote-home/share/medical/public/ImageCLEF2016/medtask/test_comfigs.jsonl' # '/remote-home/share/medical/public/MedICaT/subfig_subcap_val.jsonl'
        train_img_root = test_img_root = '/remote-home/share/medical/public/ImageCLEF2016/medtask/new_id_comfigs'
        vocab_file = '/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/'+config.pretrained_bert+'/vocab.txt'
    elif config.benchmark == 'medicat2downloads':
        train_file = '/remote-home/share/medical/public/MedICaT/compound_figures/reid_train_and_test.jsonl'
        val_file = '/remote-home/zihengzhao/CompoundFigure/Dataset/(300 samples)caption_T060_filtered_top4.jsonl' # '/remote-home/share/medical/public/MedICaT/subfig_subcap_val.jsonl'
        train_img_root = '/remote-home/share/medical/public/MedICaT/compound_figures/figures'
        test_img_root = '/remote-home/zihengzhao/CompoundFigure/Dataset/(300 samples)caption_T060_filtered_top4'
        vocab_file = '/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/'+config.pretrained_bert+'/vocab.txt'

    # Set dataset and loader
    if config.dataset == 'ours_syn':
        with open(config.synthetic_params) as f:
            param = json.load(f)
        shutil.copy(config.synthetic_params, os.path.join(config.output_dir, config.synthetic_params.split('/')[-1]))
        if config.aug_params:
            with open(config.aug_params) as f:
                aug_param = json.load(f)
            shutil.copy(config.aug_params, os.path.join(config.output_dir, config.aug_params.split('/')[-1]))
        else:
            aug_param = None
        dataset_code = 'dataset_ours_syn.py'
        TrainSet = Synthetic_Dataset(train_file, train_img_root, param, aug_param, trainset_size_ratio=config.trainset_size, input_size=config.input_size)
        TrainLoader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=False, num_workers=16, collate_fn=fig_collate)
        ValSet = Fig_Dataset(None, val_file, test_img_root, vocab_file, config.normalization, input_size=config.input_size)
        ValLoader = DataLoader(ValSet, batch_size=config.val_batch_size, shuffle=False, num_workers=16, collate_fn=fig_collate)
    elif config.dataset == 'real_layout':
        with open(config.synthetic_params) as f:
            param = json.load(f)
        shutil.copy(config.synthetic_params, os.path.join(config.output_dir, config.synthetic_params.split('/')[-1].split('/')[-1]))
        if config.aug_params:
            with open(config.aug_params) as f:
                aug_param = json.load(f)
            shutil.copy(config.aug_params, os.path.join(config.output_dir, config.aug_params.split('/')[-1]))
        else:
            aug_param = None
        dataset_code = 'dataset_reallayout_syn.py'
        TrainSet = Real_Layout_Synthetic_Dataset(train_file, train_img_root, param, aug_param, trainset_size_ratio=config.trainset_size, input_size=config.input_size)
        TrainLoader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=False, num_workers=16, collate_fn=fig_collate)
        ValSet = Fig_Dataset(None, val_file, test_img_root, vocab_file, config.normalization, input_size=config.input_size)
        ValLoader = DataLoader(ValSet, batch_size=config.val_batch_size, shuffle=False, num_workers=16, collate_fn=fig_collate)
    elif config.dataset == 'simcfs':
        with open(config.synthetic_params) as f:
            param = json.load(f)
        shutil.copy(config.synthetic_params, os.path.join(config.output_dir, config.synthetic_params.split('/')[-1].split('/')[-1]))
        if config.aug_params:
            with open(config.aug_params) as f:
                aug_param = json.load(f)
            shutil.copy(config.aug_params, os.path.join(config.output_dir, config.aug_params.split('/')[-1]))
        else:
            aug_param = None
        dataset_code = 'simcfs_reimp.py'    
        TrainSet = SimCFS_Dataset(train_file, train_img_root, param, aug_param, trainset_size_ratio=config.trainset_size, input_size=config.input_size)
        TrainLoader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=False, num_workers=16, collate_fn=fig_collate)
        ValSet = Fig_Dataset(None, val_file, test_img_root, vocab_file, config.normalization, input_size=config.input_size)
        ValLoader = DataLoader(ValSet, batch_size=config.val_batch_size, shuffle=False, num_workers=16, collate_fn=fig_collate)
    elif config.dataset == 'real_det':
        if config.aug_params:
            with open(config.aug_params) as f:
                aug_param = json.load(f)
            shutil.copy(config.aug_params, os.path.join(config.output_dir, config.aug_params.split('/')[-1]))
        else:
            aug_param = None
        dataset_code = 'dataset.py'
        TrainSet = Fig_Dataset(aug_param, train_file, train_img_root, vocab_file, config.normalization, trainset_size_ratio=config.trainset_size, input_size=config.input_size)
        TrainLoader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=True, num_workers=16, collate_fn=fig_collate)
        ValSet = Fig_Dataset(None, val_file, test_img_root, vocab_file, config.normalization, input_size=config.input_size)
        ValLoader = DataLoader(ValSet, batch_size=config.val_batch_size, shuffle=False, num_workers=16, collate_fn=fig_collate)
    elif config.dataset == 'real_align':
        if config.aug_params:
            with open(config.aug_params) as f:
                aug_param = json.load(f)
            shutil.copy(config.aug_params, os.path.join(config.output_dir, config.aug_params.split('/')[-1]))
        else:
            aug_param = None
        dataset_code = 'dataset.py'
        TrainSet = FigCap_Dataset(aug_param, train_file, train_img_root, vocab_file, config.normalization, trainset_size_ratio=config.trainset_size, input_size=config.input_size)
        TrainLoader = DataLoader(TrainSet, batch_size=config.batch_size, shuffle=True, num_workers=16, collate_fn=figcap_collate)
        ValSet = FigCap_Dataset(None, val_file, test_img_root, vocab_file, config.normalization, input_size=config.input_size)
        ValLoader = DataLoader(ValSet, batch_size=config.val_batch_size, shuffle=False, num_workers=16, collate_fn=figcap_collate)

    # Log location
    print('Output file locate at : %s' % os.path.join(config.output_dir))
    Path(os.path.join(config.output_dir)).mkdir(parents=True, exist_ok=True)
    # Backup code for reproducibility
    shutil.copy('train_detect_align.py', os.path.join(config.output_dir, 'train_detect_align.py'))
    shutil.copy('inference_detect_align.py', os.path.join(config.output_dir, 'inference_detect_align.py'))
    shutil.copy(os.path.join('dataset_code', dataset_code), os.path.join(config.output_dir, dataset_code))
    shutil.copy(model_code, os.path.join(config.output_dir, model_code))
    shutil.copy('transformer_module.py', os.path.join(config.output_dir, 'transformer_module.py'))
    shutil.copy('detect_metric.py', os.path.join(config.output_dir, 'detect_metric.py'))
    shutil.copy('align_metric.py', os.path.join(config.output_dir, 'align_metric.py'))
    shutil.copy('augmentation_tools.py', os.path.join(config.output_dir, 'augmentation_tools.py'))

    train_det_and_align(model=model,
                        optimizer=optimizer,
                        lr_sch=lr_sch,
                        start_epoch=start_epoch,
                        best_val_f1=best_val_f1,
                        best_val_mAP=best_val_mAP,
                        config=config,
                        trainSet=TrainSet, trainLoader=TrainLoader,
                        valSet=ValSet, valLoader=ValLoader)
    
if __name__ == "__main__":
    config = get_train_args_parser()
    main(config)