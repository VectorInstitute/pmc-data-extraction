import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tkinter import W

import torch
import torch.nn as nn
from align_metric import SubfigureSubcaptionAlignmentMetric
from dataset_code.dataset_det_align import (Fig_Dataset, FigCap_Dataset,
                                            fig_collate, figcap_collate)
from dataset_code.dataset_ours_syn import Synthetic_Dataset
from dataset_code.dataset_reallayout_syn import Real_Layout_Synthetic_Dataset
from dataset_code.simcfs_reimp import SimCFS_Dataset
from detect_metric import calculate_mAP, calculate_mAP_voc12
from matplotlib import pyplot as plt
from regex import P
from torch.utils.data import DataLoader
from tqdm import tqdm
from visualization_tools import visualization, visualization_noComparision


def contain(name, key_words_list):
    for keys in key_words_list:
        if keys in name:
            return True
    return False

# Visualization the dataset
def inference(model, valSet, valLoader, save_path='./Inference', iou_threshold=0.75, score_threshold=0.5, similarity_threshold=0.5):

    # Subfig Subcap Metric
    subcaption_metric = SubfigureSubcaptionAlignmentMetric(iou_threshold)
    subcaption_metric.reset()

    Path(save_path).mkdir(parents=True, exist_ok=True)

    # validate one epoch
    with torch.no_grad():
        model.eval()
        for batch in tqdm(valLoader):
            image = batch['image'].cuda() # (bs, 3, max_w, max_h)
            caption = batch['caption'].cuda()   # (bs, max_l)
            subfigures = batch['subfigs']
            img_ids = batch['image_id'] # [bs]
            unpadded_hws = batch['unpadded_hws'] # [bs * [2]] 没resize+pad前的hw
            untokenized_caption = batch['untokenized_caption'] # [bs * 'string']
            output_det_class, output_box, output_sim = model(image, caption)    # (bs, query_num, 1), (bs, query_num, 4), (bs, query_num, caption_length)
            cpu_output_box = output_box.cpu()
            cpu_output_sim = output_sim.cpu()
            cpu_output_det_class = output_det_class.cpu()
            cpu_caption = caption.cpu()

            # evaluation detection(mAP) and alignment(f1)
            filter_mask = cpu_output_det_class.squeeze() > score_threshold # [bs, query_num], True or False
            index_matrix = torch.arange(0, cpu_output_sim.shape[-1])  # [caption_length], (0, 1, 2 ...)
            for i in range(image.shape[0]):
                # accumulate results as coco format
                det_boxes=[cpu_output_box[i, filter_mask[i,:], :]]   # [1 * (filter_pred_num, 4)]
                # Note the boxex are kept (cx, cy, w, h) until calculating IOU in calculate_mAP()
                det_scores=[cpu_output_det_class.squeeze()[i, filter_mask[i,:]]]  # [1 * (filter_pred_num)]
                det_labels=[torch.ones_like(det_scores[-1])]  # [1 * (filter_pred_num)] all 1
                true_boxes=[subfigures[i][0]] # [1 * (subfig_num, 4)]
                true_labels=[torch.ones(true_boxes[-1].shape[0])]  # [1 * (subfig_num)] all 1 
                true_difficulties=[torch.zeros_like(true_labels[-1])]  # [1 * (subfig_num)] all zeros 
                
                # calcualte mAP, recall, precision
                mAP, recall, precision, _ = calculate_mAP_voc12(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, iou_threshold, score_threshold)

                # accumulate results as subfig-subcap format
                ls_caption = [valSet.id_to_token(cpu_caption[i].tolist())]   # [1 * cap_len] before evaluation, convert ids to tokens
                ls_pred_boxes = [cpu_output_box[i, filter_mask[i], :].tolist()] # [1 * [filtered_query_num * [4]]], (cx, cy, w, h)
                ls_gt_boxes = [subfigures[i][0].tolist()]  # [bs * [subfigure * [4]]], (cx, cy, w, h)
                filter_tmp = cpu_output_sim[i, filter_mask[i], :] > similarity_threshold    # (filtered_query_num, caption_length) True or False
                pred_tokens = [index_matrix[filter_tmp[j, :]].tolist() for j in range(filter_tmp.shape[0])] # [filtered_query_num * [aligned_token_num]] index of tokens in the caption
                ls_pred_tokens = [pred_tokens]  # [1 * [filtered_query_num * [aligned_token_num]]]
                filter_tmp = subfigures[i][1] > similarity_threshold    # (subfig_num, caption_length) True or False
                gt_tokens = [index_matrix[filter_tmp[j, :]].tolist() for j in range(filter_tmp.shape[0])] # [subfig_num * [subcap_token_num]] index of tokens in the caption
                ls_gt_tokens = [gt_tokens]  # [1 * [subfig_num * [subcap_token_num]]]
                # calculate mAP, recall, precision
                match_matrix = subcaption_metric.update(predicted_subfigures=ls_pred_boxes, predicted_tokens=ls_pred_tokens, 
                                                        gold_subfigures=ls_gt_boxes, gold_tokens=ls_gt_tokens, wordpieces=ls_caption)
                                                        # [num_gt] index of the best matched prediction, -1 if not matched
                f1, rcl, prec = subcaption_metric.get_metric(reset=True)

                img_path = None
                if config.name_by_f1:
                    img_path = "%s/f1(%.2f)%s" % (save_path, f1, img_ids[i])
                elif config.name_by_mAP:
                    img_path = "%s/mAP(%.2f)%s" % (save_path, mAP, img_ids[i])
                    
                visualization(image=image[i].cpu(), original_h=unpadded_hws[i][0], original_w=unpadded_hws[i][1],
                              pred_boxes=det_boxes[-1], pred_texts=pred_tokens, gt_boxes=true_boxes[-1], gt_texts=gt_tokens, match_matrix = match_matrix[-1], 
                              cap=ls_caption[-1], untokenized_cap=untokenized_caption[i], path=img_path, mAP=mAP, f1=f1)

# Evaluate det and align on the dataset
def evaluate(model, valSet, valLoader, iou_threshold=0.75, score_threshold=0.5, similarity_threshold=0.5):

    # Subfig Subcap Metric
    subcaption_metric = SubfigureSubcaptionAlignmentMetric(iou_threshold)
    subcaption_metric.reset()

    # validate one epoch
    with torch.no_grad():
        model.eval()
        det_boxes = []
        det_labels = []
        det_scores = []
        true_boxes = []
        true_labels = []
        true_difficulties = []
        for batch in tqdm(valLoader):
            image = batch['image'].cuda() # (bs, 3, max_w, max_h)
            caption = batch['caption'].cuda()   # (bs, max_l)
            subfigures = batch['subfigs']
            output_det_class, output_box, output_sim  = model(image, caption)    # (bs, query_num, 1), (bs, query_num, 4), (bs, query_num, caption_length)
            cpu_output_box = output_box.cpu()
            cpu_output_sim = output_sim.cpu()
            cpu_output_det_class = output_det_class.cpu()

            # accumulate results as coco format
            filter_index = cpu_output_det_class.squeeze() > 0.0   # [bs, pred_num] True or False
            for i in range(filter_index.shape[0]):
                det_boxes.append(cpu_output_box[i, filter_index[i,:], :])   # [bs * (filter_pred_num, 4)]
                # Note the boxex are kept (cx, cy, w, h) until calculating IOU in calculate_mAP()
                det_scores.append(cpu_output_det_class.squeeze()[i, filter_index[i,:]])  # [bs * (filter_pred_num)]
                det_labels.append(torch.ones_like(det_scores[-1]))  # [bs * (filter_pred_num)] all 1
                true_boxes.append(subfigures[i][0]) # [bs * (subfig_num, 4)]
                true_labels.append(torch.ones(true_boxes[-1].shape[0]))  # [bs * (subfig_num)] all 1 
                true_difficulties.append(torch.zeros_like(true_labels[-1]))  # [bs * (subfig_num)] all zeros

            # accumulate results as subfig-subcap format 
            cpu_caption = caption.cpu()
            ls_caption = [valSet.id_to_token(cpu_caption[i].tolist()) for i in range(cpu_caption.shape[0])]   # [bs * cap_len] before evaluation, convert ids to tokens
            #
            filter_mask = cpu_output_det_class.squeeze() > 0.0 # [bs, query_num], True or False
            ls_pred_boxes = [cpu_output_box[i, filter_mask[i], :].tolist() for i in range(image.shape[0])] # [bs * [filtered_query_num * [4]]], (cx, cy, w, h)
            #
            ls_gt_boxes = [ls[0].tolist() for ls in subfigures]  # [bs * [subfigure * [4]]], (cx, cy, w, h)
            #
            index_matrix = torch.arange(0, cpu_output_sim.shape[-1])  # [caption_length], (0, 1, 2 ...)
            ls_pred_tokens = [] # [bs * [filtered_query_num * [aligned_token_num]]]
            ls_gt_tokens = []   # [bs * [subfig_num * [subcap_token_num]]]
            for i in range(image.shape[0]):
                filter_tmp = cpu_output_sim[i, filter_mask[i], :] > similarity_threshold    # (filtered_query_num, caption_length) True or False
                pred_tokens = [index_matrix[filter_tmp[j, :]].tolist() for j in range(filter_tmp.shape[0])] # [filtered_query_num * [aligned_token_num]] index of tokens in the caption
                ls_pred_tokens.append(pred_tokens)
                filter_tmp = subfigures[i][1] > similarity_threshold    # (subfig_num, caption_length) True or False
                gt_tokens = [index_matrix[filter_tmp[j, :]].tolist() for j in range(filter_tmp.shape[0])] # [subfig_num * [subcap_token_num]] index of tokens in the caption
                ls_gt_tokens.append(gt_tokens)
            subcaption_metric.update(predicted_subfigures=ls_pred_boxes, predicted_tokens=ls_pred_tokens, 
                                     gold_subfigures=ls_gt_boxes, gold_tokens=ls_gt_tokens, wordpieces=ls_caption)

        gold_subfig_num = 0
        gold_subcap_num = 0
        for subfigs in true_boxes:
            gold_subfig_num += len(subfigs)
        for subcaps in true_boxes:
            gold_subcap_num += len(subcaps)
        print('evaluate on %d compound figures, total %d gold subfigs, %d gold subcaps' % (len(valSet), gold_subfig_num, gold_subcap_num))


        mAP, recall, precision, _ = calculate_mAP_voc12(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, iou_threshold, score_threshold)
        f1, rcl, prec = subcaption_metric.get_metric(reset=True)

        return f1, rcl, prec, mAP, recall, precision
       

def str2bool(v):
    return v.lower() in ('true')

def get_eval_args_parser():
    parser = argparse.ArgumentParser()

    # ----------------- Eval Config
    parser.add_argument('--checkpoint', default=None, help='checkpoint to load')
    parser.add_argument('--eval_file', default='/remote-home/share/medical/public/MedICaT/compound_figures/reid_test.jsonl', help='dataset for evaluation')
    parser.add_argument('--rcd_file', default=None, help='file to record')
    parser.add_argument('--vis_path', default=None, help='file to record the visualizations')

    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--resnet', type=int, default=34)
    parser.add_argument('--pretrained_bert', type=str, default='PubMed_BERT') 

    parser.add_argument('--gpu', type=str, default='0') 

    # ----------------- Dataset
    parser.add_argument('--benchmark', type=str, default='medicat')
    parser.add_argument('--dataset', type=str, default='None', help = "ours_syn, real_layout, simcfs, real_det, real_align")
    parser.add_argument('--synthetic_params', type=str, default='synthetic/parameters/baseline.jsonl')
    parser.add_argument('--aug_params', type=str, default=None)
    parser.add_argument('--normalization', type=str2bool, default=False)
    parser.add_argument('--val_batch_size', type=int, default=64)

    parser.add_argument('--name_by_f1', type=str2bool, default=True, help='name the visualization by f1 score')
    parser.add_argument('--name_by_mAP', type=str2bool, default=True, help='name the visualization by f1 score')

    # ----------------- Eval Hyperparameters
    parser.add_argument('--iou_threshold', default=0.75, type=float)
    parser.add_argument('--score_threshold', default=0.5, type=float)
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
    parser.add_argument('--num_queries', default=32, type=int,
                        help="Number of query slots")

    config = parser.parse_args()
    return config

def main(config):
    torch.multiprocessing.set_sharing_strategy('file_system')

    if config.benchmark == 'medicat':
        img_root = '/remote-home/share/medical/public/MedICaT/compound_figures/figures'
        vocab_file = '/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/'+config.pretrained_bert+'/vocab.txt'
    elif config.benchmark == 'imageclef':
        img_root = '/remote-home/share/medical/public/ImageCLEF2016/medtask/new_id_comfigs'
        vocab_file = '/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/'+config.pretrained_bert+'/vocab.txt'
    elif config.benchmark == 'our_downloads':
        img_root = '/remote-home/share/medical/public/PMC_OA/figures'
        vocab_file = '/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/'+config.pretrained_bert+'/vocab.txt'

    if config.dataset == 'ours_syn':
        with open(config.synthetic_params) as f:
            param = json.load(f)
        if config.aug_params:
            with open(config.aug_params) as f:
                aug_param = json.load(f)
        else:
            aug_param = None
        Set = Synthetic_Dataset(param, aug_param, epoch_maximum=600)
        Loader = DataLoader(Set, batch_size=config.val_batch_size, shuffle=False, num_workers=8, collate_fn=fig_collate)
    elif config.dataset == 'real_layout':
        with open(config.synthetic_params) as f:
            param = json.load(f)
        if config.aug_params:
            with open(config.aug_params) as f:
                aug_param = json.load(f)
        else:
            aug_param = None
        Set = Real_Layout_Synthetic_Dataset(config.eval_file, img_root, param, aug_param)
        Loader = DataLoader(Set, batch_size=config.batch_size, shuffle=False, num_workers=8, collate_fn=fig_collate)
    elif config.dataset == 'simcfs':
        with open(config.synthetic_params) as f:
            param = json.load(f)
        if config.aug_params:
            with open(config.aug_params) as f:
                aug_param = json.load(f)
        else:
            aug_param = None 
        Set = SimCFS_Dataset(config.eval_file, img_root, param, aug_param)
        Loader = DataLoader(Set, batch_size=config.batch_size, shuffle=False, num_workers=8, collate_fn=fig_collate)
    elif config.dataset == 'real_det':
        if config.aug_params:
            with open(config.aug_params) as f:
                aug_param = json.load(f)
        else:
            aug_param = None
        Set = Fig_Dataset(aug_param, config.eval_file, img_root, vocab_file, config.normalization)
        Loader = DataLoader(Set, batch_size=config.val_batch_size, shuffle=False, num_workers=8, collate_fn=fig_collate)
    elif config.dataset == 'real_align':
        if config.aug_params:
            with open(config.aug_params) as f:
                aug_param = json.load(f)
        else:
            aug_param = None
        Set = FigCap_Dataset(aug_param, config.eval_file, img_root, vocab_file, normalization=config.normalization)
        Loader = DataLoader(Set, batch_size=config.val_batch_size, shuffle=False, num_workers=8, collate_fn=figcap_collate)
    else:
        print('Unregconized Dataset Type % s' % config.dataset)
        exit()

    if config.model == 'baseline':
        from model import FigCap_Former
    else:
        print('Unsupported Model Type %s' % config.model)
        exit()

    if config.dataset in ["ours_syn", "real_layout", "simcfs", "real_det"]:
        alignment_network = False
    elif config.dataset == "real_align":
        alignment_network = True
    else:
        print('Unregconized Dataset Type % s' % config.dataset)
        exit()
    
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
                          resnet=config.resnet,
                          resnet_pretrained=False)

    checkpoint = torch.load(config.checkpoint, map_location='cpu')
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

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    device = torch.device('cuda')
    model = nn.DataParallel(model)
    model.to(device)
        
    print('Load checkpoint from %s, epoch %d, trn_mAP %.2f, trn_F1 %.2f, val_mAP %.2f, val_F1 %.2f' % \
        (config.checkpoint, checkpoint['epoch'], checkpoint['train_mAP'], checkpoint['train_f1'], checkpoint['val_mAP'], checkpoint['val_f1']))

    f1, rcl, prec, mAP, recall, precision = evaluate(model, Set, Loader, config.iou_threshold, config.score_threshold, config.similarity_threshold)
    out_str = "mAP:%.3f    recall:%.3f    precsision:%.3f    f1:%.3f    recall:%.3f    precsision:%.3f" % (mAP, recall, precision, f1, rcl, prec)
    print(out_str)

    if config.vis_path:
        inference(model, Set, Loader, config.vis_path, config.iou_threshold, config.score_threshold, config.similarity_threshold)
    if config.rcd_file:
        with open(config.rcd_file, 'a') as f:
            SHA_TZ = timezone(timedelta(hours=8), name='Asia/Shanghai')   
            utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
            beijing_now = utc_now.astimezone(SHA_TZ)    # 北京时间
            f.write('Time: %d.%d.%d--%d:%d\n' % (beijing_now.year, beijing_now.month, beijing_now.day, beijing_now.hour, beijing_now.minute))
            f.write('Load checkpoint from: %s\n' % (config.checkpoint))
            f.write('Evaluate on: %s\n' % config.eval_file)
            f.write('IOU > %.2f    Score > %.2f    Similarity > %.2f\n' % (config.iou_threshold, config.score_threshold, config.similarity_threshold))
            f.write('Results: mAP--%.3f  recall--%.3f  precision--%.3f  f1--%.3f  recall--%.3f  precision--%.3f\n' % (mAP, recall, precision, f1, rcl, prec))
            f.write('\n\n')
            f.close()

if __name__ == '__main__':
    config = get_eval_args_parser()
    main(config)