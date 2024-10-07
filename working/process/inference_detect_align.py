"""
Inference % Evaluation functions of subfig detection, OR subfig detection & subfig-subcap token-wise align     
"""

import argparse
import json
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tkinter import W

import numpy as np
import torch
from align_metric import SubfigureSubcaptionAlignmentMetric
from dataset_code.dataset_det_align import (Fig_Dataset,
                                            Fig_Separation_Dataset,
                                            FigCap_Dataset, fig_collate,
                                            fig_separation_collate,
                                            figcap_collate)
from detect_metric import (box_cxcywh_to_xyxy, calculate_mAP_voc12,
                           find_jaccard_overlap)
from matplotlib import pyplot as plt
from regex import P
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm import tqdm
from visualization_tools import visualization, visualization_noComparision


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
            # TRANSFORM 
            # image = image_transform(image)
            #
            caption = batch['caption'].cuda()   # (bs, max_l)
            subfigures = batch['subfigs']
            img_ids = batch['image_id'] # [bs]
            original_hws = batch['original_hws'] # [bs * [2]] 没resize+pad前的hw
            untokenized_caption = batch['untokenized_caption'] # [bs * 'string']
            output_det_class, output_box, output_sim = model(image, caption)    # (bs, query_num, 1), (bs, query_num, 4), (bs, query_num, caption_length)
            cpu_output_box = output_box.cpu()
            cpu_output_sim = output_sim.cpu()
            cpu_output_det_class = output_det_class.cpu()
            cpu_caption = caption.cpu()

            # evaluation detection(mAP) and alignment(f1)
            filter_mask = cpu_output_det_class.squeeze() > 0.0 # [bs, query_num], True or False
            index_matrix = torch.arange(0, cpu_output_sim.shape[-1])  # [caption_length], (0, 1, 2 ...)
            for i in range(image.shape[0]):
                # accumulate results as coco format
                det_boxes=[cpu_output_box[i, filter_mask[i,:], :]]   # [1 * (filter_pred_num, 4)]
                # Note the boxex are kept (cx, cy, w, h) until calculating IOU in calculate_mAP_voc12()
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
                f1, rcl, prec = subcaption_metric.get_metric(reset=True)

                img_path = []
                if config.name_by_f1:
                    img_path.append("%s/f1(%.2f)%s" % (save_path, f1, img_ids[i]))
                elif config.name_by_mAP:
                    img_path.append("%s/mAP(%.2f)%s" % (save_path, mAP, img_ids[i]))
                    
                visualization(image=image[i].cpu(), original_h=original_hws[i][0], original_w=original_hws[i][1],
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
                # Note the boxex are kept (cx, cy, w, h) until calculating IOU in calculate_mAP_voc12()
                det_scores.append(cpu_output_det_class.squeeze()[i, filter_index[i,:]])  # [bs * (filter_pred_num)]
                det_labels.append(torch.ones_like(det_scores[-1]))  # [bs * (filter_pred_num)] all 1
                true_boxes.append(subfigures[i][0]) # [bs * (subfig_num, 4)]
                true_labels.append(torch.ones(true_boxes[-1].shape[0]))  # [bs * (subfig_num)] all 1 
                true_difficulties.append(torch.zeros_like(true_labels[-1]))  # [bs * (subfig_num)] all zeros

            # accumulate results as subfig-subcap format 
            cpu_caption = caption.cpu()
            ls_caption = [valSet.id_to_token(cpu_caption[i].tolist()) for i in range(cpu_caption.shape[0])]   # [bs * cap_len] before evaluation, convert ids to tokens
            #
            filter_mask = cpu_output_det_class.squeeze() > score_threshold # [bs, query_num], True or False
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
        out_str = "mAP:%.3f    recall:%.3f    precsision:%.3f" % (mAP, recall, precision)
        print(out_str)
        f1, rcl, prec = subcaption_metric.get_metric(reset=True)
        out_str = "f1:%.3f    recall:%.3f    precsision:%.3f" % (mAP, rcl, prec)
        print(out_str)
        return f1, rcl, prec, mAP, recall, precision
       
# Evaluate det and align on the dataset
def evaluate_det(model, valSet, valLoader, iou_threshold=0.75, score_threshold=0.5):
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
            # cpu_output_sim = output_sim.cpu()
            cpu_output_det_class = output_det_class.cpu()

            # accumulate results as coco format
            filter_index = cpu_output_det_class.squeeze() > 0.0   # [bs, pred_num] True or False
            for i in range(filter_index.shape[0]):
                det_boxes.append(cpu_output_box[i, filter_index[i,:], :])   # [bs * (filter_pred_num, 4)]
                # Note the boxex are kept (cx, cy, w, h) until calculating IOU in calculate_mAP_voc12()
                det_scores.append(cpu_output_det_class.squeeze()[i, filter_index[i,:]])  # [bs * (filter_pred_num)]
                det_labels.append(torch.ones_like(det_scores[-1]))  # [bs * (filter_pred_num)] all 1
                true_boxes.append(subfigures[i][0]) # [bs * (subfig_num, 4)]
                true_labels.append(torch.ones(true_boxes[-1].shape[0]))  # [bs * (subfig_num)] all 1 
                true_difficulties.append(torch.zeros_like(true_labels[-1]))  # [bs * (subfig_num)] all zeros

        gold_subfig_num = 0
        for subfigs in true_boxes:
            gold_subfig_num += len(subfigs)
        print('evaluate on %d compound figures, total %d gold subfigs' % (len(valSet), gold_subfig_num))

        mAP, recall, precision, _ = calculate_mAP_voc12(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, iou_threshold, score_threshold)
        out_str = "mAP:%.3f    recall:%.3f    precsision:%.3f" % (mAP, recall, precision)
        print(out_str)

        return mAP, recall, precision
        
# Separation compound figures
def separation(model, valLoader, save_path='./Segmentation', rcd_file='./Segmentation/separation.jsonl', score_threshold=0.75, nms_threshold=0.4):

    Path(save_path).mkdir(parents=True, exist_ok=True)
    subfig_list = []
    subfig_count = 0

    # validate one epoch
    with torch.no_grad():
        model.eval()
        try:
            for batch in tqdm(valLoader):
                image = batch['image'].cuda() # (bs, 3, max_h, max_w)
                _, _, height, width = image.shape
                caption = batch['caption'].cuda()   # (bs, max_l)
                img_ids = batch['image_id'] # [bs]
                img_idxes = batch['image_index']
                original_images = batch['original_image']    # [bs * (3, h, w)]
                unpadded_hws = batch['unpadded_hws']    # [bs * [2]]
                output_det_class, output_box, _ = model(image, caption)    # (bs, query_num, 1), (bs, query_num, 4), (bs, query_num, caption_length)
                cpu_output_box = output_box.cpu()
                cpu_output_det_class = output_det_class.cpu()
                filter_mask = cpu_output_det_class.squeeze() > score_threshold # [bs, query_num], True or False
                for i in range(image.shape[0]):
                    det_boxes=cpu_output_box[i, filter_mask[i,:], :]   # (filter_pred_num, 4)
                    det_scores=cpu_output_det_class.squeeze()[i, filter_mask[i,:]].numpy()  # (filter_pred_num)
                    img_id = img_ids[i].split('.jpg')[0] # xxxx.jpg
                    img_idx = img_idxes[i]    # x
                    unpadded_image = original_images[i]
                    original_h, original_w = unpadded_hws[i]
                    # 求scale
                    if original_h > original_w:
                        scale = original_h / 512
                    else:
                        scale = original_w / 512
                    # 非极大值抑制
                    order = np.argsort(det_scores)
                    picked_bboxes = []  # [suppressed_num, 4]
                    picked_scores = []  # [suppressed_num]
                    while order.size > 0:
                        # 最大score的index
                        index = order[-1]
                        if order.size == 1:
                            picked_bboxes.append(det_boxes[index].tolist())
                            picked_scores.append(det_scores[index])
                            break
                        else:
                            iou_with_left = find_jaccard_overlap(box_cxcywh_to_xyxy(det_boxes[index]), box_cxcywh_to_xyxy(det_boxes[order[:-1]])).squeeze().numpy()    # (left_bboxes_num)
                            left = np.where(iou_with_left < nms_threshold)  # (left_bboxes_num)
                            order = order[left] # (new_filtered_num)
                            picked_bboxes.append(det_boxes[index].tolist())
                            picked_scores.append(det_scores[index])
                    # Crop and Save Image 
                    for bbox, score in zip(picked_bboxes, picked_scores):
                        try:
                            subfig_path = '%s/%s_%d.jpg' % (save_path, img_id, subfig_count)
                            cx, cy, w, h = bbox
                            # 计算在原图中的bbox坐标
                            x1 = round((cx - w/2)*width*scale)
                            x2 = round((cx + w/2)*width*scale)
                            y1 = round((cy - h/2)*height*scale)
                            y2 = round((cy + h/2)*height*scale)
                            # 纠正如果超出了compound figure的boundary
                            x1 = min(max(x1, 0), original_w-1)
                            x2 = min(max(x2, 0), original_w-1)
                            y1 = min(max(y1, 0), original_h-1)
                            y2 = min(max(y2, 0), original_h-1)
                            # 抠图
                            subfig = unpadded_image[:, y1:y2, x1:x2]  # （3, h, w）
                            subfig = subfig.to(torch.device('cpu'))
                            vutils.save_image(subfig, subfig_path)
                            # Record the Subfig Info in Jsonl file
                            subfig_list.append({'id':'%d.jpg'%subfig_count, 'source_fig_id':img_id, 'position':[(x1, y1), (x2, y2)], 'score':score.item()})
                            subfig_count += 1
                        except (ValueError):
                            print('Crop Error: [x1 x2 y1 y2]:[%.2f %.2f %.2f %.2f], w:%.2f, h:%.2f' % (x1, x2, y1, y2, original_w, original_h))
        
        # 若中途报错，暂时保存现有的分割结果
        except Exception as e:
            print(repr(e))
            print('Break at %s' % img_idx)

        f = open(rcd_file, 'a')
        for line in subfig_list:
            f.write(json.dumps(line)+'\n')
        f.close()

# grid search align 
def grid_search(config, model, Set, Loader):
    # 测试不同的confidence score，在recall和precision之间平衡
    for score_th in np.arange(0.3, 1.0, 0.05):
        config.score_threshold = score_th
        f1, rcl, prec, mAP, recall, precision = evaluate(model, Set, Loader, config.iou_threshold, config.score_threshold, config.similarity_threshold)
        out_str = "mAP:%.3f    recall:%.3f    precsision:%.3f    f1:%.3f    recall:%.3f    precsision:%.3f" % (mAP, recall, precision, f1, rcl, prec)
        print(out_str)
        if config.rcd_file:
            with open(config.rcd_file, 'a') as f:
                f.write('IOU > %.2f    Score > %.2f    Similarity > %.2f\n' % (config.iou_threshold, config.score_threshold, config.similarity_threshold))
                f.write('Results: mAP--%.3f  recall--%.3f  precision--%.3f  f1--%.3f  recall--%.3f  precision--%.3f\n' % (mAP, recall, precision, f1, rcl, prec))
                f.write('\n\n')
                f.close()
           
    if config.vis_path:
        inference(model, Set, Loader, config.vis_path, config.iou_threshold, config.score_threshold, config.similarity_threshold)

# grid search detection
def grid_search_det(config, model, Set, Loader):
    # 测试不同的confidence score，在recall和precision之间平衡
    for score_th in np.arange(0.3, 1.0, 0.05):
        config.score_threshold = score_th
        mAP, recall, precision = evaluate_det(model, Set, Loader, config.iou_threshold, config.score_threshold)
        out_str = "mAP:%.3f    recall:%.3f    precsision:%.3f" % (mAP, recall, precision)
        print(out_str)
        if config.rcd_file:
            with open(config.rcd_file, 'a') as f:
                f.write('IOU > %.2f    Score > %.2f    Similarity > %.2f\n' % (config.iou_threshold, config.score_threshold, config.similarity_threshold))
                f.write('Results: mAP--%.3f  recall--%.3f  precision--%.3f' % (mAP, recall, precision))
                f.write('\n\n')
                f.close()
    # the inference function should be simplified for det only       
    if config.vis_path:
        inference(model, Set, Loader, config.vis_path, config.iou_threshold, config.score_threshold, config.similarity_threshold)


def str2bool(v):
    return v.lower() in ('true')


def get_eval_args_parser():
    parser = argparse.ArgumentParser()

    # ----------------- Eval Config

    parser.add_argument('--checkpoint', default=None, help='checkpoint to load')
    parser.add_argument('--eval_file', default='/remote-home/share/medical/public/MedICaT/compound_figures/reid_test.jsonl', help='dataset for evaluation')
    parser.add_argument('--img_root', default='/remote-home/share/medical/public/MedICaT/compound_figures/figures', help='root path for figures')
    parser.add_argument('--rcd_file', default=None, help='file to record')
    parser.add_argument('--vis_path', default=None, help='file to record the visualizations')

    parser.add_argument('--model', default='baseline', type=str, help='baseline')
    parser.add_argument('--resnet', type=int, default=34)
    parser.add_argument('--pretrained_bert', type=str, default='PubMed_BERT') 

    parser.add_argument('--platform', type=str, default='DBCloud')
    parser.add_argument('--gpu', type=str, default='0') 

    parser.add_argument('--task', type=str, default='det', help='det, det_grid_search, align, align_grid_search, sep')

    parser.add_argument('--normalization', type=str2bool, default=False)
    parser.add_argument('--val_batch_size', type=int, default=64)

    parser.add_argument('--name_by_f1', type=str2bool, default=True, help='name the visualization by f1 score')
    parser.add_argument('--name_by_mAP', type=str2bool, default=True, help='name the visualization by mAP score')

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


def prepare_model_and_dataset(config):
    torch.multiprocessing.set_sharing_strategy('file_system')

    vocab_file = 'path to bert vocab.txt, needed to tokenize compound caption; not necessary for subfigure separation'

    if 'det' in config.task:
        Set = Fig_Dataset(None, config.eval_file, config.img_root, vocab_file, config.normalization)
        Loader = DataLoader(Set, batch_size=config.val_batch_size, shuffle=False, num_workers=8, collate_fn=fig_collate)
    if 'separate' in config.task:
        Set = Fig_Separation_Dataset(None, config.eval_file, config.img_root, config.normalization)
        Loader = DataLoader(Set, batch_size=config.val_batch_size, shuffle=False, num_workers=8, collate_fn=fig_separation_collate)
    else:
        Set = FigCap_Dataset(None, config.eval_file, config.img_root, vocab_file, normalization=config.normalization)
        Loader = DataLoader(Set, batch_size=config.val_batch_size, shuffle=False, num_workers=8, collate_fn=figcap_collate)
  
    if config.model == 'baseline':
        from model import FigCap_Former
    else:
        print('Unsupported Model Type %s' % config.model)
        exit()
    
    
    model = FigCap_Former(num_query=config.num_queries, 
                          num_encoder_layers=config.enc_layers, 
                          num_decoder_layers=config.dec_layers,
                          feature_dim=config.hidden_dim, 
                          atten_head_num=config.nheads, 
                          mlp_ratio=config.mlp_ratio, 
                          dropout=config.dropout, 
                          activation='relu',
                          bert_path = '/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/'+config.pretrained_bert,
                          num_text_decoder_layers=config.txt_dec_layers, 
                          text_atten_head_num=config.text_nheads, 
                          text_mlp_ratio=config.mlp_ratio, 
                          text_dropout=config.dropout, 
                          text_activation='relu',
                          resnet=config.resnet,
                          resnet_pretrained=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    # model = nn.DataParallel(model)
    model.cuda()

    checkpoint = torch.load(config.checkpoint, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    model_checkpoint = {k: v for k, v in state_dict.items()}
    model_dict.update(model_checkpoint)
    model.load_state_dict(model_dict)

    cp_mAP = checkpoint['val_mAP'] if 'val_mAP' in checkpoint else 0
    cp_f1 = checkpoint['val_f1'] if 'val_f1' in checkpoint else 0
        
    print('Load checkpoint from %s, epoch %d, val_mAP %.2f, val_F1 %.2f' % \
        (config.checkpoint, checkpoint['epoch'], cp_mAP, cp_f1))

    return model, Set, Loader


if __name__ == '__main__':
    config = get_eval_args_parser()
    model, Set, Loader = prepare_model_and_dataset(config)
    if config.task == 'det':
        evaluate_det(model, Set, Loader, config.iou_threshold, config.score_threshold)
    elif config.task == 'det_grid_search':
        grid_search_det(config, model, Set, Loader)
    elif config.task == 'align':
        evaluate(config, model, Set, Loader, config.iou_threshold, config.score_threshold, config.similarity_threshold)
    elif config.task == 'align_grid_search':
        grid_search(config, model, Set, Loader)
    elif config.task == "sep":
        separation(model, Loader, config.save_path, config.rcd_file, config.score_threshold, nms_threshold=0.4)
    else:
        print('Undefined Evaluation Task')