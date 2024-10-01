"""
Infer and Eval function of subfig-subcap sentencewise alignment (finetune PMC-CLIP-Beta based on MediCAT)
"""

import argparse
import json

import torch
import torch.nn as nn
from dataset_code.dataset_clip_align import (
    SentenceWise_Align_Dataset, SentenceWise_Align_Dataset_Infer,
    SentenceWise_Align_Dataset_Infer_V1, sentencewise_align_collate,
    sentencewise_align_collate_infer, sentencewise_align_collate_infer_v1)
from model_clip_align import (SentenceWise_Align_Former,
                              SentenceWise_Align_Former_V1)
from torch.utils.data import DataLoader
from tqdm import tqdm


def contain(name, key_words_list):
    for keys in key_words_list:
        if keys in name:
            return True
    return False

# 使用CLIP计算subfig--[all subcap]的相似度，V1版本处理使用    
def inference_v1(model, valLoader):
    # validate one epoch
    with torch.no_grad():
        model.eval()
        subfig_subcap_ls = []
        count = 0
        for batch in tqdm(valLoader):
            # forward pass
            images = batch['image'].cuda()   # (bs, 3, max_w, max_h)
            subcaptions = batch['subcaptions'].cuda()   # (bs, subcap_num, 77)
            untokenized_subcaption = batch['untokenized_subcaptions']   # [bs, [sub_cap_num]]
            nopad_subcap_idx = batch['nopad_subcap_idx']  # [bs]
            subfigure_info = batch['subfigure_info']  # [bs * {comfig_id:xxx, subfig_loc:xxx, ......}]

            output_sim = model(images, subcaptions) 
            
            for i in range(images.shape[0]):
                tmp_sim = output_sim[i, :nopad_subcap_idx[i]]
                # print(tmp_sim)
                # if torch.max(tmp_sim) > confidence_threshold:
                index = torch.argmax(tmp_sim).item()
                confid = torch.max(tmp_sim).item()
                # print(index)
                # subfig_id = image_ids[i]
                # print(image_ids)
                
                subcap_text = untokenized_subcaption[i][index]
                # print(untokenized_subcaption[i])
                
                subfigure_info[i]['subcaption'] = subcap_text
                subfigure_info[i]['subcap_score'] = confid
                
                subfig_subcap_ls.append(subfigure_info[i])
                if len(subfig_subcap_ls) >= 5000:
                    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(after)(subfig-subcap-CLIP)subfigure_v1.jsonl', 'a') as f:
                        for line in subfig_subcap_ls:
                            f.write(json.dumps(line) + '\n')
                    count += len(subfig_subcap_ls)
                    print('Checkpoint : %d'%count)
                    subfig_subcap_ls = []
                
        if len(subfig_subcap_ls) > 0:
            with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(after)(subfig-subcap-CLIP)subfigure_v1.jsonl', 'a') as f:
                for line in subfig_subcap_ls:
                    f.write(json.dumps(line) + '\n')
            count += len(subfig_subcap_ls)

        return count
   
 
########################################### 分割线：V0版本代码 before MICCAI


# Evaluate sentence-wise align on the dataset
def evaluate(model, valLoader, confidence_threshold=0.5):
    # validate one epoch
    with torch.no_grad():
        model.eval()
        hit = 0
        miss = 0
        not_confid = 0
        random_guess_acc_ls = []
        for batch in tqdm(valLoader):
            # forward pass
            image = batch['image'].cuda()   # (bs, 3, max_w, max_h)
            subcaptions = batch['subcaptions'].cuda()   # (bs, subcap_num, 77)
            location = batch['location'].cuda()    # (bs, 4)
            gt = batch['subcaption_gts'].cuda()  # (bs, subcap_num)
            # noaug_subcap_idx = batch['noaug_subcap_idx']  # [bs]
            nopad_subcap_idx = batch['nopad_subcap_idx']  # [bs]

            output_sim = model(image, subcaptions, location) 

            for i in range(image.shape[0]):
                tmp_sim = output_sim[i, :nopad_subcap_idx[i]]
                tmp_gt_sim = gt[i, :nopad_subcap_idx[i]]
                if torch.max(tmp_sim) > confidence_threshold:
                    if torch.argmax(tmp_sim) == torch.argmax(tmp_gt_sim):
                        hit += 1
                    else:
                        miss += 1
                else:
                    not_confid += 1

        val_acc = hit/(hit+miss)

        return val_acc, (hit+miss)/(hit+miss+not_confid)

def inference(model, valLoader):
    # validate one epoch
    with torch.no_grad():
        model.eval()
        subfig_subcap_ls = []
        count = 0
        for batch in tqdm(valLoader):
            # forward pass
            images = batch['image'].cuda()   # (bs, 3, max_w, max_h)
            subcaptions = batch['subcaptions'].cuda()   # (bs, subcap_num, 77)
            locations = batch['location'].cuda()    # (bs, 4)
            image_ids = batch['image_id']
            untokenized_subcaption = batch['untokenized_subcaptions']   # [bs, [sub_cap_num]]
            nopad_subcap_idx = batch['nopad_subcap_idx']  # [bs]

            output_sim = model(images, subcaptions, locations) 
            
            for i in range(images.shape[0]):
                tmp_sim = output_sim[i, :nopad_subcap_idx[i]]
                # print(tmp_sim)
                # if torch.max(tmp_sim) > confidence_threshold:
                index = torch.argmax(tmp_sim).item()
                confid = torch.max(tmp_sim).item()
                # print(index)
                subfig_id = image_ids[i]
                # print(image_ids)
                subcap_text = untokenized_subcaption[i][index]
                # print(untokenized_subcaption[i])
                subfig_subcap_ls.append({'subfig_id':subfig_id, 'subcaption':subcap_text, 'confidence_score':confid})
                if len(subfig_subcap_ls) >= 5000:
                    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_subfig2subcap.jsonl', 'a') as f:
                        for line in subfig_subcap_ls:
                            f.write(json.dumps(line) + '\n')
                    count += len(subfig_subcap_ls)
                    print('Checkpoint : %d'%count)
                    subfig_subcap_ls = []
                
        if len(subfig_subcap_ls) > 0:
            with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_subfig2subcap.jsonl', 'a') as f:
                for line in subfig_subcap_ls:
                    f.write(json.dumps(line) + '\n')
            count += len(subfig_subcap_ls)

        return count   
       
def str2bool(v):
    return v.lower() in ('true')

def get_eval_args_parser():
    parser = argparse.ArgumentParser()

    # ----------------- Eval Config
    parser.add_argument('--checkpoint', default=None, help='checkpoint to load')
    parser.add_argument('--eval_file', default=None, help='dataset for evaluation')
    parser.add_argument('--task', type=str, default='evaluate')

    # ----------------- Dataset
    parser.add_argument('--normalization', type=str2bool, default=False)
    parser.add_argument('--val_batch_size', type=int, default=40)
    parser.add_argument('--input_size', type=int, default=224)

    # ----------------- Eval Hyperparameters
    parser.add_argument('--confidence_threshold', default=0.8, type=float)

    config = parser.parse_args()
    return config

def main(config):
    torch.multiprocessing.set_sharing_strategy('file_system')

    model = SentenceWise_Align_Former_V1() if config.task == 'infer_v1' else SentenceWise_Align_Former()
    device = torch.device('cuda')
    model = nn.DataParallel(model)
    model.to(device)

    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Load checkpoint from %s, Epoch %d' % (config.checkpoint, int(checkpoint['epoch'])))

    ############# 对齐subcap和subfig
    if config.task == 'infer_v1':
        ValSet = SentenceWise_Align_Dataset_Infer_V1(config.eval_file, input_size=config.input_size)
        ValLoader = DataLoader(ValSet, batch_size=config.val_batch_size, shuffle=False, num_workers=8, collate_fn=sentencewise_align_collate_infer_v1)
        confid_num = inference_v1(model, ValLoader)
        print("%d Confidence Subfigures Out of %d Subfigures" % (confid_num, len(ValSet)))
        exit()
    elif config.task == 'infer':
        test_img_root = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_subfigures'
        ValSet = SentenceWise_Align_Dataset_Infer(config.eval_file, test_img_root, input_size=config.input_size)
        ValLoader = DataLoader(ValSet, batch_size=config.val_batch_size, shuffle=False, num_workers=8, collate_fn=sentencewise_align_collate_infer)
        confid_num = inference(model, ValLoader)
        print("%d Confidence Subfigures Out of %d Subfigures" % (confid_num, len(ValSet)))
        exit()
    ############# 在MediCAT上验证对齐的性能
    elif config.task == 'evaluate':
        test_img_root = '/remote-home/share/medical/public/MedICaT/compound_figures/subfigures'
        ValSet = SentenceWise_Align_Dataset(None, config.eval_file, test_img_root, config.normalization, input_size=config.input_size, mode='Test')
        ValLoader = DataLoader(ValSet, batch_size=config.val_batch_size, shuffle=False, num_workers=8, collate_fn=sentencewise_align_collate)
        acc, confi_ratio = evaluate(model, ValLoader, confidence_threshold=config.confidence_threshold)
        print('Confident threshold %.2f : hit acc %.2f, confident ratio %.2f' % (config.confidence_threshold, acc, confi_ratio))
        exit()

if __name__ == '__main__':
    config = get_eval_args_parser()
    main(config)