"""
Pipeline to filter out nonmedical subfigs
"""

import argparse
import csv
import json
import logging
import os

import torch
import torch.nn as nn
import torchvision.transforms as standard_transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import datetime
import json

import numpy as np
from spacy.tokens import Span
from torch.utils.data import Dataset
from tqdm import tqdm


def fig_classification(fig_class_model_path):
    fig_model =  models.resnext101_32x8d()
    num_features = fig_model.fc.in_features
    fc = list(fig_model.fc.children()) # Remove last layer
    fc.extend([nn.Linear(num_features, 28)]) # Add our layer with 4 outputs
    fig_model.fc = nn.Sequential(*fc)
    fig_model = fig_model.to(device)
    fig_model.load_state_dict(torch.load(fig_class_model_path))
    fig_model.eval()
    
    return fig_model


class PMC_OA(Dataset):
    def __init__(self, csv_path):
        # self.data_info = json.load(open(csv_path,'r'))
        f = open(csv_path)
        lines = f.readlines()
        self.data_info = [json.loads(line) for line in lines]
        
        self.root_dir = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v1/caption_T060_filtered_top4_sep_v1_subfigures/'
        mean_std = ( [.485, .456, .406], [.229, .224, .225])
        self.fig_class_trasform = standard_transforms.Compose([
            standard_transforms.Resize((384, 384), interpolation=Image.ANTIALIAS),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)])

        """
        self.data_info = []
        ab_count = 0
        for datum in data_info:
            img_path = self.root_dir + datum['subfig_id']
            try:
                tmp = Image.open(img_path).convert('RGB')
                self.data_info.append(datum)
            except:
                ab_count += 1
        print('Cant Open %d SubFigure'%ab_count)
        """

    def __getitem__(self, index):
        try:
            img_path = self.root_dir + self.data_info[index]['subfig_id']
            image = Image.open(img_path).convert('RGB')
            image = self.fig_class_trasform(image)
        except:
            image = torch.zeros((3, 384, 384))
            img_path = None
            print(self.root_dir + self.data_info[index]['subfig_id'])
        return {
            "image_path": img_path,
            "image": image
            }
        
    def __len__(self):
        return len(self.data_info)


if __name__ == "__main__":
    # 加载所有comfig-fullcaption
    with open('/remote-home/share/medical/public/PMC_OA/comfig2cap_dict.json', 'r') as f:
        figcap_dict = json.load(f)

    # 对每个subfig分类
    args = argparse.ArgumentParser(description='DocFigure trained model')
    args.add_argument('-p', type=str)
    args.add_argument('-bs', default=4, type=int)
    args = args.parse_args()
    # 加载sep得到的所有subfig
    json_path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v1/caption_T060_filtered_top4_sep_v1_before_cap/caption_T060_filtered_top4_sep_v1_before_filter/sub2com_part%s.jsonl'%args.p
    # figure classification model
    print('Model loading...')  
    fig_model = fig_classification('/remote-home/zihengzhao/CompoundFigure/Ours/dataset_code/resnext101_figure_class.pth')
    print('Model loaded.')  
    # dataset
    print('Dataset loading...')  
    test_dataset = PMC_OA(json_path)
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.bs,
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )
    print('Dataset loaded.') 
    # class each subfig as medical or nonmedical
    pred = {} # subfig_id : class_index
    for sample in tqdm(test_dataloader):
        image_path = sample['image_path']  
        img_tensor = sample['image'].to(device)    
        fig_label = fig_model(img_tensor)
        #print(fig_label.shape)
        fig_prediction = fig_label
        #print(fig_label.shape)
        for index in range(len(image_path)):
            if image_path[index]:
                subfig_id = image_path[index].split('/')[-1]
                pred[subfig_id] = torch.argmax(fig_prediction[index].cpu().detach()).item()
    print('%d Subfigs are Reabable Out of %d'%(len(pred), len(test_dataset)))
    # 加载unfilter数据，根据class结果筛选subfig，并和captions匹配
    f = open(json_path)
    lines = f.readlines()
    unfiltered_data = [json.loads(line) for line in lines]
    filtered_data = []  # dict of all medical subfigs
    ab_count = 0
    # {'subfig_id':'part_%d_%d.jpg'%(part_id, subfig_id), 'comfig_id':comfig_id, 'subfig_loc':[x1/w, y1/h, x2/w, y2/h], 'subfig_score':score, 'caption'}
    for datum in tqdm(unfiltered_data):
        subfig_id = datum['subfig_id']
        comfig_id = datum['comfig_id']
        if subfig_id not in pred:
            continue
        if pred[subfig_id] == 15:
            filtered_data.append(datum)
            datum['caption'] = figcap_dict[comfig_id]
    # 所有medical的subfig
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v1/caption_T060_filtered_top4_sep_v1_before_cap/sub2com_part%s.jsonl'%args.p, 'w')
    for line in filtered_data:
        f.write(json.dumps(line)+'\n')
    f.close()
    print('%d Medical Subfigs Out of %d'%(len(filtered_data), len(unfiltered_data)))
    # 根据过滤的subfig，生成comfig数据，comfig中所有nonmedical的subfig将不被记录（如果comfig内所有subfig都是nonmedical，那么就会被过滤掉
    agg_comfig = []
    cur_comfig = filtered_data[0]['comfig_id']
    cur_comfig_datum = {'comfig_id':cur_comfig, 'subfig_ids':[], 'subfig_locs':[], 'subfig_scores':[], 'caption':filtered_data[0]['caption']}
    for datum in tqdm(filtered_data):
        if datum['comfig_id'] != cur_comfig:    # comfig中所有的subfig必须顺序相连
            if len(cur_comfig_datum['subfig_ids'])>0:
                agg_comfig.append(cur_comfig_datum)
            cur_comfig = datum['comfig_id']
            cur_comfig_datum = {'comfig_id':cur_comfig, 'subfig_ids':[], 'subfig_locs':[], 'subfig_scores':[], 'caption':datum['caption']}
        cur_comfig_datum['subfig_ids'].append(datum['subfig_id'])
        cur_comfig_datum['subfig_locs'].append(datum['subfig_loc'])
        cur_comfig_datum['subfig_scores'].append(datum['subfig_score'])
        # print('cur_comfig_datum:', cur_comfig_datum)
    if len(cur_comfig_datum['subfig_ids'])>0:
        agg_comfig.append(cur_comfig_datum)
    # 所有含有medical的subfig的comfig
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v1/caption_T060_filtered_top4_sep_v1_before_cap/com2sub_part%s.jsonl'%args.p, 'w')
    for line in agg_comfig:
        f.write(json.dumps(line)+'\n')
    f.close()
    print('%d Comfigs Have Medical Subfig'%(len(agg_comfig)))

    exit()

    ##############################################################
    # 下面是filter nonmedical + filter不能被exsclaim分割caption #
    ##############################################################

    """
    # 加载每个comfig的subcaps和caption
    subcap_num = 0
    separable_comfig_num = 0
    comfig2subcaps = {}
    file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_sep_cap_v0.json' # exsclaim分割的cap(没有分割subfig也没有对齐，所以都在unassign里)
    with open(file_path, 'r') as f:
        results = json.load(f)
    for comfig_id, datum in tqdm(results.items()):
        cap_ls = [datum['full_caption']]    # [caption, subcap1, ......]
        for cap in datum['unassigned']['captions']:
            if len(cap['description']) > 0:
                cap_ls += cap['description']
        for subfigure in datum['master_images']:
            if "caption" in subfigure and len(subfigure['caption']) > 0:
                cap_ls += subfigure['caption']
        comfig2subcaps[comfig_id] = cap_ls
        if len(cap_ls) > 1:
            separable_comfig_num += 1
            subcap_num += (len(cap_ls) - 1)
    print('%d Separable Out of %d, Avg %.2f Subcaptions'%(separable_comfig_num, len(comfig2subcaps), subcap_num/separable_comfig_num))

    # 加载unfilter数据，根据class结果筛选subfig，并和captions匹配
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v1_sub2com_unfiltered_part%s.jsonl'%args.p)
    lines = f.readlines()
    unfiltered_data = [json.loads(line) for line in lines]
    filtered_data = []
    # {'subfig_id':'part_%d_%d.jpg'%(part_id, subfig_id), 'comfig_id':comfig_id, 'subfig_loc':[x1/w, y1/h, x2/w, y2/h], 'subfig_score':score}
    for datum in tqdm(unfiltered_data):
        subfig_id = datum['subfig_id']
        comfig_id = datum['comfig_id']
        if subfig_id not in pred:
            continue
        if pred[subfig_id] == 15:
            # print('medical')
            if comfig_id in comfig2subcaps:
                print('caption')
                datum['caption'] = comfig2subcaps[comfig_id][0]
                # {'subfig_id':'part_%d_%d.jpg'%(part_id, subfig_id), 'comfig_id':comfig_id, 'subfig_loc':[x1/w, y1/h, x2/w, y2/h], 'subfig_score':score, 'caption':caption}
                filtered_data.append(datum)
            else:
                print(comfig_id)
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_after_extract_filter/sub2com_part%s.jsonl'%args.p, 'w')
    for line in filtered_data:
        f.write(json.dumps(line)+'\n')
    f.close()
    print('%d Subfig Has Caption Out of %d'%(len(filtered_data), len(unfiltered_data)))

    # 根据过滤的subfig，生成comfig数据
    agg_comfig = []
    cur_comfig = filtered_data[0]['comfig_id']
    cur_comfig_datum = {'comfig_id':cur_comfig, 'subfig_ids':[], 'subfig_locs':[], 'subfig_scores':[], 'caption':filtered_data[0]['caption'], 'subcaptions':comfig2subcaps[cur_comfig][1:]}
    # {'comfig_id':cur_comfig_id, 'subfig_ids':[], 'subfig_locs':[], 'subfig_scores':[]}
    for datum in tqdm(filtered_data):
        if datum['comfig_id'] != cur_comfig:
            if len(cur_comfig_datum['subfig_ids'])>0:
                agg_comfig.append(cur_comfig_datum)
                # print('agg_comfig:', agg_comfig)
            cur_comfig = datum['comfig_id']
            cur_comfig_datum = {'comfig_id':cur_comfig, 'subfig_ids':[], 'subfig_locs':[], 'subfig_scores':[], 'caption':datum['caption'], 'subcaptions':comfig2subcaps[cur_comfig][1:]}
        cur_comfig_datum['subfig_ids'].append(datum['subfig_id'])
        cur_comfig_datum['subfig_locs'].append(datum['subfig_loc'])
        cur_comfig_datum['subfig_scores'].append(datum['subfig_score'])
        # print('cur_comfig_datum:', cur_comfig_datum)
    if len(cur_comfig_datum['subfig_ids'])>0:
        agg_comfig.append(cur_comfig_datum)
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_after_extract_filter/com2sub_part%s.jsonl'%args.p, 'w')
    for line in agg_comfig:
        f.write(json.dumps(line)+'\n')
    f.close()
    print('%d Comfig Has Caption'%(len(agg_comfig)))
    """