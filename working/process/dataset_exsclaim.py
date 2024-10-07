""" 
将Exsclaim配对的subfig-subcap用于训练Sentence-Wise Alignment
"""

import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from augmentation_tools import Augmentation_Tool
from PIL import Image
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer


def convert_to_wordpieces(tokenizer, tokens, subcaps):
    """
    Tokenize the tokens accroding to the vocab of the pretrained bert, and re-index the subcaption spans

    Args:
        tokenizer (BertTokenizer): pretrained tokenizer
        tokens (list): a lits of tokens, e.g. ['This', 'is' ...]
        subcaps (dict): a dict of subcaptions label and corresponding span, e.g. {'a':[1, 2, 3, 4] ...}

    Returns:
        new tokens and new subcaps after
    """
    # 首先，对token进行进一步细粒度的切分，例如lover-->lov+er，使得token符合vocab中的规范
    token_to_wordpiece_map = {}
    new_tokens = []
    for index, token in enumerate(tokens):
        token_to_wordpiece_map[index] = len(new_tokens) # 第i个token对应的起始wordpiece index
        new_tokens += tokenizer.tokenize(token) # 第i个token分成的wordpiece
    token_to_wordpiece_map[len(tokens)] = len(new_tokens) # 最后一个token无意义
    # 然后，根据更细的新“token”（又称wordpiece），重新划定token和span
    new_subcaps = {}
    for label in subcaps:
        wordpieces_idx = []
        for idx in subcaps[label]:
            wordpieces_idx += [tmp for tmp in range(token_to_wordpiece_map[idx], token_to_wordpiece_map[idx+1])]
        new_subcaps[label] = wordpieces_idx
    return new_tokens, new_subcaps

################################ Compound Figures + Captions ################################

class SentenceWise_Align_EM_Dataset(Dataset):
    def __init__(self, aug_params, exsclaim_filepath, medicat_filepath, exsclaim_image_root, medicat_image_root, normalization=False, medicat_ratio=1.0, exsclaim_ratio=1.0, input_size=512, aug_ratio=0.0):
        print('Sentence-Wise Align Dataset')
        self.images = []        # list of {'path':'xxx/xxx.png', 'w':256, 'h':256, ......}

        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

        if normalization:
            self.image_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
        else:
            self.image_transform = transforms.Compose([
                                transforms.ToTensor()
                                ])

        # aug param
        self.aug_tool = Augmentation_Tool(aug_params)

        self.all_subcap_list = []
        abandon_count = 0

        # preprocessing
        f = open(exsclaim_filepath)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        print("%d Compound Figure" % len(lines))

        avg_subcaps = 0
        for datum in tqdm(data):
            # Don't include figures that don't have subfigures
            if 'subfigures' not in datum or datum['subfigures']==None or len(datum['subfigures']) == 0:
                continue
            # Don't include figures that lack subcaptions (detection不需要这一步
            if "subcaptions" not in datum or datum["subcaptions"] == None or len(datum["subcaptions"]) == 0:
                abandon_count += len(datum['subfigures'])
                continue
            
            subcaptions = []
            subcap_dict = {}
            start = len(self.all_subcap_list)
            for subfig_id, subcap in datum["subcaptions"].items():
                subcap_token_ls = self.tokenizer.tokenize(subcap)
                tmp = ['[CLS]'] + subcap_token_ls + ['[SEP]']
                tmp = torch.tensor(self.tokenizer.convert_tokens_to_ids(tmp))
                if tmp.shape[0] < 78 and tmp.shape[0] > 2:
                    pad = (0, 77-tmp.shape[0])
                    tmp = F.pad(tmp, pad, 'constant', 0)    # [subcap_num, (77)]
                    subcap_dict[subfig_id] = len(subcaptions)
                    subcaptions.append(tmp)
                    self.all_subcap_list.append(tmp)
                else:
                    subcap_dict[subfig_id] = -1
            subcap_global_idx_range = [start, len(self.all_subcap_list)]

            avg_subcaps += len(subcaptions)

            # check every subfigure
            width = datum['width']
            height = datum['height']
            for subfigure in datum["subfigures"]:
                if subfigure["label"] in datum["subcaptions"] and subcap_dict[subfigure["label"]] != -1:
                    x = [point[0] for point in subfigure["points"]]
                    y = [point[1] for point in subfigure["points"]]
                    x1 = min(x)
                    x2 = max(x)
                    y1 = min(y)
                    y2 = max(y)
                    # 过滤不在图中的bbox (medicat没有这一步，不过在数据集中没有发现不符合的subfig
                    if y2 < 0 or x2 < 0 or x1 > width or y1 > height:
                        print(image_info['id'])
                        continue
                    # 规范bbox
                    if y1 < 0:
                        y1 = 0
                    if x1 < 0:
                        x1 = 0
                    if x2 > width:
                        x2 = width
                    if y2 > height:
                        y2 = height
                    if x1 == x2 or y1 == y2:
                        continue
                    # quanlified train data  
                    image_info = {}      
                    image_info["path"] = exsclaim_image_root+'/'+subfigure["id"]
                    image_info['id'] = subfigure["id"]
                    image_info['subcap_global_idx_range'] = subcap_global_idx_range   # indexes of the subcaps in whole dataset
                    image_info['subcap_gt_index'] = subcap_dict[subfigure["label"]]   # index of the subcap in compound figure
                    image_info['subcaptions'] = subcaptions # all the candidate subcaps, [subcap_num, tensor(77)]
                    image_info['caption_string'] = datum['text'] if 'text' in datum else datum['caption']
                    image_info['location'] = [x1/width, y1/height, x2/width, y2/height]
                    self.images.append(image_info)

        self.exsclaim_idx = len(self.images)
        f = open(medicat_filepath)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

        for datum in tqdm(data):
            # Don't include figures that don't have subfigures
            if 'subfigures' not in datum or datum['subfigures']==None or len(datum['subfigures']) == 0:
                continue
            # Don't include figures that lack subcaptions (detection不需要这一步
            if "subcaptions" not in datum or datum["subcaptions"] == None or len(datum["subcaptions"]) == 0:
                abandon_count += len(datum['subfigures'])
                continue

            # caption tokenization
            old_tokens = ["[CLS]"] + [token["text"] for token in datum["tokens"]] + ["[SEP]"]  # list of ['CLS', 'This', 'figure' ...]
            old_subcaps = {}      # {'a':[0,1,2,3...], ...}
            for subcap_key in datum["subcaptions"]:
                old_subcaps[subcap_key] = [idx+1 for idx in datum["subcaptions"][subcap_key]]
            tokens, subcaps = convert_to_wordpieces(self.tokenizer, old_tokens, old_subcaps)    # tokenize后新的token和subcap spans
            # group the subcaps
            subcaptions = []
            subcap_dict = {}
            subcap_global_idx_range = [len(self.all_subcap_list)]
            for label, token_index in subcaps.items():  # 'a', [0, 1, 2 ......]
                tmp = ["[CLS]"] + [tokens[i] for i in token_index] + ["[SEP]"]  # subcap span转成具体的tokens并加CLS和SEP
                tmp = torch.tensor(self.tokenizer.convert_tokens_to_ids(tmp))
                if tmp.shape[0] < 78 and tmp.shape[0] > 2:
                    pad = (0, 77-tmp.shape[0])
                    tmp = F.pad(tmp, pad, 'constant', 0)    # [subcap_num, (77)]
                    subcap_dict[label] = len(subcaptions)
                    subcaptions.append(tmp)    # [subcap_num, (77)]
                    self.all_subcap_list.append(tmp)
                else:
                    subcap_dict[label] = -1
            subcap_global_idx_range.append(len(self.all_subcap_list))

            # check every subfigure
            width = datum['width']
            height = datum['height']
            for subfigure in datum["subfigures"]:
                if subfigure["label"] in datum["subcaptions"] and subcap_dict[subfigure["label"]] != -1:
                    x = [point[0] for point in subfigure["points"]]
                    y = [point[1] for point in subfigure["points"]]
                    x1 = min(x)
                    x2 = max(x)
                    y1 = min(y)
                    y2 = max(y)
                    # 过滤不在图中的bbox (medicat没有这一步，不过在数据集中没有发现不符合的subfig
                    if y2 < 0 or x2 < 0 or x1 > width or y1 > height:
                        print(image_info['id'])
                        continue
                    # 规范bbox
                    if y1 < 0:
                        y1 = 0
                    if x1 < 0:
                        x1 = 0
                    if x2 > width:
                        x2 = width
                    if y2 > height:
                        y2 = height
                    if x1 == x2 or y1 == y2:
                        continue
                    # quanlified train data  
                    image_info = {}      
                    image_info["path"] = medicat_image_root+'/'+subfigure["id"]
                    image_info['id'] = subfigure["id"]
                    image_info['subcap_global_idx_range'] = subcap_global_idx_range   # indexes of the subcaps in whole dataset
                    image_info['subcap_gt_index'] = subcap_dict[subfigure["label"]]   # index of the subcap in compound figure
                    image_info['subcaptions'] = subcaptions # all the candidate subcaps, [subcap_num, tensor(77)]
                    image_info['caption_string'] = datum['text'] if 'text' in datum else datum['caption']
                    image_info['location'] = [x1/width, y1/height, x2/width, y2/height]
                    self.images.append(image_info)

        self.e_num = int(exsclaim_ratio*self.exsclaim_idx) # 一轮epoch中train sample的数目
        self.m_num = int(medicat_ratio*(len(self.images)-self.exsclaim_idx))

        self.shuffle_index()

        self.input_size = input_size
        self.aug_ratio = aug_ratio

        print('Abandon %d samples (excessive caption tokens), Keep %d samples' % (abandon_count, len(self.images)))

    def shuffle_index(self):
        """
            when train dataset set to be smaller or larger than real dataset, need to randomly select a new set,
            mapping the index to a real sample

            self.dataset_size (int) : size of train/val set
        """
        e = np.random.choice(self.exsclaim_idx, self.e_num, replace=False)
        m = np.random.choice((len(self.images)-self.exsclaim_idx), self.m_num, replace=False) + self.exsclaim_idx
        self.index_map = np.concatenate((e, m), axis=0)
        print(e.shape, m.shape)

    def __len__(self):
        return self.e_num + self.m_num

    def __getitem__(self, index):
        index = self.index_map[index]    # transfer to new sample index
        
        unpadded_image = Image.open(self.images[index]['path']).convert('RGB')
        unpadded_image = self.image_transform(unpadded_image)
        
        unpadded_image, _ = self.aug_tool(unpadded_image, []) # tensor (3, h, w)

        subcaptions = self.images[index]['subcaptions'] # [subcap_num, tensor(77)]
        real_subcap_num = len(subcaptions)
        subcap_gt_index = self.images[index]['subcap_gt_index']
        subcap_global_idx_range = self.images[index]['subcap_global_idx_range']
        if self.aug_ratio > 0.0:
            other_subcaps = random.choices([*range(0, subcap_global_idx_range[0])]+[*range(subcap_global_idx_range[1], len(self.all_subcap_list))], \
                                        k=round(self.aug_ratio*(subcap_global_idx_range[1]-subcap_global_idx_range[0])))
        else:
            other_subcaps = []
        subcaptions += [self.all_subcap_list[idx] for idx in other_subcaps] # [50%subcap_num, tensor(77)]
        subcaptions = torch.stack(subcaptions, dim=0)   # (150%subcap_num, 77)
        caption_string = self.images[index]['caption_string']
        image_id = self.images[index]['id']
        loc = self.images[index]['location']
        
        return unpadded_image, loc, subcaptions, real_subcap_num, subcap_gt_index, image_id, caption_string, self.input_size
