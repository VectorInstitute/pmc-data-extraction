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


# 以subfig为单位读取subfig--[all subcap]，V1版本处理使用
class SentenceWise_Align_Dataset_Infer_V1(Dataset):
    def __init__(self, filepath, input_size=224):
        print('Sentence-Wise Align Dataset')
        self.images = []        # list of {'path':'xxx/xxx.png', 'w':256, 'h':256, ......}

        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

        self.image_transform = transforms.Compose([
                            transforms.ToTensor()
                            ])

        # preprocessing
        f = open(filepath)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        print("%d Subfigure" % len(lines))

        abandon_count = 0
        for datum in tqdm(data):
            """
            {
            'image_path': '/remote-home/.../PMC212319_Fig3_0.jpg',	# subfigure path
            'full_caption': 'A. Real time image ...', 				# full caption
            'media_name': 'PMC212319_Fig3.jpg', 				# fig id
            'subfig_loc': [x1, y1, x2, y2],		# normed to 0~1
            'subfig_score': 0.9, 	# subfigure's detection confidence score
            'subfig': 'PMC212319_Fig3_0.jpg', 
            'subcaptions': ['collected ...', ...]
            } 	
            """
            subcaptions = []
            untokenized_subcaptions = []
            for subcap_label, subcap in datum["subcaptions"]['subcaptions'].items():  # {A:xxx, B:xxx, ...}
                subcap_token_ls = self.tokenizer.tokenize(subcap)
                tmp = ['[CLS]'] + subcap_token_ls + ['[SEP]']
                tmp = torch.tensor(self.tokenizer.convert_tokens_to_ids(tmp))
                if tmp.shape[0] < 78 and tmp.shape[0] > 2:
                    pad = (0, 77-tmp.shape[0])
                    tmp = F.pad(tmp, pad, 'constant', 0)    # [subcap_num, (77)]
                    subcaptions.append(tmp)
                    untokenized_subcaptions.append(subcap)
                elif tmp.shape[0] > 77:
                    tmp = tmp[:77]
                    subcaptions.append(tmp)
                    untokenized_subcaptions.append(subcap)
                    
            if len(subcaptions) == 0:
                if len(datum["subcaptions"]) > 0:
                    abandon_count += 1
                continue

            # quanlified train data  
            image_info = {}      
            image_info["path"] = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/subfigures/' + datum['subfig_id']
            image_info['tokenized_subcaptions'] = subcaptions   # all the candidate subcaps, [subcap_num, tensor(77)]
            image_info['untokenized_subcaptions'] = untokenized_subcaptions
            del datum['subcaptions']
            image_info['subfigure_info'] = datum
            
            self.images.append(image_info)

        self.input_size = input_size
        print('Keep %d Subfigures, %d are Aborted' % (len(self.images), abandon_count))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        unpadded_image = Image.open(self.images[index]['path']).convert('RGB')
        unpadded_image = self.image_transform(unpadded_image)

        subcaptions = self.images[index]['tokenized_subcaptions'] # [subcap_num, tensor(77)]
        subcaptions = torch.stack(subcaptions, dim=0)   # (subcap_num, 77)
        
        untokenized_subcaptions = self.images[index]['untokenized_subcaptions'] # [subcap_num of strings]
        subfigure_info = self.images[index]['subfigure_info']

        return unpadded_image, subcaptions, untokenized_subcaptions, subfigure_info, self.input_size

def sentencewise_align_collate_infer_v1(data):
    """
    Pad input samples in a batch

    Args:
        data : list of samples

    Returns:
        a dictionary for batch input
    """
    max_subcap_num = 0  # 先统计最大的subcap num
    for unit in data:   
        _, subcaptions, _, _, _ = unit
        if subcaptions.shape[0] > max_subcap_num:
            max_subcap_num = subcaptions.shape[0]
        
    pad_imgs = []  
    pad_subcaps = []        # (bs, subcap_num, 77)
    untokenized_subcaption_ls = []
    nopad_subcap_idx_ls = []   # [bs]
    subfigure_info_ls = []
    for unit in data:
        unpadded_image, subcaptions, untokenized_subcaptions, subfigure_info, input_size = unit
        untokenized_subcaption_ls.append(untokenized_subcaptions)
        subfigure_info_ls.append(subfigure_info)
        # bbox info
        # resize+pad image
        _, h, w = unpadded_image.shape
        scale = min(input_size/h, input_size/w)
        resize_transform = transforms.Resize([round(scale*h), round(scale*w)])
        resized_img = resize_transform(unpadded_image) # reize到input_size(512)以内
        pad = (0, input_size-round(scale*w), 0, input_size-round(scale*h))  
        padded_img = F.pad(resized_img, pad, 'constant', 0) # 对img加pad成512x512
        pad_imgs.append(padded_img)
        # padd subcaptions
        pad = (0, 0, 0, max_subcap_num-subcaptions.shape[0])
        padded_subcap = F.pad(subcaptions, pad, 'constant', 0) # (max_subcap_num, 77)
        pad_subcaps.append(padded_subcap)
        nopad_subcap_idx_ls.append(subcaptions.shape[0])

    pad_imgs = torch.stack(pad_imgs, dim=0) # (bs, 3, max_w, max_h)
    pad_subcaps = torch.stack(pad_subcaps, dim=0)   # (bs, subcap_num, 77)

    return {'image':pad_imgs, 'subcaptions':pad_subcaps, 'untokenized_subcaptions':untokenized_subcaption_ls, 'nopad_subcap_idx':nopad_subcap_idx_ls, 'subfigure_info':subfigure_info_ls}


########################################### 分割线：V0版本代码 before MICCAI


# 基于分割好的Subfig和Subcap，以figure为单位采样
class SentenceWise_Align_Dataset_Infer(Dataset):
    def __init__(self, filepath, image_root, input_size=224):
        print('Sentence-Wise Align Dataset')
        self.images = []        # list of {'path':'xxx/xxx.png', 'w':256, 'h':256, ......}

        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

        self.image_transform = transforms.Compose([
                            transforms.ToTensor()
                            ])

        # preprocessing
        f = open(filepath)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        print("%d Compound Figure" % len(lines))

        self.all_subcap_list = []
        abandon_count = 0
        for datum in tqdm(data):
            #{
            # 'comfig_id':cur_comfig, 
            # 'subfig_ids':[], 
            # 'subfig_locs':[], 
            # 'subfig_scores':[], 
            # 'caption':'caption', 
            # 'subcaptions':['subcaps' ...]
            # }

            # Don't include figures that don't have subfigures
            if 'subfig_ids' not in datum or datum['subfig_ids']==None or len(datum['subfig_ids']) == 0:
                print('No subfig but %d subcap'%len(datum["subcaptions"]))
                continue
            # Don't include figures that lack subcaptions (detection不需要这一步
            if "subcaptions" not in datum or datum["subcaptions"] == None or len(datum["subcaptions"]) == 0:
                continue

            subcaptions = []
            untokenized_subcaptions = []
            for subcap in datum["subcaptions"]:
                subcap_token_ls = self.tokenizer.tokenize(subcap)
                tmp = ['[CLS]'] + subcap_token_ls + ['[SEP]']
                tmp = torch.tensor(self.tokenizer.convert_tokens_to_ids(tmp))
                if tmp.shape[0] < 78 and tmp.shape[0] > 2:
                    pad = (0, 77-tmp.shape[0])
                    tmp = F.pad(tmp, pad, 'constant', 0)    # [subcap_num, (77)]
                    subcaptions.append(tmp)
                    untokenized_subcaptions.append(subcap)

            if len(subcaptions) == 0:
                if len(datum["subcaptions"]) > 0:
                    abandon_count += len(datum["subfig_ids"])
                continue

            # check every subfigure
            for normed_subfig_loc, subfig_id, subfig_score in zip(datum["subfig_locs"],datum["subfig_ids"],datum["subfig_scores"]):
                x1 = normed_subfig_loc[0]
                x2 = normed_subfig_loc[2]
                y1 = normed_subfig_loc[1]
                y2 = normed_subfig_loc[3]
                # quanlified train data  
                image_info = {}      
                image_info["path"] = image_root+'/'+subfig_id
                image_info['id'] = subfig_id
                image_info['subcaptions'] = subcaptions # all the candidate subcaps, [subcap_num, tensor(77)]
                image_info['caption_string'] = untokenized_subcaptions
                image_info['location'] = [x1, y1, x2, y2]
                self.images.append(image_info)

        self.input_size = input_size
        print('Keep %d Subfigures, %d are Aborted for Too Long Sucaps' % (len(self.images), abandon_count))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        unpadded_image = Image.open(self.images[index]['path']).convert('RGB')
        unpadded_image = self.image_transform(unpadded_image)

        subcaptions = self.images[index]['subcaptions'] # [subcap_num, tensor(77)]
        subcaptions = torch.stack(subcaptions, dim=0)   # (subcap_num, 77)
        
        subcaption_strings = self.images[index]['caption_string'] # [subcap_num of strings]
        subfig_id = self.images[index]['id']
        loc = self.images[index]['location']

        return unpadded_image, loc, subcaptions, subfig_id, subcaption_strings, self.input_size

def sentencewise_align_collate_infer(data):
    """
    Pad input samples in a batch

    Args:
        data : list of samples

    Returns:
        a dictionary for batch input
    """
    max_subcap_num = 0  # 先统计最大的subcap num
    for unit in data:   
        unpadded_image, loc, subcaptions, image_id, caption_string, input_size = unit
        if subcaptions.shape[0] > max_subcap_num:
            max_subcap_num = subcaptions.shape[0]
        
    image_ids = []          # list of strings
    image_captions = []     # [bs * [subcap_num * 'subcap']]
    pad_imgs = []  
    pad_subcaps = []        # (bs, subcap_num, 77)
    nopad_subcap_idx = []
    locations = []      
    for unit in data:
        unpadded_image, loc, subcaptions, image_id, caption_string, input_size = unit
        image_ids.append(image_id)
        image_captions.append(caption_string)
        # bbox info
        locations.append(loc)
        # resize+pad image
        _, h, w = unpadded_image.shape
        scale = min(input_size/h, input_size/w)
        resize_transform = transforms.Resize([round(scale*h), round(scale*w)])
        resized_img = resize_transform(unpadded_image) # reize到input_size(512)以内
        pad = (0, input_size-round(scale*w), 0, input_size-round(scale*h))  
        padded_img = F.pad(resized_img, pad, 'constant', 0) # 对img加pad成512x512
        pad_imgs.append(padded_img)
        # padd subcaptions
        pad = (0, 0, 0, max_subcap_num-subcaptions.shape[0])
        padded_subcap = F.pad(subcaptions, pad, 'constant', 0) # (max_subcap_num, 77)
        pad_subcaps.append(padded_subcap)
        nopad_subcap_idx.append(subcaptions.shape[0])

    pad_imgs = torch.stack(pad_imgs, dim=0) # (bs, 3, max_w, max_h)
    pad_subcaps = torch.stack(pad_subcaps, dim=0)   # (bs, subcap_num, 77)
    locations = torch.tensor(locations)   # (bs, 4)

    return {'image':pad_imgs, 'location':locations, 'image_id':image_ids, 'subcaptions':pad_subcaps, 'untokenized_subcaptions':image_captions, 'nopad_subcap_idx':nopad_subcap_idx}


# 基于MediCAT数据集，以subfig为单位采样
class SentenceWise_Align_Dataset(Dataset):
    def __init__(self, aug_params, filepath, image_root, normalization=False, trainset_size_ratio=1.0, input_size=224, aug_ratio=0.0):
        """
        Args:
            aug_params (dict): Dict of augmentation params
            filepath (str): Path to dataset file
            image_root (str): Root path to all subfigures
            normalization (bool, optional): Normalization option. Defaults to False.
            trainset_size_ratio (float, optional): Train dataset size in an epoch. Defaults to 1.0, i.e. enumerate
            input_size (int, optional): Input image size . Defaults to 512.
            aug_ratio (float, optional): Random aug samples from other compoundfigures. Defaults to 0.0.
        """
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

        # preprocessing
        f = open(filepath)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        print("%d Compound Figure" % len(lines))

        self.all_subcap_list = []
        abandon_count = 0
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
            subcaps_token_ids = []
            subcap_dict = {}
            subcap_global_idx_range = [len(self.all_subcap_list)]
            for label, token_index in subcaps.items():  # 'a', [0, 1, 2 ......]
                subcap_dict[label] = len(subcaps_token_ids)
                tmp = ["[CLS]"] + [tokens[i] for i in token_index] + ["[SEP]"]  # subcap span转成具体的tokens并加CLS和SEP
                tmp = torch.tensor(self.tokenizer.convert_tokens_to_ids(tmp))
                pad = (0, 77-tmp.shape[0])
                subcaps_token_ids.append(F.pad(tmp, pad, 'constant', 0))    # [subcap_num, (77)]
                self.all_subcap_list.append(F.pad(tmp, pad, 'constant', 0))
            # subcaps_token_ids = torch.stack(subcaps_token_ids, dim=0)   # (subcap_num, 77)
            subcap_global_idx_range.append(len(self.all_subcap_list))

            # check every subfigure
            width = datum['width']
            height = datum['height']
            for subfigure in datum["subfigures"]:
                if subfigure["label"] in subcaps and len(subcaps[subfigure["label"]]) > 0 and len(subcaps[subfigure["label"]]) < 76:
                    x = [point[0] for point in subfigure["points"]]
                    y = [point[1] for point in subfigure["points"]]
                    x1 = min(x)
                    x2 = max(x)
                    y1 = min(y)
                    y2 = max(y)
                    # 过滤不在图中的bbox (medicat没有这一步，不过在数据集中没有发现不符合的subfig
                    if y2 < 0 or x2 < 0 or x1 > datum['width'] or y1 > datum['height']:
                        print(image_info['id'])
                        continue
                    # 规范bbox
                    if y1 < 0:
                        y1 = 0
                    if x1 < 0:
                        x1 = 0
                    if x2 > datum['width']:
                        x2 = datum['width']
                    if y2 > datum['height']:
                        y2 = datum['height']
                    if x1 == x2 or y1 == y2:
                        continue
                    # quanlified train data  
                    image_info = {}      
                    image_info["path"] = image_root+'/'+subfigure["id"]
                    image_info['id'] = subfigure["id"]
                    image_info['subcap_global_idx_range'] = subcap_global_idx_range   # indexes of the subcaps in whole dataset
                    image_info['subcap_gt_index'] = subcap_dict[subfigure["label"]]   # index of the subcap in compound figure
                    image_info['subcaptions'] = subcaps_token_ids # all the candidate subcaps, [subcap_num, tensor(77)]
                    image_info['caption_string'] = datum['text']
                    # image_info['caption_token'] = tokens # ['[CLS]', 'This', ... '[SEP]']
                    # image_info['caption_input'] = token_ids # [2, 43, 59, .... 3, 3]
                    # image_info['subcaption'] = subcaps[subfigure["label"]]  # [1,2,3,4 ...]
                    image_info['location'] = [x1/width, y1/height, x2/width, y2/height]
                    self.images.append(image_info)

        self.dataset_size = int(trainset_size_ratio * len(self.images))  # 一轮epoch中train sample的数目
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
        if self.dataset_size < len(self.images):
            self.index_map = np.random.choice(len(self.images), self.dataset_size, replace=False)
        elif self.dataset_size == len(self.images):
            self.index_map = np.arange(len(self.images))
        else:
            self.index_map = np.concatenate(np.arange(len(self.images)), np.random.choice(len(self.images), self.dataset_size-len(self.images), replace=False))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        index = self.index_map[index]    # transfer to new sample index
        
        unpadded_image = Image.open(self.images[index]['path']).convert('RGB')
        unpadded_image = self.image_transform(unpadded_image)
        
        unpadded_image, _ = self.aug_tool(unpadded_image, []) # tensor (3, h, w)

        subcaptions = self.images[index]['subcaptions'] # [subcap_num, tensor(77)]
        real_subcap_num = len(subcaptions)
        subcap_gt_index = self.images[index]['subcap_gt_index']
        subcap_global_idx_range = self.images[index]['subcap_global_idx_range']
        if self.aug_ratio > 0:
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

def sentencewise_align_collate(data):
    """
    Pad input samples in a batch

    Args:
        data : list of samples

    Returns:
        a dictionary for batch input
    """
    max_subcap_num = 0  # 先统计最大的subcap num
    for unit in data:   
        unpadded_image, loc, subcaptions, noaug_subcap_num, subcap_gt_index, image_id, caption_string, input_size = unit
        if subcaptions.shape[0] > max_subcap_num:
            max_subcap_num = subcaptions.shape[0]
        
    image_ids = []          # list of strings
    image_captions = []     # list of strings
    pad_imgs = []  
    pad_subcaps = []        # (bs, subcap_num, 77)
    subcaption_gts = []     # (bs, subcap_num)
    nopad_subcap_idx = []    # list of int
    noaug_subcap_idx = []    # list of int
    locations = []      
    for unit in data:
        unpadded_image, loc, subcaptions, noaug_subcap_num, subcap_gt_index, image_id, caption_string, input_size = unit
        image_ids.append(image_id)
        image_captions.append(caption_string)
        # bbox info
        locations.append(loc)
        # resize+pad image
        _, h, w = unpadded_image.shape
        scale = min(input_size/h, input_size/w)
        resize_transform = transforms.Resize([round(scale*h), round(scale*w)])
        resized_img = resize_transform(unpadded_image) # reize到input_size(512)以内
        pad = (0, input_size-round(scale*w), 0, input_size-round(scale*h))  
        padded_img = F.pad(resized_img, pad, 'constant', 0) # 对img加pad成512x512
        pad_imgs.append(padded_img)
        # padd subcaptions
        pad = (0, 0, 0, max_subcap_num-subcaptions.shape[0])
        padded_subcap = F.pad(subcaptions, pad, 'constant', 0) # (max_subcap_num, 77)
        pad_subcaps.append(padded_subcap)
        nopad_subcap_idx.append(subcaptions.shape[0])
        # hard samples and easy samples
        noaug_subcap_idx.append(noaug_subcap_num)
        # one-hot gt vector 
        tmp = torch.zeros(max_subcap_num) # (subcap_num)
        tmp[subcap_gt_index] = 1
        subcaption_gts.append(tmp)

    pad_imgs = torch.stack(pad_imgs, dim=0) # (bs, 3, max_w, max_h)
    subcaption_gts = torch.stack(subcaption_gts, dim=0) # (bs, subcap_num)
    pad_subcaps = torch.stack(pad_subcaps, dim=0)   # (bs, subcap_num, 77)
    locations = torch.tensor(locations)   # (bs, 4)

    return {'image':pad_imgs, 'location':locations, 'image_id':image_ids, 'subcaptions':pad_subcaps, 'subcaption_gts':subcaption_gts, 'untokenized_caption':image_captions, 'nopad_subcap_idx':nopad_subcap_idx, 'noaug_subcap_idx':noaug_subcap_idx}


# 基于MediCAT数据集，以compound figure为单位采样
class Bidirection_SentenceWise_Align_Dataset(Dataset):
    def __init__(self, aug_params, filepath, image_root, normalization=False, trainset_size_ratio=1.0, input_size=512):
        print('Bidirection Sentence-Wise Align Dataset')
        self.samples = [] # list of data samples(dict)

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

        # preprocessing
        f = open(filepath)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        print("%d Compound Figure" % len(lines))

        for datum in tqdm(data):
            # Don't include figures without subfigures
            if 'subfigures' not in datum or datum['subfigures']==None or len(datum['subfigures']) == 0:
                continue
            # Don't include figures without subcaptions
            if "subcaptions" not in datum or datum["subcaptions"] == None or len(datum["subcaptions"]) == 0:
                continue

            # caption tokenization
            old_tokens = ["[CLS]"] + [token["text"] for token in datum["tokens"]] + ["[SEP]"] # list of ['CLS', 'This', 'figure' ...]
            old_subcaps = {} # {'a':[0,1,2,3...], ...}
            for subcap_key in datum["subcaptions"]:
                old_subcaps[subcap_key] = [idx+1 for idx in datum["subcaptions"][subcap_key]]
            tokens, subcaps = convert_to_wordpieces(self.tokenizer, old_tokens, old_subcaps)  # tokenize后新的token和subcap spans
            # group the subcaps
            subcaps_tensor_ls = [] # list of tensors(token ids of a subcap, int)
            subcap_dict = {} # subcap label to tensor index in subcaps_tensor, e.g. 'a':0
            for label, token_index in subcaps.items():  # 'a', [0, 1, 2 ......]
                subcap_dict[label] = len(subcaps_tensor_ls)
                tmp = ["[CLS]"] + [tokens[i] for i in token_index] + ["[SEP]"] # subcap span转成具体的tokens并加CLS和SEP
                tmp = torch.tensor(self.tokenizer.convert_tokens_to_ids(tmp)) # subcap tokens转id
                pad = (0, 77-tmp.shape[0])
                tmp = F.pad(tmp, pad, 'constant', 0)
                subcaps_tensor_ls.append(tmp) # pad to tensor of shape (77)
            padded_subcaps = torch.stack(subcaps_tensor_ls, dim=0) # (num_subcap, 77)

            # check every subfigure
            image_paths = [] # list of image paths(str)
            image_ids = [] # list of image ids(str)
            image_gt_subcap = [] # list of gt one-hot tensor
            image_locations = [] # tensor shape (num_subfig, 4)
            width = datum['width']
            height = datum['height']
            for subfigure in datum["subfigures"]:
                if subfigure["label"] in subcaps and len(subcaps[subfigure["label"]]) > 0 and len(subcaps[subfigure["label"]]) < 76:
                    x = [point[0] for point in subfigure["points"]]
                    y = [point[1] for point in subfigure["points"]]
                    x1 = min(x)
                    x2 = max(x)
                    y1 = min(y)
                    y2 = max(y)
                    # 过滤不在图中的bbox (medicat没有这一步，不过在数据集中没有发现不符合的subfig
                    if y2 < 0 or x2 < 0 or x1 > datum['width'] or y1 > datum['height']:
                        print(subfigure["id"])
                        continue
                    # 规范bbox
                    if y1 < 0:
                        y1 = 0
                    if x1 < 0:
                        x1 = 0
                    if x2 > datum['width']:
                        x2 = datum['width']
                    if y2 > datum['height']:
                        y2 = datum['height']
                    if x1 == x2 or y1 == y2:
                        continue

                    image_paths.append(image_root+'/'+subfigure["id"])
                    image_ids.append(subfigure["id"])
                    tmp_idx = subcap_dict[subfigure["label"]]
                    tmp_one_hot = torch.zeros(padded_subcaps.shape[0])
                    tmp_one_hot[tmp_idx] = 1
                    image_gt_subcap.append(tmp_one_hot)
                    image_locations.append([x1/width, y1/height, x2/width, y2/height])

            if len(image_paths) > 0:
                sample_info = {}
                sample_info['img2cap_gt'] = torch.stack(image_gt_subcap, dim=0) # one-hot matrix (num_subfig, num_subcap)
                sample_info['img_path'] = image_paths
                sample_info['img_id'] = image_ids
                sample_info['padded_subcap'] = padded_subcaps
                sample_info['cap_str'] = datum['text']
                sample_info['img_loc'] =  torch.tensor(image_locations)
                self.samples.append(sample_info)

        self.dataset_size = int(trainset_size_ratio * len(self.samples))  # 一轮epoch中train sample的数目
        self.shuffle_index()

        self.input_size = input_size

        print('Keep %d Compound Figure' % (len(self.samples)))

    def shuffle_index(self):
        """
            when train dataset set to be smaller or larger than real dataset, need to randomly select a new set,
            mapping the index to a real sample

            self.dataset_size (int) : size of train/val set
        """
        if self.dataset_size < len(self.samples):
            self.index_map = np.random.choice(len(self.samples), self.dataset_size, replace=False)
        elif self.dataset_size == len(self.samples):
            self.index_map = np.arange(len(self.samples))
        else:
            self.index_map = np.concatenate(np.arange(len(self.samples)), np.random.choice(len(self.samples), self.dataset_size-len(self.samples), replace=False))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        index = self.index_map[index]    # transfer to new sample index
        
        padded_img_ls = []
        for img_path in self.samples[index]['img_path']:
            unpadded_image = Image.open(img_path).convert('RGB')
            unpadded_image = self.image_transform(unpadded_image)
            _, h, w = unpadded_image.shape
            scale = min(self.input_size/h, self.input_size/w)
            resize_transform = transforms.Resize([round(scale*h), round(scale*w)])
            resized_img = resize_transform(unpadded_image) # reize到input_size(224)以内
            resized_img, _ = self.aug_tool(resized_img, []) # augmentation
            pad = (0, self.input_size-round(scale*w), 0, self.input_size-round(scale*h))  
            padded_img_ls.append(F.pad(resized_img, pad, 'constant', 0)) # 对img加pad成224x224
        padded_img = torch.stack(padded_img_ls) # (num_subfig, 3, 224, 224)

        padded_subcap = self.samples[index]['padded_subcap'] # (num_sucap, 77)
        img2cap_gt = self.samples[index]['img2cap_gt'] # (num_subfig, num_subcap)
        img_loc = self.samples[index]['img_loc'] # (num_subfig, 4)
        cap_str = self.samples[index]['cap_str']  # string
        
        return padded_img, img_loc, padded_subcap, img2cap_gt, cap_str

def bidirection_sentencewise_align_collate(data):
    """
    Stack captions and images in a batch

    Args:
        data: a list of [padded_img, img_loc, padded_subcap, img2cap_gt, cap_str], refer to __getitem__() in OnlyCap_Dataset

    Returns:
        images: tensor (bs, 3, max_h, max_w)
        captions: tensor (bs, max_l)
        subfigs: list of lists  [ ... [box(tensor, (subfig_num, 4)), class(tensor, (subfig_num, 1)), alignment(tensor, (subfig_num, max_l))], ... ]
    """
    img_split_idx = []    # num_subfig in each sample/compound-figure
    cap_split_idx = []     # num_subcap in each sample/compound-figure
    padded_img_ls = []
    padded_cap_ls = []
    loc_ls = []
    img2cap_gt_ls = []
    cap_str_ls = []
    for unit in data:
        padded_img, img_loc, padded_subcap, img2cap_gt, cap_str = unit
        img_split_idx.append(padded_img.shape[0])
        padded_img_ls.append(padded_img)
        loc_ls.append(img_loc)
        cap_split_idx.append(padded_subcap.shape[0])
        padded_cap_ls.append(padded_subcap)
        cap_str_ls.append(cap_str)
        img2cap_gt_ls.append(img2cap_gt)
    # batch内的img和cap拼成一个matrix
    padded_img = torch.cat(padded_img_ls, dim=0) # (bs*num_subimg, 3, max_w, max_h)
    padded_cap = torch.cat(padded_cap_ls, dim=0) # (bs*num_subcap, 77)
    loc = torch.cat(loc_ls, dim=0) # (bs*num_subimg, 4)
    # 每个compound figure的gt matrix放在整个batch的大gt matrix的对角线上
    img2cap_gt_martix = torch.zeros(padded_img.shape[0], padded_cap.shape[0])
    i_cursor = 0
    t_cursor = 0
    for subfig_num, subcap_num, gt_matrix in zip(img_split_idx, cap_split_idx, img2cap_gt_ls):
        img2cap_gt_martix[i_cursor:i_cursor+subfig_num, t_cursor:t_cursor+subcap_num] = gt_matrix
        i_cursor += subfig_num
        t_cursor += subcap_num

    return {'padded_img':padded_img, 'loc':loc, 'img_split_idx':img_split_idx, 
            'padded_cap':padded_cap, 'cap_str_ls':cap_str_ls, 'cap_split_idx':cap_split_idx,
            'img2cap_gt_ls':img2cap_gt_martix}
