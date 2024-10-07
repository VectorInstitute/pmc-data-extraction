import json

import numpy as np
import torch
import torch.nn.functional as F
from augmentation_tools import Augmentation_Tool
from PIL import Image
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


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

class FigCap_Dataset(Dataset):
    def __init__(self, aug_params, filepath, image_root, vocab_file, normalization=False, trainset_size_ratio=1.0, input_size=512):
        print('Fig Cap Dataset')
        self.images = []        # list of {'path':'xxx/xxx.png', 'w':256, 'h':256}
        self.subfig_bbox = []   # list [comfig_num, [subfig_num, [4]]]
        self.subfig_token = []  # list [comfig_num, [subfig_num, [subcap_len]]]
        self.captions = []      # list of [500, 3971, ...] i.e. token id in the vocab

        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

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
        anno_subfig_num = 0
        filtered_compound_fig_num = 0
        filtered_subfig_num = 0
        filtered_token_num = 0
        print("%d Compound Figure" % len(lines))

        #min_max = 10000
        #max_max = 0

        for datum in tqdm(data):
            # Don't include figures that don't have subfigures
            if 'subfigures' not in datum or datum['subfigures']==None or len(datum['subfigures']) == 0:
                continue
            # Don't include figures that lack subcaptions (detection不需要这一步
            if "subcaptions" not in datum or datum["subcaptions"] == None or len(datum["subcaptions"]) == 0:
                continue
            # basic info of compound figure
            image_info = {}
            image_info["path"] = image_root+'/'+datum["id"]
            image_info['id'] = datum["id"]
            image_info["w"] = datum["width"]
            image_info["h"] = datum["height"]
            image_info['caption'] = datum['text']

            """
            if image_info["w"] > image_info["h"]:
                if min_max > image_info["w"]:
                    min_max = image_info["w"]
                if max_max < image_info["w"]:
                    max_max = image_info["w"]
            else:
                if min_max > image_info["h"]:
                    min_max = image_info["h"]
                if max_max < image_info["h"]:
                    max_max = image_info["h"]
            """
            
            # caption tokens
            old_tokens = ["[CLS]"] + [token["text"] for token in datum["tokens"]] + ["[SEP]"]  # list of ['CLS', 'This', 'figure' ...]
            old_subcaps = {}      # {'a':[0,1,2,3...], ...}
            for subcap_key in datum["subcaptions"]:
                old_subcaps[subcap_key] = [idx+1 for idx in datum["subcaptions"][subcap_key]]
            tokens, subcaps = convert_to_wordpieces(self.tokenizer, old_tokens, old_subcaps)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # 这里len > 512直接扔掉是否合理？（detection不需要这一步，medicat没有这一步，少了一个sample
            if len(token_ids) > 512:
                print('abandon len(cap) > 512')
                continue

            # subfigure and subcaption
            subfig_list = []    # [subfignum, [4]]
            subcap_list = []    # [subfignum, [subcap_len]]
            for subfigure in datum["subfigures"]:
                # (detection不需要这一步
                if subfigure["label"] in subcaps and len(subcaps[subfigure["label"]]) > 0 :# and len(subfigure['points'])==4:
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
                    # subfig_subcap['label'] = subfigure["label"]
                    subfig_list.append([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
                    subcap_list.append(subcaps[subfigure["label"]])

            anno_subfig_num += len(datum["subfigures"])
            if len(subfig_list) > 0:
                self.images.append(image_info)
                self.captions.append(token_ids)
                self.subfig_bbox.append(subfig_list)
                self.subfig_token.append(subcap_list)
                filtered_subfig_num += len(subfig_list)
            
            filtered_compound_fig_num += 1

        self.dataset_size = int(trainset_size_ratio * len(self.images))  # 一轮epoch中train sample的数目
        self.shuffle_index()

        self.input_size = input_size

        print('Filter %d Compound Figures'%filtered_compound_fig_num)
        print('Total %d Subfig'%anno_subfig_num)
        print('Filter %d Subfig'%filtered_subfig_num)
        # print('Min Max', min_max)
        # print('Max Max', max_max)
        # print(filtered_token_num)

    def id_to_token(self, id_tensor):
        return self.tokenizer.convert_ids_to_tokens(id_tensor)

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

        subcap_tokens = self.subfig_token[index] # [subfig_num, [subcap_len]]
        unnorm_bboxes = self.subfig_bbox[index]  # [subfig_num, [cx, cy, w, h]]
        
        unpadded_image, unnorm_bboxes = self.aug_tool(unpadded_image, unnorm_bboxes) # tensor (3, h, w), unnormalized [... [cx, cy, w, h], ...]
        
        return unpadded_image, unnorm_bboxes, subcap_tokens, torch.tensor(self.captions[index]).type(torch.IntTensor), self.images[index]['id'], self.images[index]['h'], self.images[index]['w'], self.images[index]['caption'], self.input_size

def figcap_collate(data):
    """
    1. Padding captions and images in a batch 
    2. Normalize box coordinates after padding 
    3. Adjust subcap indexes after padding

    Args:
        data: a list of [image(tensor), subfigures(list of dicts), caption(tensor)], refer to __getitem__() in FigCap_Dataset

    Returns:
        images: tensor (bs, 3, max_h, max_w)
        captions: tensor (bs, max_l)
        subfigs: list of lists  [ ... [box(tensor, (subfig_num, 4)), class(tensor, (subfig_num, 1)), alignment(tensor, (subfig_num, max_l))], ... ]
    """
    image_ids = []
    image_hws = []
    image_captions = []
    max_l = 0
    for unit in data:
        image, unnorm_bboxes, subcap_tokens, caption, image_id, original_h, original_w, untokenized_cap, input_size = unit

        image_ids.append(image_id)
        image_hws.append([original_h, original_w])
        image_captions.append(untokenized_cap)

        cap_len = caption.shape[0]  # 统计cap长度
        if cap_len > max_l:
            max_l = cap_len

    pad_imgs = []
    pad_caps = []
    subfigures = []
    for unit in data:
        image, unnorm_bboxes, subcap_tokens, caption, image_id, original_h, original_w, untokenized_cap, input_size = unit

        # resize+pad成512x512
        _, h, w = image.shape
        scale = min(input_size/h, input_size/w)
        resize_transform = transforms.Resize([round(scale*h), round(scale*w)])
        resized_img = resize_transform(image) # reize到input_size(512)以内
        pad = (0, input_size-round(scale*w), 0, input_size-round(scale*h))  
        padded_img = F.pad(resized_img, pad, 'constant', 0) # 对img加pad成512x512
        pad_imgs.append(padded_img)

        for i in range(len(unnorm_bboxes)):
            unnorm_bboxes[i] = [unnorm_bboxes[i][0]*scale/input_size, 
                                unnorm_bboxes[i][1]*scale/input_size,
                                unnorm_bboxes[i][2]*scale/input_size, 
                                unnorm_bboxes[i][3]*scale/input_size]  # pad后normalize坐标(cx, cy, w, h)
        pad = (0, max_l-caption.shape[0])
        pad_caps.append(F.pad(caption, pad, 'constant', 0))    # 对seq加[PAD], scibert的vocab中PAD的id是0

        subfig_box = [] # (subfigure_num, 4)
        subfig_tokens = []  # (subfigure_num, max_length_in_this_batch)
        for subfig in unnorm_bboxes:
            subfig_box.append(subfig)
        for subcap in subcap_tokens:
            tmp = torch.zeros(max_l)
            tmp[subcap] = 1
            subfig_tokens.append(tmp)
        subfig_box = torch.tensor(subfig_box)
        subfig_tokens = torch.stack(subfig_tokens, dim=0)
        subfigures.append([subfig_box, subfig_tokens])

    pad_imgs = torch.stack(pad_imgs, dim=0) # (bs, 3, max_w, max_h)
    pad_caps = torch.stack(pad_caps, dim=0) # (bs, max_cap_l)

    # To Continue
    return {'image':pad_imgs, 'image_id':image_ids, 'original_hws':image_hws, 'caption':pad_caps, 'subfigs':subfigures, 'untokenized_caption':image_captions}

################################ Compound Figures ################################

class Fig_Dataset(Dataset):
    def __init__(self, aug_param, filepath, image_root, vocab_file, normalization=False, trainset_size_ratio=1.0, input_size=512):
        print('Only Fig Dataset')
        self.images = []        # list of {'path':'xxx/xxx.png', 'w':256, 'h':256}
        self.subfigures = []    # list of [{'subfig_coord':[cx, cy, w, h], 'subcap_token_idx':[1,2,3...]}, ...]
        self.captions = []      # list of [500, 3971, ...] i.e. token id in the vocab

        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

        # aug param
        self.aug_tool = Augmentation_Tool(aug_param)

        if normalization:
            self.image_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
        else:
            self.image_transform = transforms.Compose([
                                transforms.ToTensor()
                                ])

        # preprocessing
        f = open(filepath)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        anno_subfig_num = 0
        filtered_compound_fig_num = 0
        filtered_subfig_num = 0
        filtered_token_num = 0
        print("%d Compound Figure" % len(lines))
        for datum in tqdm(data):
            # Don't include figures that don't have subfigures
            if 'subfigures' not in datum or datum['subfigures']== None or len(datum['subfigures']) == 0:
                continue

            # basic info of compound figure
            image_info = {}
            image_info["path"] = image_root+'/'+datum["id"]
            image_info["image"] = Image.open(image_root+'/'+datum["id"]).convert('RGB')
            image_info['id'] = datum["id"]
            image_info["w"] = datum["width"]
            image_info["h"] = datum["height"]

            # subfigure and subcaption
            subfig_list = []
            for subfigure in datum["subfigures"]:
                x = [point[0] for point in subfigure["points"]]
                y = [point[1] for point in subfigure["points"]]
                x1 = min(x)
                x2 = max(x)
                y1 = min(y)
                y2 = max(y)
                # 过滤不在图中的bbox (medicat没有这一步，不过在数据集中没有发现不符合的subfig
                if y2 < 0 or x2 < 0 or x1 > datum['width'] or y1 > datum['height']:
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
                # 过滤重合的bbox
                if x1 == x2 or y1 == y2:
                    continue
                # subfig_subcap['label'] = subfigure["label"]
                subfig = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]    # [cx, cy, w, h]
                subfig_list.append(subfig)

            anno_subfig_num += len(datum["subfigures"])
            if len(subfig_list) > 0:
                self.images.append(image_info)
                self.subfigures.append(subfig_list)  # [comfigs, [subfigs, [4]]]
                filtered_subfig_num += len(subfig_list)
            
            filtered_compound_fig_num += 1

        self.dataset_size = int(trainset_size_ratio * len(self.images))  # 一轮epoch中train sample的数目
        self.shuffle_index()

        self.input_size = input_size

        print('Filter %d Compound Figures'%filtered_compound_fig_num)
        print('Total %d Subfig'%anno_subfig_num)
        print('Filter %d Subfig'%filtered_subfig_num)

    def id_to_token(self, id_tensor):
        return self.tokenizer.convert_ids_to_tokens(id_tensor)

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

        unpadded_image = self.images[index]['image']
        unpadded_image = self.image_transform(unpadded_image)
        unnorm_bboxes = self.subfigures[index] # [subfig_num, [cx, cy, w, h]]
        
        for bbox in unnorm_bboxes:  # subfig augmentation
            cx, cy, w, h = bbox
            x1 = int(cx - w/2)
            x2 = int(cx + w/2)
            y1 = int(cy - h/2)
            y2 = int(cy + h/2)
            subfig = unpadded_image[:, y1:y2, x1:x2]
            subfig, _ = self.aug_tool(subfig, [])
            unpadded_image[:, y1:y2, x1:x2] = subfig

        unpadded_image, unnorm_bboxes = self.aug_tool(unpadded_image, unnorm_bboxes) # tensor (3, h, w), unnormalized [... [cx, cy, w, h], ...]
        
        return unpadded_image, unnorm_bboxes, self.images[index]['h'], self.images[index]['w'], self.images[index]['id'], self.input_size

def fig_collate(data):
    """
    1. Padding captions and images in a batch 
    2. Normalize box coordinates after padding 

    Args:
        data: refer to __getitem__() in FigCap_Dataset

    Returns:
        images: tensor (bs, 3, max_h, max_w)
        captions: tensor (bs, max_l)
        subfigs: list of lists  [ ... [box(tensor, (subfig_num, 4)), alignment(tensor, (subfig_num, max_l))], ... ]
        other info: ......
    """
    pad_imgs = []
    subfigs = []
    unpadded_hws = []
    image_ids = []
    untokenized_captions = []
    for sample in data:
        unpadded_image, unnorm_bboxes, unpadded_h, unpadded_w, sample_id, input_size = sample
        # image : tensor (3, h, w)
        # bbox : unnormalized [subfig_num * [cx, cy, h, w]]
        # original_hw / unpadded_hw : scalar, original h/w of the real images, or the unpadded h/w of the syn images
        # id : image id or syn id
        image_ids.append(sample_id)
        unpadded_hws.append([unpadded_h, unpadded_w])
        untokenized_captions.append('[PAD]')

        _, h, w = unpadded_image.shape
        scale = min(input_size/h, input_size/w)
        resize_transform = transforms.Resize([round(scale*h), round(scale*w)])
        resized_img = resize_transform(unpadded_image) # reize到input_size以内
        pad = (0, input_size-round(scale*w), 0, input_size-round(scale*h))  
        padded_img = F.pad(resized_img, pad, 'constant', 0) # 对img加pad成input_sizexinput_size
        pad_imgs.append(padded_img)

        subfig_box = [] # (subfigure_num, 4)
        for bboxes in unnorm_bboxes:
            subfig_box.append([bboxes[0]*scale/input_size, 
                               bboxes[1]*scale/input_size, 
                               bboxes[2]*scale/input_size, 
                               bboxes[3]*scale/input_size])  # pad后normalize坐标(cx, cy, w, h)
        subfig_box = torch.tensor(subfig_box)
        subfigs.append([subfig_box, torch.zeros(len(subfig_box), 1)])

    pad_imgs = torch.stack(pad_imgs, dim=0) # (bs, 3, max_w, max_h)

    return {'image':pad_imgs, 'subfigs':subfigs, 
            'caption':torch.zeros([len(data), 1], dtype=torch.int32), 
            'unpadded_hws':unpadded_hws, 
            'image_id':image_ids, 
            'untokenized_caption':untokenized_captions}

def fig_detr_collate(data):
    """
    1. Padding captions and images in a batch 
    2. Normalize box coordinates after padding 

    Args:
        data: refer to __getitem__() in FigCap_Dataset

    Returns:
        images: tensor (bs, 3, max_h, max_w)
        captions: tensor (bs, max_l)
        subfigs: list of lists  [ ... [box(tensor, (subfig_num, 4)), alignment(tensor, (subfig_num, max_l))], ... ]
        other info: ......
    """
    pad_imgs = []
    subfigs = []
    unpadded_hws = []
    image_ids = []
    untokenized_captions = []
    input_size = data[0][-1]
    image_mask = torch.zeros(len(data), input_size, input_size)
    for i, sample in enumerate(data):
        unpadded_image, unnorm_bboxes, unpadded_h, unpadded_w, sample_id, input_size = sample
        # image : tensor (3, h, w)
        # bbox : unnormalized [subfig_num * [cx, cy, h, w]]
        # original_hw / unpadded_hw : scalar, original h/w of the real images, or the unpadded h/w of the syn images
        # id : image id or syn id
        image_ids.append(sample_id)
        unpadded_hws.append([unpadded_h, unpadded_w])
        untokenized_captions.append('[PAD]')

        _, h, w = unpadded_image.shape
        scale = min(input_size/h, input_size/w)
        resize_transform = transforms.Resize([round(scale*h), round(scale*w)])
        resized_img = resize_transform(unpadded_image) # reize到input_size以内
        pad = (0, input_size-round(scale*w), 0, input_size-round(scale*h))  # (padding_left,padding_right, padding_top, padding_bottom)
        padded_img = F.pad(resized_img, pad, 'constant', 0) # 对img加pad成input_sizexinput_size
        pad_imgs.append(padded_img)
        image_mask[i, round(scale*h):, round(scale*w):] = 1

        subfig_box = [] # (subfigure_num, 4)
        for bboxes in unnorm_bboxes:
            subfig_box.append([bboxes[0]*scale/input_size, 
                               bboxes[1]*scale/input_size, 
                               bboxes[2]*scale/input_size, 
                               bboxes[3]*scale/input_size])  # pad后normalize坐标(cx, cy, w, h)
        subfig_box = torch.tensor(subfig_box)
        subfigs.append([subfig_box, torch.zeros(len(subfig_box), 1)])

    pad_imgs = torch.stack(pad_imgs, dim=0) # (bs, 3, max_h, max_w)

    return {'image':pad_imgs, 'image_mask':image_mask, 'subfigs':subfigs, 
            'caption':torch.zeros([len(data), 1], dtype=torch.int32), 
            'unpadded_hws':unpadded_hws, 
            'image_id':image_ids, 
            'untokenized_caption':untokenized_captions}


################################ Compound Figures for Separation Inference ################################

class Fig_Separation_Dataset(Dataset):
    def __init__(self, aug_param, filepath, image_root, normalization=False, start=0, end=-1, input_size=512):
        print('Only Fig Dataset')
        self.images = []        # list of {'path':'xxx/xxx.png', 'w':256, 'h':256}
        # self.subfigures = []    # list of [{'subfig_coord':[cx, cy, w, h], 'subcap_token_idx':[1,2,3...]}, ...]
        # self.captions = []      # list of [500, 3971, ...] i.e. token id in the vocab

        # aug param
        self.aug_tool = Augmentation_Tool(aug_param)

        if normalization:
            self.image_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
        else:
            self.image_transform = transforms.Compose([
                                transforms.ToTensor()
                                ])

        # preprocessing
        f = open(filepath)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        data = data[start:end]
        anno_subfig_num = 0
        filtered_compound_fig_num = 0
        filtered_subfig_num = 0
        filtered_token_num = 0
        print("%d Compound Figure" % len(lines))
        count = start
        for datum in tqdm(data):
            # # Don't include figures that don't have subfigures
            # if 'subfigures' not in datum or datum['subfigures']== None or len(datum['subfigures']) == 0:
            #     continue

            # basic info of compound figure
            image_info = {}
            image_info["path"] = image_root+'/'+datum["id"]
            image_info['id'] = datum["id"]
            image_info['index'] = count
            count += 1
            image_info["w"] = datum["width"]
            image_info["h"] = datum["height"]

            self.images.append(image_info)
            
            # # subfigure and subcaption
            # subfig_list = []
            # for subfigure in datum["subfigures"]:
            #     x = [point[0] for point in subfigure["points"]]
            #     y = [point[1] for point in subfigure["points"]]
            #     x1 = min(x)
            #     x2 = max(x)
            #     y1 = min(y)
            #     y2 = max(y)
            #     # 过滤不在图中的bbox (medicat没有这一步，不过在数据集中没有发现不符合的subfig
            #     if y2 < 0 or x2 < 0 or x1 > datum['width'] or y1 > datum['height']:
            #         continue
            #     # 规范bbox
            #     if y1 < 0:
            #         y1 = 0
            #     if x1 < 0:
            #         x1 = 0
            #     if x2 > datum['width']:
            #         x2 = datum['width']
            #     if y2 > datum['height']:
            #         y2 = datum['height']
            #     # 过滤重合的bbox
            #     if x1 == x2 or y1 == y2:
            #         continue
            #     # subfig_subcap['label'] = subfigure["label"]
            #     subfig = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]    # [cx, cy, w, h]
            #     subfig_list.append(subfig)

            # anno_subfig_num += len(datum["subfigures"])
            # if len(subfig_list) > 0:
            #     self.images.append(image_info)
            #     self.subfigures.append(subfig_list)  # [comfigs, [subfigs, [4]]]
            #     filtered_subfig_num += len(subfig_list)
            
            filtered_compound_fig_num += 1

        self.input_size = input_size

        # print('Filter %d Compound Figures'%filtered_compound_fig_num)
        # print('Total %d Subfig'%anno_subfig_num)
        # print('Filter %d Subfig'%filtered_subfig_num)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        unpadded_image = Image.open(self.images[index]['path']).convert('RGB')
        unpadded_image = self.image_transform(unpadded_image)
        
        # unnorm_bboxes = self.subfigures[index] # [subfig_num, [cx, cy, w, h]]
        
        # for bbox in unnorm_bboxes:  # subfig augmentation
        #     cx, cy, w, h = bbox
        #     x1 = int(cx - w/2)
        #     x2 = int(cx + w/2)
        #     y1 = int(cy - h/2)
        #     y2 = int(cy + h/2)
        #     subfig = unpadded_image[:, y1:y2, x1:x2]
        #     subfig, _ = self.aug_tool(subfig, [])
        #     unpadded_image[:, y1:y2, x1:x2] = subfig
        
        # unpadded_image, unnorm_bboxes = self.aug_tool(unpadded_image, unnorm_bboxes) # tensor (3, h, w), unnormalized [... [cx, cy, w, h], ...]
        
        # return unpadded_image, unnorm_bboxes, self.images[index]['h'], self.images[index]['w'], self.images[index]['id'], self.images[index]['index'], self.input_size
        return unpadded_image, self.images[index]['h'], self.images[index]['w'], self.images[index]['id'], self.images[index]['index'], self.input_size

def fig_separation_collate(data):
    """
    1. Padding captions and images in a batch 
    2. Normalize box coordinates after padding 

    Args:
        data: refer to __getitem__() in FigCap_Dataset

    Returns:
        images: tensor (bs, 3, max_h, max_w)
        captions: tensor (bs, max_l)
        # subfigs: list of lists  [ ... [box(tensor, (subfig_num, 4)), alignment(tensor, (subfig_num, max_l))], ... ]
        other info: ......
    """
    pad_imgs = []
    # subfigs = []
    unpadded_hws = []
    image_ids = []
    image_index = []
    untokenized_captions = []
    unpadded_images = []
    for sample in data:
        # unpadded_image, unnorm_bboxes, unpadded_h, unpadded_w, sample_id, index, input_size = sample
        unpadded_image, unpadded_h, unpadded_w, sample_id, index, input_size = sample
        # image : tensor (3, h, w)
        # bbox : unnormalized [subfig_num * [cx, cy, h, w]]
        # original_hw / unpadded_hw : scalar, original h/w of the real images, or the unpadded h/w of the syn images
        # id : image id or syn id
        image_ids.append(sample_id)
        image_index.append(index)
        unpadded_hws.append([unpadded_h, unpadded_w])
        untokenized_captions.append('[PAD]')

        _, h, w = unpadded_image.shape
        scale = min(input_size/h, input_size/w)
        resize_transform = transforms.Resize([round(scale*h), round(scale*w)])
        resized_img = resize_transform(unpadded_image) # reize到input_size以内
        pad = (0, input_size-round(scale*w), 0, input_size-round(scale*h))  
        padded_img = F.pad(resized_img, pad, 'constant', 0) # 对img加pad成input_sizexinput_size
        pad_imgs.append(padded_img)

        # subfig_box = [] # (subfigure_num, 4)
        # for bboxes in unnorm_bboxes:
        #     subfig_box.append([bboxes[0]*scale/input_size, 
        #                        bboxes[1]*scale/input_size, 
        #                        bboxes[2]*scale/input_size, 
        #                        bboxes[3]*scale/input_size])  # pad后normalize坐标(cx, cy, w, h)
        # subfig_box = torch.tensor(subfig_box)
        # subfigs.append([subfig_box, torch.zeros(len(subfig_box), 1)])
        
        unpadded_images.append(unpadded_image)  # [bs * (3, h, w)]

    pad_imgs = torch.stack(pad_imgs, dim=0) # (bs, 3, max_w, max_h)

    return {'image':pad_imgs, # 'subfigs':subfigs, 
            'caption':torch.zeros([len(data), 1], dtype=torch.int32), 
            'unpadded_hws':unpadded_hws, 
            'image_id':image_ids, 
            'image_index':image_index,
            'untokenized_caption':untokenized_captions,
            'original_image':unpadded_images}


        