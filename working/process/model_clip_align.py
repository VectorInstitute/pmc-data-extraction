import math
from cgitb import text

import numpy as np
import torch
# from clip_code.clip_module import ModifiedResNet
import torch.nn.functional as F
from einops import repeat
# from pytorch_pretrained_bert.modeling import BertModel
from torch import nn
from torchvision import models, transforms
# from torchsummary import summary
from transformer_module import *


# 直接zero-shot用CLIP计算image和text的相似度（不考虑position），V1版本使用代码
class SentenceWise_Align_Former_V1(nn.Module):
    def __init__(self):
        super().__init__()
        
        EXTRACT_DIR = '/remote-home/share/weixionglin/medclip/vlp/open_clip/src/extract_component/extracted'
        
        img_checkpoint_path = f"{EXTRACT_DIR}/2023_06_29-08_02_01-model_CLIP-RN50-p_amp/ImageEncoder.bin"
        self.img_encoder = torch.load(img_checkpoint_path)

        text_checkpoint_path = f'{EXTRACT_DIR}/2023_06_29-07_58_12-model_CLIP-RN50-p_amp/TextEncoder.bin'
        self.text_encoder = torch.load(text_checkpoint_path)
        self.text_proj = torch.load(f'{EXTRACT_DIR}/2023_06_29-07_58_12-model_CLIP-RN50-p_amp/text_projection.bin')

    def forward(self, images, texts):
        """
        Assign Img(Subfigure) to the Subcaption (img2txt 

        Args:
            images (subfigure): shape (bs, c, h, w)
            texts (subcaption tokens): shape (bs, subcap_num, max_len)

        Returns:
            similarity: list of tensor (bs, subcap_num), image's similarity to each subcaption(one positive, N negative)
        """

        # Img Embed
        img_feature = self.img_encoder(images)['image_features']  # (bs, 768)

        img_feature = F.normalize(img_feature, dim=-1)

        # Text Embed
        texts = torch.reshape(texts, (-1, texts.shape[-1])) # (bs*subcap_num, 77)
        txt_feature = self.text_encoder(texts)['pooler_output']   # (bs*subcap_num, 768)
        txt_feature = txt_feature @ self.text_proj
        
        txt_feature = F.normalize(txt_feature, dim=-1)
        txt_feature = torch.reshape(txt_feature, (img_feature.shape[0], -1, txt_feature.shape[-1])) # (bs, subcap_num, 768)

        # Img Similarity to Each Subcap
        cos_simi = torch.bmm(txt_feature, img_feature.unsqueeze(-1))  # (bs, subcap_num, 1) bmm requires 3D input (bs, h, w)
        cos_simi = (cos_simi + 1)/2

        return cos_simi.squeeze(-1)


########################################### 分割线：V0版本代码 before MICCAI


class SentenceWise_Align_Former(nn.Module):
    def __init__(self):
        super().__init__()
        img_checkpoint_path = "./clip_code/extracted/2022_12_30-05_55_11-model_RN50_PubmedBERT_MLMG-p_amp/ImageEncoder.bin"
        self.img_encoder = torch.load(img_checkpoint_path)

        text_checkpoint_path = "./clip_code/extracted/2022_12_30-05_50_30-model_RN50_PubmedBERT_MLMG-p_amp/TextEncoder.bin"
        self.text_encoder = torch.load(text_checkpoint_path)

        self.bbox_embed = nn.Sequential(
            nn.Linear(4, 768),
            nn.ReLU(inplace=True),
        )

    def forward(self, images, texts, location):
        """
        Assign Img(Subfigure) to the Subcaption (img2txt 

        Args:
            images (subfigure): shape (bs, c, h, w)
            texts (subcaption tokens): shape (bs, subcap_num, max_len)
            location (subfigure bbox location): shape (bs, 4)

        Returns:
            similarity: list of tensor (bs, subcap_num), image's similarity to each subcaption(one positive, N negative)
        """

        # Img Embed
        img_feature = self.img_encoder(images)  # (bs, 768)
        loc_feature = self.bbox_embed(location) # (bs, 768)
        img_feature = F.normalize(img_feature+loc_feature, dim=-1)

        # Text Embed
        texts = torch.reshape(texts, (-1, texts.shape[-1])) # (bs*subcap_num, 77)
        txt_feature = self.text_encoder(texts)['pooler_output']   # (bs*subcap_num, 768)
        txt_feature = F.normalize(txt_feature, dim=-1)
        txt_feature = torch.reshape(txt_feature, (img_feature.shape[0], -1, txt_feature.shape[-1])) # (bs, subcap_num, 768)

        # Img Similarity to Each Subcap
        cos_simi = torch.bmm(txt_feature, img_feature.unsqueeze(-1))  # (bs, subcap_num, 1) bmm requires 3D input (bs, h, w)
        cos_simi = (cos_simi + 1)/2

        return cos_simi.squeeze(-1)


class SentenceWise_Align_Former_Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        img_checkpoint_path = "./clip_code/extracted/2022_12_30-05_55_11-model_RN50_PubmedBERT_MLMG-p_amp/ImageEncoder.bin"
        self.img_encoder = torch.load(img_checkpoint_path)

        text_checkpoint_path = "./clip_code/extracted/2022_12_30-05_50_30-model_RN50_PubmedBERT_MLMG-p_amp/TextEncoder.bin"
        self.text_encoder = torch.load(text_checkpoint_path)

        self.bbox_embed = nn.Sequential(
            nn.Linear(4, 768),
            nn.ReLU(inplace=True),
        )

    def forward(self, images, texts, location):
        """
        Align Img(Subfigure) with Subcaptions (img2txt 

        Args:
            images (subfigure): shape (bs, c, h, w)
            texts (subcaption tokens): shape (bs, subcap_num, max_len)
            location (subfigure bbox location): shape (bs, 4)

        Returns:
            similarity: list of tensor (bs, subcap_num), image's similarity to each subcaption(one positive, N negative)
        """
        
        # Img Embed
        img_feature = self.img_encoder(images)  # (bs, 768)
        loc_feature = self.bbox_embed(location) # (bs, 768)
        img_feature = F.normalize(img_feature+loc_feature, dim=-1)

        # Text Embed
        txt_feature = self.text_encoder(texts)['pooler_output']   # (bs, subcap_num, 768)
        txt_feature = F.normalize(txt_feature, dim=-1)

        # Img Similarity to Each Subcap
        cos_simi = torch.bmm(txt_feature, img_feature.unsqueeze(-1))  # (bs, subcap_num)
        cos_simi = F.softmax(cos_simi, dim=1)

        return cos_simi.squeeze()


class SentenceWise_Align_Former_SCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = ModifiedResNet(
                layers=[3,4,6,3],
                output_dim=768,
                heads=8,
                image_size=224,
                width=64
            )

        bert_path = '/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/PubMed_BERT'
        self.text_encoder = BertModel.from_pretrained(bert_path)
        self.text_projection = nn.Parameter(torch.rand(768, 768))

    def forward(self, images, texts, location):
        """
        Align Img(Subfigure) with Subcaptions (img2txt 

        Args:
            images (subfigure): shape (bs, c, h, w)
            texts (subcaption tokens): shape [bs * (subcap_num, max_len)]
            location (subfigure bbox location): shape (bs, 4)

        Returns:
            similarity: list of tensor [bs * (subcap_num)], image's similarity to each subcaption(one positive, N negative)
        """

        # Img Embed
        img_feature = self.img_encoder(images)[0]  # (bs, 768)
        img_feature = F.normalize(img_feature, dim=-1)

        # Text Embed
        cursor = 0
        index_range = []    # [[0,2], [2, 10], ......]
        for sample in texts:
            index_range.append([cursor, cursor+sample.shape[0]])
            cursor += sample.shape[0]
        flatten_texts = torch.cat(texts, dim=0).cuda()  # (bs*subcap_num, max_length_in_this_batch)
        txt_feature = self.text_encoder(flatten_texts)[0][-1][:, 0, :]   # (bs*subcap_num, 768) 取CLS对应的Feature

        txt_feature = txt_feature @ self.text_projection    # (bs*subcap_num, 768)  

        txt_feature = F.normalize(txt_feature, dim=-1)
        unflatten_txt_feature = []
        for r in index_range:
            unflatten_txt_feature.append(txt_feature[r[0]:r[1], :]) # [bs * (subcap_num, 768)]

        # Img Similarity to Each Subcap
        cos_simi = []
        for i in range(img_feature.shape[0]):
            cos_simi.append((unflatten_txt_feature[i] @ img_feature[i] + 1)/2)    # [bs * (subcap_num)]

        return cos_simi


class SentenceWise_Align_Former_Bidirection(nn.Module):
    def __init__(self):
        super().__init__()
        img_checkpoint_path = "/remote-home/share/weixionglin/checkpoint_demo/extracted/2022_12_30-05_55_11-model_RN50_PubmedBERT_MLMG-p_amp/ImageEncoder.bin"
        self.img_encoder = torch.load(img_checkpoint_path)

        text_checkpoint_path = "/remote-home/share/weixionglin/checkpoint_demo/extracted/2022_12_30-05_50_30-model_RN50_PubmedBERT_MLMG-p_amp/TextEncoder.bin"
        self.text_encoder = torch.load(text_checkpoint_path)

    def forward(self, img, cap):
        """
        Contrast subfigs and subcaps within the same compound figure

        Args:
            img (subfigure): shape (bs*subfig_num, c, h, w)
            cap (subcaption tokens): shape (bs*subcap_num, 77)

        Returns:
            similarity: list of tensor [bs * (num_subfig, num_subcap)]
        """

        # Img Embed
        img_feature = self.img_encoder(img)  # (bs*subfig_num, 768)
        img_feature = F.normalize(img_feature, dim=-1)
        # img_feature_ls = torch.split(img_feature, img_split_idx, dim=0)

        # Text Embed
        txt_feature = self.text_encoder(cap)['pooler_output']   # (bs*subcap_num, 768)
        txt_feature = F.normalize(txt_feature, dim=-1)
        # txt_feature_ls = torch.split(txt_feature, cap_split_idx, dim=0)

        # Img Similarity to Each Subcap
        cos_simi = (img_feature @ txt_feature.T + 1)/2

        return cos_simi


class SentenceWise_Align_Former_Bidirection_Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        img_checkpoint_path = "/remote-home/share/weixionglin/checkpoint_demo/extracted/2022_12_30-05_55_11-model_RN50_PubmedBERT_MLMG-p_amp/ImageEncoder.bin"
        self.img_encoder = torch.load(img_checkpoint_path)

        text_checkpoint_path = "/remote-home/share/weixionglin/checkpoint_demo/extracted/2022_12_30-05_50_30-model_RN50_PubmedBERT_MLMG-p_amp/TextEncoder.bin"
        self.text_encoder = torch.load(text_checkpoint_path)

    def forward(self, img, cap):
        """
        Contrast subfigs and subcaps within the same compound figure

        Args:
            img (subfigure): shape (bs*subfig_num, c, h, w)
            cap (subcaption tokens): shape (bs*subcap_num, 77)

        Returns:
            similarity: list of tensor [bs * (num_subfig, num_subcap)]
        """

        # Img Embed
        img_feature = self.img_encoder(img)  # (bs*subfig_num, 768)
        img_feature = F.normalize(img_feature, dim=-1)
        # img_feature_ls = torch.split(img_feature, img_split_idx, dim=0)

        # Text Embed
        txt_feature = self.text_encoder(cap)['pooler_output']   # (bs*subcap_num, 768)
        txt_feature = F.normalize(txt_feature, dim=-1)
        # txt_feature_ls = torch.split(txt_feature, cap_split_idx, dim=0)

        # Img Similarity to Each Subcap
        cos_simi = F.softmax(img_feature @ txt_feature.T, dim=1)

        return cos_simi


class PositionEncoding(nn.Module):
    def __init__(self, normalize=True, scale=100.0, num_pos_feats=256, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats//2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
    def forward(self, bs, h, w, device):
        # 输入是b,c,h,w
        mask = torch.ones(bs, h, w, device=device)
        # 因为图像是2d的，所以位置编码也分为x,y方向
        # 1 1 1 1 ..  2 2 2 2... 3 3 3...
        y_embed = mask.cumsum(1, dtype=torch.float32)   # (b, h, w) the 'y-index' of each position
        # 1 2 3 4 ... 1 2 3 4...
        x_embed = mask.cumsum(2, dtype=torch.float32)   # (b, h, w) the 'x-index' of each position
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # num_pos_feats = 128
        # 0~127 self.num_pos_feats=128,因为前面输入向量是256，编码是一半sin，一半cos
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 输出shape=b,h,w,128
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # 每个特征图的xy位置都编码成256的向量，其中前128是y方向编码，而128是x方向编码
        return pos  # (b,n=256,h,w)


if __name__ == '__main__':
    device = torch.device('cuda')
    model = FigCap_Former()
    # model = nn.DataParallel(model)
    model.to(device)
    img = torch.rand(4, 3, 128, 128).to(device)
    text = torch.randint(1, 30000, (4, 100)).to(device)
    a, b, c = model(img, text)
    print(a.shape)
    print(a[0,0,0])
    print(b.shape)
    print(c.shape)
    print(c[0,0,0])
    """
    print(resnet)
    print(summary(resnet, (3, 128, 128)))
    x = torch.ones(4, 3, 128, 128).cuda()
    print(x.shape)
    y = resnet(x)
    print(y.shape)

    x = torch.ones(1, 3, 4, 4)
    print(pe(x))
    """