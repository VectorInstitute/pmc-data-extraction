import math

import torch
from einops import repeat
from pytorch_pretrained_bert.modeling import BertModel
from torch import nn
from torchvision import models

from openpmcvl.granular.models.transformer_module import *


class FigCap_Former(nn.Module):
    def __init__(
        self,
        num_query=50,
        num_encoder_layers=6,
        num_decoder_layers=6,
        feature_dim=256,
        atten_head_num=8,
        mlp_ratio=4,
        dropout=0.0,
        activation="relu",
        alignment_network=False,
        bert_path="/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/PubMed_BERT",
        num_text_decoder_layers=6,
        text_atten_head_num=8,
        text_mlp_ratio=4,
        text_dropout=0.0,
        text_activation="relu",
        resnet=34,
        resnet_pretrained=False,
    ):
        super().__init__()
        # Followings are modules for fig detection
        if resnet == 18:
            self.img_embed = nn.Sequential(
                *list(models.resnet18(pretrained=resnet_pretrained).children())[:8]
            ).cuda()
            self.img_channel_squeeze = nn.Conv2d(512, feature_dim, 1)
        elif resnet == 34:
            self.img_embed = nn.Sequential(
                *list(models.resnet34(pretrained=resnet_pretrained).children())[:8]
            ).cuda()
            self.img_channel_squeeze = nn.Conv2d(512, feature_dim, 1)
        elif resnet == 50:
            self.img_embed = nn.Sequential(
                *list(models.resnet50(pretrained=resnet_pretrained).children())[:8]
            ).cuda()
            self.img_channel_squeeze = nn.Conv2d(2048, feature_dim, 1)
        else:
            print("ResNet Error: Unsupported Version ResNet%d" % resnet)
            exit()
        self.pos_embed = PositionEncoding(num_pos_feats=feature_dim)

        encoder_layer = TransformerEncoderLayer(
            feature_dim, atten_head_num, mlp_ratio * feature_dim, dropout, activation
        )
        self.img_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.query = nn.Parameter(torch.rand(num_query, feature_dim))
        decoder_layer = TransformerDecoderLayer(
            feature_dim, atten_head_num, mlp_ratio * feature_dim, dropout, activation
        )
        self.img_decoder = TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_decoder_layers
        )

        self.box_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 4),
            nn.Sigmoid(),
        )
        self.det_class_head = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())

        # Followings are modules for fig-cap alignment
        self.alignment_network = alignment_network  # exclude alignment modules(BERT) to allow multi-gpu acceleration
        if self.alignment_network:
            self.text_embed = BertModel.from_pretrained(bert_path)

            self.text_channel_squeeze = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(768, feature_dim),
                nn.Dropout(p=dropout),
            )

            text_decoder_layer = TransformerDecoderLayer(
                feature_dim,
                text_atten_head_num,
                text_mlp_ratio * feature_dim,
                text_dropout,
                text_activation,
            )
            self.text_decoder = TransformerDecoder(
                decoder_layer=text_decoder_layer, num_layers=num_text_decoder_layers
            )

            self.simi_head = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 1),
                nn.Sigmoid(),
            )

            self.img_proj = nn.Parameter(torch.rand(feature_dim, feature_dim))

    def forward(self, images, texts):
        """
        1. Detect the subfigures (nonobject/object binary classification  + box coordinates linear regression)
        2. Align captions to each of the detection output

        Args:
            images (compound figure): shape (bs, c, h, w)
            texts (caption tokens): shape (bs, max_length_in_this_batch)

        Returns
        -------
            output_det_class: tensor (bs, query_num, 1), 0~1 indicate subfigure or no-subfigure
            output_box: tensor (bs, query_num, 4), prediction of [cx, cy, w, h]
            similarity: tensor (bs, query_num, caption_length), 0~1 indicate belong or not belong to the subfigure
        """
        # Img Embed
        x = self.img_embed(images)  # (bs, 2048, h/32, w/32)
        x = self.img_channel_squeeze(x)  # (bs, 256, h/32, w/32)

        pos = self.pos_embed(
            x.shape[0], x.shape[2], x.shape[3], x.device
        )  # (bs, 256, h/32, w/32)
        x = x + pos
        x = x.view(x.shape[0], x.shape[1], -1)  # (bs, 256, (w*h)/(32*32))
        x = x.transpose(1, 2)  # (bs, (w*h)/(32*32), 256)

        # Detect
        x = self.img_encoder(x)  # (bs, (w*h)/(32*32), 256)
        query = repeat(self.query, "l d -> bs l d", bs=x.shape[0])  # (bs, 50, 256)
        query, _ = self.img_decoder(x, query)  # (bs, 50, 256)

        output_det_class = self.det_class_head(query)  # (bs, 50, 1)
        output_box = self.box_head(query)  # (bs, 50, 4)

        # Text Embed
        if self.alignment_network:
            t = self.text_embed(texts)[0][-1]  # (bs, l, 768)
            t = self.text_channel_squeeze(t)  # (bs, l, 256)

            # Align
            query = query @ self.img_proj  # (bs, 50, 256)
            t, _ = self.text_decoder(query, t)  # (bs, l, 256)

            query = query.unsqueeze(2).repeat(1, 1, t.shape[1], 1)  # (bs, 50, l, 256)
            t = t.unsqueeze(1).repeat(1, query.shape[1], 1, 1)  # (bs, 50, l, 256)
            similarity = torch.cat((query, t), -1)  # (bs, 50, l, 512)
            similarity = self.simi_head(similarity).squeeze(-1)  # (bs, 50, l)
        else:
            # We wont use similarity, set to 0 to make code faster
            similarity = (
                0  # torch.zeros(query.shape[0], query.shape[1], texts.shape[-1]).cuda()
            )

        return output_det_class, output_box, similarity


class PositionEncoding(nn.Module):
    def __init__(
        self, normalize=True, scale=100.0, num_pos_feats=256, temperature=10000
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, bs, h, w, device):
        # Input is b,c,h,w
        mask = torch.ones(bs, h, w, device=device)
        # Since image is 2D, position encoding is split into x,y directions
        # 1 1 1 1 ..  2 2 2 2... 3 3 3...
        y_embed = mask.cumsum(
            1, dtype=torch.float32
        )  # (b, h, w) the 'y-index' of each position
        # 1 2 3 4 ... 1 2 3 4...
        x_embed = mask.cumsum(
            2, dtype=torch.float32
        )  # (b, h, w) the 'x-index' of each position
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # num_pos_feats = 128
        # 0~127 self.num_pos_feats=128, since input vector is 256, encoding is half sin, half cos
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Output shape=b,h,w,128
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # Each feature map position is encoded as a 256d vector, first 128d for y direction, last 128d for x direction
        return pos  # (b,n=256,h,w)
