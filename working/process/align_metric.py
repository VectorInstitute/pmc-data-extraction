from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union

    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Calculate GIoU between box pairs

    Input boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def pair_wise_bce(output, target):
    """
    Calculate BCE between tensor pairs 

    Args:
        output & target (tensor): prob sequences with shape (N/M, *)

    Returns:
        loss_martix: [N, M] pairwise matrix
    """
    assert output.shape[1:] == target.shape[1:], "Cant Match Shape of Output, Target for Binary Cross Entrop Loss"

    loss_matrix = torch.zeros([output.shape[0], target.shape[0]])
    for i in range(output.shape[0]):
        for j in range(target.shape[0]):
            try:
                loss_matrix[i, j] = F.binary_cross_entropy(output[i], target[j])
            except:
                print(output[i])
                print(target[j])
                print(output[i]>=0)
                print(output[i]<=1)

    return loss_matrix


def iou(box1, box2):
    # Box1 and Box2 should be (x1, y1, x2, y2), 1 is the left top point, 2 is the right bottom point
    if max(box1[0], box2[0]) > min(box1[2], box2[2]) or max(box1[1], box2[1]) > min(box1[3], box2[3]):
        return 0
    intersect_box = [max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])]
    intersect_area = (intersect_box[2]-intersect_box[0])*(intersect_box[3]-intersect_box[1])
    union_area = (box1[3]-box1[1])*(box1[2]-box1[0])+(box2[3]-box2[1])*(box2[2]-box2[0])-intersect_area
    if union_area == 0:
        return 0
    return intersect_area / union_area


class SubfigureSubcaptionAlignmentMetric():
    def __init__(self, iou_threshold: float):
        self.iou_threshold = iou_threshold
        self.reset()

    def update(self,
               predicted_subfigures: List[List[List[float]]], # [bs * [num_gt * [4]]]
               predicted_tokens: List[List[List[int]]], # [bs * [num_pred * [pred_cap_len]]]
               gold_subfigures: List[List[List[float]]], # [bs * [num_gt * [4]]]
               gold_tokens: List[List[List[int]]],    # [bs * [num_gt * [gt_cap_len]]]
               wordpieces: List[List[str]]): # [bs * [cap_len]]

        match_matrix = []   # [bs * [num_gt]]

        assert len(predicted_subfigures) == len(predicted_tokens) == len(gold_subfigures) == len(gold_tokens) == len(wordpieces)

        batch_size = len(wordpieces)
        for i in range(batch_size):
            match_idx = []
            # 对Predict Token过滤
            pred_filtered_tokens = []
            for subcaption in predicted_tokens[i]:
                filtered_subcaption = []
                for t in subcaption:
                    if wordpieces[i][t][0] != "#" and wordpieces[i][t].isalnum() and (len(wordpieces[i][t]) > 0 or not wordpieces[i][t].isalpha()):
                        # if t not in common_wordpieces[i]:
                        filtered_subcaption.append(t)
                pred_filtered_tokens.append(filtered_subcaption)  # [num_pred * [pred_cap_len] 

            # 对于每个GT Subfigure，找到与之最匹配的Predict Subfigure（没有则直接f1=0）
            for k in range(len(gold_subfigures[i])):
                max_iou = 0
                max_iou_index = None
                for p in range(len(predicted_subfigures[i])):
                    iou_value = iou(
                        box_cxcywh_to_xyxy(torch.tensor(predicted_subfigures[i][p])), 
                        box_cxcywh_to_xyxy(torch.tensor(gold_subfigures[i][k]))
                    )
                    """
                    print('Sample%d, GT%d:(%.2f, %.2f, %.2f, %.2f) Pred%d:(%.2f, %.2f, %.2f, %.2f), IOU%.1f' % (i, k, 
                    gold_subfigures[i][k][0], gold_subfigures[i][k][1], gold_subfigures[i][k][2], gold_subfigures[i][k][3],
                    p, 
                    predicted_subfigures[i][p][0], predicted_subfigures[i][p][1], predicted_subfigures[i][p][2], predicted_subfigures[i][p][3], 
                    iou_value))
                    """
                    if iou_value > max_iou:
                        max_iou = iou_value
                        max_iou_index = p
                if max_iou < self.iou_threshold:
                    self.f1s.append(0)
                    self.recall.append(0)
                    self.prec.append(0)
                    match_idx.append(-1)
                    continue
                else:
                    match_idx.append(max_iou_index)

                # 对GT Token过滤
                gold_filtered_tokens = []
                for t in gold_tokens[i][k]:
                    if wordpieces[i][t][0] != "#" and wordpieces[i][t].isalnum() and (len(wordpieces[i][t]) > 0 or not wordpieces[i][t].isalpha()):
                        gold_filtered_tokens.append(t)
                if len(gold_filtered_tokens) == 0:
                    continue
                
                # 最匹配的Predict Subfigure对应的Subcaption，与GT Subcaption计算F1
                matching_pred_tokens = pred_filtered_tokens[max_iou_index]
                # print([wordpieces[i][ind] for ind in matching_pred_tokens])
                intersection = set(gold_filtered_tokens).intersection(set(matching_pred_tokens))
                recall = float(len(intersection))/float(len(gold_filtered_tokens))
                if recall == 0:
                    self.f1s.append(0)
                    self.recall.append(0)
                    self.prec.append(0)
                    continue
                precision = float(len(intersection))/float(len(matching_pred_tokens))

                self.recall.append(recall)
                self.prec.append(precision)
                self.f1s.append(2.0*precision*recall/(precision+recall))
        
            match_matrix.append(match_idx)

        return match_matrix


        

    def get_metric(self, reset: bool = False):
        if len(self.f1s) == 0:
            return 0.0, 0.0, 0.0

        avg_f1 = np.mean(self.f1s)
        avg_rcl = np.mean(self.recall)
        avg_prec = np.mean(self.prec)

        if reset:
            self.reset()
        return avg_f1, avg_rcl, avg_prec

    def reset(self):
        self.f1s = []
        self.recall = []
        self.prec = []


if __name__=="__main__":
    similarity_threshold = 0.5
    iou_threshold = 0.5
    from dataset import FigCap_Dataset, my_collate
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    val_file = '/remote-home/share/medical/public/MedICaT/subfig_subcap_val.jsonl' 
    img_root = '/remote-home/share/medical/public/MedICaT/release/figures'
    vocab_file = '/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/scibert/vocab.txt'
    ValSet = FigCap_Dataset(val_file, img_root, vocab_file)
    ValLoader = DataLoader(ValSet, batch_size=2, shuffle=False, num_workers=4, collate_fn=my_collate)
    val_subcaption_metric = SubfigureSubcaptionAlignmentMetric(iou_threshold)
    val_subcaption_metric.reset()
    for batch in tqdm(ValLoader):
        image = batch['image']#.cuda() # (bs, 3, max_w, max_h)
        caption = batch['caption']#.cuda()   # (bs, max_l)
        subfigures = batch['subfigs']
        # output_det_class, output_box, output_sim = model(image, caption)    # (bs, query_num, 1), (bs, query_num, 4), (bs, query_num, caption_length)
        # accumulate results as subfig-subcap format 
        cpu_caption = caption#.cpu()
        ls_caption = [ValSet.id_to_token(cpu_caption[i].tolist()) for i in range(cpu_caption.shape[0])]   # [bs * cap_len] before evaluation, convert ids to tokens
        #
        # filter_mask = output_det_class.squeeze() > config.score_threshold # [bs, query_num], True or False
        # ls_pred_boxes = [output_box[i, filter_mask[i], :].tolist() for i in range(image.shape[0])] # [bs * [filtered_query_num * [4]]], (cx, cy, w, h)
        #
        ls_gt_boxes = [ls[1].tolist() for ls in subfigures]  # [bs * [subfigure * [4]]], (cx, cy, w, h)
        #
        index_matrix = torch.arange(0, caption.shape[-1])  # [caption_length], (0, 1, 2 ...)
        ls_pred_tokens = [] # [bs * [filtered_query_num * [aligned_token_num]]]
        ls_gt_tokens = []   # [bs * [subfig_num * [subcap_token_num]]]
        for i in range(image.shape[0]):
            # filter_tmp = output_sim[i, filter_mask[i], :] > config.similarity_threshold    # (filtered_query_num, caption_length) True or False
            # pred_tokens = [index_matrix[filter_tmp[j, :]].tolist() for j in range(filter_tmp.shape[0])] # [filtered_query_num * [aligned_token_num]] index of tokens in the caption
            # ls_pred_tokens.append(pred_tokens)
            filter_tmp = subfigures[i][2] > similarity_threshold    # (subfig_num, caption_length) True or False
            gt_tokens = [index_matrix[filter_tmp[j, :]].tolist() for j in range(filter_tmp.shape[0])] # [subfig_num * [subcap_token_num]] index of tokens in the caption
            ls_gt_tokens.append(gt_tokens)
        val_subcaption_metric.update(predicted_subfigures=ls_gt_boxes, predicted_tokens=ls_gt_tokens, 
                                     gold_subfigures=ls_gt_boxes, gold_tokens=ls_gt_tokens, wordpieces=ls_caption)
            
    f1, rcl, prc = val_subcaption_metric.get_metric(reset=True)


