"""
Train functions of detection, detection & token-wise alignment(fail), clip-style sentence-wise alignment 
"""

import os
import time
from builtins import print
from datetime import datetime, timedelta, timezone

import torch
import torch.nn.functional as F
from align_metric import (SubfigureSubcaptionAlignmentMetric,
                          box_cxcywh_to_xyxy, generalized_box_iou,
                          pair_wise_bce)
from detect_metric import calculate_mAP_voc12, side_loss
from einops import repeat
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_sentencewise_align(model=None, optimizer=None, lr_sch=None, config=None, start_epoch=0, best_val_acc=0.0, trainLoader=None, trainSet=None, valLoader=None, valSet=None):
    """
    基于PMC-CLIP-BETA在Alignment数据上训练subfig和subcap对齐（在train时只考虑img2text

    Args:
        model : DETR + BERT + AlignmentDecoder
        config : args parser
        start_epoch : start from a checkpoint or from 0
        best_val_acc : load from a checkpoint or 0.0 when train from scratch
        ......
    """
    # log file
    f_path = os.path.join(config.output_dir, 'log.txt')
    with open(f_path, 'a') as f:
        SHA_TZ = timezone(timedelta(hours=8),
                          name='Asia/Shanghai')   
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        beijing_now = utc_now.astimezone(SHA_TZ)    # 北京时间
        configDict = config.__dict__
        f.write('%d--%d--%d--%d--%d Configs :\n' % (beijing_now.year, beijing_now.month, beijing_now.day, beijing_now.hour, beijing_now.minute))
        for eachArg, value in configDict.items():
            f.write(eachArg + ' : ' + str(value) + '\n')
        f.write('\n')
        f.close()
    log_path = os.path.join(config.output_dir, 'log')
    writer = SummaryWriter(log_path)
    # model checkpoint
    val_align_best_model_path = os.path.join(config.output_dir, 'best_align.pth')
    cp_model_path = os.path.join(config.output_dir, 'checkpoint.pth')
    # start train from here
    for epoch in range(start_epoch, config.epoch_num+1):
        # train one epoch
        model.train()
        train_loss_align = 0.0
        hit = 0
        noaug_hit = 0
        random_guess_acc_ls = []
        for batch in tqdm(trainLoader):
            image = batch['image'].cuda()   # (bs, 3, max_w, max_h)
            subcaptions = batch['subcaptions'].cuda()   # (bs, subcap_num, 77)
            location = batch['location'].cuda()    # (bs, 4)
            gt = batch['subcaption_gts'].cuda()  # (bs, subcap_num)
            noaug_subcap_idx = batch['noaug_subcap_idx']  # [bs]
            nopad_subcap_idx = batch['nopad_subcap_idx']  # [bs]
            # forward pass
            output_sim = model(image, subcaptions, location)    # (bs, subcap_num)
            batch_loss = F.binary_cross_entropy(torch.clamp(output_sim, min=1e-10, max=1-1e-10), gt)
            train_loss_align += batch_loss.detach().item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # train set的对齐命中率
            for i in range(image.shape[0]):
                tmp_sim = output_sim[i, :noaug_subcap_idx[i]]
                tmp_gt_sim = gt[i, :noaug_subcap_idx[i]]
                if torch.argmax(tmp_sim) == torch.argmax(tmp_gt_sim):
                    noaug_hit += 1
            for i in range(image.shape[0]):
                tmp_sim = output_sim[i, :nopad_subcap_idx[i]]
                tmp_gt_sim = gt[i, :nopad_subcap_idx[i]]
                if torch.argmax(tmp_sim) == torch.argmax(tmp_gt_sim):
                    hit += 1
            # 对每个sample随机猜的命中率
            for noaug_subcap_num in noaug_subcap_idx:
                random_guess_acc_ls.append(1/noaug_subcap_num)
        lr_sch.step()
        noaug_train_acc = noaug_hit/len(trainSet)
        train_acc = hit/len(trainSet)
        random_guess_acc = sum(random_guess_acc_ls)/len(random_guess_acc_ls)
        print('train random guess acc:', random_guess_acc)
        train_loss_align /= len(trainLoader)

        # validate one epoch
        with torch.no_grad():
            model.eval()
            hit = 0
            noaug_hit = 0
            random_guess_acc_ls = []
            for batch in tqdm(valLoader):
                # forward pass
                image = batch['image'].cuda()   # (bs, 3, max_w, max_h)
                subcaptions = batch['subcaptions'].cuda()   # (bs, subcap_num, 77)
                location = batch['location'].cuda()    # (bs, 4)
                gt = batch['subcaption_gts'].cuda()  # (bs, subcap_num)
                noaug_subcap_idx = batch['noaug_subcap_idx']  # [bs]
                nopad_subcap_idx = batch['nopad_subcap_idx']  # [bs]
                output_sim = model(image, subcaptions, location)    # (bs, subcap_num)
                # val set的对齐命中率
                for i in range(image.shape[0]):
                    tmp_sim = output_sim[i, :noaug_subcap_idx[i]]
                    tmp_gt_sim = gt[i, :noaug_subcap_idx[i]]
                    if torch.argmax(tmp_sim) == torch.argmax(tmp_gt_sim):
                        noaug_hit += 1
                for i in range(image.shape[0]):
                    tmp_sim = output_sim[i, :nopad_subcap_idx[i]]
                    tmp_gt_sim = gt[i, :nopad_subcap_idx[i]]
                    if torch.argmax(tmp_sim) == torch.argmax(tmp_gt_sim):
                        hit += 1
                # 对每个sample随机猜的命中率
                for noaug_subcap_num in noaug_subcap_idx:
                    random_guess_acc_ls.append(1/noaug_subcap_num)
            noaug_val_acc = noaug_hit/len(valSet)
            val_acc = hit/len(valSet)
            random_guess_acc = sum(random_guess_acc_ls)/len(random_guess_acc_ls)
            print('val random guess acc:', random_guess_acc)

        # record and save
        if noaug_val_acc > best_val_acc:
            best_val_acc = noaug_val_acc
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_sch_state_dict': lr_sch.state_dict(),                        
                        'val_acc': noaug_val_acc,
                        'train_acc': noaug_train_acc,
                        'train_loss_align': train_loss_align,
                        }, val_align_best_model_path)

        if epoch % config.cp_interval == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_sch_state_dict': lr_sch.state_dict(),                        
                        'val_acc': noaug_val_acc,
                        'train_acc': noaug_train_acc,
                        'train_loss_align': train_loss_align,
                        }, cp_model_path)

        writer.add_scalar('train/noaug_train_acc', noaug_train_acc, epoch)
        writer.add_scalar('train/train_loss_align', train_loss_align, epoch)
        writer.add_scalar('val/noaug_val_acc', noaug_val_acc, epoch)
        writer.add_scalar('train/train_acc', train_acc, epoch)
        writer.add_scalar('val/val_acc', val_acc, epoch)

        out_str = "Epoch:%d    Learning Rate:%.4e\n" \
        "trn-aln:%.4f   trn_acc:%.4f    noaug_trn_acc:%.4f    val_acc:%.4f    noaug_val_acc:%.4f"% \
        (epoch, optimizer.param_groups[0]['lr'],
        train_loss_align, train_acc, noaug_train_acc, val_acc, noaug_val_acc)
        
        with open(f_path, 'a') as f:
            f.write(out_str + '\n\n')
        print(out_str)

    out_str = 'Best val_acc:%.2f' % (best_val_acc)
    print(out_str)
    with open(f_path, 'a') as f:
        f.write(out_str )


def train_sentencewise_align_bidirection(model=None, optimizer=None, lr_sch=None, config=None, start_epoch=0, best_val_acc=0.0, trainLoader=None, valLoader=None):
    """
    基于PMC-CLIP-BETA在Alignment数据上训练subfig和subcap对齐（在训练时考虑一个compoundfigure内img2text和text2img

    Args:
        model : DETR + BERT + AlignmentDecoder
        config : args parser
        start_epoch : start from a checkpoint or from 0
        best_val_acc : load from a checkpoint or 0.0 when train from scratch
        ......
    """
    # log file
    f_path = os.path.join(config.output_dir, 'log.txt')
    with open(f_path, 'a') as f:
        SHA_TZ = timezone(timedelta(hours=8),
                          name='Asia/Shanghai')   
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        beijing_now = utc_now.astimezone(SHA_TZ)    # 北京时间
        configDict = config.__dict__
        f.write('%d--%d--%d--%d--%d Configs :\n' % (beijing_now.year, beijing_now.month, beijing_now.day, beijing_now.hour, beijing_now.minute))
        for eachArg, value in configDict.items():
            f.write(eachArg + ' : ' + str(value) + '\n')
        f.write('\n')
        f.close()
    log_path = os.path.join(config.output_dir, 'log')
    writer = SummaryWriter(log_path)
    # model checkpoint
    val_align_best_model_path = os.path.join(config.output_dir, 'best_align.pth')
    cp_model_path = os.path.join(config.output_dir, 'checkpoint.pth')

    for epoch in range(start_epoch, config.epoch_num+1):
        # train one epoch
        model.train()
        train_loss_align = 0.0
        img2txt_hit = 0
        img2txt_miss = 0
        # img2txt_random_guess_acc_ls = []
        for batch in tqdm(trainLoader):
            # forward pass
            padded_img = batch['padded_img'].cuda() # (bs*num_subfig, 3, 224, 224)
            padded_cap = batch['padded_cap'].cuda() # (bs*num_sucap, 77)
            loc = batch['loc'].cuda()    # (bs*num_subfig, 4)
            img2cap_gt = batch['img2cap_gt_ls'].cuda() # (bs*num_subfig, bs*num_subcap)
            img_split_idx = batch['img_split_idx'] # [bs]
            cap_split_idx = batch['cap_split_idx'] # [bs]
            similarity_matrix = model(padded_img, padded_cap, loc, img_split_idx, cap_split_idx) # (bs*subfig_num, bs*subcap_num)

            i_cursor = 0
            t_cursor = 0
            batch_i_loss = 0.0
            batch_t_loss = 0.0
            for subfig_num, subcap_num in zip(img_split_idx, cap_split_idx):
                pred = similarity_matrix[i_cursor:i_cursor+subfig_num, t_cursor:t_cursor+subcap_num]
                gt = img2cap_gt[i_cursor:i_cursor+subfig_num, t_cursor:t_cursor+subcap_num]
                i_cursor += subfig_num
                t_cursor += subcap_num
                for i in range(pred.shape[0]):
                    batch_i_loss += F.binary_cross_entropy(torch.clamp(pred[i,:], min=1e-10, max=1-1e-10), gt[i,:])
                    if torch.argmax(pred[i,:]) == torch.argmax(gt[i,:]):
                        img2txt_hit += 1
                    else:
                        img2txt_miss += 1
                for i in range(pred.shape[1]):
                    batch_t_loss += F.binary_cross_entropy(torch.clamp(pred[:,i], min=1e-10, max=1-1e-10), gt[:,i])
            batch_i_loss /= similarity_matrix.shape[0]
            batch_t_loss /= similarity_matrix.shape[1]

            batch_loss = (batch_i_loss + batch_t_loss)/2
            train_loss_align += batch_loss.detach().item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            """
            # 对每个sample随机猜的命中率
            # for gt_matrix in img2cap_gt: # (num_subfig, num_subcap)
            #    img2txt_random_guess_acc_ls += [1/gt_matrix.shape[1]] * gt_matrix.shape[0]

            # bce loss weight for each subfig-subcap pair in the batch
            coef_matrix = torch.ones_like(similarity_matrix) * config.easy_sample_weight
            i_cursor = 0
            t_cursor = 0
            for subfig_num, subcap_num in zip(img_split_idx, cap_split_idx):
                coef_matrix[i_cursor:i_cursor+subfig_num, t_cursor:t_cursor+subcap_num] = 1.0
                i_cursor += subfig_num
                t_cursor += subcap_num

            # bp
            assert similarity_matrix.shape == coef_matrix.shape == img2cap_gt.shape
            batch_loss_img = 0.0
            batch_loss_cap = 0.0
            bce_matrix = F.binary_cross_entropy(torch.clamp(similarity_matrix, min=1e-10, max=1-1e-10), img2cap_gt, reduce=False)
            bce_matrix = torch.mul(bce_matrix, coef_matrix)
            for i in range(bce_matrix.shape[0]): # bs*num_subfig
                batch_loss_img += torch.mean(bce_matrix[i, :])
                if torch.argmax(similarity_matrix[i, :]) == torch.argmax(img2cap_gt[i, :]):
                    img2txt_hit += 1
                else:
                    img2txt_miss += 1
            for i in range(bce_matrix.shape[1]): # bs*num_subcap
                batch_loss_cap += torch.mean(bce_matrix[:, i])
            batch_loss_img /= bce_matrix.shape[0]
            batch_loss_cap /= bce_matrix.shape[1]
            batch_loss = (batch_loss_cap + batch_loss_img)/2
            train_loss_align += batch_loss.detach().item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            """

        lr_sch.step()
        train_acc = img2txt_hit/(img2txt_hit+img2txt_miss)
        # random_guess_acc = sum(img2txt_random_guess_acc_ls)/len(img2txt_random_guess_acc_ls)
        # print('train random guess acc:', random_guess_acc)
        train_loss_align /= len(trainLoader)

        # validate one epoch
        with torch.no_grad():
            model.eval()
            img2txt_hit = 0
            img2txt_miss = 0
            # img2txt_random_guess_acc_ls = []
            for batch in tqdm(valLoader):
                # forward pass
                padded_img = batch['padded_img'].cuda() # (bs*num_subfig, 3, 224, 224)
                padded_cap = batch['padded_cap'].cuda() # (bs*num_sucap, 77)
                loc = batch['loc'].cuda()    # (bs*num_subfig, 4)
                img2cap_gt = batch['img2cap_gt_ls'].cuda() # (bs*num_subfig, bs*num_subcap)
                img_split_idx = batch['img_split_idx'] # [bs]
                cap_split_idx = batch['cap_split_idx'] # [bs]
                similarity_matrix = model(padded_img, padded_cap, loc, img_split_idx, cap_split_idx) # (bs*subfig_num, bs*subcap_num)

                i_cursor = 0
                t_cursor = 0
                for subfig_num, subcap_num in zip(img_split_idx, cap_split_idx):
                    pred = similarity_matrix[i_cursor:i_cursor+subfig_num, t_cursor:t_cursor+subcap_num]
                    gt = img2cap_gt[i_cursor:i_cursor+subfig_num, t_cursor:t_cursor+subcap_num]
                    i_cursor += subfig_num
                    t_cursor += subcap_num
                    for i in range(pred.shape[0]):
                        if torch.argmax(pred[i,:]) == torch.argmax(gt[i,:]):
                            img2txt_hit += 1
                        else:
                            img2txt_miss += 1

                # 对每个sample随机猜的命中率
                # for gt_matrix in img2cap_gt: # (num_subfig, num_subcap)
                #    img2txt_random_guess_acc_ls += [1/gt_matrix.shape[1]] * gt_matrix.shape[0]

                """
                # evaluate
                for i in range(similarity_matrix.shape[0]): # bs*num_subfig
                    if torch.argmax(similarity_matrix[i, :]) == torch.argmax(img2cap_gt[i, :]):
                        img2txt_hit += 1
                    else:
                        img2txt_miss += 1
                """

            val_acc = img2txt_hit/(img2txt_hit+img2txt_miss)
            # random_guess_acc = sum(img2txt_random_guess_acc_ls)/len(img2txt_random_guess_acc_ls)
            # print('val random guess acc:', random_guess_acc)

        # record and save
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_sch_state_dict': lr_sch.state_dict(),                        
                        'val_acc': val_acc,
                        'train_acc': train_acc,
                        'train_loss_align': train_loss_align,
                        }, val_align_best_model_path)

        if epoch % config.cp_interval == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_sch_state_dict': lr_sch.state_dict(),                        
                        'val_acc': val_acc,
                        'train_acc': train_acc,
                        'train_loss_align': train_loss_align,
                        }, cp_model_path)

        writer.add_scalar('train/train_acc', train_acc, epoch)
        writer.add_scalar('train/train_loss_align', train_loss_align, epoch)
        writer.add_scalar('val/val_acc', val_acc, epoch)

        out_str = "Epoch:%d    Learning Rate:%.4e\n" \
        "trn-aln:%.4f   trn_acc:%.4f    val_acc:%.4f"% \
        (epoch, optimizer.param_groups[0]['lr'],
        train_loss_align, train_acc, val_acc)
        
        with open(f_path, 'a') as f:
            f.write(out_str + '\n\n')
        print(out_str)

    out_str = 'Best val_acc:%.2f' % (best_val_acc)
    print(out_str)
    with open(f_path, 'a') as f:
        f.write(out_str )


def train_det_and_align(model=None, optimizer=None, lr_sch=None, config=None, start_epoch=0, best_val_f1=0.0, best_val_mAP=0.0, trainSet=None, trainLoader=None, valLoader=None):
    """
    耦合DETR和AlignmentNetwork训练（Alignment效果不好

    Args:
        model : DETR + BERT + AlignmentDecoder
        config : args parser
        start_epoch : start from a checkpoint or from 0
        best_val_f1 & best_val_mAP : load from a checkpoint or 0.0 when train from scratch
        ......
    """
    # log file
    f_path = os.path.join(config.output_dir, 'log.txt')
    with open(f_path, 'a') as f:
        SHA_TZ = timezone(timedelta(hours=8),
                          name='Asia/Shanghai')   
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        beijing_now = utc_now.astimezone(SHA_TZ)    # 北京时间
        configDict = config.__dict__
        f.write('%d--%d--%d--%d--%d Configs :\n' % (beijing_now.year, beijing_now.month, beijing_now.day, beijing_now.hour, beijing_now.minute))
        for eachArg, value in configDict.items():
            f.write(eachArg + ' : ' + str(value) + '\n')
        f.write('\n')
        f.close()
    log_path = os.path.join(config.output_dir, 'log')
    writer = SummaryWriter(log_path)
    # path to model checkpoint
    val_align_best_model_path = os.path.join(config.output_dir, 'best_align.pth')
    val_det_best_model_path = os.path.join(config.output_dir, 'best_det.pth')
    cp_model_path = os.path.join(config.output_dir, 'checkpoint.pth')
    # subfig-subcap align metric
    train_subcaption_metric = SubfigureSubcaptionAlignmentMetric(config.iou_threshold)
    train_subcaption_metric.reset()
    val_subcaption_metric = SubfigureSubcaptionAlignmentMetric(config.iou_threshold)
    val_subcaption_metric.reset()
    # start train here
    for epoch in range(start_epoch, config.epoch_num+1):
        # train one epoch
        t0 = time.time()
        model.train()
        train_loss_box = train_loss_iou = train_loss_align = train_loss_class = train_loss_side = 0.0
        train_det_boxes = []
        train_det_labels = []
        train_det_scores = []
        train_true_boxes = []
        train_true_labels = []
        train_true_difficulties = []
        for batch in tqdm(trainLoader):
            # forward pass
            image = batch['image'].cuda() # (bs, 3, max_w, max_h)
            caption = batch['caption'].cuda()   # (bs, max_cap_l)
            subfigures = batch['subfigs']   # [... [ (subfig_num, 4), (subfig_num, max_cap_l)] ...]
            output_det_class, output_box, output_sim = model(image, caption)    # (bs, query_num, 1), (bs, query_num, 4), (bs, query_num, max_cap_l)
            # hungarian match (refer to DETR for details
            cpu_output_box = output_box.cpu()
            cpu_output_sim = output_sim.cpu()
            cpu_output_det_class = output_det_class.cpu()
            with torch.no_grad():
                best_match = [] # [bs, (subfigure)]
                bs_cost_class = F.binary_cross_entropy(cpu_output_det_class, torch.ones_like(cpu_output_det_class), reduce=False) # (bs, query_num, 1)
                for i in range(image.shape[0]):
                    gold_box, gold_align = subfigures[i]
                    cost_class = repeat(bs_cost_class[i], 'q c -> q (repeat c)', repeat=gold_box.shape[0]) # bce (query_num, subfigure_num)
                    cost_box = torch.cdist(cpu_output_box[i], gold_box, p=1)   # iou + l1 (query_num, sugfigure_num)
                    cost_iou = -generalized_box_iou(box_cxcywh_to_xyxy(cpu_output_box[i]), box_cxcywh_to_xyxy(gold_box))
                    cost_side = side_loss(cpu_output_box[i], gold_box)  # side loss (query_num, subfigure_num)
                    cost_align = pair_wise_bce(cpu_output_sim[i], gold_align) # ce (query_num, subfigure)  
                    C = config.match_cost_align * cost_align + \
                        config.match_cost_bbox * cost_box + \
                        config.match_cost_giou * cost_iou + \
                        config.match_cost_class * cost_class + \
                        config.match_cost_side * cost_side    
                    # find best bipartite match
                    _, col_idx = linear_sum_assignment(C.t())  # (subfigure) 
                    best_match.append(col_idx)
            # loss calculation according to the match
            batch_loss_box = batch_loss_iou = batch_loss_align = batch_loss_class = batch_loss_side = 0.0
            for i in range(image.shape[0]):
                gold_box, gold_align = subfigures[i]
                gold_box = gold_box.cuda()
                gold_align = gold_align.cuda()
                # bbox l1 loss + iou loss
                tmp = F.l1_loss(gold_box, output_box[i, best_match[i], :], reduction='none') # (subfig_num, 4)
                batch_loss_box += tmp.sum() / gold_box.shape[0]
                tmp = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(gold_box),
                                                         box_cxcywh_to_xyxy(output_box[i, best_match[i], :])))  # (subfig)
                batch_loss_iou += tmp.sum() / gold_box.shape[0]
                # classify loss
                tmp = torch.zeros_like(output_det_class[i]) # (query_num, 1) 
                tmp[best_match[i], :] = 1.0 # GT: set matched queries as 1.0, otherwise 0.0
                batch_loss_class += F.binary_cross_entropy(output_det_class[i], tmp)  
                # align loss
                matched_sim = output_sim[i, best_match[i], :]  # prob (subfig_num, caption_length)
                focal_weight = torch.abs(matched_sim - gold_align) ** config.focal_sigma    # (1-p_t)**sigma = (y-p)**sigma
                unweighted_bce = F.binary_cross_entropy(matched_sim, gold_align, reduce=False)  # prob (subfig_num, caption_length)
                batch_loss_align += torch.mean(focal_weight * unweighted_bce)  # prob (subfig_num, caption_length)
                # side loss
                tmp = torch.diag(side_loss(output_box[i, best_match[i], :], gold_box))  # (subfig_num)
                batch_loss_side += tmp.sum() / gold_box.shape[0]
            train_loss_box += batch_loss_box.detach().item()
            train_loss_iou += batch_loss_iou.detach().item()
            train_loss_align += batch_loss_align.detach().item()
            train_loss_class += batch_loss_class.detach().item()
            train_loss_side += batch_loss_side.detach().item()
            # accumulate results for mAP
            filter_index = cpu_output_det_class.squeeze() > 0.0   # [bs, pred_num] True or False
            for i in range(filter_index.shape[0]):
                train_det_boxes.append(cpu_output_box[i, filter_index[i,:], :])   # [bs * (filter_pred_num, 4)]
                # Note the boxex are kept (cx, cy, w, h) until calculating IOU in calculate_mAP()
                train_det_scores.append(cpu_output_det_class.squeeze()[i, filter_index[i,:]])  # [bs * (filter_pred_num)]
                train_det_labels.append(torch.ones_like(train_det_scores[-1]))  # [bs * (filter_pred_num)] all 1
                train_true_boxes.append(subfigures[i][0]) # [bs * (subfig_num, 4)]
                train_true_labels.append(torch.ones(train_true_boxes[-1].shape[0]))  # [bs * (subfig_num)] all 1 
                train_true_difficulties.append(torch.zeros_like(train_true_labels[-1]))  # [bs * (subfig_num)] all zeros
            # accumulate results as subfig-subcap format 
            cpu_caption = caption.cpu()
            ls_caption = [trainSet.id_to_token(cpu_caption[i].tolist()) for i in range(cpu_caption.shape[0])]   # [bs * cap_len] before evaluation, convert ids to tokens
            # filter out nonobject outputs
            filter_mask = cpu_output_det_class.squeeze() > 0.0 # [bs, query_num], True or False
            ls_pred_boxes = [cpu_output_box[i, filter_mask[i], :].tolist() for i in range(image.shape[0])] # [bs * [filtered_query_num * [4]]], (cx, cy, w, h)
            ls_gt_boxes = [ls[0].tolist() for ls in subfigures]  # [bs * [subfigure * [4]]], (cx, cy, w, h)
            # for detected and gt subfigs, find indexes of aligned tokens
            index_matrix = torch.arange(0, cpu_output_sim.shape[-1])  # [caption_length], (0, 1, 2 ...)
            ls_pred_tokens = [] # [bs * [filtered_query_num * [aligned_token_num]]]
            ls_gt_tokens = []   # [bs * [subfig_num * [subcap_token_num]]]
            for i in range(image.shape[0]):
                filter_tmp = cpu_output_sim[i, filter_mask[i], :] > config.similarity_threshold    # (filtered_query_num, caption_length) True or False
                pred_tokens = [index_matrix[filter_tmp[j, :]].tolist() for j in range(filter_tmp.shape[0])] # [filtered_query_num * [aligned_token_num]] index of tokens in the caption
                ls_pred_tokens.append(pred_tokens)
                filter_tmp = subfigures[i][1] > config.similarity_threshold    # (subfig_num, caption_length) True or False
                gt_tokens = [index_matrix[filter_tmp[j, :]].tolist() for j in range(filter_tmp.shape[0])] # [subfig_num * [subcap_token_num]] index of tokens in the caption
                ls_gt_tokens.append(gt_tokens)
            train_subcaption_metric.update(predicted_subfigures=ls_pred_boxes, predicted_tokens=ls_pred_tokens, 
                                           gold_subfigures=ls_gt_boxes, gold_tokens=ls_gt_tokens, wordpieces=ls_caption)
            # back prop
            batch_loss = config.bbox_loss_coef*batch_loss_box + \
                         config.giou_loss_coef*batch_loss_iou + \
                         config.align_loss_coef*batch_loss_align + \
                         config.class_loss_coef*batch_loss_class + \
                         config.side_loss_coef*batch_loss_side
            batch_loss /= image.shape[0]    # avg within a batch
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        lr_sch.step()
        train_mAP, train_det_recall, train_det_precision, _ = calculate_mAP_voc12(train_det_boxes, train_det_labels, train_det_scores, train_true_boxes, train_true_labels, train_true_difficulties, config.iou_threshold, config.score_threshold)
        train_f1, train_aln_recall, train_aln_precision = train_subcaption_metric.get_metric(reset=True)
        train_loss_box /= len(trainSet) # avg within an epoch
        train_loss_iou /= len(trainSet)
        train_loss_align /= len(trainSet)
        train_loss_class /= len(trainSet)
        train_loss_side /= len(trainSet)

        # torch.cuda.empty_cache()

        # validate one epoch
        with torch.no_grad():
            model.eval()
            val_loss_box = val_loss_iou = val_loss_align = val_loss_class = 0.0000
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
                output_det_class, output_box, output_sim = model(image, caption)    # (bs, query_num, 1), (bs, query_num, 4), (bs, query_num, caption_length)
                cpu_output_box = output_box.cpu()
                cpu_output_sim = output_sim.cpu()
                cpu_output_det_class = output_det_class.cpu()
                # accumulate results for mAP
                filter_index = cpu_output_det_class.squeeze() >  0.0  # [bs, pred_num] True or False
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
                ls_caption = [trainSet.id_to_token(cpu_caption[i].tolist()) for i in range(cpu_caption.shape[0])]   # [bs * cap_len] before evaluation, convert ids to tokens
                # filter out nonobject outputs
                filter_mask = output_det_class.squeeze() > 0.0 # [bs, query_num], True or False
                ls_pred_boxes = [output_box[i, filter_mask[i], :].tolist() for i in range(image.shape[0])] # [bs * [filtered_query_num * [4]]], (cx, cy, w, h)
                ls_gt_boxes = [ls[0].tolist() for ls in subfigures]  # [bs * [subfigure * [4]]], (cx, cy, w, h)
                # for detected and gt subfigs, find indexes of aligned tokens
                index_matrix = torch.arange(0, output_sim.shape[-1])  # [caption_length], (0, 1, 2 ...)
                ls_pred_tokens = [] # [bs * [filtered_query_num * [aligned_token_num]]]
                ls_gt_tokens = []   # [bs * [subfig_num * [subcap_token_num]]]
                for i in range(image.shape[0]):
                    filter_tmp = output_sim[i, filter_mask[i], :] > config.similarity_threshold    # (filtered_query_num, caption_length) True or False
                    pred_tokens = [index_matrix[filter_tmp[j, :]].tolist() for j in range(filter_tmp.shape[0])] # [filtered_query_num * [aligned_token_num]] index of tokens in the caption
                    ls_pred_tokens.append(pred_tokens)
                    filter_tmp = subfigures[i][1] > config.similarity_threshold    # (subfig_num, caption_length) True or False
                    gt_tokens = [index_matrix[filter_tmp[j, :]].tolist() for j in range(filter_tmp.shape[0])] # [subfig_num * [subcap_token_num]] index of tokens in the caption
                    ls_gt_tokens.append(gt_tokens)
                val_subcaption_metric.update(predicted_subfigures=ls_pred_boxes, predicted_tokens=ls_pred_tokens, 
                                             gold_subfigures=ls_gt_boxes, gold_tokens=ls_gt_tokens, wordpieces=ls_caption)
        val_mAP, val_det_recall, val_det_precision, _ = calculate_mAP_voc12(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, config.iou_threshold, config.score_threshold)
        val_f1, val_aln_recall, val_aln_precision = val_subcaption_metric.get_metric(reset=True)
        """
        val_loss_box /= len(valSet) # avg within an epoch
        val_loss_iou /= len(valSet)
        val_loss_align /= len(valSet)
        val_loss_class /= len(valSet)
        """
        # record and save
        if val_mAP > best_val_mAP:
            best_val_mAP = val_mAP
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_sch_state_dict': lr_sch.state_dict(),
                        #
                        'val_f1': val_f1,
                        'val_aln_recall' : val_aln_recall,
                        'val_aln_precision' : val_aln_precision,
                        'val_mAP' : val_mAP,
                        'val_det_recall' : val_det_recall,
                        'val_det_precision' : val_det_precision,
                        #
                        'train_f1': train_f1,
                        'train_aln_recall' : train_aln_recall,
                        'train_aln_precision' : train_aln_precision,
                        'train_mAP' : train_mAP,
                        'train_det_recall' : train_det_recall,
                        'train_det_precision' : train_det_precision,
                        #
                        'train_loss_box': train_loss_box,
                        'train_loss_iou': train_loss_iou,
                        'train_loss_class': train_loss_class,
                        'train_loss_align': train_loss_align,
                        }, val_det_best_model_path)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_sch_state_dict': lr_sch.state_dict(),
                        #
                        'val_f1': val_f1,
                        'val_aln_recall' : val_aln_recall,
                        'val_aln_precision' : val_aln_precision,
                        'val_mAP' : val_mAP,
                        'val_det_recall' : val_det_recall,
                        'val_det_precision' : val_det_precision,
                        #
                        'train_f1': train_f1,
                        'train_aln_recall' : train_aln_recall,
                        'train_aln_precision' : train_aln_precision,
                        'train_mAP' : train_mAP,
                        'train_det_recall' : train_det_recall,
                        'train_det_precision' : train_det_precision,
                        #
                        'train_loss_box': train_loss_box,
                        'train_loss_iou': train_loss_iou,
                        'train_loss_class': train_loss_class,
                        'train_loss_align': train_loss_align,
                        }, val_align_best_model_path)

        if epoch % config.cp_interval == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_sch_state_dict': lr_sch.state_dict(),
                        #
                        'val_f1': val_f1,
                        'val_aln_recall' : val_aln_recall,
                        'val_aln_precision' : val_aln_precision,
                        'val_mAP' : val_mAP,
                        'val_det_recall' : val_det_recall,
                        'val_det_precision' : val_det_precision,
                        #
                        'train_f1': train_f1,
                        'train_aln_recall' : train_aln_recall,
                        'train_aln_precision' : train_aln_precision,
                        'train_mAP' : train_mAP,
                        'train_det_recall' : train_det_recall,
                        'train_det_precision' : train_det_precision,
                        #
                        'train_loss_box': train_loss_box,
                        'train_loss_iou': train_loss_iou, 
                        'train_loss_class': train_loss_class,
                        'train_loss_align': train_loss_align,
                        }, cp_model_path)

        writer.add_scalar('train/train_f1', train_f1, epoch)
        writer.add_scalar('train/train_aln_recall', train_aln_recall, epoch)
        writer.add_scalar('train/train_aln_precision', train_aln_precision, epoch) 
        writer.add_scalar('train/train_mAP', train_mAP, epoch)
        writer.add_scalar('train/train_det_recall', train_det_recall, epoch)
        writer.add_scalar('train/train_det_precision', train_det_precision, epoch)      
        #          
        writer.add_scalar('train/train_loss_align', train_loss_align, epoch)
        writer.add_scalar('train/train_loss_box', train_loss_box, epoch)
        writer.add_scalar('train/train_loss_class', train_loss_class, epoch)
        writer.add_scalar('train/train_loss_iou', train_loss_iou, epoch)
        #
        writer.add_scalar('val/val_f1', val_f1, epoch)
        writer.add_scalar('val/val_aln_recall', val_aln_recall, epoch)
        writer.add_scalar('val/val_aln_precision', val_aln_precision, epoch)
        writer.add_scalar('val/val_mAP', val_mAP, epoch)
        writer.add_scalar('val/val_det_recall', val_det_recall, epoch)
        writer.add_scalar('val/val_det_precision', val_det_precision, epoch)

        out_str = "Epoch:%d    Learning Rate:%.4e\n" \
        "trn-cls:%.4f    trn-iou:%.4f    trn-box:%.4f    trn-aln:%.4f\n" \
        "trn_mAP:%.4f    trn_det_R:%.4f    trn_det_P:%.4f    trn_f1:%.4f    trn_aln_R:%.4f    trn_aln_P:%.4f\n" \
        "val_mAP:%.4f    val_det_R:%.4f    val_det_P:%.4f    val_f1:%.4f    val_aln_R:%.4f    val_aln_P:%.4f"% \
        (epoch, optimizer.param_groups[0]['lr'],
        train_loss_class, train_loss_iou, train_loss_box, train_loss_align,
        train_mAP, train_det_recall, train_det_precision, train_f1, train_aln_recall, train_aln_precision,
        val_mAP, val_det_recall, val_det_precision, val_f1, val_aln_recall, val_aln_precision
        )
        
        with open(f_path, 'a') as f:
            f.write(out_str + '\n\n')
        print(out_str)

    out_str = 'Best val_F1:%.2f, val_mAP:%.2f' % (best_val_f1, best_val_mAP)
    print(out_str)
    with open(f_path, 'a') as f:
            f.write(out_str )

