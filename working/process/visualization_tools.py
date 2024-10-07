import os
import random

from matplotlib import pyplot as plt

color_ls = ['red', 'darkred', 'lightcoral', 'orangered', 'sienna', 'sandybrown',
            'gold', 'olive', 'olivedrab', 'yellowgreen', 'darkseagreen', 'green',
            'lightseagreen', 'skyblue', 'steelblue', 'deepskyblue', 'dodgerblue',
            'blue', 'darkblue', 'blueviolet', 'violet']
random.shuffle(color_ls)

def idx_2_cap(text_idxs, cap):
    text = ''
    if len(text_idxs) > 0:
        sorted_tokens = sorted(text_idxs)
        curr_end = sorted_tokens[0]     
        for token in sorted_tokens[1:]: 
            if token > curr_end+1:
                if curr_end != sorted_tokens[0]:
                    text += '[SEP] '
                text += cap[token]
                text += ' '
            else:
                assert curr_end+1 == token, "curr_end %d, token %d, len(text_ids) %d, len(cap) %d" % (curr_end, token, len(text_idxs), len(cap))
                curr_end = token
                text += cap[token]
                text += ' '
    return text

def span_2_text(spans, caption):
    text = ""
    for span in spans:
        if span[0] == span[1]:
            text += (caption[span[0]] + '/')
        else:
            text += (caption[span[0]] + ' ' + caption[span[0]+1] + '......' + caption[span[1]-1] + ' ' + caption[span[1]] + '/')
    return text

def concat_caption(caption):
    text = ""
    for token in caption:
        if token != '[CLS]' and token != '[SEP]' and token != '[PAD]':
            text += token
            text += " "
    return text

# Visualization function for a compound figure (only infer)
def visualization_noComparision(image, original_h, original_w, boxes, normalized_coord, texts, cap, untokenized_cap, path):
    """
    Visualization a compound figure inference result 

    Args:
        image tensor: (3, h, w)
        original_h/w: scalar
        boxes tensor or list of tensors: (pred_num, 4) / [pred_num, (4)], (cx, cy, w, h) ratio of the image
        texts list: [pred_num, [subcap_len]], index
        cap list: [cap_len], string tokens
        untokenized_cap string: untokenized caption
        path string: path to save the figure
    """
    _, padded_h, padded_w = image.shape
    if original_h > original_w:
        unpadded_h = padded_h
        unpadded_w = (original_w/original_h)*padded_h
    else:
        unpadded_w = padded_w
        unpadded_h = (original_h/original_w)*padded_w
    np_image = image.permute(1, 2, 0).numpy()[:int(unpadded_h), :int(unpadded_w), :]
    subcap = []

    fig = plt.figure(dpi=300)#, figsize=(5, 5*unpadded_h/unpadded_w))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(np_image, interpolation=None)
    plt.axis("off")
    for i in range(boxes.shape[0]): 
        if normalized_coord:
            x1 = (boxes[i][0] - 0.5*boxes[i][2]) * padded_w
            y1 = (boxes[i][1] - 0.5*boxes[i][3]) * padded_h
            rec_w = boxes[i][2] * padded_w
            rec_h = boxes[i][3] * padded_h
        else:
            x1 = (boxes[i][0] - 0.5*boxes[i][2])
            y1 = (boxes[i][1] - 0.5*boxes[i][3])
            rec_w = boxes[i][2]
            rec_h = boxes[i][3]
        rect = plt.Rectangle((x1, y1), rec_w, rec_h, fill=False, edgecolor=color_ls[i%21], linewidth=1)   # the same color as the matched gt
        ax.add_patch(rect)
        if texts:
            subcap.append([i, str(i) + ' : ' + idx_2_cap(texts[i], cap)])   # the same color as the matched gt
            ax.text(x1+2, y1-2, str(i), wrap=True, color=color_ls[i%21],fontsize=10)

    ax = fig.add_subplot(2, 2, 2)
    plt.axis("off")
    if cap:
        for cap in subcap:
            ax.text(0, cap[0]/len(cap), cap[1], wrap=True, color=color_ls[cap[0]%21], fontsize=5)

    if untokenized_cap:
        ax = fig.add_subplot(2, 2, 3)
        ax.text(0, 0, untokenized_cap, wrap=True, color='black',fontsize=8)
        plt.axis("off")
    
    # plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Visualization function for a compound figure (infer and gt)
def visualization(image, original_h, original_w,
                  pred_boxes, pred_texts, gt_boxes, gt_texts, match_matrix, 
                  cap, untokenized_cap, path, f1, mAP):
    """
    Visualization a compound figure inference result

    对于Pred，同F1metric函数一样，需要和GT匹配上，对应的用同一颜色；
    匹配不上的呢？再用另外的颜色并（在Box上）额外标注是匹配不上的，无需再列出文本；

    Args:
        image tensor: (3, h, w)
        original_h/w: scalar
        boxes tensor: (pred_num, 4), (cx, cy, w, h) ratio of the image
        texts list: [pred_num, [subcap_len]], index
        cap list: [cap_len], string tokens
        match_matrix list: [num_gt], the best matched prediction index of each gold subfigure
        untokenized_cap string: untokenized caption
        path string: path to save the figure
        metric dict: {'metric':number ...}
    """
    _, padded_h, padded_w = image.shape
    if original_h > original_w:
        unpadded_h = padded_h
        unpadded_w = (original_w/original_h)*padded_h
    else:
        unpadded_w = padded_w
        unpadded_h = (original_h/original_w)*padded_w
    np_image = image.permute(1, 2, 0).numpy()[:int(unpadded_h), :int(unpadded_w), :]
    # np_image = image.permute(1, 2, 0).numpy() # 不去除pad
    pred_subcap = []
    gt_subcap = []

    fig = plt.figure(dpi=300)#, figsize=(5, 5*unpadded_h/unpadded_w))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(np_image, interpolation=None)
    plt.axis("off")
    for i in range(pred_boxes.shape[0]):
        if i in match_matrix:   
            gt_idx = match_matrix.index(i)
            x1 = (pred_boxes[i][0] - 0.5*pred_boxes[i][2]) * padded_w
            y1 = (pred_boxes[i][1] - 0.5*pred_boxes[i][3]) * padded_h
            rec_w = pred_boxes[i][2] * padded_w
            rec_h = pred_boxes[i][3] * padded_h
            rect = plt.Rectangle((x1, y1), rec_w, rec_h, fill=False, edgecolor=color_ls[gt_idx%21], linewidth=1)   # the same color as the matched gt
            ax.add_patch(rect)
            pred_subcap.append([gt_idx, str(gt_idx) + ' : ' + idx_2_cap(pred_texts[i], cap)])   # the same color as the matched gt
            ax.text(x1+2, y1-2, str(gt_idx), wrap=True, color=color_ls[gt_idx%21],fontsize=10)
        else:
            x1 = (pred_boxes[i][0] - 0.5*pred_boxes[i][2]) * padded_w
            y1 = (pred_boxes[i][1] - 0.5*pred_boxes[i][3]) * padded_h
            rec_w = pred_boxes[i][2] * padded_w
            rec_h = pred_boxes[i][3] * padded_h
            rect = plt.Rectangle((x1, y1), rec_w, rec_h, fill=False, edgecolor='grey', linewidth=1)
            ax.add_patch(rect)
            # pred_subcap.append(str(gt_idx) + ' : ' + idx_2_cap(pred_texts[i], cap))
            ax.text(x1+2, y1-2, 'None', wrap=True, color='grey',fontsize=10)

    ax = fig.add_subplot(3, 2, 2)
    plt.axis("off")
    for matched_subcap in pred_subcap:
        ax.text(0, matched_subcap[0]/len(pred_subcap), matched_subcap[1], wrap=True, color=color_ls[matched_subcap[0]%21], fontsize=5)

    ax = fig.add_subplot(3, 2, 3)
    ax.imshow(np_image, interpolation=None)
    plt.axis("off")
    for i in range(gt_boxes.shape[0]):
        x1 = (gt_boxes[i][0] - 0.5*gt_boxes[i][2]) * padded_w
        y1 = (gt_boxes[i][1] - 0.5*gt_boxes[i][3]) * padded_h
        rec_w = gt_boxes[i][2] * padded_w
        rec_h = gt_boxes[i][3] * padded_h
        rect = plt.Rectangle((x1, y1), rec_w, rec_h, fill=False, edgecolor=color_ls[i%21], linewidth=1)
        ax.add_patch(rect)
        gt_subcap.append(str(i) + ' : ' + idx_2_cap(gt_texts[i], cap))
        ax.text(x1+2, y1-2, str(i), wrap=True, color=color_ls[i%21],fontsize=10)

    ax = fig.add_subplot(3, 2, 4)
    plt.axis("off")
    for i in range(len(gt_subcap)):
        ax.text(0, i/len(gt_subcap), gt_subcap[i], wrap=True, color=color_ls[i%21],fontsize=5)

    ax = fig.add_subplot(3, 2, 5)
    ax.text(0, 0, untokenized_cap, wrap=True, color='black',fontsize=8)
    plt.axis("off")
    
    plt.suptitle('F1:%.2f  mAP:%.2f' % (f1, mAP))
    # plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Visualization function for a compound figure (only detection results)
def visualization_detection(image, original_h, original_w,
                            pred_boxes, gt_boxes, match_matrix, 
                            path, mAP):
    """
    Visualization a compound figure inference result

    对于Pred，同F1metric函数一样，需要和GT匹配上，对应的用同一颜色；
    匹配不上的呢？再用另外的颜色并（在Box上）额外标注是匹配不上的；

    Args:
        image tensor: (3, h, w)
        original_h/w: scalar
        boxes tensor: (pred_num, 4), (cx, cy, w, h) ratio of the image
        match_matrix list: [num_gt], the best matched prediction index of each gold subfigure
        path string: path to save the figure
        metric dict: {'metric':number ...}
    """
    _, padded_h, padded_w = image.shape
    if original_h > original_w:
        unpadded_h = padded_h
        unpadded_w = (original_w/original_h)*padded_h
    else:
        unpadded_w = padded_w
        unpadded_h = (original_h/original_w)*padded_w
    np_image = image.permute(1, 2, 0).numpy()[:int(unpadded_h), :int(unpadded_w), :]

    fig = plt.figure(dpi=300)#, figsize=(5, 5*unpadded_h/unpadded_w))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(np_image, interpolation=None)
    plt.axis("off")
    for i in range(pred_boxes.shape[0]):
        if i in match_matrix:   
            gt_idx = match_matrix.index(i)
            x1 = (pred_boxes[i][0] - 0.5*pred_boxes[i][2]) * padded_w
            y1 = (pred_boxes[i][1] - 0.5*pred_boxes[i][3]) * padded_h
            rec_w = pred_boxes[i][2] * padded_w
            rec_h = pred_boxes[i][3] * padded_h
            rect = plt.Rectangle((x1, y1), rec_w, rec_h, fill=False, edgecolor=color_ls[gt_idx%21], linewidth=1)   # the same color as the matched gt
            ax.add_patch(rect)
            ax.text(x1+2, y1-2, str(gt_idx), wrap=True, color=color_ls[gt_idx%21],fontsize=10)
        else:
            x1 = (pred_boxes[i][0] - 0.5*pred_boxes[i][2]) * padded_w
            y1 = (pred_boxes[i][1] - 0.5*pred_boxes[i][3]) * padded_h
            rec_w = pred_boxes[i][2] * padded_w
            rec_h = pred_boxes[i][3] * padded_h
            rect = plt.Rectangle((x1, y1), rec_w, rec_h, fill=False, edgecolor='grey', linewidth=1)
            ax.add_patch(rect)
            ax.text(x1+2, y1-2, 'None', wrap=True, color='grey',fontsize=10)

    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(np_image, interpolation=None)
    plt.axis("off")
    for i in range(gt_boxes.shape[0]):
        x1 = (gt_boxes[i][0] - 0.5*gt_boxes[i][2]) * padded_w
        y1 = (gt_boxes[i][1] - 0.5*gt_boxes[i][3]) * padded_h
        rec_w = gt_boxes[i][2] * padded_w
        rec_h = gt_boxes[i][3] * padded_h
        rect = plt.Rectangle((x1, y1), rec_w, rec_h, fill=False, edgecolor=color_ls[i%21], linewidth=1)
        ax.add_patch(rect)
        ax.text(x1+2, y1-2, str(i), wrap=True, color=color_ls[i%21],fontsize=10)
    
    plt.suptitle('mAP:%.2f' % (mAP))
    # plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Load visualiztaion from medicat and ours, concat and save for comparison
def compare_visualization():
    """
    load visualiztaion from medicat and ours, concat and save for comparison
    """
    ours_root = 'Log/ContinueAlign/(0.646)newdataset_open_detection/visualization'
    medicat_root = "../medicat/code/subcaption/log/9.27/visualization"
    save_root = 'Log/ContinueAlign/(0.646)newdataset_open_detection/compare_visual'

    # rcd ours prediction
    ours_dict = {}
    ours = os.walk(ours_root)
    for path, dir_list, file_list in ours:
        file_list.sort()
        for file_name in file_list:
            metric, other = file_name.split('(')
            value, image_id = other.split(')')
            if image_id not in ours_dict:
                ours_dict[image_id] = {metric:float(value), "path":'%s/%s'%(ours_root, file_name)}
            else:
                ours_dict[image_id][metric] = float(value)

    medicat = os.walk(medicat_root)
    for path, dir_list, file_list in medicat:
        file_list.sort()
        for file_name in file_list:
            # compare metric
            metric, other = file_name.split('(')
            value, image_id = other.split(')')
            if image_id == '1706.png':
                continue
            ours_metric = ours_dict[image_id][metric]   # float
            gap = ours_metric - float(value)
            save_path = "%s/%s(%.2f)%s" % (save_root, metric, gap, image_id)
            # concat image
            our_img = Image.open(ours_dict[image_id]['path'])
            w, h = our_img.size
            medicat_img = Image.open(os.path.join(medicat_root, file_name))
            result = Image.new(our_img.mode, (w*2, h))
            result.paste(our_img, box=(0, 0))
            result.paste(medicat_img, box=(w, 0))
            result.save(save_path)

# 可视化synthetic图像和bbox
def visual_synthetic_data():
    torch.multiprocessing.set_sharing_strategy('file_system')

    with open('synthetic_parameters/parameters/target_depth_v0.jsonl') as f:
        param = json.load(f)

    with open('augmentation_parameters/flip_color_grey_noise_blur.jsonl') as f:
        aug_param = json.load(f)

    # eval_synthetic_data(aug_param, param, False)

    filepath = '/remote-home/share/medical/public/MedICaT/compound_figures/reid_train.jsonl'
    image_root = '/remote-home/share/medical/public/MedICaT/compound_figures/figures'
    Set = Synthetic_Dataset(filepath, image_root, param, aug_param, eval_mode=True)
    Loader = DataLoader(Set, 64, collate_fn=fig_collate)
    for batch in tqdm(Loader):
        # forward pass
        image = batch['image'] # (bs, 3, max_w, max_h)
        caption = batch['caption'] # (bs, max_cap_l)
        subfigures = batch['subfigs'] # [... [(subfig_num, 4), (subfig_num, max_cap_l)] ...]
        unpadded_hws = batch['unpadded_hws'] # [bs, [2]]
        image_ids = batch['image_id'] # [bs]
        exit()

# 可视化synthetic(基于真实的Layout替换图像)图像和bbox
def visual_reallayout_synthetic_data():
    torch.multiprocessing.set_sharing_strategy('file_system')

    with open('synthetic_parameters/parameters/sim_cfs.jsonl') as f:
        param = json.load(f)
    # eval_synthetic_data(param, eval_mode=False)

    with open('augmentation_parameters/flip_color_grey_noise_blur.jsonl') as f:
        aug_param = json.load(f)

    filepath = '/remote-home/share/medical/public/MedICaT/compound_figures/reid_train.jsonl'
    image_root = '/remote-home/share/medical/public/MedICaT/compound_figures/figures'
    Set = Real_Layout_Synthetic_Dataset(filepath, image_root, param, aug_param, eval_mode=True)
    Loader = DataLoader(Set, 64, collate_fn=fig_collate)
    for batch in tqdm(Loader):
        # forward pass
        image = batch['image'] # (bs, 3, max_w, max_h)
        caption = batch['caption'] # (bs, max_cap_l)
        subfigures = batch['subfigs'] # [... [(subfig_num, 4), (subfig_num, max_cap_l)] ...]
        unpadded_hws = batch['unpadded_hws'] # [bs, [2]]
        image_ids = batch['image_id'] # [bs]
        exit()

# 可视化simcfs synthetic图像和bbox
def visual_simcfs_data():
    torch.multiprocessing.set_sharing_strategy('file_system')

    with open('synthetic_parameters/parameters/real_layout.jsonl') as f:
        param = json.load(f)
    # eval_synthetic_data(param, eval_mode=False)

    with open('augmentation_parameters/flip_color_grey_noise_blur.jsonl') as f:
        aug_param = json.load(f)

    Set = SimCFS_Dataset(param, aug_param, epoch_maximum=1500, eval_mode=True)
    Loader = DataLoader(Set, 64, collate_fn=fig_collate)
    for batch in tqdm(Loader):
        # forward pass
        image = batch['image'] # (bs, 3, max_w, max_h)
        caption = batch['caption'] # (bs, max_cap_l)
        subfigures = batch['subfigs'] # [... [(subfig_num, 4), (subfig_num, max_cap_l)] ...]
        unpadded_hws = batch['unpadded_hws'] # [bs, [2]]
        image_ids = batch['image_id'] # [bs]
        exit()