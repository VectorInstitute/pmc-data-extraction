import matplotlib.pyplot as plt
import numpy as np
import torch
from align_metric import box_cxcywh_to_xyxy
from einops import repeat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def side_loss(pred_bboxes, gt_bboxes, x1y1x2y2=True):
    # Returns the IoU of pred_bboxes to gt_bboxes. pred_bboxes is (m, 4), gt_bboxes is (n, 4)
    gt_bboxes = gt_bboxes.T
    pred_bboxes = pred_bboxes.T
    gt_num = gt_bboxes.shape[1]
    pred_bboxes = repeat(pred_bboxes, 'x m -> x m n', n=gt_num) # (4, m, n)

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        p_x1, p_y1, p_x2, p_y2 = pred_bboxes[0], pred_bboxes[1], pred_bboxes[2], pred_bboxes[3] # mxn
        g_x1, g_y1, g_x2, g_y2 = gt_bboxes[0], gt_bboxes[1], gt_bboxes[2], gt_bboxes[3] # n
    else:  # transform from xywh to xyxy
        p_x1, p_x2 = pred_bboxes[0] - pred_bboxes[2] / 2, pred_bboxes[0] + pred_bboxes[2] / 2
        p_y1, p_y2 = pred_bboxes[1] - pred_bboxes[3] / 2, pred_bboxes[1] + pred_bboxes[3] / 2
        g_x1, g_x2 = gt_bboxes[0] - gt_bboxes[2] / 2, gt_bboxes[0] + gt_bboxes[2] / 2
        g_y1, g_y2 = gt_bboxes[1] - gt_bboxes[3] / 2, gt_bboxes[1] + gt_bboxes[3] / 2

    x1_loss = torch.max(torch.zeros((p_x1-g_x1).shape).to(g_x1.device), g_x1-p_x1)    # mxn
    y1_loss = torch.max(torch.zeros((p_y1-g_y1).shape).to(g_x1.device), g_y1-p_y1)   
    x2_loss = torch.max(torch.zeros((g_x2-p_x2).shape).to(g_x1.device), p_x2-g_x2)
    y2_loss = torch.max(torch.zeros((g_y2-p_y2).shape).to(g_x1.device), p_y2-g_y2)

    side_loss = x1_loss+x2_loss+y1_loss+y2_loss # mxn
    return side_loss

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4) -- (x1, y1, x2, y2)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # (n1, 1, 2)  (1, n2, 2) --> (n1, n2, 2)

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    # print('lt')
    # print(lower_bounds)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    # print('rb')
    # print(upper_bounds)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    # print(intersection_dims)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    if set_1.dim() == 1 and set_1.shape[0] == 4:
        set_1 = set_1.unsqueeze(0)
    if set_2.dim() == 1 and set_2.shape[0] == 4:
        set_2 = set_2.unsqueeze(0)

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # print('intersection')
    # print(intersection)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    # print('union')
    # print(union)

    return intersection / union  # (n1, n2)

def calculate_mAP_voc07(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, iou_threshold):
    """
    Calculate the Mean Average Precision (mAP) of detected objects. 

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    det_boxes: list of tensors, [bs * (num_pred, 4)], one tensor for each image containing detected objects' bounding boxes
    det_labels: list of tensors, [bs * (num_pred)], one tensor for each image containing detected objects' labels
    det_scores: list of tensors, [bs * (num_pred)], one tensor for each image containing detected objects' labels' scores
    true_boxes: list of tensors, [bs * (num_gt, 4)], one tensor for each image containing actual objects' bounding boxes
    true_labels: list of tensors, [bs * (num_gt)], one tensor for each image containing actual objects' labels
    true_difficulties: list of tensors, [bs * (num_gt)], one tensor for each image containing actual objects' difficulty (0 or 1)
    
    return: list of average precisions for all classes, mean average precision (mAP)
    """

    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels) == len(true_difficulties) # these are all lists of tensors of the same length, i.e. number of images
    n_classes = 2 # len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].shape[0]) # [n_objects = bs * num_gt]
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4), true_box[i] indicate the image index of i-th object
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # print('=='*10)
    # print('true_images')
    # print(true_images)
    # print('true_boxes')
    # print(true_boxes)
    # print('true_labels')
    # print(true_labels)
    # print('true_difficulties')
    # print(true_difficulties)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].shape[0]) # [n_detections = bs * num_pred]
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    # print('=='*10)
    # print('det_images')
    # print(det_images)
    # print('det_boxes')
    # print(det_boxes)
    # print('det_labels')
    # print(det_labels)
    # print('det_scores')
    # print(det_scores)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects ？ difficulty of this class

        # Keep track of which true objects with this class have already been 'detected' -- 重复匹配同一个true box的后续detect box会被算作False Positive
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            cumul_recall = cumul_precision = torch.tensor([0.0])
            continue

        # Sort detections in decreasing order of confidence/scores -- 重复匹配同一个true box的后续detect box会被算作False Positive
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections) -- 每个detect box要么true_positive位置上是1，要么false_positive位置上是1
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img, 4)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue
            
            # print('=='*10)
            # print('this_detection_box')
            # print(this_detection_box)
            # print('this_image')
            # print(this_image)
            # print('object_boxes')
            # print(object_boxes)
            # print('object_difficulties')
            # print(object_difficulties)

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(box_cxcywh_to_xyxy(this_detection_box), box_cxcywh_to_xyxy(object_boxes))  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # print('overlaps')
            # print(overlaps)
            # print('max_overlap & ind')
            # print(max_overlap, ind)

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]  # 与detected box overlap最大的true box在所有true box中的index
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # print(original_ind)

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > iou_threshold:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

            # print('true_positives')
            # print(true_positives)
            # print('false_positives')
            # print(false_positives)

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections）
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections) -- 更多的detect box一定会提升recall，但precision可能会下降或上升
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t # 需要多少个detect box来达到recall的阈值
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()  # 在count这么多detect box时，最高的precision
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()
    recall = cumul_recall[-1].item()
    precision = cumul_precision[-1].item()
    
    return mean_average_precision, recall, precision

def calculate_mAP_voc12(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, iou_threshold=0.75, score_threshold=0.5, ap_curve_path=None):
    """
    Calculate the Mean Average Precision (mAP) of detected objects. 
    列出所有的detection bbox，按照得分从高到低排序，然后每个与gt bbox去匹配，匹配上算作true pos，否则是false pos（每个gt bbox不可被重复匹配）；
    然后对true pos累积，可以得到recall增长的曲线；同时对false pos累计，可以得到prec变化的曲线；
    取每个true positive(recall值增长的点)对应的最大prec，计算AP折线图像下的面积；

    See https://zhuanlan.zhihu.com/p/70667071 for an explanation

    det_boxes: list of tensors, [bs * (num_pred, 4)], one tensor for each image containing detected objects' bounding boxes
    det_labels: list of tensors, [bs * (num_pred)], one tensor for each image containing detected objects' labels
    det_scores: list of tensors, [bs * (num_pred)], one tensor for each image containing detected objects' labels' scores
    true_boxes: list of tensors, [bs * (num_gt, 4)], one tensor for each image containing actual objects' bounding boxes
    true_labels: list of tensors, [bs * (num_gt)], one tensor for each image containing actual objects' labels
    true_difficulties: list of tensors, [bs * (num_gt)], one tensor for each image containing actual objects' difficulty (0 or 1)
    
    return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels) == len(true_difficulties) # these are all lists of tensors of the same length, i.e. number of images
    n_classes = 2 # len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].shape[0]) # [n_objects = bs * num_gt]
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4), true_box[i] indicate the image index of i-th object
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].shape[0]) # [n_detections = bs * num_pred]
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class  （对于所有subfig默认都算作一类
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects ？ difficulty of this class

        # Keep track of which true objects with this class have already been 'detected' -- 重复匹配同一个true box的后续detect box会被算作False Positive
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(device)  # (n_class_objects)
        match_matrice = torch.ones(true_class_difficulties.size(0)) * -1    # WARNING 仅针对只有一类检测目标, 且输入bs为1的时候

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            cumul_recall = cumul_precision = torch.tensor([0.0])
            continue

        # Sort detections in decreasing order of confidence/scores -- 重复匹配同一个true box的后续detect box会被算作False Positive
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = np.zeros(n_class_detections)  # (n_class_detections) -- 每个detect box要么true_positive位置上是1，要么false_positive位置上是1
        false_positives = np.zeros(n_class_detections)  # (n_class_detections)
        threshold_index = -1
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # 标记score threshold所在的位置，用于找到这个点的recall和precision
            if threshold_index == -1 and det_class_scores[d] < score_threshold:
                threshold_index = d-1

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img, 4)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(box_cxcywh_to_xyxy(this_detection_box), box_cxcywh_to_xyxy(object_boxes))  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]  # 与detected box overlap最大的true box在所有true box中的index
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > iou_threshold:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                        match_matrice[original_ind] = d
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = np.concatenate(([0.], np.cumsum(true_positives, axis=0)))  # (n_class_detections）
        cumul_false_positives = np.concatenate(([0.], np.cumsum(false_positives, axis=0)))  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections) -- 更多的detect box一定会提升recall，但precision可能会下降或上升
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # checkpoint recall and precision
        checkpoint_recall = cumul_recall[threshold_index].item()
        checkpoint_precision = cumul_precision[threshold_index].item()
        print(threshold_index, checkpoint_recall, checkpoint_precision)

        # save fig
        if ap_curve_path:
            plt.scatter(cumul_recall, cumul_precision, s=5)
            plt.savefig(ap_curve_path)

        for i in range(cumul_precision.shape[0]-1, 0, -1):  # 每个recall值，都取右边最大的prec
            cumul_precision[i-1] = max(cumul_precision[i-1], cumul_precision[i])

        change_points = np.where(cumul_recall[1:]!=cumul_recall[:-1])[0]

        average_precisions[c - 1] = np.sum((cumul_recall[change_points+1]-cumul_recall[change_points])*cumul_precision[change_points+1])    

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()
    
    return mean_average_precision, checkpoint_recall, checkpoint_precision, match_matrice.tolist()

if __name__=="__main__":
    from dataset import FigCap_Dataset, my_collate
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    val_file = '/remote-home/share/medical/public/MedICaT/subfig_subcap_val.jsonl' 
    img_root = '/remote-home/share/medical/public/MedICaT/release/figures'
    vocab_file = '/remote-home/zihengzhao/CompoundFigure/medicat/code/pretrained_model/scibert/vocab.txt'
    ValSet = FigCap_Dataset(val_file, img_root, vocab_file)
    ValLoader = DataLoader(ValSet, batch_size=1, shuffle=False, num_workers=4, collate_fn=my_collate)

    true_boxes = []
    true_labels = []
    true_difficulties = []
    for batch in tqdm(ValLoader):
        subfigures = batch['subfigs']
        for i in range(len(subfigures)):
            true_boxes.append(subfigures[i][1]) # [bs * (subfig_num, 4)]
            true_labels.append(torch.ones(true_boxes[-1].shape[0]))  # [bs * (subfig_num)] all 1 
            true_difficulties.append(torch.zeros_like(true_labels[-1]))  # [bs * (subfig_num)] all zeros

        # print(true_boxes)
        # print(true_labels)
        # print(true_difficulties)
    mAP, recall, precision = calculate_mAP(true_boxes, true_labels, true_labels, true_boxes, true_labels, true_difficulties, 0.7)
    print(mAP, recall, precision)
