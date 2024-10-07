""" 
Pipeline to OCR the label of each subfig and align them with subcaptions
"""

import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from skimage import io
from subfigure_ocr.models.network import *
from subfigure_ocr.models.yolov3 import *
from subfigure_ocr.separator import process
from torch.autograd import Variable
from tqdm import tqdm


class classifier():
    def __init__(self):
        model_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/exsclaim/figures/'
        configuration_file = model_path + "config/yolov3_default_subfig.cfg"
        with open(configuration_file, 'r') as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)

        self.image_size            = configuration['TEST']['IMGSIZE']
        self.nms_threshold         = configuration['TEST']['NMSTHRE']
        self.confidence_threshold  = 0.0001
        self.dtype = torch.cuda.FloatTensor
        self.device = torch.device('cuda')

        object_detection_model = YOLOv3(configuration['MODEL'])
        self.object_detection_model = self.load_model_from_checkpoint(object_detection_model, "object_detection_model.pt")
        ## Load text recognition model
        text_recognition_model = resnet152()
        self.text_recognition_model = self.load_model_from_checkpoint(text_recognition_model, 'text_recognition_model.pt')

        self.object_detection_model.eval()
        self.text_recognition_model.eval()

    def load_model_from_checkpoint(self, model, model_name):
        """ load checkpoint weights into model """
        checkpoints_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/exsclaim/figures/checkpoints/'
        checkpoint = checkpoints_path + model_name
        model.load_state_dict(torch.load(checkpoint))
        # model = nn.DataParallel(model)
        model.to(self.device)
        return model
    
    def detect_subfigure_boundaries(self, figure_path):
        """ Detects the bounding boxes of subfigures in figure_path

        Args:
            figure_path: A string, path to an image of a figure
                from a scientific journal
        Returns:
            subfigure_info (list of lists): Each inner list is
                x1, y1, x2, y2, confidence 
        """

        ## Preprocess the figure for the models
        img = io.imread(figure_path)
        if len(np.shape(img)) == 2:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)

        img, info_img = process.preprocess(img, self.image_size, jitter=0)
        img = np.transpose(img / 255., (2, 0, 1))
        img = np.copy(img)
        img = torch.from_numpy(img).float().unsqueeze(0)
        img = Variable(img.type(self.dtype))

        img_raw = Image.open(figure_path).convert("RGB")
        width, height = img_raw.size

        ## Run model on figure
        with torch.no_grad():
            outputs = self.object_detection_model(img.to(self.device))
            outputs = process.postprocess(outputs, dtype=self.dtype, 
                        conf_thre=self.confidence_threshold, nms_thre=self.nms_threshold)

        ## Reformat model outputs to display bounding boxes in our desired format
        ## List of lists where each inner list is [x1, y1, x2, y2, confidence]
        subfigure_info = list()

        if outputs[0] is None:
            return subfigure_info

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
            box = process.yolobox2label([y1.data.cpu().numpy(), x1.data.cpu().numpy(), y2.data.cpu().numpy(), x2.data.cpu().numpy()], info_img)
            box[0] = int(min(max(box[0],0),width-1))
            box[1] = int(min(max(box[1],0),height-1))
            box[2] = int(min(max(box[2],0),width))
            box[3] = int(min(max(box[3],0),height))
            # ensures no extremely small (likely incorrect) boxes are counted
            small_box_threshold = 5
            if (box[2]-box[0] > small_box_threshold and 
                box[3]-box[1] > small_box_threshold):
                box.append("%.3f"%(cls_conf.item()))
                subfigure_info.append(box)
        return subfigure_info

    def detect_subfigure_labels(self, figure_path, subfigure_info):
        """ Uses text recognition to read subfigure labels from figure_path
        
        Note: 
            To get sensible results, should be run only after
            detect_subfigure_boundaries has been run
        Args:
            figure_path (str): A path to the image (.png, .jpg, or .gif)
                file containing the article figure
            subfigure_info (list of lists): Details about bounding boxes
                of each subfigure from detect_subfigure_boundaries(). Each
                inner list has format [x1, y1, x2, y2, confidence] where
                x1, y1 are upper left bounding box coordinates as ints, 
                x2, y2, are lower right, and confidence the models confidence
        Returns:
            subfigure_info (list of tuples): Details about bounding boxes and 
                labels of each subfigure in figure. Tuples for each subfigure are
                (x1, y1, x2, y2, label) where x1, y1 are upper left x and y coord
                divided by image width/height and label is the an integer n 
                meaning the label is the nth letter
            concate_img (np.ndarray): A numpy array representing the figure.
                Used in classify_subfigures. Ideally this will be removed to 
                increase modularity. 
        """
        img_raw = Image.open(figure_path).convert("RGB")
        img_raw = img_raw.copy()
        width, height = img_raw.size
        binary_img = np.zeros((height,width,1))

        detected_label_and_bbox = None
        max_confidence = 0.0
        for subfigure in subfigure_info:
            ## Preprocess the image for the model
            bbox = tuple(subfigure[:4])
            img_patch = img_raw.crop(bbox)
            img_patch = np.array(img_patch)[:,:,::-1]
            img_patch, _ = process.preprocess(img_patch, 28, jitter=0)
            img_patch = np.transpose(img_patch / 255., (2, 0, 1))
            img_patch = torch.from_numpy(img_patch).type(self.dtype).unsqueeze(0)

            ## Run model on figure
            label_prediction = self.text_recognition_model(img_patch.to(self.device))
            label_confidence = np.amax(F.softmax(label_prediction, dim=1).data.cpu().numpy())
            x1,y1,x2,y2, box_confidence = subfigure
            total_confidence = float(box_confidence)*label_confidence
            if total_confidence < max_confidence:
                continue
            label_value = chr(label_prediction.argmax(dim=1).data.cpu().numpy()[0]+ord("a"))
            if label_value == "z":
                continue
            if (x2-x1) < 64 and (y2-y1)< 64:
                detected_label_and_bbox = [label_value, x1,y1,x2,y2]
        
        return detected_label_and_bbox
    
    def run(self, figure_path):
        subfigure_info = self.detect_subfigure_boundaries(figure_path)
        subfigure_info = self.detect_subfigure_labels(figure_path, subfigure_info)      

        return subfigure_info


def subfigure_ocr():
    """
    提取subfigure的label信息
    """
    # 所有可以分出subcap的comfig
    old_dict = {}
    for i in range(10):
        file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_part_%d/exsclaim.json'%i
        with open(file_path, 'r') as f:
            old_dict.update(json.load(f))

    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4.jsonl')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]  
    comfig_dict = {}
    for datum in tqdm(data):
        comfig_id = datum['id']
        comfig_dict[comfig_id] = {'h':datum['height'], 'w':datum['width']}

    model = classifier()
    # root path to all subfigs
    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_subfigures/'
    # all comfigs
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2sub(new).jsonl')
    lines = f.readlines()
    comfig = [json.loads(line) for line in lines]  
    new_dict = {}
    # for datum in tqdm(comfig[:100000]):
    for datum in tqdm(comfig):
        comfig_id = datum['comfig_id']
        if not (comfig_id in old_dict): # 只ocr那些caption可以被打开的结果
            continue
        comfig_h = comfig_dict[comfig_id]['h']
        comfig_w = comfig_dict[comfig_id]['w']
        # 记录每个comfig的subfig信息（包含OCR 
        master_images=[]
        for subfig, locs, scores in zip(datum['subfig_ids'], datum['subfig_locs'], datum['subfig_scores']):
            label_info = model.run(path+subfig)
            if label_info:
                x1, y1, x2, y2 = locs
                x1 = round(x1 * comfig_w)
                x2 = round(x2 * comfig_w)
                y1 = round(y1 * comfig_h)
                y2 = round(y2 * comfig_h)
                w = x2 - x1
                h = y2 - y1
                geometry = [{'x':x1, 'y':y1}, {'x':x1, 'y':y2}, {'x':x2, 'y':y1}, {'x':x2, 'y':y2}]
                label, label_x1, label_y1, label_x2, label_y2 = label_info
                label_geometry = [{'x':label_x1+x1, 'y':label_y1+y1}, {'x':label_x1+x1, 'y':label_y2+y1}, {'x':label_x2+x1, 'y':label_y1+y1}, {'x':label_x2+x1, 'y':label_y2+y1}]
                subfig_label = {"text":label, "geometry":label_geometry}
                master_images.append({
                    'classification':subfig, 
                    'confidence':scores, 
                    'height':h, 'width':w, 
                    'geometry':geometry, 
                    "subfigure_label":subfig_label, 
                    "scale_bars":[], 
                    "caption": [],
                    "keywords": [],
                    "general": []})
        new_dict[comfig_id] = old_dict[comfig_id]
        new_dict[comfig_id]['master_images'] = master_images
                
    with open('/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_ocr_subfig.json', 'w', encoding="utf-8") as f:
        json.dump(new_dict, f, indent=3)

# change this file path for different subcaption separation method
def subfigure_ocr_link_exsclaim_subcap():
    """
    ocr之后的subfigure和subcaption合并到同一个exsclaim中
    """
    def are_same_text(t1, t2, tokenizer):
        if t1 == t2:
            return True
        if t1[:-1] == t2:
            return True
        if t1 == t2[:-1]:
            return True
        if tokenizer.tokenize(t1) == tokenizer.tokenize(t2):
            return True
        else:
            return False
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

    file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_sep_cap_v0.json'    
    with open(file_path, 'r') as f:
        cap_dict = json.load(f)

    file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/detr_subfigure_ocr_align/exsclaim_no_subcaps.json'
    with open(file_path, 'r') as f:
        data_dict = json.load(f)

    """
    check = {}
    for pmc_id, datum in tqdm(data_dict.items()):
        check[pmc_id] = datum
        if len(check) > 2:
            break
    with open('/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/detr_subfigure_ocr_align/check_before_link_subcaps.json', 'w', encoding="utf-8") as f:
        json.dump(check, f, indent=3)
    """
        
    filtered_subcap = 0
    unfiltered_subcap = 0
    for pmc_id, datum in tqdm(data_dict.items()):
        subcaps = []
        for subcap in cap_dict[pmc_id]['unassigned']['captions']:
            if len(subcap["description"]) == 0:
                # 没有
                continue
            filtered_description = []
            for text in subcap["description"]:
                # subcap中每一段不等于整个caption
                if not are_same_text(text, datum['full_caption'], tokenizer):
                    filtered_description.append(text)
            if len(filtered_description) == 0:
                # 都过滤掉了
                continue
            joint_subcap = " ".join(filtered_description)
            if not are_same_text(joint_subcap, datum['full_caption'], tokenizer):
                # subcap连在一起不等于整个caption
                subcap["description"] = filtered_description
                subcaps.append(subcap)
                unfiltered_subcap += 1
            else:
                filtered_subcap += 1
        data_dict[pmc_id]['unassigned']['captions'] = subcaps

    """
    check = {}
    for pmc_id, datum in tqdm(data_dict.items()):
        check[pmc_id] = datum
        if len(check) > 2:
            break
    with open('/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/detr_subfigure_ocr_align/check_after_link_subcaps.json', 'w', encoding="utf-8") as f:
        json.dump(check, f, indent=3)
    """

    with open('/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/detr_subfigure_ocr_align/exsclaim.json', 'w', encoding="utf-8") as f:
        json.dump(data_dict, f, indent=3)

# 将/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/detr_subfigure_ocr_align/exsclaim.json送入exsclaim的pipeline将subfig和subcap对齐

def ocr_replace_clip_alignment():
    """
    将exsclaim输出的subfigure和subcaption对齐结果，替换掉CLIP的对齐结果
    """
    file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/detr_subfigure_ocr_align/exsclaim.json'
    with open(file_path, 'r') as f:
        ocr_dict = json.load(f)

    file_path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2sub(new).jsonl'
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    new_data = []
    missing_comfig = abandon_1 = abandon_2 = replace = 0
    for datum in tqdm(data):
        comfig_id = datum['comfig_id']
        old_subfig_ids = datum['subfig_ids']
        old_subcaptions = datum['subcaptions']
        old_subcap_indexes = datum['subcap_indexes']
        old_subcap_scores = datum['subcap_scores']
        new_subfig_ids = []
        new_subcap_indexes = []
        new_subcap_scores = []
        if comfig_id not in ocr_dict:
            missing_comfig += 1
            continue
        ocr_results = ocr_dict[comfig_id]
        for subfig in ocr_results['master_images']:
            if len(subfig['caption']) > 0:
                if subfig['classification'] not in old_subfig_ids:
                    print('Subfig ID not found Error : ', subfig['classification'])
                    abandon_1 += 1
                    continue
                subcaption = " ".join(subfig['caption'])
                if subcaption not in old_subcaptions:
                    print('Caption not found Error : ', subfig['classification'])
                    print('Subcaption : %s'% subcaption)
                    for i, tmp in enumerate(old_subcaptions):
                        print('Subcaption Option %d : %s \n', (i,tmp))
                    abandon_2 += 1
                    continue
                # 有ocr匹配的subcaption结果, 而且可以替换进来
                new_subfig_ids.append(subfig['classification'])
                new_subcap_indexes.append(old_subcaptions.index(subcaption))
                new_subcap_scores.append(1.0)
                replace += 1
        for idx, subfig in enumerate(old_subfig_ids):
            if subfig in new_subfig_ids:
                # 已经用ocr的匹配结果替换掉了
                continue
            else:
                new_subfig_ids.append(subfig)
                new_subcap_indexes.append(old_subcap_indexes[idx])
                new_subcap_scores.append(old_subcap_scores[idx])
        datum['subfig_ids'] = new_subfig_ids
        datum['subcap_indexes'] = new_subcap_indexes
        datum['subcap_scores'] = new_subcap_scores
        new_data.append(datum)

    print('Missing comfig in exsclaim:', missing_comfig)
    print('Missing subfigure id:', abandon_1)
    print('Missing subcaption:', abandon_2)
    print('Successfully replace:', replace)

    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2sub(plus_ocr).jsonl', 'w')
    for datum in tqdm(new_data):
        f.write(json.dumps(datum)+'\n')
    f.close()
                

if __name__ == "__main__":    
    file_path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2sub(new).jsonl'
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    for datum in data:
        if datum['comfig_id'] == "PMC4608103_Fig2.jpg":
            for subcap in datum['subcaptions']:
                print(subcap, '\n')
            break

    file_path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_sub2com(new).jsonl'
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    for datum in data:
        if datum['comfig_id'] == "PMC4608103_Fig2.jpg":
            print(datum['subcaption'])

    

    

