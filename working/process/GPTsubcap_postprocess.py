import json
import os

import jsonpickle
from tqdm import tqdm

# 把GPTsubcaption和OCRfigure放在一个exsclaim.json后，然后用exsclaim进行配对

def pair_GPTsubcap_with_OCRsubfig():
    """
    将chatgpt并行分割的subcap和subfigOCR结果整理成exsclaim.json（并把ROCO过滤掉
    然后用exscliam配对
    """
    with open('/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/detr_subfigure_ocr_align/chatgpt_subcaptions.json', 'r') as f: # chatgpt_subcaptions，已过滤roco
        subcap_dict = json.load(f)  # pmcid: {status:xxxx, caption:xxxx, subcaptions:{A:xxx, ....}}
        
    # OCR的结果
    file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/detr_subfigure_ocr_align/subfigure_ocr_no_subcaps.json'   # exsclaim subfigs ocr results，没过滤roco
    with open(file_path, 'r') as f:
        subfig_dict = json.load(f)

    # 填入chatgpt的结果
    new_subfig_dict = {}
    for fig_id, data in subfig_dict.items():
        fig_id = fig_id[:-4]
        # 如果pmcid对不上（why？可能是roco被过滤掉了，chatgpt版本的subcap是基于huggingface版本的，应该以它为准
        if fig_id not in subcap_dict:
            continue
        # 如果分不开，那么subcap直接置空
        if subcap_dict[fig_id]['status'] != 'Separable':
            data['unassigned']['captions'] = []
        # 可以被分开
        else:
            subcaptions = []
            for label, subcap in subcap_dict[fig_id]['subcaptions'].items():
                subcaptions.append({'label':label, 'description':[subcap], 'keywords':[], 'general':[]})
            data['unassigned']['captions'] = subcaptions
        new_subfig_dict[fig_id] = data

    # 保存
    with open('/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/detr_subfigure_ocr_align/exsclaim.json', 'w') as f:    # 已经过滤了roco
        json.dump(new_subfig_dict, f, indent=3)

# 挑出所有nonROCO的subfig-->subfigure_v1.jsonl，和nonROCO的figure-->woROCO_fig_id_set.json
# OCR配对的结果-->(subfig-subcap-OCR)subfigure_v1.jsonl，顺便把noncompound figure-caption挑出来-->(noncompound)subfigure_v1.jsonl

def update_subfigure_with_paired_GTPsubcap_OCRsubfig():
    """
    配对后的subfig-subcap整理成规范格式；
    更新之前的subfig-subcap数据；
    顺便过滤掉ROCO（chatgpt的subcap是过滤好的；
    把noncompound figure-caption检测出来；
    """
    with open('/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/detr_subfigure_ocr_align/aligned_GPTsubcap_OCRsubfig.json', 'r') as f: # EXSCLAIM处理过后的subcap-subfig pairs（已经过滤掉ROCO
        data_dict = json.load(f)

    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/subfigure_v0.jsonl', 'r') as f:  # v0的subfig（基于EXSCLAIMsubcaps，有ROCO
        subfig_data = [json.loads(line) for line in lines]

    # chatgpt+ocr的结果，先从exsclaim格式转成subfig:subcap的字典，woROCO
    subfig_subcap_dict = {} # PMC8253756_Fig2_0.jpg:xxx
    for _, datum in data_dict.items():
        for subfig in datum['master_images']:
            if len(subfig['caption']) == 0:
                continue
            else:
                subfig_id = subfig['classification']
                subfig_subcap_dict[subfig_id] = subfig['caption'][0]
    print('%d chatgpt+ocr processed subfig-subcap pairs'%len(subfig_subcap_dict))

    # 读取所有不在roco中的fig的id
    with open('/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/detr_subfigure_ocr_align/intermediate_files/chatgpt_subcaptions.json', 'r') as f:  # chatGPT输出的subcap结果，没有ROCO
        none_roco_dict = json.load(f)

    # 查找字典，将subfigure中的subcaption更新一遍（有更新的单独保存
    chatgpt_ocr_subfigure = []    # 所有pair好的subfig-subcap
    none_roco_subfigure = []      # 用pair好的subfig-subcap更新一下subfigure，同时滤掉roco，即包含了没有和subcap配上的subfig
    roco_subfigs = 0
    no_roco_fig_set = set()
    for datum in tqdm(subfig_data):
        fig_id = datum['media_name'][:-4]
        if fig_id not in none_roco_dict:
            roco_subfigs += 1
            continue
        else:
            no_roco_fig_set.add(fig_id)
        if datum['subfig_id'] in subfig_subcap_dict:
            datum['subcaption'] = subfig_subcap_dict[datum['subfig_id']]
            datum['subcap_score'] = 1.0
            chatgpt_ocr_subfigure.append(datum)
        else:
            # 没有match上的直接放弃
            # 如果之前有match上exsclaim-subcap，也直接放弃
            datum['subcaption'] = ''
            datum['subcap_score'] = 0.0
        none_roco_subfigure.append(datum)
    print('%d None ROCO SubFigures'%len(none_roco_subfigure))
    print('%d None ROCO Compound Figures'%len(no_roco_fig_set))
    print('%d subfigs are in ROCO'%roco_subfigs)
    # chatgpt_ocr_subfigure和none_roco_subfigure分开保存
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(subset)subfigure_v1.jsonl', 'w') as f:
        for datum in chatgpt_ocr_subfigure:
            f.write(json.dumps(datum)+'\n')
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/subfigure_v1.jsonl', 'w') as f:
        for datum in none_roco_subfigure:
            f.write(json.dumps(datum)+'\n')
    # no_roco_fig_set也要保存
    json_data = jsonpickle.encode(no_roco_fig_set)
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/woROCO_fig_id_set.json', 'w') as f:
        json.dump(json_data, f, indent=3)

    # 读取fig信息，把noncompound过滤出来
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/com2sub_v0.jsonl', 'r') as f:
        lines = f.readlines()
        fig_data = [json.loads(line) for line in lines]
    noncom_fig_set = set()
    for datum in tqdm(fig_data):
        if len(datum['subfig_ids']) == 1:
            # 这个subfig就是nonfig
            fig_id = datum['fig_id'][:-4]
            noncom_fig_set.add(fig_id)
    print('%d compound figure has only one subfigure (include ROCO and None ROCO)' % len(noncom_fig_set))
    # nonfig_set也要保存
    json_data = jsonpickle.encode(noncom_fig_set)
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/wROCO_nonecom_fig_id_set.json', 'w') as f:
        json.dump(json_data, f, indent=3)

    # 对于每个subfig，如果没有subcap可以对上，但确定所在fig中只有这一个subfig&caption经过chatgpt判定为unseparable，则可判定是noncompound fig，这个subfig-caption质量也很高才对
    none_roco_noncompound_subfigure =[]
    none_roco_noncompound_set = set()
    for datum in tqdm(none_roco_subfigure):
        fig_id = datum['media_name'][:-4]
        if fig_id in noncom_fig_set and none_roco_dict[fig_id]['status']!='Separable':
            # 直接让subcaption等于caption
            datum['subcaption'] = datum['caption']
            datum['subcaption_score'] = 1.0
            none_roco_noncompound_set.add(fig_id)
            none_roco_noncompound_subfigure.append(datum)
    print('%d none compound figures (exclude roco)'%len(none_roco_noncompound_subfigure))
    # noncompoundfigure-fullcaption也要保存，可以视作，和subfig-subcap同样低噪声的一部分数据
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(non-compound)subfigure_v1.jsonl', 'w') as f:
        for datum in none_roco_noncompound_subfigure:
            f.write(json.dumps(datum)+'\n')
    # none_roco_noncompound_set也要保存
    json_data = jsonpickle.encode(none_roco_noncompound_set)
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/woROCO_nonecom_fig_id_set.json', 'w') as f:
        json.dump(json_data, f, indent=3)
 
# 找到所有不可分的caption对应的subfigure，如果不是compoundfigure，那么把所有的subfigure都指向full caption
    
def find_subfigure_with_unseparable_caption():
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/subfigure_v1.jsonl') as f:    # all subfigures，woROCO
        lines = f.readlines()
        all_nonroco_subfig = [json.loads(line) for line in lines]
        
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/chatgpt_subcaptions.json', 'r') as f: # chatgpt_subcaptions，woROCO
        subcap_dict = json.load(f)  # pmcid: {status:xxxx, caption:xxxx, subcaptions:{A:xxx, ....}}
        
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/woROCO_nonecom_fig_id_set.json') as f:   # noncompound figures，woROCO
        noncompound_dict = jsonpickle.decode(f.read())
        
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/woROCO_fig_id_set.json') as f:   # nonROCO
        nonROCO_dict = jsonpickle.decode(f.read())
        
    subfigure2fullcaption = []  # caption不可分且不是noncompound-figure的subfigures
    for subfig in tqdm(all_nonroco_subfig):
        fig_id = subfig['media_name'][:-4]
        if fig_id not in subcap_dict:
            print(fig_id)
            print('is ROCO: ', fig_id not in nonROCO_dict)
            continue
        if subcap_dict[fig_id]['status'] !='Separable':   # 找到所有，Unseparable
            if fig_id not in noncompound_dict:    # 而且不是noncompound figure
                subfig['subcaption'] = subfig['caption']    # full caption描述对应了所有的subfigure
                subfig['subcap_score'] = 1.0
                subfigure2fullcaption.append(subfig)
    
    print(len(subfigure2fullcaption))
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(subfig-fullcap)subfigure_v1.jsonl', 'w') as f:
        for datum in subfigure2fullcaption:
            f.write(json.dumps(datum)+'\n')
            
# 找到所有caption可分，但是又没能用OCR对上的subfigure，整理一个subfigure--subfigure_info--all_subcap_in_this_figure的list出来
    
def find_subfig_subcap_cannotOCR():
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/subfigure_v1.jsonl') as f:    # all subfigures，woROCO
        lines = f.readlines()
        all_nonroco_subfig = [json.loads(line) for line in lines]
        
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(subfig-subcap-ocr)subfigure_v1.jsonl') as f:    # part3
        lines = f.readlines()
        OCR_subfig = [json.loads(line)['subfig_id'] for line in lines]
        OCR_subfig_set = set(OCR_subfig)
        
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(subfig-fullcap)subfigure_v1.jsonl') as f:    # part2
        lines = f.readlines()
        cap2all_subfig = [json.loads(line)['subfig_id'] for line in lines]
        cap2all_subfig_set = set(cap2all_subfig)
        
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(noncompound)subfigure_v1.jsonl') as f:    # part1
        lines = f.readlines()
        noncompound_subfig = [json.loads(line)['subfig_id'] for line in lines]
        noncompound_subfig_set = set(noncompound_subfig)
        
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/chatgpt_subcaptions.json', 'r') as f: # chatgpt_subcaptions，woROCO
        subcap_dict = json.load(f)  # pmcid: {status:xxxx, caption:xxxx, subcaptions:{A:xxx, ....}}
    
    CLIP_Align_subfig = []
    for datum in tqdm(all_nonroco_subfig):
        subfig_id = datum['subfig_id']
        if subfig_id not in OCR_subfig_set and subfig_id not in cap2all_subfig_set and subfig_id not in noncompound_subfig_set:
            fig_id = datum['media_name'][:-4]
            datum['subcaptions'] = subcap_dict[fig_id]
            if datum['subcaption'] != '':
                print(datum['subcaption'])
            del datum['subcaption']
            del datum['subcap_score']
            CLIP_Align_subfig.append(datum)
     
    print(len(CLIP_Align_subfig))
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(subfig-subcap-CLIP)subfigure_v1.jsonl', 'w') as f:
        for datum in CLIP_Align_subfig:
            f.write(json.dumps(datum)+'\n')
    
# V1版本对jsonl的字段重新规范化，参考README中的格式

def reformat_all_jsonl():
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(subfig-subcap-clip)subfigure_v1.jsonl', 'r') as f:
        lines = f.readlines()
        clip_align = [json.loads(line) for line in lines]
        new_clip_align = []
        for datum in tqdm(clip_align):
            new_datum = {'subfigure':datum['subfig_id'], 'subfig_loc':datum['subfig_loc'], 'subfig_score':datum['subfig_score'], 'subcaption':datum['subcaption'], 'subcap_score':datum['subcap_score'], 'alignment_type':'CLIP', 'origin_figure':datum['media_name'], 'full_caption':datum['caption']}
            new_clip_align.append(new_datum)
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(subfig-subcap-CLIP)subfigure_v1.jsonl', 'a') as f:
        for line in new_clip_align:
            f.write(json.dumps(line)+'\n')
    
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(subfig-subcap-ocr)subfigure_v1.jsonl', 'r') as f:
        lines = f.readlines()
        ocr_align = [json.loads(line) for line in lines]
        new_ocr_align = []
        for datum in tqdm(ocr_align):
            new_datum = {'subfigure':datum['subfig_id'], 'subfig_loc':datum['subfig_loc'], 'subfig_score':datum['subfig_score'], 'subcaption':datum['subcaption'], 'subcap_score':datum['subcap_score'], 'alignment_type':'OCR', 'origin_figure':datum['media_name'], 'full_caption':datum['caption']}
            new_ocr_align.append(new_datum)
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(subfig-subcap-OCR)subfigure_v1.jsonl', 'a') as f:
        for line in new_ocr_align:
            f.write(json.dumps(line)+'\n')
        
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(noncompound)subfigure_v1.jsonl', 'r') as f:
        lines = f.readlines()
        noncompound = [json.loads(line) for line in lines]
        new_noncompound = []
        for datum in tqdm(noncompound):
            new_datum = {'subfigure':datum['subfig_id'], 'subfig_loc':datum['subfig_loc'], 'subfig_score':datum['subfig_score'], 'subcaption':datum['subcaption'], 'subcap_score':datum['subcap_score'], 'alignment_type':'nonCompound-figure', 'origin_figure':datum['media_name'], 'full_caption':datum['caption']}
            new_noncompound.append(new_datum)
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(nonCompound-figure)subfigure_v1.jsonl', 'a') as f:
        for line in new_noncompound:
            f.write(json.dumps(line)+'\n')
        
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(subfig-fullcap)subfigure_v1.jsonl', 'r') as f:
        lines = f.readlines()
        noncompound = [json.loads(line) for line in lines]
        new_noncompound = []
        for datum in tqdm(noncompound):
            new_datum = {'subfigure':datum['subfig_id'], 'subfig_loc':datum['subfig_loc'], 'subfig_score':datum['subfig_score'], 'subcaption':datum['subcaption'], 'subcap_score':datum['subcap_score'], 'alignment_type':'nonCompound-caption', 'origin_figure':datum['media_name'], 'full_caption':datum['caption']}
            new_noncompound.append(new_datum)
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/(nonCompound-caption)subfigure_v1.jsonl', 'a') as f:
        for line in new_noncompound:
            f.write(json.dumps(line)+'\n')

def concat_all_jsonl():
    data = []
    for jsonl in [
        '(subfig-subcap-CLIP)subfigure_v1.jsonl',
        '(subfig-subcap-OCR)subfigure_v1.jsonl',
        '(nonCompound-caption)subfigure_v1.jsonl',
        '(nonCompound-figure)subfigure_v1.jsonl'
    ]:
        with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/'+jsonl, 'r') as f:
            lines = f.readlines()
            print(lines[0])
            data += [json.loads(line) for line in lines]
                
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/subfigure_v2.jsonl', 'a') as f:
        for datum in data:
            f.write(json.dumps(datum)+'\n')
    
if __name__ == "__main__":
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/subfigure_v2.jsonl', 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        
    new_data = []
    for datum in data:
        new_datum = {'image':datum['subfigure'], 'caption':datum['subcaption'], 'alignment_type':datum['alignment_type'], 'alignment_score':datum['subcap_score']}
        new_data.append(new_datum)
             
    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/DETR_sep/pmc_oa.jsonl', 'w') as f:
        for datum in new_data:
            f.write(json.dumps(datum)+'\n')