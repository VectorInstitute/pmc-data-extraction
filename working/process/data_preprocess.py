import json
import os
import random
import shutil
from builtins import print
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def reid_and_extract_subfigs_fromDETR():
    rename_dict = {}    # {"xxxxxxx.png" : "index.png"}

    # For Train Set

    new_comfigs = []    # {除去pdf_hash和fig_uri}
    new_subfigs = []    # {"id":"index-1.png", "h":x, "w":x, "subcaption_tokens":[{"text": "Figure", "start": 0, "end": 6, "id": 0}, ...], 'text':'xxxxxxxxxxx'}

    f = open('/remote-home/share/medical/public/MedICaT/compound_figures/old/subfig_subcap_train.jsonl')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    root_path = '/remote-home/share/medical/public/MedICaT/compound_figures/old/figures/'
    target_path = '/remote-home/share/medical/public/MedICaT/compound_figures/figures/'
    subfig_path = '/remote-home/share/medical/public/MedICaT/compound_figures/subfigures/'
    for i, datum in enumerate(tqdm(data)):
        # convert to new comfig id
        old_id = datum["pdf_hash"]+"_"+datum["fig_uri"]
        if old_id in rename_dict:
            print('Repeated ID %s', old_id)
            exit()

        # Read Image File
        com_image = np.array(Image.open(root_path+old_id).convert('RGB'))

        # change comfig id in comfig_data.json
        rename_dict[old_id] = str(len(rename_dict)) + '.png'
        new_datum = {'id': str(len(rename_dict)) + '.png'}
        new_datum.update(datum)
        del new_datum['pdf_hash']
        del new_datum['fig_uri']
        del new_datum['fig_key']
        # change subfig id in comfig_data.json
        id = 0
        tmp = []
        for subfigure in new_datum['subfigures']:
            subfigure['id'] = str(len(rename_dict)) + '_' + str(id) + '.png'
            tmp.append(subfigure)

            x = [point[0] for point in subfigure["points"]]
            y = [point[1] for point in subfigure["points"]]
            x1 = round(min(x))
            x2 = round(max(x))
            y1 = round(min(y))
            y2 = round(max(y))
            # 规范bbox
            if y1 < 0:
                print(y1)
                y1 = 0
            if x1 < 0:
                print(x1)
                x1 = 0
            if x2 > new_datum['width']-1:
                print(x2, '>', new_datum['width']-1)
                x2 = new_datum['width']-1
            if y2 > new_datum['height']-1:
                print(y2, '>', new_datum['height']-1)
                y2 = new_datum['height']-1
            if x2 == x1 or y2 == y1:
                continue

            # rcd the subfig basic info in subfig_data.json
            new_subfig = {}
            new_subfig['id'] = str(len(rename_dict)) + '_' + str(id) + '.png'
            new_subfig['h'] = y2 - y1
            new_subfig['w'] = x2 - x1
            if new_datum['subcaptions']!= None and subfigure['label'] in new_datum['subcaptions']:
                subcaption = new_datum['subcaptions'][subfigure['label']] # [3, 4, 5, 6 ......]
            else:
                subcaption = []
            new_subfig['subcaption_tokens'] = [ token for token in new_datum['tokens'] if token['id'] in subcaption ]
            new_subfig['caption'] = new_datum['text']
            new_subfigs.append(new_subfig)
            # extract the subfig and save it
            # print(x1, x2, y1, y2)
            sub_image = com_image[y1:y2, x1:x2]
            # print(com_image.shape, sub_image.shape)
            try:
                sub_image = Image.fromarray(sub_image)
            except:
                print(x1, x2, y1, y2)
                print(com_image.shape, sub_image.shape)
                exit()
            sub_image.save(subfig_path+new_subfig['id'])

            id += 1

        new_datum['subfigures'] = tmp
        new_comfigs.append(new_datum)
            
        # rename/move the comfig
        # shutil.copyfile(root_path+old_id, target_path+str(len(rename_dict)) + '.png')

    f = open('/remote-home/share/medical/public/MedICaT/compound_figures/reid_train.jsonl', "w")
    for line in new_comfigs:
        f.write(json.dumps(line)+'\n')
    f.close()
    f = open('/remote-home/share/medical/public/MedICaT/compound_figures/reid_train_subfigs.jsonl', "w")
    for line in new_subfigs:
        f.write(json.dumps(line)+'\n')
    f.close()

    # Repeat for Val Set

    new_comfigs = []    # {除去pdf_hash和fig_uri}
    new_subfigs = []    # {"id":"index-1.png", "h":x, "w":x, "subcaption_tokens":[{"text": "Figure", "start": 0, "end": 6, "id": 0}, ...], 'text':'xxxxxxxxxxx'}

    f = open('/remote-home/share/medical/public/MedICaT/compound_figures/old/subfig_subcap_val.jsonl')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    root_path = '/remote-home/share/medical/public/MedICaT/compound_figures/old/figures/'
    target_path = '/remote-home/share/medical/public/MedICaT/compound_figures/figures/'
    subfig_path = '/remote-home/share/medical/public/MedICaT/compound_figures/subfigures/'
    for i, datum in enumerate(data):
        # convert to new comfig id
        old_id = datum["pdf_hash"]+"_"+datum["fig_uri"]
        if old_id in rename_dict:
            print('Repeated ID %s', old_id)
            exit()

        # Read Image File
        com_image = np.array(Image.open(root_path+old_id).convert('RGB'))

        # change comfig id in comfig_data.json
        rename_dict[old_id] = str(len(rename_dict)) + '.png'
        new_datum = {'id': str(len(rename_dict)) + '.png'}
        new_datum.update(datum)
        del new_datum['pdf_hash']
        del new_datum['fig_uri']
        del new_datum['fig_key']
        # change subfig id in comfig_data.json
        id = 0
        tmp = []
        for subfigure in new_datum['subfigures']:
            subfigure['id'] = str(len(rename_dict)) + '_' + str(id) + '.png'
            tmp.append(subfigure)

            x = [point[0] for point in subfigure["points"]]
            y = [point[1] for point in subfigure["points"]]
            x1 = round(min(x))
            x2 = round(max(x))
            y1 = round(min(y))
            y2 = round(max(y))
            # 规范bbox
            if y1 < 0:
                print(y1)
                y1 = 0
            if x1 < 0:
                print(x1)
                x1 = 0
            if x2 > new_datum['width']-1:
                print(x2, '>', new_datum['width']-1)
                x2 = new_datum['width']-1
            if y2 > new_datum['height']-1:
                print(y2, '>', new_datum['height']-1)
                y2 = new_datum['height']-1
            if x2 == x1 or y2 == y1:
                continue

            # rcd the subfig basic info in subfig_data.json
            new_subfig = {}
            new_subfig['id'] = str(len(rename_dict)) + '_' + str(id) + '.png'
            new_subfig['h'] = y2 - y1
            new_subfig['w'] = x2 - x1
            if subfigure['label'] in new_datum['subcaptions']:
                subcaption = new_datum['subcaptions'][subfigure['label']] # [3, 4, 5, 6 ......]
            else:
                subcaption = []
            new_subfig['subcaption_tokens'] = [ token for token in new_datum['tokens'] if token['id'] in subcaption ]
            new_subfig['caption'] = new_datum['text']
            new_subfigs.append(new_subfig)
            # extract the subfig and save it
            # print(x1, x2, y1, y2)
            sub_image = com_image[y1:y2, x1:x2]
            # print(com_image.shape, sub_image.shape)
            try:
                sub_image = Image.fromarray(sub_image)
            except:
                print(x1, x2, y1, y2)
                print(com_image.shape, sub_image.shape)
                exit()
            sub_image.save(subfig_path+new_subfig['id'])

            id += 1

        new_datum['subfigures'] = tmp
        new_comfigs.append(new_datum)
            
        # rename/move the comfig
        # shutil.copyfile(root_path+old_id, target_path+str(len(rename_dict)) + '.png')

    f = open('/remote-home/share/medical/public/MedICaT/compound_figures/reid_test.jsonl', "w")
    for line in new_comfigs:
        f.write(json.dumps(line)+'\n')
    f.close()
    f = open('/remote-home/share/medical/public/MedICaT/compound_figures/reid_test_subfigs.jsonl', "w")
    for line in new_subfigs:
        f.write(json.dumps(line)+'\n')
    f.close()
    f = open('/remote-home/share/medical/public/MedICaT/compound_figures/convert_id_tab.jsonl', 'w')
    f.write(json.dumps(rename_dict)+'\n')


def imageclef_xml_2_jsonl():
    comfig_dict = []
    subfig_dict = []
    id_convert_tab = {} # old_id : new_id
    import xml.dom.minidom
    from xml.dom.minidom import parse
    xml_file = "/remote-home/share/medical/public/ImageCLEF2016/medtask/FigureSeparationTraining2016-GT.xml"
    # xml_file = "/remote-home/share/medical/public/ImageCLEF2016/medtask/FigureSeparationTest2016GT.xml"
    DOMTree = xml.dom.minidom.parse(xml_file)
    doc = DOMTree.documentElement
    annos = doc.getElementsByTagName("annotation")
    comfig_count = 1
    # comfig_count = 6783
    for anno in annos:  # a compound figure
        print(comfig_count)
        f = anno.getElementsByTagName("filename")[0].childNodes[0].data # old id
        # convert to new id
        id_convert_tab['%s.jpg'%f] = '%d.jpg'%comfig_count  
        a_comfig = {'id':'%d.jpg'%comfig_count}
        # read images and record h w
        comfig = np.array(Image.open('/remote-home/share/medical/public/ImageCLEF2016/medtask/FigureSeparationTraining2016/%s.jpg'%f))
        # comfig = np.array(Image.open('/remote-home/share/medical/public/ImageCLEF2016/medtask/FigureSeparationTest2016/%s.jpg'%f))
        try:
            h, w, _ = comfig.shape
        except:
            h, w = comfig.shape
        a_comfig['height'] = h
        a_comfig['width'] = w   
        # copy to new id
        # shutil.copyfile('/remote-home/share/medical/public/ImageCLEF2016/medtask/FigureSeparationTraining2016/%s.jpg'%f, '/remote-home/share/medical/public/ImageCLEF2016/medtask/new_id_comfigs/%d.jpg'%comfig_count)
        # shutil.copyfile('/remote-home/share/medical/public/ImageCLEF2016/medtask/FigureSeparationTest2016/%s.jpg'%f, '/remote-home/share/medical/public/ImageCLEF2016/medtask/new_id_comfigs/%d.jpg'%comfig_count)
        # crop subfigs
        subfig_list = []
        bboxes = anno.getElementsByTagName("object")
        subfig_count = 0
        for bbox in bboxes: # a subfig bbox
            points = bbox.getElementsByTagName("point")
            # id the subfig
            a_subfig = {'id':"%d_%d.png"%(comfig_count, subfig_count)}
            # read bbox
            x_values = []
            y_values = []
            for point in points:    # four points
                x_values.append(int(point.getAttribute("x")))
                y_values.append(int(point.getAttribute("y")))
            x1 = min(x_values) 
            y1 = min(y_values)
            x2 = max(x_values)
            y2 = max(y_values)
            a_subfig['height'] = y2 - y1
            a_subfig['width'] = x2 - x1
            # record the subfig
            subfig_dict.append(a_subfig)
            subfig_list.append({'id':"%d_%d.png"%(comfig_count, subfig_count), "points":[[x1, y1], [x1, y2], [x2, y1], [x2, y2]]})    
            # crop and save 
            sub_image = comfig[y1:y2, x1:x2]
            try:
                sub_image = Image.fromarray(sub_image)
                # sub_image.save('/remote-home/share/medical/public/ImageCLEF2016/medtask/new_id_subfigs/%d_%d.png'%(comfig_count, subfig_count))
            except:
                print ("%d_%d.png"%(comfig_count, subfig_count))
            subfig_count += 1
        a_comfig['subfigures'] = subfig_list
        comfig_count += 1
        comfig_dict.append(a_comfig)
    # Save comfig_dict and subfig_dict and id convert dict
    f = open('/remote-home/share/medical/public/ImageCLEF2016/medtask/train_comfigs.jsonl', "w")
    for line in comfig_dict:
        f.write(json.dumps(line)+'\n')
    f.close()
    f = open('/remote-home/share/medical/public/ImageCLEF2016/medtask/train_subfigs.jsonl', "w")
    for line in subfig_dict:
        f.write(json.dumps(line)+'\n')
    f.close()
    f = open('/remote-home/share/medical/public/ImageCLEF2016/medtask/convert_id_tab.jsonl', 'a')
    f.write(json.dumps(id_convert_tab)+'\n')
    f.close()


def prepare_pubmed_from_folder():
    """
    将non-compound图片转化为medicat的json格式
    """
    data_list = []

    root_path = '/remote-home/share/medical/public/PMC_OA/Evaluate_Compound_Separation/100_samples(NoneCompound)'
    for path, dir_list, file_list in os.walk(root_path):
        for image_file in file_list:
            image = Image.open(os.path.join(root_path, image_file)).convert('RGB')
            image = transforms.ToTensor()(image)
            c, h, w = image.shape
            data_list.append({"id":image_file, "height":h, "width":w, "subfigures":[{"id":'None', "points":[[0, 0], [w, h]]}]})        
        
    f = open('/remote-home/share/medical/public/PMC_OA/100_samples(NoneCompound).jsonl', 'w')
    for line in data_list:
        f.write(json.dumps(line)+'\n')
    f.close()


def prepare_pubmed():
    """
    将wcy和lwx爬取过滤的数据转化为medicat的json格式
    """
    caption_book = json.load(open('/remote-home/share/chaoyiwu/caption_T060_filtered_top4.json','r'))
    total_num = len(caption_book)
    data_list = []
    for index in tqdm(range(total_num)):
        # index = random.randint(0, total_num)
        data = caption_book[index]
        image_path = data['image']
        image_id = image_path.split('/')[-1]
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        # id: str
        # h&w: int
        c, h, w = image.shape
        # {"id": "6783_1.png", "points": [[0, 0], [0, 481], [484, 0], [484, 481]]}
        data_dict = {"id":image_id, "height":h, "width":w, "subfigures":[{"id":'None', "points":[[0, 0], [0, h], [w, 0], [w, h]]}]}
        data_list.append(data_dict)
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4.jsonl', 'w')
    for line in data_list:
        f.write(json.dumps(line)+'\n')
    f.close()


def prepare_labelme():
    """
    labelme标注的数据转化为medicat的json格式
    """
    data_list = []

    root_path = '/remote-home/share/medical/public/PMC_OA/Test_Compound_Separation/(500 samples)caption_T060_filtered_top4_Annotation'
    for path, dir_list, file_list in os.walk(root_path):
        for anno_dict in file_list:
            image_id = anno_dict.split('.json')[0]
            # image_path = '/remote-home/zihengzhao/CompoundFigure/Dataset/(300 samples)caption_T060_filtered_top4/' + image_id + '.jpg'
            caption_book = json.load(open(os.path.join(root_path, anno_dict),'r'))
            h = caption_book['imageHeight']
            w = caption_book['imageWidth']

            subfigure_list = []
            bboxes = caption_book['shapes']
            count = 0
            for bbox in bboxes:
                xy_list = bbox['points']
                x1 = min(xy_list[0][0], xy_list[1][0])
                x2 = max(xy_list[0][0], xy_list[1][0])
                y1 = min(xy_list[0][1], xy_list[1][1])
                y2 = max(xy_list[0][1], xy_list[1][1])
                subfigure_list.append({"id":str(count), "points":[[x1, y1], [x1, y2], [x2, y1], [x2, y2]]})

            data_list.append({"id":image_id+".jpg", "height":h, "width":w, "subfigures":subfigure_list})        
        
    f = open('/remote-home/share/medical/public/PMC_OA/Test_Compound_Separation/(500 samples)caption_T060_filtered_top4.jsonl', 'w')
    for line in data_list:
        f.write(json.dumps(line)+'\n')
    f.close()


def select_separation_samples(sample_num=300):
    """
    从wcy和lwx爬取过滤的数据中随机抽取一些图像
    """
    caption_book = json.load(open('/remote-home/share/chaoyiwu/caption_T060_filtered_top4.json','r'))
    save_path = '/remote-home/share/medical/public/PMC_OA/(300~600 samples)caption_T060_filtered_top4'

    from pathlib import Path
    Path(save_path).mkdir(parents=True, exist_ok=True)

    total_num = len(caption_book)
    selected_indexes = np.random.choice(total_num, sample_num, False)
    for i in range(sample_num):
        index = selected_indexes[i]
        # index = random.randint(0, total_num)
        data = caption_book[index]
        image_path = data['image']
        image_id = image_path.split('/')[-1]
        print('copy %s to %s' % (image_path, os.path.join(save_path, image_id)))
        shutil.copy(image_path, os.path.join(save_path, image_id))


def pair_compoundfigure2caption():
    """
    将compound figure和caption一一对应起来
    """
    filepath = "/remote-home/share/medical/public/PMC_OA/pairs.jsonl"
    f = open(filepath)
    lines = f.readlines()
    pmc_data = [json.loads(line) for line in lines]
    processed_fig = []
    dict_lsit = []
    for datum in tqdm(pmc_data):
        comfig_id = datum["media_name"]     # PMC212319_Fig3
        if comfig_id in processed_fig:
            continue
        else:
            processed_fig.append(comfig_id)
            dict_lsit.append({'comfig_id':comfig_id, 'caption':datum['caption']})
    f = open('/remote-home/share/medical/public/PMC_OA/compoundfigure_pairs.jsonl', 'w')
    for line in dict_lsit:
        f.write(json.dumps(line)+'\n')
    f.close()


def pmc2exsclaim():
    """
    将caption_T060_filtered_top4中的compoundfigure--caption数据转为exsclaim的输入格式, 以便分割caption
    分成10份保存
    """
    filepath = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/'
    f = open(filepath)
    lines = f.readlines()
    pmc_data = [json.loads(line) for line in lines]
    random.shuffle(pmc_data)
    for i in range(10):
        start = i*(len(pmc_data)//10)
        end = min(start+len(pmc_data)//10, len(pmc_data))
        fraction = pmc_data[start:end]
        exsclaim_data = {}
        for datum in tqdm(fraction):
            pmc_id = datum['comfig_id']
            caption = datum['caption']
            tmp = {
                "title": "None",
                "article_url": "None",
                "article_name": "None",
                "full_caption": caption,
                "caption_delimiter": "",
                "figure_name": pmc_id,
                "image_url": pmc_id,
                "license": "None",
                "open": True,
                "figure_path": "/remote-home/share/medical/public/PMC_OA/figures/"+pmc_id,
                "master_images": [],
                "unassigned": {
                    "master_images": [],
                    "dependent_images": [],
                    "inset_images": [],
                    "subfigure_labels": [],
                    "scale_bar_labels": [],
                    "scale_bar_lines": [],
                    "captions": []
                }
            }
            exsclaim_data[pmc_id] = tmp
        
        Path("/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_part_%d"%i).mkdir()
        with open("/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_part_%d/exsclaim.json"%i, 'w', encoding="utf-8") as f:
            json.dump(exsclaim_data, f, indent=3)


def prepare_exsclaim_data():
    """
    将subfig-subcap对齐后的exsclaim.json转化成规定的jsonl格式用于训练clip

    {"id": "1537.png", "text": "Fig...", "width": 850, "height": 644, 
     "subcaptions": {"1537_0.png": ['This...'], ...}, 
     "subfigures": [{"id": "1537_0.png", "label": "1537_0.png", "points":[[x1, y1],[x2, y2]]}, ...]}
    """
    data_list = []

    comfig_root_path = '/remote-home/share/medical/public/PMC_OA/figures/'
    subfig_root_path = '/remote-home/share/medical/public/MedICaT/exsclaim_extracted/subfigures/'
    file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim/exsclaim.json' # result file path
    with open(file_path, 'r') as f:
        results = json.load(f)
    for comfig_id, datum in tqdm(results.items()):
        full_caption = datum["full_caption"]
        subfigure_list = []
        subcap_dict = {}
        count = 0
        com_image = None
        for subfigure in datum['master_images']:
            #  print(subfigure)
            if "caption" in subfigure and len(subfigure['caption']) > 0:
                # print(subfigure['caption'])
                caption = ''.join(subfigure['caption'])
                x1, y1 = subfigure['geometry'][0]['x'], subfigure['geometry'][0]['y']
                x2, y2 = subfigure['geometry'][3]['x'], subfigure['geometry'][3]['y']
                # print(x1, x2, y1, y2)
                # print(w, h)
                subfig_id = comfig_id.split('.jpg')[0] + '_' + str(count) + '.jpg'
                subcap_dict[subfig_id] = caption
                # Read Compound Image File
                if count == 0:
                    com_image = np.array(Image.open(comfig_root_path+comfig_id).convert('RGB'))
                    h, w, _ = com_image.shape
                try:
                    sub_image = com_image[y1:y2, x1:x2]
                    # print(com_image.shape, sub_image.shape)
                    sub_image = Image.fromarray(sub_image)
                    sub_image.save(subfig_root_path+subfig_id)
                    tmp_dict = {'id':subfig_id, 'label':subfig_id, 'points':[[x1, y1],[x2, y2]]}
                    subfigure_list.append(tmp_dict)
                    count += 1
                    # print('yes')
                except:
                    print(x1, x2, y1, y2)
                    print(com_image.shape, sub_image.shape)
                    continue
            else:
                continue

        if count == 0:
            continue
        else:
            data_dict = {"id":comfig_id, "height":h, "width":w, "caption":full_caption, "subfigures":subfigure_list, "subcaptions":subcap_dict}
            data_list.append(data_dict)

    print(data_list)
    f = open('/remote-home/zihengzhao/CompoundFigure/exsclaim-data/data.jsonl', 'w')
    for line in data_list:
        f.write(json.dumps(line)+'\n')
    f.close()


def divide_before_sep():
    """
    将caption_T060_filtered_top4分成50份再分割
    """
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4.jsonl')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

    for i in range(0, len(data), len(data)//50):
        f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/%d.jsonl'%(i//(len(data)//50)), 'w')
        for line in data[i : min(i+len(data)//50, len(data))]:
            f.write(json.dumps(line)+'\n')
        f.close()
        print(i, min(i+len(data)//50, len(data)))


def search_division_by_figure_id(figure_id):
    """
    将caption_T060_filtered_top4分成50份后, 查找某一figure id在哪一份中
    """
    for i in range(51):
        f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/%d.jsonl'%i)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        for datum in data:
            if datum['id'] == figure_id:
                print('in %d.jsonl'%i)
                exit()
            else:
                continue


def filter_unsep_figure():
    """
    筛选caption_T060_filtered_top4被分成50份后, 每一份中尚未被sep的figure
    """
    for i in tqdm(range(51)):
        processed_data = []
        if Path('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_%d.jsonl'%i).exists():
            f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_%d.jsonl'%i)
            lines = f.readlines()
            processed_data += [json.loads(line)['image_id'] for line in lines]
        if Path('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v2_%d.jsonl'%i).exists():
            f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v2_%d.jsonl'%i)
            lines = f.readlines()
            processed_data += [json.loads(line)['image_id'] for line in lines]
        if Path('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v3_%d.jsonl'%i).exists():
            f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v3_%d.jsonl'%i)
            lines = f.readlines()
            processed_data += [json.loads(line)['image_id'] for line in lines]

        unprocessed_data = []
        f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/%d.jsonl'%i)
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        for datum in data:
            if datum['id'] in processed_data:
                continue
            else:
                unprocessed_data.append(datum)

        if len(unprocessed_data) > 0:
            print('unsep %d in %d' % (len(unprocessed_data), i))
            f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/unsep_%d.jsonl'%i, 'w')
            for line in unprocessed_data:
                f.write(json.dumps(line)+'\n')
            f.close()


def aggregate_sep_figure():
    """
    caption_T060_filtered_top4被分成50份分割后, 聚合分割结果
    """
    processed_data = []
    for i in tqdm(range(51)):
        total_num = 7621 if i < 50 else 46
        tmp_data_count = []
        if Path('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_%d.jsonl'%i).exists():
            f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_%d.jsonl'%i)
            lines = f.readlines()
            tmp_data_count += [json.loads(line) for line in lines]
        if Path('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v2_%d.jsonl'%i).exists():
            f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v2_%d.jsonl'%i)
            lines = f.readlines()
            tmp_data_count += [json.loads(line) for line in lines]
        if Path('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v3_%d.jsonl'%i).exists():
            f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v3_%d.jsonl'%i)
            lines = f.readlines()
            tmp_data_count += [json.loads(line) for line in lines]
        if len(tmp_data_count) == 7621 or len(tmp_data_count) == 46:
            processed_data += tmp_data_count
        else:
            print('Not Aligned! Sep %d : Total %d'%(len(tmp_data_count), total_num))
            # 有重复的分割
            tmp_data_count = []
            tmp_comfig_count = []
            if Path('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_%d.jsonl'%i).exists():
                f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_%d.jsonl'%i)
                lines = f.readlines()
                data = [json.loads(line) for line in lines]
                for datum in data:
                    if datum['image_id'] not in tmp_comfig_count:
                        tmp_comfig_count.append(datum['image_id'])
                        tmp_data_count.append(datum)
                    else:
                        continue
            if Path('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v2_%d.jsonl'%i).exists():
                f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v2_%d.jsonl'%i)
                lines = f.readlines()
                data = [json.loads(line) for line in lines]
                for datum in data:
                    if datum['image_id'] not in tmp_comfig_count:
                        tmp_comfig_count.append(datum['image_id'])
                        tmp_data_count.append(datum)
                    else:
                        continue
            if Path('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v3_%d.jsonl'%i).exists():
                f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_sep/sep_results_v3_%d.jsonl'%i)
                lines = f.readlines()
                data = [json.loads(line) for line in lines]
                for datum in data:
                    if datum['image_id'] not in tmp_comfig_count:
                        tmp_comfig_count.append(datum['image_id'])
                        tmp_data_count.append(datum)
                    else:
                        continue
            assert len(tmp_data_count) == 7621 or len(tmp_data_count) == 46, print(len(tmp_data_count))
            processed_data += tmp_data_count

    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_result_v1.jsonl', 'w')
    for line in processed_data:
        f.write(json.dumps(line)+'\n')
    f.close()    


def aggregate_sep_caption():
    """
    caption_T060_filtered_top4被分成10份断句后, 聚合结果
    """
    aggregate_dict = {}
    for i in tqdm(range(10)):
        file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_part_%d/exsclaim.json'%i
        with open(file_path, 'r') as f:
            results = json.load(f)
        for comfig_id, datum in tqdm(results.items()):
            aggregate_dict[comfig_id] = datum

    f = open('/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_sep_cap_v1.json', 'w')
    f.write(json.dumps(aggregate_dict))
    f.close()    


def reformat_sep_results_v0():
    """
    将caption_T060_filtered_top4的分割结果(detr, v0)reformat成comfig--subfig_list的形式,
    并和exsclaim的分割结果拼起来
    """
    # 先统计每个comfig分出来的caption
    comfig2subcaps = {}
    file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim/exsclaim.json' # result file path
    with open(file_path, 'r') as f:
        results = json.load(f)
    for comfig_id, datum in tqdm(results.items()):
        cap_ls = []
        for cap in datum['unassigned']['captions']:
            if len(cap['description']) > 0:
                cap_ls += cap['description']
        for subfigure in datum['master_images']:
            if "caption" in subfigure and len(subfigure['caption']) > 0:
                cap_ls += subfigure['caption']
        if len(cap_ls) > 0:
            comfig2subcaps[comfig_id] = cap_ls

    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_result.jsonl')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    reformat_results = []   # {comfig_id, subfig_ids:[...], subfig_locs:[...]}
    cur_comfig_id = data[0]['source_fig_id']+'.jpg'
    cur_comfig_dict = {'comfig_id':cur_comfig_id, 'subfig_ids':[], 'subfig_locs':[], 'subfig_scores':[]}
    for datum in tqdm(data):
        source_fig_id = datum['source_fig_id']+'.jpg'
        if source_fig_id not in comfig2subcaps:
            continue
        if source_fig_id != cur_comfig_id:
            if len(cur_comfig_dict['subfig_ids'])>0:
                reformat_results.append(cur_comfig_dict)
            cur_comfig_id = source_fig_id
            cur_comfig_dict = {'comfig_id':cur_comfig_id, 'subfig_ids':[], 'subfig_locs':[], 'subfig_scores':[], 'subcaps':comfig2subcaps[cur_comfig_id]}
        cur_comfig_dict['subfig_ids'].append(datum['id'])
        cur_comfig_dict['subfig_locs'].append(datum['position'])
        cur_comfig_dict['subfig_scores'].append(datum['score'])

    if len(cur_comfig_dict['subfig_ids'])>0:
        reformat_results.append(cur_comfig_dict)

    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_results_v0_com2sub.jsonl', 'w')
    for line in reformat_results:
        f.write(json.dumps(line)+'\n')
    f.close()


def reformat_sep_results_v1():
    """
    将caption_T060_filtered_top4的分割结果(fasterRCNN, v1)reformat成comfig--subfig_list的形式, 以及subfig--comfig--score--caption的形式
    并和exsclaim的分割结果拼起来
    """
    # 先统计每个comfig分出来的caption
    subcap_num = 0
    separable_comfig_num = 0
    comfig2subcaps = {}
    file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_sep_cap_v0.json' # exsclaim分割的cap(没有分割subfig也没有对齐，所以都在unassign里)
    with open(file_path, 'r') as f:
        results = json.load(f)
    for comfig_id, datum in tqdm(results.items()):
        cap_ls = [datum['full_caption']]    # [caption, subcap1, ......]
        for cap in datum['unassigned']['captions']:
            if len(cap['description']) > 0:
                cap_ls += cap['description']
        for subfigure in datum['master_images']:
            if "caption" in subfigure and len(subfigure['caption']) > 0:
                cap_ls += subfigure['caption']
        comfig2subcaps[comfig_id] = cap_ls
        if len(cap_ls) > 1:
            separable_comfig_num += 1
            subcap_num += (len(cap_ls) - 1)

    print('%d Separable Out of %d, Avg %.2f Subcaptions'%(separable_comfig_num, len(comfig2subcaps), subcap_num/separable_comfig_num))

    exception = 0
    subfig_id = 0
    comfig_path = '/remote-home/share/medical/public/PMC_OA/figures'
    subfig_path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_result_v1'
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v1.jsonl')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    reformat_comfig2subfig_results = []   # [ ... {comfig_id, subfig_ids:[...], subfig_locs:[...], subfig_scores:[...], subcaps:[...], caption} ... ]
    reformat_subfig2comfig_results = []   # [ ... {subfig_id, comfig_id, subfig_loc:[...], subfig_score, caption} ... ]
    for datum in tqdm(data):
        comfig_id = datum['image_id']
        com_image = np.array(Image.open('%s/%s'%(comfig_path, comfig_id)).convert('RGB'))
        h, w, _ = com_image.shape
        comfig_dict = {'comfig_id':comfig_id, 'subfig_ids':[], 'subfig_locs':[], 'subfig_scores':[], 'subcaps':comfig2subcaps[comfig_id][1:], 'caption':comfig2subcaps[comfig_id][0]}   
        for loc, score in zip(datum['predictions']['boxes'], datum['predictions']['scores']):   # [x1 y1 x2 y2], 0.9
            # print(com_image.shape, sub_image.shape)
            try:
                x1, y1, x2, y2 = [int(pos) for pos in loc]
                sub_image = com_image[y1:y2, x1:x2]
                sub_image = Image.fromarray(sub_image)
                sub_image.save('%s/%d.jpg'%(subfig_path, subfig_id))
                reformat_subfig2comfig_results.append({'subfig_id':'%d.jpg'%subfig_id, 'comfig_id':comfig_id, 'subfig_loc':[x1/w, y1/h, x2/w, y2/h], 'subfig_score':score, 'caption':comfig2subcaps[comfig_id][0]})
                comfig_dict['subfig_ids'].append('%d.jpg'%subfig_id)
                comfig_dict['subfig_locs'].append([x1/w, y1/h, x2/w, y2/h])
                comfig_dict['subfig_scores'].append(score)
                subfig_id += 1
            except:
                print(x1, x2, y1, y2)
                print(com_image.shape)
                exception += 1
            
        reformat_comfig2subfig_results.append(comfig_dict)

    print('Avg %d Subfigures, Abandon %d Subfigures'%(len(reformat_subfig2comfig_results)/len(reformat_comfig2subfig_results)), exception)

    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v1_com2sub.jsonl', 'w')
    for line in reformat_comfig2subfig_results:
        f.write(json.dumps(line)+'\n')
    f.close()

    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v1_sub2com.jsonl', 'w')
    for line in reformat_subfig2comfig_results:
        f.write(json.dumps(line)+'\n')
    f.close()


def divide_before_extract_and_filter():
    """
    将整合去重后的fasterRCNN分割结果分成10份以便并行提取subfig
    """
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v1.jsonl')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    for i in range(0, len(data), len(data)//10):
        f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_filter/%d.jsonl'%(i//(len(data)//10)), 'w')
        for line in data[i : min(i+len(data)//10, len(data))]:
            f.write(json.dumps(line)+'\n')
        f.close()
        print(i, min(i+len(data)//10, len(data)))


def extract_reid_subfigs_fromFasterRCNN(part_id):
    """
    根据fasterRCNN的分割结果, 把subfig提取出来(并行10份)
    """
    comfig_path = '/remote-home/share/medical/public/PMC_OA/figures'
    subfig_path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_result_v1'
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_v1_divide_before_extract_filter/%d.jsonl'%part_id)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    exception = 0
    subfig_id = 0
    reformat_subfig2comfig_results = []   # [ ... {subfig_id, comfig_id, subfig_loc:[...], subfig_score, caption} ... ]
    for datum in tqdm(data):
        comfig_id = datum['image_id']
        com_image = np.array(Image.open('%s/%s'%(comfig_path, comfig_id)).convert('RGB'))
        h, w, _ = com_image.shape
        for loc, score in zip(datum['predictions']['boxes'], datum['predictions']['scores']):   # [x1 y1 x2 y2], 0.9
            # print(com_image.shape, sub_image.shape)
            x1, y1, x2, y2 = [int(pos) for pos in loc]
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x1 == x2 or y1 == y2:
                continue
            if Path('%s/part_%d_%d.jpg'%(subfig_path, part_id, subfig_id)).exists():
                # 已经save过subfig了
                print('part_%d_%d.jpg'%(part_id, subfig_id))
            else:
                sub_image = com_image[y1:y2, x1:x2]
                sub_image = Image.fromarray(sub_image)
                sub_image.save('%s/part_%d_%d.jpg'%(subfig_path, part_id, subfig_id))
            reformat_subfig2comfig_results.append({'subfig_id':'part_%d_%d.jpg'%(part_id, subfig_id), 'comfig_id':comfig_id, 'subfig_loc':[x1/w, y1/h, x2/w, y2/h], 'subfig_score':score})
            subfig_id += 1
    print('%d Compoundfigures Has %d Subfigures, Avg %.1f Subfigures, Abandon %d Subfigures'%(len(data), len(reformat_subfig2comfig_results), len(reformat_subfig2comfig_results)/len(data), exception))

    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v1_sub2com_unfiltered_part%d.jsonl'%part_id, 'w')
    for line in reformat_subfig2comfig_results:
        f.write(json.dumps(line)+'\n')
    f.close()


def pair_subfig_and_loc_score():
    """
    将v0的subfig&caption和subfig_loc, subfig_score配对起来, 得到每个（经过resnext过滤的）subfig的完整信息
    """
    comfig_hw_dict = {}
    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4.jsonl'
    f = open(path)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    for datum in data:
        comfig_hw_dict[datum['id']] = [datum['height'], datum['width']]

    subfig_loc_score_dict = {}
    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_sub2com(score).jsonl'
    f = open(path)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    for datum in tqdm(data):
        comfig_id = datum['source_fig_id'] + '.jpg'
        subfig_id = datum['source_fig_id'] + '_' + datum['id']
        locs = datum['position']
        x1, y1 = locs[0]
        x2, y2 = locs[1]
        h, w = comfig_hw_dict[comfig_id]
        normed_locs_and_score = [x1/w, y1/h, x2/w, y2/h, datum['score']]
        subfig_loc_score_dict[subfig_id] = normed_locs_and_score

    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_sub2com(caption).jsonl'
    f = open(path)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    new_data = []
    for datum in tqdm(data):
        subfig_id = datum['image'].split('/')[-1]
        normed_locs_and_score = subfig_loc_score_dict[subfig_id]
        datum['subfig_loc'] = normed_locs_and_score[:4]
        datum['subfig_score'] = normed_locs_and_score[-1]
        new_data.append(datum)

    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_sub2com.jsonl'
    f = open(path, 'w')
    for datum in tqdm(new_data):
        f.write(json.dumps(datum)+'\n')


def pair_comfig_and_subfig_subcap():
    """
    聚合subfig的完整信息，得到comfig的完整信息：subfig的完整信息 + subcaptions
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
    # 加载所有的subcap，和comfig对上
    comfig2subcaps = {}
    subcap_num = 0
    separable_comfig_num = 0
    file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_sep_cap_v0.json' # exsclaim分割的cap(没有分割subfig也没有对齐，所以都在unassign里)
    with open(file_path, 'r') as f:
        results = json.load(f)
    for comfig_id, datum in tqdm(results.items()):
        cap_ls = [datum['full_caption']]    # [caption, subcap1, ......]
        for cap in datum['unassigned']['captions']:
            if len(cap['description']) > 0:
                for text in cap['description']:
                    if not are_same_text(text, datum['full_caption'], tokenizer):
                        cap_ls.append(text)
        for subfigure in datum['master_images']:
            if "caption" in subfigure and len(subfigure['caption']) > 0:
                for text in subfigure['caption']:
                    if not are_same_text(text, datum['full_caption'], tokenizer):
                        cap_ls.append(text)
        comfig2subcaps[comfig_id] = cap_ls
        if len(cap_ls) > 1:
            separable_comfig_num += 1
            subcap_num += (len(cap_ls) - 1)
    print('%d Separable Out of %d, Avg %.2f Subcaptions'%(separable_comfig_num, len(comfig2subcaps), subcap_num/separable_comfig_num))

    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2subcap.jsonl', 'w') as f:
        for comfig, subcap_ls in comfig2subcaps.items():
            item = {'comfig_id':comfig, 'caption':subcap_ls[0], 'subcaptions':subcap_ls[1:]}
            f.write(json.dumps(item) + '\n')

    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_sub2com.jsonl'
    f = open(path)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    new_data = []   # {comfig_id, subfig_ids:[...], subfig_locs:[...], subfig_scores:[...], caption, subcaptions:[...]}
    cur_comfig_id = data[0]['media_name']
    subcaps = comfig2subcaps[cur_comfig_id][1:] if cur_comfig_id in comfig2subcaps else []
    cur_comfig_dict = {'comfig_id':cur_comfig_id, 'subfig_ids':[], 'subfig_locs':[], 'subfig_scores':[], 'caption':data[0]['caption'], 'subcaptions':subcaps}
    for datum in tqdm(data):
        comfig_id = datum['media_name']
        if comfig_id != cur_comfig_id:
            new_data.append(cur_comfig_dict)
            cur_comfig_id = comfig_id
            subcaps = comfig2subcaps[cur_comfig_id][1:] if cur_comfig_id in comfig2subcaps else []
            cur_comfig_dict = {'comfig_id':cur_comfig_id, 'subfig_ids':[], 'subfig_locs':[], 'subfig_scores':[], 'caption':datum['caption'], 'subcaptions':subcaps}
        cur_comfig_dict['subfig_ids'].append(datum['image'].split('/')[-1])
        cur_comfig_dict['subfig_locs'].append(datum['subfig_loc'])
        cur_comfig_dict['subfig_scores'].append(datum['subfig_score'])

    with open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2sub.jsonl', 'w') as f:
        for line in new_data:
            f.write(json.dumps(line) + '\n')


def pair_subcap_and_subfig():
    """
    将exsclaim的subcap和detr的分割subfig配对起来
    """
    #{
    # 'comfig_id':cur_comfig, 
    # 'subfig_ids':[], 
    # 'subfig_locs':[], 
    # 'subfig_scores':[], 
    # 'caption':'caption', 
    # 'subcaptions':['subcaps' ...]
    # }

    # 加载所有的subcap，和comfig对上
    comfig2subcaps = {}
    subcap_num = 0
    separable_comfig_num = 0
    file_path = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_sep_cap_v0.json' # exsclaim分割的cap(没有分割subfig也没有对齐，所以都在unassign里)
    with open(file_path, 'r') as f:
        results = json.load(f)
    for comfig_id, datum in tqdm(results.items()):
        cap_ls = [datum['full_caption']]    # [caption, subcap1, ......]
        for cap in datum['unassigned']['captions']:
            if len(cap['description']) > 0:
                cap_ls += cap['description']
        for subfigure in datum['master_images']:
            if "caption" in subfigure and len(subfigure['caption']) > 0:
                cap_ls += subfigure['caption']
        comfig2subcaps[comfig_id] = cap_ls
        if len(cap_ls) > 1:
            separable_comfig_num += 1
            subcap_num += (len(cap_ls) - 1)
    print('%d Separable Out of %d, Avg %.2f Subcaptions'%(separable_comfig_num, len(comfig2subcaps), subcap_num/separable_comfig_num))

    # 加载所有的comfig其他信息
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2sub.jsonl')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    new_data = []
    for datum in data:
        comfig_id = datum['comfig_id']
        subfig_ids_new = ['%s_%s'%(comfig_id[:-4], subfig) for subfig in datum['subfig_ids']]
        datum['subfig_ids'] = subfig_ids_new
        datum['full_caption'] = comfig2subcaps[comfig_id][0]
        datum['sub_captions'] = comfig2subcaps[comfig_id][1:]
        new_data.append(datum)

    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2sub(caption).jsonl')
    for line in new_data:
        f.write(json.dumps(line))
    f.close()


def stat_subcap_subfig_pair():
    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2sub(new).jsonl'
    f = open(path)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    comfig_w_subfig = 0
    comfig_w_subcap = 0
    comfig_w_both = 0
    for datum in data:
        if len(datum['subfig_ids']) > 0:
            comfig_w_subfig += 1
        if len(datum['subcaptions']) > 0:
            comfig_w_subcap += 1
        if len(datum['subfig_ids']) > 0 and len(datum['subcaptions']) > 0:
            comfig_w_both += 1

    print('%d Comfig has Subfigs, %d has Subcaptions, %d has Both'%(comfig_w_subfig, comfig_w_subcap, comfig_w_both))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_sub2com(new).jsonl'
    f = open(path)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    threshold = [0 for i in range(10)]
    total_token_len = 0
    total_subcap_num = 0
    for datum in data:
        if datum['subcaption'] != '':
            total_token_len += len(tokenizer.tokenize(datum['subcaption']))
            total_subcap_num += 1
        for i in range(9, -1, -1):
            if datum['subcap_score'] > i/10:
                for j in range(i+1):
                    threshold[j] += 1
                break
    for i in range(10):
        print('threshold %f has %d'%((i)/10, threshold[i]))       

    print('Avg %.2f Tokens'%(total_token_len/total_subcap_num))         


def aggregate_subcap_info():
    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/others/caption_T060_filtered_top4_sep_v0_sub2com(没有subcap).jsonl'
    f = open(path)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    sub2com = {}
    for datum in tqdm(data):
        subfig_id = datum['image'].split('/')[-1]
        datum['subfig_id'] = subfig_id
        datum['subcaption'] = ''
        datum['subcap_score'] = 0.0
        sub2com[subfig_id] = datum

    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/others/caption_T060_filtered_top4_sep_v0_subfig2subcap.jsonl'
    f = open(path)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    for datum in tqdm(data):
        subfig_id = datum['subfig_id']
        sub2com[subfig_id]['subcaption'] = datum['subcaption']
        sub2com[subfig_id]['subcap_score'] = datum['confidence_score']

    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_sub2com(new).jsonl', 'w')
    for _, datum in tqdm(sub2com.items()):
        f.write(json.dumps(datum)+'\n')
    f.close()

    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/others/caption_T060_filtered_top4_sep_v0_com2sub(没有对齐subfig和subcap).jsonl'
    f = open(path)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    new_data = []
    count_assigned_subcap = 0
    for datum in tqdm(data):
        subcaps = datum['subcaptions']
        subcap_scores = []
        subcap_indexes = []
        for subfig_id in datum['subfig_ids']:
            subcap = sub2com[subfig_id]['subcaption']
            subcap_scores.append(sub2com[subfig_id]['subcap_score'])
            if subcap == "":
                subcap_indexes.append(-1)
            else:
                subcap_indexes.append(subcaps.index(subcap))
            count_assigned_subcap += 1
        datum['subcap_indexes'] = subcap_indexes
        datum['subcap_scores'] = subcap_scores 
        new_data.append(datum)

    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2sub(new).jsonl', 'w')
    for datum in tqdm(new_data):
        f.write(json.dumps(datum)+'\n')
    f.close()

    print('%d Assigned'%(count_assigned_subcap))


def visualization_subfig_subcap_align(np_image, normed_subfig_locs, subfig_scores, subcaps, subcap_scores, subcap_indexes, full_caption, path):
    """
    Visualization a compound figure inference result 

    Args:
        image tensor: (3, h, w)
        original_h/w: scalar
        boxes tensor or list of tensors: (pred_num, 4) / [pred_num, (4)], (x1, y1, x2, y2) ratio of the image

        path string: path to save the figure
    """
    color_ls = ['red', 'darkred', 'lightcoral', 'orangered', 'sienna', 'sandybrown',
                'gold', 'olive', 'olivedrab', 'yellowgreen', 'darkseagreen', 'green',
                'lightseagreen', 'skyblue', 'steelblue', 'deepskyblue', 'dodgerblue',
                'blue', 'darkblue', 'blueviolet', 'violet']
    random.shuffle(color_ls)

    padded_h, padded_w, _ = np_image.shape

    fig = plt.figure(dpi=300)#, figsize=(5, 5*unpadded_h/unpadded_w))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(np_image, interpolation=None)
    plt.axis("off")

    subcaps_to_plot = []
    for i in range(len(normed_subfig_locs)): 
        x1 = (normed_subfig_locs[i][0]) * padded_w
        y1 = (normed_subfig_locs[i][1]) * padded_h
        rec_w = (normed_subfig_locs[i][2]-normed_subfig_locs[i][0]) * padded_w
        rec_h = (normed_subfig_locs[i][3]-normed_subfig_locs[i][1]) * padded_h
        if subcap_indexes[i]!=-1:
            rect = plt.Rectangle((x1, y1), rec_w, rec_h, fill=False, edgecolor=color_ls[i%21], linewidth=1)   # the same color as the matched gt
            ax.add_patch(rect)
            subcaps_to_plot.append('%d(%.2f):%s'%(i, subcap_scores[i], subcaps[subcap_indexes[i]]))   # the same color as the matched gt
            ax.text(x1+2, y1-2, '%.2f'%subfig_scores[i], wrap=True, color=color_ls[i%21],fontsize=10)
        else:
            rect = plt.Rectangle((x1, y1), rec_w, rec_h, fill=False, edgecolor='grey', linewidth=1)   # the same color as the matched gt
            ax.add_patch(rect)

    ax = fig.add_subplot(3, 2, 2)
    plt.axis("off")
    for i, cap in enumerate(subcaps_to_plot):
        ax.text(0, i/len(subcaps_to_plot), cap, wrap=True, color=color_ls[i%21], fontsize=5)

    subcaps_to_plot = []
    for i in range(len(subcaps)):
        if i not in subcap_indexes:
            subcaps_to_plot.append(subcaps[i])

    ax = fig.add_subplot(3, 2, 3)
    plt.axis("off")
    for i, cap in enumerate(subcaps_to_plot):
        ax.text(0, i/len(subcaps_to_plot), cap, wrap=True, color='grey', fontsize=5)

    ax = fig.add_subplot(3, 2, 5)
    ax.text(0, 0, full_caption, wrap=True, color='black',fontsize=8)
    plt.axis("off")
    
    # plt.tight_layout()
    plt.savefig(path)
    plt.close()


def cherry_pick():
    """
    找一些subcap-subfig对齐的sample
    """
    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2sub(new).jsonl'
    f = open(path)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    for datum in tqdm(data[0:1000]):
        if "PMC8253756_Fig2.jpg" == datum['comfig_id']:
            for sub_cap in datum['subcaptions']:
                print(sub_cap)
            exit()
        else:
            continue
            img = Image.open('/remote-home/share/medical/public/PMC_OA/figures/'+datum['comfig_id']).convert('RGB')
            np_image = np.array(img)
            normed_subfig_locs = datum['subfig_locs']
            subfig_scores = datum['subfig_scores']
            subcaps = datum['subcaptions']
            subcap_scores = datum['subcap_scores']
            subcap_indexes = datum['subcap_indexes']
            full_caption = datum['caption']
            path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/挑图/%s'%datum['comfig_id']
            visualization_subfig_subcap_align(np_image, normed_subfig_locs, subfig_scores, subcaps, subcap_scores, subcap_indexes, full_caption, path)


def cherry_pick_forT060():
    """
    找一些被T060过滤掉的nonmedical的sample
    """
    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_com2sub(new).jsonl'
    f = open(path)
    lines = f.readlines()
    filtered = [json.loads(line)['comfig_id'] for line in lines]

    path = '/remote-home/share/medical/public/PMC_OA/0.jsonl'
    f = open(path)
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    for datum in data:
        if datum['media_name'] not in filtered:
            print('filtered:', datum['media_name'])
            shutil.copy('/remote-home/share/medical/public/PMC_OA/figures/'+datum['media_name'], '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/挑图(T060)/'+datum['media_name'])
        else:
            print(datum['media_name'])


def cherry_pick_forEXSCALIM():
    """
    找一些subcap-subfig对齐的sample
    """
    with open('/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_sep_cap_v0.json', 'r') as f:
        exsclaim_data = json.load(f)

    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/挑图(subfig&subcap)'
    picked_samples = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for comfig in files:
            picked_samples[comfig] = exsclaim_data[comfig]

    with open('/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_1000_samples/exsclaim.json', 'w') as f:
        json.dump(picked_samples, f, indent=3)


def compare_EXSCLAIM_MedCLIP():
    """
    在可视化我们的方法和EXSCLAIM之后，两张可视化拼在一起，
    对比我们的方法和EXSCLAIM的差距
    """
    path = '/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/挑图(subfig&subcap)/'
    exsclaim_visual_root = '/remote-home/zihengzhao/CompoundFigure/exsclaim/extracted/pmc2exsclaim_1000_samples/visualization/'
    for root, dirs, files in os.walk(path, topdown=False):
        for comfig in tqdm(files):
            save_path = "/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/挑图(subfig&subcap)/compare_visual/" + comfig
            # concat image
            our_img = Image.open(path+comfig)
            w, h = our_img.size
            exsclaim_img = Image.open(exsclaim_visual_root+comfig)
            result = Image.new(our_img.mode, (w*2, h))
            result.paste(our_img, box=(0, 0))
            result.paste(exsclaim_img, box=(w, 0))
            result.save(save_path)


def statis_pmc_ids():
    """
    统计PMC_id，也即所有爬到过的paper的数量
    """
    ids = []
    for i in range(10):
        last_num = len(ids)
        f = open('/remote-home/share/medical/public/PMC_OA/%d.jsonl'%i, 'r')
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        last_pmc = data[0]['PMC_ID']
        for datum in data:
            if datum['PMC_ID'] != last_pmc:
            # a new id
                ids.append(last_pmc)
                last_pmc = datum['PMC_ID']
        ids.append(last_pmc)
        print('in %d.jsonl, %d'%(i, len(ids)-last_num))
    print('total num %d'%len(ids))


def statis_caption_length():
    """
    所有subcap的长度和（不能打开的）caption的长度
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    f = open('/remote-home/share/medical/public/PMC_OA/caption_T060_filtered_top4/caption_T060_filtered_top4_sep_v0/caption_T060_filtered_top4_sep_v0_sub2com(new).jsonl', 'r')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    subcap_num = 0
    subcap_len = 0
    cap_num = 0
    cap_len = 0
    for datum in tqdm(data):
        if datum['subcaption'] != "":
            subcap_len += len(tokenizer.tokenize(datum['subcaption']))
            subcap_num += 1
        else:
            cap_len += len(tokenizer.tokenize(datum['caption']))
            cap_num += 1
    print('Avg Cap Len %.2f'%(cap_len/cap_num))
    print('Avg Subcap Len %.2f'%(subcap_len/subcap_num))


def filter_medical_from_imageclef2016():
    """
    将imageclef2016中的medical图像过滤出来
    """
    score_dict = {}
    score_file = json.load(open('/remote-home/share/medical/public/ImageCLEF2016/medtask/filtered_diagnosis_comfig.json', 'r'))
    for image_path in score_file.keys():
        image_id = image_path.split('/')[-1]
        score_dict[image_id] = np.argmax(np.array(score_file[image_path]))

    f = open('/remote-home/share/medical/public/ImageCLEF2016/medtask/train_and_test_comfigs.jsonl')
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    filtered_data = []
    for datum in tqdm(data):
        if score_dict[datum['id']] == 15:
            shutil.copy(os.path.join('/remote-home/share/medical/public/ImageCLEF2016/medtask/new_id_comfigs', datum['id']), os.path.join('/remote-home/share/medical/public/MedICaT/compound_figures/comfig_with_imageclef2016', str(len(filtered_data) + 2119) + '.jpg'))
            new_id_datum = datum
            new_id_datum['id'] = str(len(filtered_data) + 2119) + '.jpg'
            filtered_data.append(new_id_datum)

    f = open('/remote-home/share/medical/public/ImageCLEF2016/medtask/(filtered)train_and_test_comfigs.jsonl', "w")
    for line in filtered_data:
        f.write(json.dumps(line)+'\n')
    f.close()




if __name__ == "__main__":
    statis_caption_length()



    
