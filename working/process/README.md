filter out non-medical figure

```
please refer to subfigure_classify.py
```

detect and separate subfigure

```
inference_det_align.py \
--task 'separate' \
--save_path path_to_save_subfigures \
--rcd_file file_to_log_results \
--img_root path_of_compound_figures \   # save all the figure in a folder
--eval_file jsonl_file_of_all_the_samples_to_infer # refer to Fig_Separation_Dataset for the format of the jsonl file
--checkpoint subfigure_detection.pth
```

filter out non-medical subfigure

```
please refer to subfigure_classify.py
```

detect and separate subcaption

```
please refer to chatgpt_subcaption.py (this is a py file converted from jupyter notebook on colab)
```

subfigure OCR

```
please refer to subfigure_ocr() in subfigure_ocr.py, its done using [exsclaim](https://github.com/MaterialEyes/exsclaim)
```

pair subfigure and subcaption

```
refer to GPTsubcap_postprocess
```
