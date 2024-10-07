for iou_th in 0.95 0.9 0.8 0.75 0.7 0.65 0.6 0.55 0.5
    do 
        for score_th in 0.9 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.15 0.1
            do
                for simi_th in 0.9 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.15 0.1
                    do
                        python inference.py --model baseline --checkpoint Log/ContinueAlign/\(0.646\)newdataset_open_detection/best_valid.pth --iou_threshold $iou_th --score_threshold $score_th --similarity_threshold $simi_th --rcd_file Log/ContinueAlign/\(0.646\)newdataset_open_detection/grid_search_threshold.txt
                    done
            done
    done 