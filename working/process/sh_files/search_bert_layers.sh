python train.py --output_dir 'Log/ContinueAlign/Bert_Froze_8' --model baseline --detr_froze True --alignment_froze False --bert_froze_depth 8 --checkpoint 'Log/Detection/(0.837)Color_Flip_Aug/best_det.pth' --warmup 300 --lr 1e-5 --align_loss_coef 1.0

python train.py --output_dir 'Log/ContinueAlign/Bert_Froze_10' --model baseline --detr_froze True --alignment_froze False --bert_froze_depth 10 --checkpoint 'Log/Detection/(0.837)Color_Flip_Aug/best_det.pth' --warmup 300 --lr 1e-5 --align_loss_coef 1.0 --gpu 1

python train.py --output_dir 'Log/ContinueAlign/Bert_Froze_4' --model baseline --detr_froze True --alignment_froze False --bert_froze_depth 4 --checkpoint 'Log/Detection/(0.837)Color_Flip_Aug/best_det.pth' --warmup 300 --lr 1e-5 --align_loss_coef 1.0 --gpu 2