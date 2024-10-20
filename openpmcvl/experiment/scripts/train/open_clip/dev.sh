cd openpmcvl/experiment/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

python -m open_clip_train.main \
        --model ViT-B-16 \
        --pretrained openai \
        --train-data $PMCOA_ROOT_DIR/train.csv \
        --train-num-samples 1317219 \
        --dataset-type csv \
        --data-separator , \
        --data-img-key image \
        --data-caption-key caption \
        --val-data $PMCOA_ROOT_DIR/valid.csv \
        --batch-size 32 \
        --accum-freq 4 \
        --workers 4 \
        --lr 5e-5 \
        --lr-scheduler cosine \
        --epochs 20 \
        --warmup 0 \
        --wd 0.1 \
        --name test \
        --resume latest \
        --gather-with-grad \
        --logs /checkpoint/$USER/$SLURM_JOBID/ \
        --zeroshot-frequency 1 \
        --report-to wandb
