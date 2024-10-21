cd openpmcvl/experiment/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"


python -m open_clip_train.main \
        --model biomedclip_tokenizer \
        --pretrained-image \
        --train-data /datasets/PMC-15M/processed/train_cleaner.jsonl \
        --val-data /datasets/PMC-15M/processed/val_cleaner.jsonl \
        --dataset-type jsonl \
        --data-img-key media_name \
        --data-caption-key caption_name \
        --data-img-rootdir /datasets/PMC-15M/figures/ \
        --data-cap-rootdir /datasets/PMC-15M/captions/ \
        --train-num-samples 13397815 \
        --val-num-samples 2512082 \
        --val-no-retrieval \
        --batch-size 256 \
        --accum-freq 4 \
        --workers 4 \
        --lr 5e-4 \
        --beta1 0.9 \
        --beta1 0.98 \
        --eps 1e-6 \
        --lr-scheduler cosine \
        --wd 0.2 \
        --epochs 32 \
        --warmup 2000 \
        --seed 0 \
        --image-mean 0.48145466 0.4578275 0.40821073 \
        --image-std 0.26862954 0.26130258 0.27577711 \
        --val-frequency 1 \
        --precision amp_bfloat16 \
        --local-loss \
        --name test \
        --resume latest \
        --logs /checkpoint/$USER/$SLURM_JOBID/ \
        --report-to wandb
