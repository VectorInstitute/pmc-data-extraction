cd openpmcvl/experiment/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

python -m open_clip_train.main \
        --model pubmedbert_ViT-B-16 \
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


python -m open_clip_train.main \
        --model biomedclip_tokenizer \
        --pretrained-image \
        --train-data /datasets/PMC-15M/processed/train_dummy_.jsonl \
        --val-data /datasets/PMC-15M/processed/test_dummy_.jsonl \
        --dataset-type jsonl \
        --data-img-key media_name \
        --data-caption-key caption_name \
        --data-img-rootdir /datasets/PMC-15M/figures/ \
        --data-cap-rootdir /datasets/PMC-15M/captions/ \
        --train-num-samples 100 \
        --val-num-samples 500 \
        --val-no-retrieval \
        --batch-size 4 \
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


python -m open_clip_train.main \
    --val-data $PMCOA_ROOT_DIR/valid.csv \
    --dataset-type csv \
    --data-separator , \
    --data-img-key image \
    --data-caption-key caption \
    --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --pretrained /path/to/checkpoints/epoch_K.pt

python -m open_clip_train.main \
    --val-data $PMCOA_ROOT_DIR/valid.csv \
    --dataset-type csv \
    --data-separator , \
    --data-img-key image \
    --data-caption-key caption \
    --model biomedclip \
