cd openpmcvl/experiment/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

python -m open_clip_train.main \
        --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
        --train-data /datasets/PMC-15M/processed/train_dummy_.jsonl \
        --val-data /datasets/PMC-15M/processed/test_dummy_.jsonl \
        --dataset-type jsonl \
        --data-img-key media_name \
        --data-caption-key caption_name \
        --data-img-rootdir /datasets/PMC-15M/figures/ \
        --data-cap-rootdir /datasets/PMC-15M/captions/ \
        --train-num-samples 1000 \
        --val-num-samples 500 \
        --val-no-retrieval \
        --batch-size 32 \
        --accum-freq 1 \
        --aug-cfg quilt_crop=True \
        --workers 4 \
        --lr 1e-4 \
        --lr-scheduler cosine \
        --epochs 100 \
        --warmup 10 \
        --wd 0.1 \
        --name test \
        --resume latest \
        --gather-with-grad \
        --logs /checkpoint/$USER/$SLURM_JOBID/ \
        --zeroshot-frequency 1
