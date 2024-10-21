cd openpmcvl/experiment/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

python -m open_clip_train.main \
    --val-data $PMCOA_ROOT_DIR/valid.csv \
    --dataset-type csv \
    --data-separator , \
    --data-img-key image \
    --data-caption-key caption \
    --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
