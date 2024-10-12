python openpmcvl/probe/classifier.py \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=test \
    strict_loading=False \
    resume_from_checkpoint="/projects/multimodal/checkpoints/openpmcvl/batch_size_tuning/bs_256/epoch18-step62149.ckpt"
