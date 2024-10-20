"""Train BiomedCLIP with PMC-17M using open_clip library."""
import open_clip

def load_biomedclip():
    """Load BiomedCLIP model and tokenizer from checkpoint."""
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    return model, preprocess_train, preprocess_val, tokenizer

def load_pmcvl_data():
    """Load OpenPMC-VL train-val-test data using open_clip functions."""
    data = open_clip.get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    model, preprocess_train, preprocess_val, tokenizer = load_biomedclip()
    print(model)
    print(preprocess_train)