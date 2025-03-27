# PMC Helper Model Checkpoints

These checkpoints contain helper models used in creating the PMC-2M dataset.

## Download Weights

Make sure to run this script before using the pipeline.

```python
from huggingface_hub import snapshot_download

weights_dir = snapshot_download(
    repo_id="vector-institute/pmc-helper-models",
    local_dir="openpmcvl/granular/checkpoints",
    allow_patterns=["*.pt", "*.pth"]
)
```

The weights will be downloaded to the specified `local_dir`. Your existing code can then load them from this location.
Change the `local_dir` to the path of the directory leading to `openpmcvl/granular/checkpoints` in your project.
