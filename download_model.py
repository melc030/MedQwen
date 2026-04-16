import os
import sys
sys.path.append('..')

from pathlib import Path
from huggingface_hub import snapshot_download
from config import Config

cfg = Config()

def main():
    target_dir = Path(cfg.model_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    model_id = os.environ.get('HF_MODEL_ID', cfg.hf_model_id)
    token    = os.environ.get('HF_TOKEN')

    print(f'downloading {model_id} → {target_dir}')

    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,       # safe to re-run if interrupted
    )
    print('download complete.')

if __name__ == '__main__':
    main()
