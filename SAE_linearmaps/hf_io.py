"""Tiny HuggingFace Hub helpers so trained artifacts live on HF, not a network volume.

Override the destination with:  export HF_REPO=youruser/your-repo
Auth comes from the HF_TOKEN env var (write scope) -- the same token also pulls gated gemma.
"""
import os
from huggingface_hub import HfApi, hf_hub_download, create_repo

HF_REPO = os.environ.get("HF_REPO", "andreayhchen/gemma2-2b-linearmap-saes")
_api = HfApi()
_repo_ready = False


def ensure_repo():
    global _repo_ready
    if not _repo_ready:
        create_repo(HF_REPO, repo_type="model", exist_ok=True, private=True)
        _repo_ready = True


def push(local_path, name=None):
    """Upload a file to the HF model repo (creates the repo on first call)."""
    ensure_repo()
    name = name or os.path.basename(local_path)
    _api.upload_file(path_or_fileobj=local_path, path_in_repo=name,
                     repo_id=HF_REPO, repo_type="model")
    print(f"  ↑ pushed {name} -> {HF_REPO}")
    return name


def pull(name, local_dir="/workspace/hf_pull"):
    """Download a file from the HF model repo; returns the local path."""
    os.makedirs(local_dir, exist_ok=True)
    path = hf_hub_download(repo_id=HF_REPO, filename=name, repo_type="model", local_dir=local_dir)
    print(f"  ↓ pulled {name} <- {HF_REPO}")
    return path
