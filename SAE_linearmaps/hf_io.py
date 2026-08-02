"""Tiny HuggingFace Hub helpers so trained artifacts live on HF, not a network volume.

Override the destination with:  export HF_REPO=youruser/your-repo
Auth comes from the HF_TOKEN env var (write scope) -- the same token also pulls gated gemma.

Artifact names (sae_{mode}_k{K}_final.pt, linear_map_layer_13.pt, ...) do NOT encode the
model variant, so a random-init run would silently OVERWRITE the trained SAEs. Rather than
tag filenames in nine scripts, each arm gets its own repo via HF_REPO -- and the guard below
refuses to write to the default repo whenever VARIANT says this is not the trained model.
"""
import os
from huggingface_hub import HfApi, hf_hub_download, create_repo

DEFAULT_REPO = "andreayhchen/gemma2-2b-linearmap-saes"   # the TRAINED-arm artifacts
HF_REPO = os.environ.get("HF_REPO", DEFAULT_REPO)
_api = HfApi()
_repo_ready = False


def _guard_repo():
    """Refuse to touch the trained-arm repo from a random-init run (see module docstring)."""
    variant = os.environ.get("VARIANT", "trained").strip() or "trained"   # `export VARIANT=` -> trained
    if variant != "trained" and HF_REPO == DEFAULT_REPO:
        raise RuntimeError(
            f"VARIANT={variant!r} but HF_REPO is still the trained-arm default "
            f"({DEFAULT_REPO!r}). This run would overwrite your trained SAEs/map. "
            f"Set HF_REPO to a per-arm repo, e.g.\n"
            f"    export HF_REPO={DEFAULT_REPO}-{variant.replace('_', '-')}-s"
            f"{os.environ.get('INIT_SEED', '0')}"
        )


def ensure_repo():
    global _repo_ready
    if not _repo_ready:
        _guard_repo()
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


# One pull dir PER REPO. A shared dir would mix arms: `linear_map_layer_13.pt` has the same
# filename in every repo, and a stale local copy is exactly the silent-wrong-artifact failure
# that has bitten this project before.
PULL_ROOT = os.environ.get("HF_PULL_DIR", "/workspace/hf_pull")


def pull(name, local_dir=None):
    """Download a file from the HF model repo; returns the local path."""
    local_dir = local_dir or os.path.join(PULL_ROOT, HF_REPO.replace("/", "__"))
    os.makedirs(local_dir, exist_ok=True)
    path = hf_hub_download(repo_id=HF_REPO, filename=name, repo_type="model", local_dir=local_dir)
    print(f"  ↓ pulled {name} <- {HF_REPO}")
    return path
