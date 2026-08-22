"""Upload a local folder into a path inside one HF repo. Used for SAEs, logs and delphi results.

sparsify writes checkpoints to local disk and never pushes. On a pod that matters more than
usual: /dev/shm is RAM and / is rebuilt from the image, so a stop destroys everything that is
not on the Hub (see ../DELPHI_SETUP.md, "Filesystem rules").

REPO LAYOUT -- one repo, and the randomized model sits at its ROOT on purpose:

    <repo>/                                  <- IS the randomized pythia-1b. `AutoModel` and
      config.json, model.safetensors,           delphi both take a bare repo id, and neither
      tokenizer.json, ...                       CLI can express `subfolder=`, so the model has
      saes/pythia1b_trained_L8_.../             to be at the root to be loadable by name.
      saes/pythia1b_rand_L8_.../
      logs/train_trained_L8.log
      delphi/pythia1b_trained_L8.tar.gz

So the trained arm's SAEs live in a repo whose root is the random model. That is deliberate --
it keeps every artifact of the comparison in one place, and the alternative (a second repo) buys
nothing except a tidier name.

    LOCAL=/dev/shm/saes/pythia1b_trained_L8_R64_k32_100M \
      HF_REPO=andreayhchen/pythia-1b-heap-replication \
      PATH_IN_REPO=saes/pythia1b_trained_L8_R64_k32_100M python -u hf_push.py

Set PRIVATE=1 to create the repo private. Existing files at the same path are overwritten, so
PATH_IN_REPO is what keeps the two arms from clobbering each other -- the same hazard hf_io.py
guards in the gemma pipeline. It is checked below rather than trusted.
"""
import os
import sys

LOCAL        = os.environ["LOCAL"]
HF_REPO      = os.environ["HF_REPO"]
PATH_IN_REPO = os.environ.get("PATH_IN_REPO", "").strip("/")
PRIVATE      = os.environ.get("PRIVATE", "0") == "1"

if not os.path.isdir(LOCAL):
    sys.exit(f"nothing at {LOCAL}")

# The arms differ ONLY by path inside the repo, so an empty/duplicated PATH_IN_REPO silently
# overwrites the other arm's SAE with weights that load fine and are the wrong model. Refuse
# the ambiguous case rather than discover it in the scores.
if not PATH_IN_REPO and not os.path.exists(os.path.join(LOCAL, "config.json")):
    sys.exit("PATH_IN_REPO is empty but LOCAL is not a model checkpoint -- refusing to write "
             "artifacts to the repo root, which is reserved for the randomized model.")

from huggingface_hub import HfApi, create_repo  # noqa: E402

create_repo(HF_REPO, repo_type="model", exist_ok=True, private=PRIVATE)
HfApi().upload_folder(
    folder_path=LOCAL,
    repo_id=HF_REPO,
    repo_type="model",
    path_in_repo=PATH_IN_REPO or None,
)
where = f"{HF_REPO}/{PATH_IN_REPO}" if PATH_IN_REPO else f"{HF_REPO} (root)"
print(f"pushed {LOCAL} -> {where}")
