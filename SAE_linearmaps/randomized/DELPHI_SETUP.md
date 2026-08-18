# Running delphi (EleutherAI) on our SAEs — the working recipe

Written 2026-08-18 after ~11 distinct failures. Read this BEFORE provisioning a pod; most of the
failures were environment mismatches that cost hours each and are avoidable in ten seconds.

## Pick the pod first — check BOTH before installing anything

```bash
python3 --version        # want 3.12.x  (delphi's CI tests ONLY on 3.12)
nvidia-smi | head -4     # want CUDA Version 13.x
```

Terminate and redeploy if either is wrong. Python version comes from the IMAGE (Ubuntu 24.04 ships
3.12; `ubuntu2204` images give 3.11); the driver comes from the HOST and is luck of the draw.

Why it matters: delphi pins nothing (`vllm>=0.10.2`, unpinned torch/transformers) and its CI runs
`pip install -e ".[dev,visualize]"` on Python 3.12. On 3.11 the resolved stack breaks — flashinfer
annotates `array.array[int]`, which is 3.12-only syntax. Chasing that by pinning vllm backwards
drags transformers and torch out of alignment; there is no consistent 3.11 combination worth finding.

If stuck on a 3.11 image with a good driver, build a 3.12 venv (see below) rather than redeploying.

## Install

```bash
apt-get update && apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential

python3.12 -m venv /root/venv312          # NOT /dev/shm (noexec) and NOT /workspace (see below)
V=/root/venv312/bin/python
export TMPDIR=/dev/shm/tmp && mkdir -p $TMPDIR
$V -m pip install --upgrade pip
$V -m pip install --no-cache-dir --ignore-installed blinker "git+https://github.com/EleutherAI/delphi"
$V -m pip install --no-cache-dir plotly "kaleido<1"    # the [visualize] extra; log_results needs it
$V -c "import sys, torch, vllm, transformers; print(sys.version.split()[0], torch.__version__,
       vllm.__version__, torch.cuda.is_available())"
```

Always invoke as `$V -m delphi ...`. Do NOT rely on `activate`: it silently left `pip` pointing at
the system 3.11 while `python` pointed at the venv, so packages installed where nothing could see them.

`python3.12-dev` is not optional — triton JIT-compiles a CUDA helper with gcc against
`/usr/include/python3.12`, and without the headers every run dies at the first GPU kernel.

## Filesystem rules (each of these cost hours)

| path | property | use for |
|---|---|---|
| `/dev/shm` | tmpfs, **noexec**, RAM, wiped on stop | caches, `results/`, `TMPDIR`, `HF_HOME` |
| `/workspace` | network mount (mfs), **I/O errors on large pip writes** | small files, log tarballs |
| `/` (overlay) | ~20 GB, executable, rebuilt from image on stop | the venv, nothing large |

- A venv on `/dev/shm` installs fine and then fails with `failed to map segment from shared object`
  — noexec. `mount -o remount,exec` is denied in an unprivileged container.
- pip unpacking multi-GB wheels on `/workspace` throws `OSError: [Errno 5] Input/output error`.
- delphi writes `results/` relative to CWD. At `--n_tokens 30000000` each arm's latent cache is
  **~14 GB**, so two arms will not fit on a 20 GB volume — `cd` somewhere on `/dev/shm` first.
  Running out mid-cache leaves a partial cache that delphi then SKIPS re-creating: it sees the
  shards, finds no `config.json`, and dies. Delete `results/<name>/latents` and redo.

## The one code edit (an actual upstream bug, not a version issue)

delphi loads SAEs onto CPU while the model is on GPU:

```bash
$V -c "
import delphi.sparse_coders.load_sparsify as m
p = m.__file__; s = open(p).read()
old = 'name_path / hookpoint, device=\"cpu\"'
open(p,'w').write(s.replace(old, 'name_path / hookpoint, device=device')) if old in s else print('ok')"
```

Reapply after any delphi reinstall.

## SAEs

Ours are `sae_lens` checkpoints; delphi wants sparsify format. `convert_sae_to_sparsify.py` handles
the fold (scale, `apply_b_dec_to_input`) and verifies pre-activations match. Train with
`SAE_ARCH=topk` so the sparsity rule converts exactly — BatchTopK does not (see that script).

delphi resolves hookpoints WITHOUT the wrapper prefix, so pass `layers.13`, and the SAE directory
must be named to match:

```bash
mv <sae_dir>/sparsify_topk_L13/model.layers.13 <sae_dir>/sparsify_topk_L13/layers.13
```

Passing `model.layers.13` gives "Could not find valid path for hookpoint". `check_hookpoint.py`
proves delphi's `layers.13` output is the same tensor as our `blocks.13.hook_resid_post`
(cosine 0.999992).

## Run

```bash
cd /dev/shm/delphi_run     # results/ lands here; needs ~14 GB per arm
CUDA_VISIBLE_DEVICES=0 nohup $V -m delphi google/gemma-2-2b <sae_dir> --hookpoints layers.13 \
  --scorers fuzz detection --log_probs --max_latents 500 --n_tokens 30000000 \
  --dataset_repo Skylion007/openwebtext --dataset_split 'train[:3%]' --name <name> > log 2>&1 &
```

Paste launches as ONE LINE. Multi-line backslash blocks were mangled by the terminal three times,
and a `sleep` between two launches discards the second if you Ctrl-C it.

- `--log_probs` is what populates the `probability` field in the score files, which is what makes
  AUROC computable. Without it (and without a local backend) delphi reports only class-balanced
  accuracy — OpenRouter cannot supply logprobs at all; its client says so explicitly.
- `--max_latents` is a CEILING, not a target. Most latents fail `min_examples=200`; 100 requested
  yielded 35/23, and 500 with 30M tokens yielded 195/161.
- `--dataset_split train[:3%]` — 1% does not contain 30M tokens.
- Caching is ~2-3 s/it for 3662 iterations (~2.5 h/arm) and looks GPU-idle: delphi materializes a
  dense (8192, 73728) latent buffer per batch, so it is memory/IO bound. Judge by it/s, not GPU%.

## Archive before stopping

`results/` is on `/dev/shm` (RAM) and the venv is on `/` — a pod **stop** wipes both; only
`/workspace` survives. Tar (excluding `latents/`) and push to HF before stopping or terminating.
