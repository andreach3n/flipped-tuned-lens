# Running delphi (EleutherAI) on our SAEs — the working recipe

Written 2026-08-18 after ~12 distinct failures, then **substantially simplified the same day** once
the actual root cause was found. Read this BEFORE provisioning a pod.

## THE ONE THING THAT MATTERS: pin the version triple

```
vllm==0.10.2   transformers==4.56.1   torch==2.8.0
```

delphi declares `vllm>=0.10.2` with **no upper bound** and leaves torch/transformers unpinned. vLLM
0.10.2 in turn pins torch exactly (`torch==2.8.0`, `torchvision==0.23.0`, `xformers==0.0.32.post1`)
but leaves **one hole: `transformers>=4.55.2`, a lower bound only.** pip therefore floats transformers
to a release far newer than vLLM understands, and the stack breaks.

That hole — not the Python version — is what broke every earlier attempt, including a direct attempt
at `vllm==0.10.2` that failed with `all_special_tokens_extended` and was wrongly blamed on Python 3.11.
Take the **oldest vllm satisfying the toml** and pin transformers to its contemporary: vllm 0.10.2
shipped 2025-09-13, transformers 4.56.1 on 2025-09-04. Use a constraints file so delphi's own install
cannot drag them forward.

With current vLLM (0.27.x) delphi dies in `input_processor._validate_model_input` with
`TypeError: '>' not supported between instances of 'str' and 'int'` — genuine API drift, delphi
passing strings where vLLM now wants token IDs. Patching that line only relocates the failure to the
detokenizer. Do not chase it; pin instead.

## Pod requirements — much weaker than this doc used to claim

```bash
python3 --version        # want 3.12.x (delphi's CI tests only 3.12); 3.11 likely fine now, see below
nvidia-smi | head -4     # driver version does NOT matter
```

**The CUDA-13 hunt is obsolete.** torch 2.8.0 ships cu128 wheels, which run on 12.8 and 13.x drivers
alike. Earlier advice to terminate and redeploy until a CUDA 13 host appeared came from newer torch
builds and no longer applies.

**Python 3.12 is probably no longer load-bearing either.** The only reason for it was flashinfer's
`array.array[int]` annotation (3.12-only syntax), and flashinfer is an *optional extra* in vLLM
0.10.2, not a dependency — so it never gets installed. delphi declares `requires-python >=3.10`.
Stay on 3.12 to match delphi's CI if it is cheap, but don't spend an hour on the PPA if it fights you.

Sizing: with a **local** judge, Llama-70B-AWQ-INT4 is ~37 GB of weights and fits one 80 GB A100
(`--num_gpus 1`; the default is 2). With a **remote** judge (OpenRouter) any small GPU works — vLLM
is still a hard import for delphi even when unused.

## Install

```bash
apt-get update && apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential

rm -rf /root/venv312 && python3.12 -m venv /root/venv312   # NOT /dev/shm (noexec), NOT /workspace
V=/root/venv312/bin/python
export TMPDIR=/dev/shm/tmp && mkdir -p $TMPDIR
export HF_HOME=/dev/shm/hf && mkdir -p $HF_HOME     # NOT optional -- see filesystem rules
$V -m pip install --upgrade pip
printf 'vllm==0.10.2\ntorch==2.8.0\ntorchvision==0.23.0\ntorchaudio==2.8.0\ntransformers==4.56.1\n' \
  > /root/constraints.txt
$V -m pip install --no-cache-dir -c /root/constraints.txt "git+https://github.com/EleutherAI/delphi"
$V -m pip install --no-cache-dir -c /root/constraints.txt plotly "kaleido<1"   # log_results needs it
$V -c "import torch,vllm,transformers;print(torch.__version__, vllm.__version__,
       transformers.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expect exactly `2.8.0 0.10.2 4.56.1 12.8 True`. Anything else means the constraints did not take —
stop there rather than debugging downstream.

**Build a FRESH venv.** Downgrading an existing one across a major torch version leaves a half-broken
environment; `rm -rf` first, always.

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
- **`export HF_HOME=/dev/shm/hf` BEFORE anything downloads a model.** Unset, the HF cache defaults to
  `/root/.cache/huggingface` on the 20 GB overlay, and the ~40 GB Llama-70B-AWQ download fills `/`
  to 100% and dies with `OSError: I/O error: No space left on device (os error 28)`. Put it in
  `~/.bashrc` — a dropped shell variable reintroduces this on every reconnect.
- `/workspace` is a **shared team volume with a quota**, not free space. Copying 28 GB of latent
  caches there failed with `Disk quota exceeded` partway, leaving unusable truncated files. Use it
  for small artifacts (log tarballs, saved patches) only; push anything large to HF instead.
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

- `--log_probs true` populates the `probability` field in the score files, which is what makes AUROC
  computable (see `plot_delphi_auroc.py`). Without it delphi gives class-balanced accuracy only.
- Run arms **one at a time** with a local judge — the 70B fills an 80 GB card.
- `--max_latents` is a CEILING, not a target. Most latents fail `min_examples=200`; 100 requested
  yielded 35/23, and 500 with 30M tokens yielded 195/161 (OpenRouter) / 198/161 (local).
- `--dataset_split train[:3%]` — 1% does not contain 30M tokens.
- Caching is ~2-3 s/it for 3662 iterations (~2.5 h/arm) and looks GPU-idle: delphi materializes a
  dense (8192, 73728) latent buffer per batch, so it is memory/IO bound. Judge by it/s, not GPU%.
- Scoring on one A100 takes ~35 min/arm at these n. `Processing items: 0it` in the log can be stale —
  tqdm writes carriage returns, so judge progress by file counts and GPU%, not by the log tail.

### A successful run still ENDS IN A TRACEBACK — ignore it

```
KeyError: 'firing_count'   ... in log/result_analysis.py, plot_firing_vs_f1
Missing firing counts for some modules. Missing modules: ['layers.13']
ERROR ... Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

That is delphi's own summary-plot step (`log_results`), which runs **after** all scoring is written
to disk, failing because our latent caches carry no `firing_count`. Harmless — we compute figures
from the per-latent score files anyway. Verify completion by counting instead:
`ls results/<name>/scores/detection | wc -l` should equal `ls results/<name>/explanations | wc -l`.

## Skip caching entirely — the latent caches are on HF

Both arms' 30M-token caches are archived, so a rerun needs **no GPU caching**:

```
andreayhchen/gemma2-2b-linearmap-saes-trained-20m : delphi_latents_L13_30M/  + sparsify_topk_L13/
andreayhchen/gemma2-2b-linearmap-saes-rand-all-s0 : delphi_latents_L13_30M/  + sparsify_topk_L13/
```

`snapshot_download` them, move `delphi_latents_L13_30M` to `results/<name>/latents`, and delphi logs
`Files found in .../latents, skipping...`. Five `.safetensors` **plus `config.json`** must be present;
shards without the config make delphi refuse to recache and die.

## Remote judges via OpenRouter — logprobs need a patch AND the right model

delphi's OpenRouter client never requests logprobs (hardcoded `logger.warning` saying so). Patching
it is straightforward — pass `logprobs`/`top_logprobs` through to the payload, add
`"provider": {"require_parameters": true}`, and convert OpenRouter's JSON into the attribute-access
shape `classifier._parse_logprobs` expects (`tok.token`, `tok.top_logprobs[i].logprob`); a
`SimpleNamespace` per token does it.

But the patch is not sufficient by itself: **the model has to actually return per-token logprobs.**
Verified empirically (advertised support is unreliable — `llama-3.3-70b` claims it and returns one
entry):

| model | result |
|---|---|
| `meta-llama/llama-3.1-70b-instruct` | DeepInfra: **0 entries**; CoreWeave: **1** (EOT only) |
| `meta-llama/llama-3.1-8b-instruct` | 9 entries ✓ |
| `openai/gpt-4o-mini` | 9 entries ✓ |
| `openai/gpt-5.x`, all `anthropic/*` | no logprobs support at all |

So the paper's own judge cannot give AUROC over OpenRouter — that is why the local vLLM route exists.
Note `_parse_logprobs` asserts that the count of tokens containing "1"/"0" equals `n_examples_shown`,
so a judge that emits reasoning or preamble around the bare `[1, 0, 1]` will break the batch. Prefer
terse non-reasoning models.

## Archive before stopping

`results/` is on `/dev/shm` (RAM) and the venv is on `/` — a pod **stop** rebuilds both; only
`/workspace` survives (and it has a quota). Tar `explanations/`, `scores/` and `run_config.json` and
push to HF before stopping or terminating. `hf` lives in the venv: `/root/venv312/bin/hf`
(`huggingface-cli` is deprecated and its old invocation now fails).
