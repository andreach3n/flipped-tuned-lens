# Running delphi (EleutherAI) on our SAEs — the working recipe

Written 2026-08-18, **substantially rewritten 2026-08-19** after finding that every delphi number
this project had produced was wrong. Read the next section before anything else.

---

## STOP: everything delphi produced before 2026-08-19 is invalid

Three independent, silent defects sat between our SAEs and delphi. Each one produces a complete set
of plausible numbers and no error. Together they made delphi report that Heap et al. replicates
(trained ≈ random) when it does not.

**1. `b_dec` was applied twice.** sparsify's `SparseCoder.encode` subtracts `b_dec` itself:

```python
def encode(self, x):
    if not self.cfg.transcode:
        x = x - self.b_dec
    return fused_encoder(x, self.encoder.weight, self.encoder.bias, self.cfg.k, self.cfg.activation)
```

`convert_sae_to_sparsify.py` *also* folded `- b_dec @ W_enc` into `encoder.bias`. The result is a
constant offset on all 73,728 pre-activations, which changes which 32 latents win the top-k.
Measured effect: 15% Jaccard against the true firing set. **Fix: `encoder.bias = b_enc`.**

**2. The decoder-norm scaling was never folded in.** `sae_lens`' `encode` returns pre-activations
multiplied by `||W_dec_j||` (verified exactly — the measured ratio equals the norm per latent). Our
decoder is not unit-norm (norms run 1.3–1.7), so omitting it re-ranks latents. **Fix: fold the
norms into the encoder and normalise `W_dec` to compensate; reconstruction is unchanged.**

**3. Activations were fed in a BOS-free regime the SAEs were never trained on.** `activations.py`
builds every training activation with `model.to_tokens(text)`, which **prepends BOS**, then drops
position 0. delphi's `filter_bos` (default `True`) deletes every BOS from the *flattened* corpus and
re-chunks, so its rows carry none. gemma-2 leans on the BOS attention sink: `|h|` falls from ~174 to
~141, and the frozen linear map `P` — fit in the BOS regime — then *overshoots*, making
`h - P[tok]` higher-variance than `h` (explained variance went **negative**, −0.07 vs the expected
0.56). **Fix: run `[BOS] + row` and drop the BOS column.**

### Why none of this was caught

`convert_sae_to_sparsify.py` verified itself by computing `h @ encoder.weight.T + encoder.bias`
**by hand** — a restatement of the fold's own assumption. It could only ever confirm the fold
against itself, and it passed at 1e-6 while delphi computed something with 0.15 Jaccard against the
truth.

> **The lesson, and it generalises: verify against the LIBRARY'S OWN CALL (`sc.encode()` vs
> `sae.encode()`), and on the quantity that determines the output (the SELECTED LATENTS), not on a
> value vector you re-derived yourself.**

### Which defect caused the false replication — measured, not guessed

`randomized/ablation_delphi.py`, random arm, plain top-k SAE, fuzz AUROC:

| | BOS-free | BOS-prefixed |
|---|---|---|
| **broken conversion** | **0.779** (n=160) — reproduces the archived 0.781 / n=161 | 0.764 (n=167) |
| **true SAE** | 0.573 (n=328) | **0.632** (n=336) |

- **The conversion is the cause**: +0.132 with BOS fixed, +0.206 without.
- **BOS acts in the OPPOSITE direction** (−0.059 with the true SAE) and partially *masked* how much
  the conversion was inflating things. There is a real interaction.
- **Mechanism, evidenced:** the broken conversion halves surviving latents (160/167 vs 328/336) and
  drops firings in latents 0–499 from 0.138 to 0.098 per token. With k=32 fixed, that is the
  dictionary collapsing into fewer, near-always-on latents — exactly what a constant bias predicts.
  TNR *rises* under the broken SAE (0.62 vs 0.39): an always-on latent has crisp surface behaviour
  the judge can describe AND correctly reject non-examples for. **The mis-conversion manufactured
  describable features out of a random transformer.**

---

## THE ROUTE THAT WORKS: write the latent cache yourself

`randomized/write_delphi_cache.py`. This is now the recommended path for **all** cells, not just
skip-embed, because it bypasses the conversion entirely — which is where two of the three defects
lived.

delphi runs three stages: **cache → explain → score**. Only `cache` touches the SAE; `explain` and
`score` read the on-disk cache and talk to the judge. So we compute the cache from `sae_lens`' own
`encode` (which *is* the trained SAE) and let delphi do everything that affects the score.

**It is also the only way to score a skip-embed (`resid`) SAE at all.** That encoder reads
`h - P[tok]`, and `P` is a (V, 2304) per-token lookup. sparsify's `SparseCoder` has exactly one bias
vector, so there is nowhere to put it — unlike `scale` and `b_dec`, which are constants and fold in
fine. A converted skip-embed SAE would silently be fed `h`.

### Validated to Jaccard 1.000000

`EMULATE_SPARSIFY` + `BACKEND=hf` makes the writer reproduce delphi's exact computation (bugs
included). Diffed against delphi's archived cache with `diff_delphi_cache.py`: **every firing
identical, 100% of tokens with the same selected-latent set, firing counts correlated at 1.000000.**
That is what licenses the whole approach. Re-run it after any change to the writer.

```bash
# validation (a few minutes): emulate delphi exactly, then diff
VARIANT=trained HF_REPO=$TRAINED_REPO MODE=full SAE_NAME=sae_full_k32_d73728_100M_topk_final.pt \
  REF_CACHE=<a cache dir> OUT_DIR=/dev/shm/delphi_run/results/valid_emul MAX_LATENTS=500 \
  MAX_ROWS=2000 BACKEND=hf EMULATE_SPARSIFY=<sparsify dir> python -u randomized/write_delphi_cache.py
REF_CACHE=<a cache dir> OURS=/dev/shm/delphi_run/results/valid_emul/latents/layers.13 \
  MAX_LATENTS=500 python -u randomized/diff_delphi_cache.py
```

### Science runs

```bash
# one line each; ~25 min per cell on an A100
VARIANT=trained HF_REPO=$TRAINED_REPO MODE=resid SAE_NAME=sae_resid_k32_d73728_100M_topk_final.pt \
  REF_CACHE=<a cache dir> OUT_DIR=/dev/shm/delphi_run/results/trained_resid MAX_LATENTS=500 \
  python -u randomized/write_delphi_cache.py
```

`BACKEND=tl` (default) for every science cell so all four share one activation pipeline; `hf` is
validation-only, and `MODE=resid` refuses it because `P` was fit against TransformerLens'
embedding table. TL vs HF differ by bf16 path noise (cosine 0.9997) which flips ~8% of firings at
the k-boundary — enough to matter for an exact diff, not enough to matter for a science comparison,
but do not mix them across cells.

**A hard gate fires at batch 0** if the linear map does not fit the activations (explained variance
< 0.1), which is what catches a missing `PREPEND_BOS`. Expect ~0.54 trained / ~0.32 rand, `mean |h|`
~174 trained / ~317 rand, and `mean |r|` within 1% of `scale * sqrt(2304)`.

---

## delphi's cache format (what the writer reproduces)

Per hookpoint, `results/<name>/latents/layers.13/`:

```
0_14744.safetensors  14745_29490.safetensors  29491_44235.safetensors
44236_58981.safetensors  58982_73727.safetensors  config.json
```

Boundaries are `torch.linspace(0, d_sae, n_splits+1).long()` with **inclusive** ranges. Each shard:

| array | shape | dtype | meaning |
|---|---|---|---|
| `locations` | (N, 3) | uint16 **or** uint32 | `[row, position, latent − start]` |
| `activations` | (N,) | float16 | firing strength |
| `tokens` | (n_rows, 256) | int64 | the **whole** token matrix, duplicated in every shard |

- uint16 only if *both* max row and max latent-offset are < 2¹⁶. At 30M tokens there are 117,184
  rows, so it is uint32.
- "Fired" is `abs(x) > 1e-5` on the dense buffer (`sae_dense_latents` scatters top-k into zeros).
- Emit in lexicographic order (row, then position, then latent) — that is what `torch.nonzero`
  produces, and it makes a byte-level diff possible.

**DO NOT REGENERATE THE `tokens` ARRAY. Read it out of an existing cache.** `filter_bos` deletes
BOS from the *flattened* stream then re-chunks, so rows do not align with documents; the corpus is
shuffled with seed 22; and it is truncated to whole batches
(`n_rows = (n_tokens // 256) // 32 * 32`). Reading it inherits all of that by construction, and
makes every arm score on identical text. delphi never re-tokenises at read time —
`LatentDataset.load_tokens` is dead code behind a `hasattr` guard that is always true.

**Any cache works as `REF_CACHE`** — the writer only reads `tokens` and copies `config.json`. Use a
~300 MB writer cache rather than re-downloading the 14 GB original.

**`--max_latents N` is `torch.arange(N)`** — the FIRST N latent indices, **not** the top-firing N
(an earlier version of this doc said otherwise). So only latents 0..N-1 need caching: ~150 MB
instead of ~14 GB. `MAX_LATENTS` in the writer must be ≥ whatever you pass delphi.

Cache-side defaults, all of which our runs used: `cache_ctx_len` 256, `batch_size` 32, `n_splits` 5,
`seed` 22, `filter_bos` True. Downstream (delphi's own, untouched): `example_ctx_len` 32,
`min_examples` 200, `n_non_activating` 50, `n_examples_test` 50,
`num_examples_per_scorer_prompt` 5.

---

## THE ONE THING THAT MATTERS FOR THE INSTALL: pin the version triple

```
vllm==0.10.2   transformers==4.56.1   torch==2.8.0
```

delphi declares `vllm>=0.10.2` with no upper bound and leaves torch/transformers unpinned. vLLM
0.10.2 pins torch exactly but leaves **one hole: `transformers>=4.55.2`, a lower bound only.** pip
floats transformers past what vLLM understands and the stack breaks. That hole — not the Python
version — is what broke every early attempt, including one wrongly blamed on Python 3.11. Take the
oldest vllm satisfying the toml and pin transformers to its contemporary (vllm 0.10.2 shipped
2025-09-13; transformers 4.56.1 is 2025-09-04). Use a constraints file.

## Pod requirements

**Python 3.12 is not load-bearing.** delphi declares `requires-python >=3.10`; flashinfer (the only
reason 3.12 was ever needed) is an optional extra in vLLM 0.10.2 and never gets installed. **Skip
the deadsnakes PPA** — on the RunPod image `add-apt-repository` dies with
`ModuleNotFoundError: No module named 'apt_pkg'`, because `/usr/bin/python3` has been swapped to
3.11.10 while apt's `apt_pkg` is built for 3.10. Just build the venv on the system python3.

**The CUDA-13 hunt is obsolete** — torch 2.8.0 ships cu128 wheels, which run on 12.8 and 13.x alike.

Sizing: Llama-70B-AWQ-INT4 is ~37 GB and fits one 80 GB A100 (`--num_gpus 1`; the default is 2).
Two 80 GB cards let you run two arms at once; two *smaller* cards is the one configuration that is
worse than one big one, since `max_memory 0.7` of 48 GB is below the weights.

```bash
apt-get install -y python3.11-dev python3.11-venv build-essential
rm -rf /root/venv312 && /usr/bin/python3 -m venv /root/venv312   # NOT /dev/shm (noexec), NOT /workspace
V=/root/venv312/bin/python
export TMPDIR=/dev/shm/tmp && mkdir -p $TMPDIR
export HF_HOME=/dev/shm/hf && mkdir -p $HF_HOME     # NOT optional -- see filesystem rules
$V -m pip install --upgrade pip
printf 'vllm==0.10.2\ntorch==2.8.0\ntorchvision==0.23.0\ntorchaudio==2.8.0\ntransformers==4.56.1\n' \
  > /root/constraints.txt
$V -m pip install --no-cache-dir -c /root/constraints.txt "git+https://github.com/EleutherAI/delphi"
$V -m pip install --no-cache-dir -c /root/constraints.txt plotly "kaleido<1"
$V -c "import torch,vllm,transformers;print(torch.__version__, vllm.__version__,
       transformers.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expect exactly `2.8.0 0.10.2 4.56.1 12.8 True`. Anything else means the constraints did not take —
stop there rather than debugging downstream. **Build a FRESH venv**; downgrading across a major
torch version leaves a half-broken environment. Always invoke as `$V -m delphi ...`; do NOT rely on
`activate`, which has silently left `pip` pointing at the system python.

`python3.11-dev` is not optional — triton JIT-compiles a CUDA helper against `Python.h`.

The **training** side (`write_delphi_cache.py`) needs `sae_lens` + `transformer_lens` and must live
in a SEPARATE environment (the base image python, via `setup.sh`). Installing those into the delphi
venv drags torch forward and breaks the pinned triple. The cache on disk is the handoff.

## The one code edit (an upstream bug, not a version issue)

delphi loads SAEs onto CPU while the model is on GPU:

```bash
$V -c "
import delphi.sparse_coders.load_sparsify as m
p = m.__file__; s = open(p).read()
old = 'name_path / hookpoint, device=\"cpu\"'
open(p,'w').write(s.replace(old, 'name_path / hookpoint, device=device')) if old in s else print('ok')"
```

Reapply after any delphi reinstall.

## Filesystem rules (each of these cost hours)

| path | property | use for |
|---|---|---|
| `/dev/shm` | tmpfs, **noexec**, RAM, wiped on stop | caches, `results/`, `TMPDIR`, `HF_HOME` |
| `/workspace` | network mount, **shared quota**, I/O errors on large pip writes | the repo, small files |
| `/` (overlay) | ~20 GB, executable, rebuilt from image on stop | the venv, nothing large |

- A venv on `/dev/shm` fails with `failed to map segment from shared object` — noexec.
- **`export HF_HOME=/dev/shm/hf` BEFORE anything downloads.** Unset, the ~37 GB judge fills the
  20 GB overlay and dies with `No space left on device`. Put it in `~/.bashrc` — a re-SSH that drops
  it reintroduces this every time.
- A pod **stop** rebuilds `/dev/shm` and `/`; only `/workspace` survives (with a quota). Push
  everything to HF.
- Shell exports are NOT `~/.bashrc`: a re-SSH silently empties them and the run dies on an empty
  path. Inline full paths in long-running launches.

## Run

```bash
cd /dev/shm/delphi_run     # results/<name> is resolved RELATIVE TO CWD
CUDA_VISIBLE_DEVICES=0 nohup $V -m delphi google/gemma-2-2b <sae_dir> --hookpoints layers.13 \
  --scorers fuzz detection --log_probs --max_latents 500 --n_tokens 29999104 --num_gpus 1 \
  --dataset_repo Skylion007/openwebtext --dataset_split 'train[:3%]' --name <name> > log 2>&1 &
```

Paste launches as ONE LINE — multi-line backslash blocks have been mangled by the terminal, and a
`sleep` between two launches discards the second if you Ctrl-C it.

- **Verify `grep -c "Files found" <log>` is ≥ 1 within two minutes.** Zero means delphi is
  recaching, which would overwrite your validated cache using the sparsify SAE. This is the one
  failure mode here that looks like normal progress.
- `<sae_dir>` only has to RESOLVE the hookpoint; with a cache present its weights are not read. It
  must contain a `layers.13/` child — delphi resolves hookpoints WITHOUT the wrapper prefix, so a
  directory named `model.layers.13` gives "Could not find valid path for hookpoint".
- `--log_probs` populates the per-example `probability` field, which is what makes AUROC computable.
- Run arms one at a time per card — the 70B fills an 80 GB GPU.
- `--dataset_split train[:3%]` — 1% does not contain 30M tokens.
- Judge progress by file counts and GPU%, not by the log tail: tqdm writes carriage returns, so
  `Processing items: 0it` can be stale. Training/writer logs likewise need `grep -a` (progress bars
  make grep call them binary).
- Pass each arm its OWN model name where possible, so that even a fallback recache uses the right
  weights. Note `andreayhchen/gemma2-2b-rand-all-s0` currently fails `AutoTokenizer` under
  transformers 4.56.1 (`extra_special_tokens` serialised as a list, not a dict) — use
  `google/gemma-2-2b`, whose tokenizer is identical, and rely on the `Files found` check instead.

### The end-of-run traceback is gone (if you write firing counts)

Runs used to end in `KeyError: 'firing_count'` from delphi's own `log_results`. `write_delphi_cache.py`
emits `log/hookpoint_firing_counts.pt`, so that step now completes and produces `visualize/` output.
The trailing `Engine core proc EngineCore_DP0 died unexpectedly` is just vLLM shutting down —
harmless, and it appears after all scores are on disk. Verify completion by counting instead:
`ls results/<name>/scores/detection | wc -l` should equal `ls results/<name>/explanations | wc -l`.

## SAEs

Ours are `sae_lens` checkpoints. Two options:

1. **Preferred: `write_delphi_cache.py`** — no conversion at all. Works for `full` and `resid`.
2. `convert_sae_to_sparsify.py` — **fixed 2026-08-19** (see the top of this file). It now verifies
   via `sc.encode()` vs `sae.encode()` and compares selected latents. Only needed if you want delphi
   to do its own caching. Cannot express `resid` at any point.

Either way, train with `SAE_ARCH=topk`: sparsify only offers per-token top-k, and it is also the
architecture Heap et al. used. `train_sae_res.py` tags `_topk` into the artifact name, so the two
archs cannot collide.

`check_hookpoint.py` proves delphi's `layers.13` is the same tensor as our
`blocks.13.hook_resid_post` — but note it does so on a **BOS-prefixed** sentence via `to_tokens`.
On BOS-free 256-token cache rows the two backends agree at cosine 0.9997, not 0.999992.

## Skip the model pass entirely — everything is on HF

```
andreayhchen/gemma2-2b-linearmap-saes-trained-20m
andreayhchen/gemma2-2b-linearmap-saes-rand-all-s0
    delphi_{arm}_{full,resid}_L13_writer.tar.gz            corrected results (2026-08-19)
    delphi_latents_L13_30M_{arm}_{full,resid}_writer/      corrected caches, ~300 MB each
    delphi_rand_full_abl_*_L13.tar.gz                      the defect ablation
    delphi_latents_L13_30M/                                the ORIGINAL, DEFECTIVE cache — kept as
                                                           evidence; do NOT score it
    sae_{full,resid}_k32_d73728_100M_topk_final.pt         the SAEs these came from
```

`plot_delphi_2x2.py` and `ablation_delphi.py` run off the extracted tarballs with no GPU, no model
and no SAE — `RESULTS_DIR=<extract dir>`.

## Remote judges via OpenRouter — logprobs need a patch AND the right model

delphi's OpenRouter client never requests logprobs. Patching it works — pass
`logprobs`/`top_logprobs` through, add `"provider": {"require_parameters": true}`, and convert the
JSON into the attribute-access shape `classifier._parse_logprobs` expects (a `SimpleNamespace` per
token). But the model must actually return them:

| model | result |
|---|---|
| `meta-llama/llama-3.1-70b-instruct` | DeepInfra: **0 entries**; CoreWeave: **1** (EOT only) |
| `meta-llama/llama-3.1-8b-instruct` | 9 entries ✓ |
| `openai/gpt-4o-mini` | 9 entries ✓ |
| `openai/gpt-5.x`, all `anthropic/*` | no logprobs support at all |

So the paper's own judge cannot give AUROC over OpenRouter — hence the local vLLM route. Note
`_parse_logprobs` asserts the count of "1"/"0" tokens equals `n_examples_shown`, so a judge that
emits reasoning around the bare `[1, 0, 1]` breaks the batch. Prefer terse non-reasoning models.

## Archive before stopping

`results/` is on `/dev/shm` (RAM) and the venv is on `/`; a stop rebuilds both. Tar
`explanations/`, `scores/`, `run_config.json`, `write_delphi_cache.json`, `log/`, `visualize/` and
push to HF. `hf` lives in the venv: `/root/venv312/bin/hf` (`huggingface-cli` is deprecated).
Verify sizes afterwards — a tar that silently failed on a missing member uploads as a near-empty
file, and `--exclude=<cell>/latents` on the whole directory is more robust than listing members.
