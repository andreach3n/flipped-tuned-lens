# Heap et al. on Pythia-1b — does the paper replicate on its own model family?

Everything in `../` tests Heap et al. (arXiv:2501.17727 / OpenReview `USyGD0eUod`,
*"Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers"*) on
**gemma-2-2b**, and finds it does **not** replicate: after fixing three conversion/BOS defects,
delphi gives trained/rand fuzz AUROC 0.781 vs 0.632 (z = 12.0). The standing caveat on that
result is **"Gemma is not Pythia."** This folder removes it.

Two arms, one layer, the paper's own model family, the paper's own training code, the paper's
own scoring pipeline and judge.

| | Heap et al. | here | why |
|---|---|---|---|
| model | Pythia 70m–6.9b | **pythia-1b** | the paper's suite |
| layer | every 2nd layer for 1b | **layer 8** of 16 | middle, on their even-layer grid; the analogue of gemma L13-of-26 |
| arms | 5 | **trained**, **re-randomized incl. embeddings** | the two the headline rests on |
| SAE | TopK, **R=64**, **k=32** | same → d_sae = 64×2048 = **131072** | §3 |
| training code | EleutherAI **sparsify** | same, **natively** | §3 ("based on [3]") |
| tokens | 100M | 48832 × 2048 = **99,991,552** | §3 |
| decoder norm | normalized each step | sparsify `normalize_decoder=True` | §3 |
| LR | unstated | sparsify's auto rule → **7.07e-5** | `2e-4/sqrt(131072/2^14)` |
| scoring | delphi, Llama-3.1-70B-Instruct-AWQ-INT4, fuzz + detection, AUROC | same | §3 |
| latents | 100 randomly sampled | **500** (`--max_latents 500`) | see "latent sampling" below |

**Training in sparsify's own format is the single most important choice here.** delphi loads
sparsify checkpoints directly, so there is no conversion step. Every delphi number this project
produced before 2026-08-19 was invalid because of `../convert_sae_to_sparsify.py` (`b_dec`
applied twice + a missing decoder-norm fold — see `../DELPHI_SETUP.md`). That entire failure
class is structurally absent from this pipeline.

## Documented deviations

1. **Corpus: openwebtext, not RedPajama.** Forced, then chosen. The paper's dataset no longer
   exists — `togethercomputer/RedPajama-Data-1T-Sample` 404s even with a valid token, as does
   `cerebras/SlimPajama-627B`; the surviving parent repo is a loading *script*, which
   `datasets>=3` will not execute. (The example command in sparsify's own README no longer
   runs, for exactly this reason.) Only community re-uploads remain. Given that RedPajama was
   already **off-distribution for Pythia** — which was pretrained on the Pile — and that every
   Gemma cell in this project used openwebtext, openwebtext makes the Pythia arms directly
   comparable to the existing 2×2. Both arms see identical text, so this moves absolute AUROC,
   not the trained-vs-random contrast the paper's claim rests on.
2. **`lr_warmup_steps` 76, not sparsify's default 1000.** 100M tokens at the default batch is
   only ~1526 steps, so the default would still be warming up two-thirds of the way through and
   then decay linearly to zero. The paper states no warmup. 76 is ~5% of steps. Rationale: do
   not hand the *"the random arm was just undertrained"* objection a free win — it already cost
   this project a full re-run cycle on Gemma, and was ruled out there empirically.
3. **500 latents, not 100.** `--max_latents N` is `torch.arange(N)` — the FIRST N latent
   indices, not the top-firing N. Since SAE latent indices are exchangeable, that *is* a random
   sample of the dictionary, and it is a superset of the paper's n. 100 latents gave standard
   errors wide enough to produce a false "no gap" read on Gemma (n=35/23), which is the specific
   mistake this raises n to avoid.

## Randomization — VERIFIED 2026-08-21, all 20 gates passed

`verify_randomization.py` is a hard gate, not a smoke test, because a partly-randomized model
trains an SAE perfectly happily and yields a complete set of plausible AUROC numbers. Record for
`rand_all` seed 0, `EleutherAI/pythia-1b`:

- **196 parameter tensors re-drawn** = 12 × 16 layers + 4. Pythia is GPT-NeoX: per layer, 2
  LayerNorms × (weight+bias) + `query_key_value` + `attention.dense` + `dense_h_to_4h` +
  `dense_4h_to_h`, each with a bias; plus `embed_in`, `final_layer_norm` (weight+bias),
  `embed_out`.
- **`embed_in` AND `embed_out` both re-drawn.** `tie_word_embeddings` is **False** on Pythia
  (it is True on Gemma), so this is the check that catches a Gemma-recipe copy-paste leaving a
  trained unembedding inside a "fully random" model.
- Moments matched: worst z_mean **3.48**, worst z_std **3.01**, both inside the 6σ band.
- Decorrelated: worst |corr(old, new)| **0.0635**, at the 1/√4096 = 0.016 noise floor scale.
- Determinism: same seed → bit-identical; different seed → 0 tensors shared. Disk round-trip
  bit-identical.
- Ordered-parameter-name hash **`245b6cc67df238e2`** (transformers 5.9.0). The draw walks
  `named_parameters()` in order, so if a transformers upgrade reorders them, the same seed gives
  a *different* model. If this hash moves, arms trained at different times are not comparable.

Activation health at layer 8, 11,264 openwebtext tokens:

| arm | \|h₈\| mean | var | max-dim-var / mean | next-token loss |
|---|---|---|---|---|
| rand_all s0 | 163.52 | 13.05 | **1.55** | **14.192** |
| trained | 58.08 | 8.83 | **879.19** | 2.957 |

- `sqrt(2048 × 13.05) = 163.5` — matches `|h|` exactly, so the random arm's per-element mean is
  ~0. Same internal-consistency signature the Gemma arm had at 317.00.
- **1.55 vs 879**: trained Pythia has massive-activation dims, the random one has none. Expected,
  and a free confirmation that randomization did what it should.
- **Loss 14.19 > ln(50304) = 10.83 is CORRECT, not a bug.** A random network is not a uniform
  predictor — it is sharply peaked and confidently wrong, which scores worse than chance. Gemma
  measured 22.6 against its own 12.45. Do not "fix" this toward chance.

## Run order

Everything except step 1 needs a GPU pod. `../DELPHI_SETUP.md` is authoritative for the delphi
install (**pin `vllm==0.10.2` / `transformers==4.56.1` / `torch==2.8.0`**, keep the delphi venv
separate from the training env, `export HF_HOME=/dev/shm/hf` before anything downloads).

```bash
# 0. clone + bootstrap. DO NOT run the repo-root setup.sh -- it installs sae_lens +
#    transformer_lens for the gemma pipeline, neither of which anything here imports, and
#    installing sae_lens is the documented cause of torch outrunning the host driver.
git clone https://github.com/<you>/flipped-tuned-lens.git && cd flipped-tuned-lens
export HF_HOME=/dev/shm/hf && mkdir -p $HF_HOME    # BEFORE anything downloads
export HF_TOKEN=...            # write access
export WANDB_API_KEY=...       # omit and training falls back to WANDB_MODE=offline
export WANDB_PROJECT=pythia1b-heap-replication
export HF_REPO=andreayhchen/pythia-1b-heap-replication
bash SAE_linearmaps/randomized/pythia/setup_pythia.sh
cd SAE_linearmaps/randomized/pythia

# 1. materialize the random arm, GATE it, then push it to the repo ROOT (see hf_push.py for
#    why the model goes at the root and not in a subfolder)
VARIANT=rand_all INIT_SEED=0 OUT_DIR=/dev/shm/pythia1b_rand_s0 python -u randomize_pythia.py
VARIANT=rand_all INIT_SEED=0 OUT_DIR=/dev/shm/pythia1b_rand_s0 python -u verify_randomization.py
LOCAL=/dev/shm/pythia1b_rand_s0 PATH_IN_REPO= python -u hf_push.py

# 2. train both arms, ONE PER GPU, simultaneously. Independent runs, so this is
#    CUDA_VISIBLE_DEVICES pinning, NOT torchrun/DDP. Stagger the second launch until the first
#    reaches "Shuffling dataset" -- tokenization is CPU-bound and runs before training.
CUDA_VISIBLE_DEVICES=0 ARM=trained nohup bash train_saes.sh > /dev/shm/logs/launch_trained.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 ARM=rand RAND_MODEL=/dev/shm/pythia1b_rand_s0 nohup bash train_saes.sh > /dev/shm/logs/launch_rand.log 2>&1 &

# 3. SAE health gate -- BEFORE spending judge time
SAE_DIR=/dev/shm/saes/pythia1b_trained_L8_R64_k32_100M ARM=trained python -u check_saes.py
SAE_DIR=/dev/shm/saes/pythia1b_rand_L8_R64_k32_100M    ARM=rand    python -u check_saes.py

# 4. delphi, one arm at a time per 80GB card (the 70B judge fills it). ONE LINE each.
cd /dev/shm/delphi_run
CUDA_VISIBLE_DEVICES=0 $V -m delphi EleutherAI/pythia-1b /dev/shm/saes/pythia1b_trained_L8_R64_k32_100M --hookpoints layers.8 --scorers fuzz detection --log_probs --max_latents 500 --n_tokens 30000000 --num_gpus 1 --dataset_repo Skylion007/openwebtext --dataset_split 'train[:3%]' --name pythia1b_trained_L8
CUDA_VISIBLE_DEVICES=0 $V -m delphi /dev/shm/pythia1b_rand_s0 /dev/shm/saes/pythia1b_rand_L8_R64_k32_100M --hookpoints layers.8 --scorers fuzz detection --log_probs --max_latents 500 --n_tokens 30000000 --num_gpus 1 --dataset_repo Skylion007/openwebtext --dataset_split 'train[:3%]' --name pythia1b_rand_L8

# 5. report
RESULTS_DIR=/dev/shm/delphi_run/results python -u report_auroc.py
```

`--log_probs` is what populates the per-example `probability` field, and AUROC is not computable
without it. It requires a **locally served** judge: over OpenRouter, Llama-3.1-70B returns no
logprobs (DeepInfra) or only the final token's (CoreWeave).

## Known risks, in the order they are likely to bite

1. **Dead latents.** sparsify's `auxk_alpha` defaults to **0**, so nothing revives dead latents,
   and 131072 latents in ~1526 steps is a cold start. delphi only scores latents with ≥200
   activations, so a dead-heavy dictionary silently changes *which* latents get sampled, per arm.
   `check_saes.py` gates on this. If alive% is low, rerun **both** arms with `AUXK=0.03125`.
2. **VRAM.** sparsify's `FusedEncoder` is fused in the backward pass only — its forward
   materializes the whole `(N, 131072)` pre-activation matrix. At batch 32 × ctx 2048 that is
   **34 GB in fp32** before anything else. `MICRO_ACC_STEPS=8` chunks it to ~4.3 GB without
   changing the effective batch, so the run is mathematically identical. Raise it if you OOM.
3. **delphi silently re-caching.** `grep -c "Files found" <log>` must be ≥1 within two minutes.
4. **n = 1 random seed.** Same caveat as every Gemma cell. Seeds 1, 2 are `INIT_SEED=1|2`.

## Files

| file | what it does |
|---|---|
| `setup_pythia.sh` | fresh-pod bootstrap **instead of** the repo-root `setup.sh`; gates torch/CUDA and the sparsify identity |
| `requirements.txt` | the small training-side stack (`eai-sparsify`, **not** `sparsify`) |
| `randomize_pythia.py` | the randomization itself + materializes an arm as an HF checkpoint |
| `verify_randomization.py` | **the gate** — 8 checks, exit 1 on any failure |
| `train_saes.sh` | sparsify launch per arm; re-runs the gate before training the random arm |
| `check_saes.py` | post-training SAE health: FVU, L0, alive%, head-latent firing rate |
| `report_auroc.py` | AUROC per latent, clustered by latent, ±1 SE + figure |
| `hf_push.py` | upload a folder into one repo; guards the two arms against overwriting |

## The HF repo

One repo, with the **randomized model at its root** — `AutoModel` and delphi both take a bare
repo id and neither CLI can express `subfolder=`, so the model has to be at the root to be
loadable by name. Everything else lives in subfolders:

```
andreayhchen/pythia-1b-heap-replication/
  config.json, model.safetensors, tokenizer.json   <- IS the randomized pythia-1b (seed 0)
  saes/pythia1b_trained_L8_R64_k32_100M/layers.8/
  saes/pythia1b_rand_L8_R64_k32_100M/layers.8/
  logs/train_{trained,rand}_L8.log
  delphi/                                          <- tar the results here before stopping the pod
```

Consequence worth knowing: the *trained* arm's SAEs live in a repo whose root is the *random*
model. Deliberate — one place for every artifact of the comparison.
