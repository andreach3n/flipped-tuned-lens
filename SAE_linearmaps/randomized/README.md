# Trained vs randomly-initialized Gemma — can skip-embed SAEs tell them apart?

The experiment: re-randomize every Gemma-2-2b weight from per-tensor moment-matched
Gaussians (= Heap et al. arXiv:2501.17727's "Re-randomized incl. embeddings" arm) and ask
whether the skip-embed decomposition separates trained from random where standard SAE
metrics don't. Model arms are built by `VARIANT`/`INIT_SEED` in `../activations.py`;
artifacts live in per-arm HF repos (`…-trained-20m` / `…-rand-all-s0`), guarded by
`../hf_io.py` against cross-arm overwrites. All SAEs matched: 16k dict, k=64, 20M tokens.

## Results so far (seed 0)

- **Tier 0 — NULL.** `context_var.py`: token-identity lookup explains ~28% of centred
  L13 variance for BOTH arms (0.2743 vs 0.2877 at min_count>=30). Random attention mixes
  context vigorously; raw variance decomposition cannot separate the arms.
- **Tier 1 — POSITIVE.** `../eval_fvu.py` + `gauss_null.py`: % below own matched-Gaussian
  floor (centred FVU) — trained/plain **47.1%**, trained/skip-embed **37.3%**,
  random/plain **14.5%**, random/skip-embed **−3.2% ≈ 0**. A plain SAE finds real
  structure in a random net; ALL of it is token-static (the two random cells differ only
  by subtracting P[tok]). Raw FVU alone shows ~no difference (2.1x vs 2.0x) — the nulls
  are what reveal the effect. Figures: `plot_trained_vs_random.py` (own-target FVU,
  headline) and `plot_trained_vs_random_h.py` (shared-target h, composite; floors derived
  via the exact identity FVU_h = FVU_r · Var(r)/Var(h)).
- **Caveats:** n=1 random seed; reconstruction only; hybrid/outbias untested on random.

## Possibility-2 retrain (is the auto-interp non-replication just undertraining?)

Heap et al. trained TopK SAEs at k=32, expansion factor 16–128, 100M RedPajama tokens; our
20M-token, ~7x-dict SAEs sit outside that range on both axes. `launch_replication.sh` retrains
both arms at the paper recipe — k=32, d_sae=73728 (R=32), 100M tokens — plus a 20M LR sweep
{1e-4, 4e-4, 1e-3} on the random arm (the 4e-4 point falls out of the 100M run's t20M
milestone). train_sae_res.py now tags non-default configs into filenames
(`sae_full_k32_d73728_100M_final.pt`) so these coexist with the 20M/16k artifacts in the same
per-arm repos, and pushes frozen milestone ckpts every 20M tokens (`…_t20M.pt`, …) so
auto-interp can be scored **vs training tokens**: if the random arm's score is still climbing
at 100M, undertraining was the story; if it is flat, possibility 3 gets more weight. Eval
scripts reach a tagged run via `SUFFIX=_d73728_100M` (+ `K=32`).

## Pipeline files

| file | what it does |
|---|---|
| `context_var.py` | Tier 0: contextual variance fraction ρ, streaming per-token lookup table |
| `gauss_null.py` | matched-covariance Gaussian floor: identical SAE trained on N(μ,Σ) |
| `plot_trained_vs_random{,_h}.py` | Tier-1 figures (hardcoded verified numbers + provenance) |
| `autointerp_common.py` | pure-python core: sampling, prompts, AUROC (unit-tested) |
| `autointerp_collect.py` | pod: PASS=freq → sample → collect → finalize (example blocks for all metrics) |
| `autointerp_explain.py` | Mac: one explanation per latent (OpenAI Batch) |
| `autointerp_score.py` | Mac: detection+fuzzing → per-latent AUROC → 2x2 figure |

The abstractness rubric runs on the SAME latents via `../judge_features.py` with
`BLIND_FILE=autointerp_blind.json KEY_FILE=autointerp_key.json OUT_TAG=_tvr`.

## Running (scripts live here; shared modules one level up)

Pod, per arm (usual `HF_TOKEN`/`HF_HOME`, arm env as for `../eval_fvu.py`):

    PASS=freq    VARIANT=trained  HF_REPO=<trained-20m> OUT_DIR=... python -u randomized/autointerp_collect.py
    PASS=freq    VARIANT=rand_all INIT_SEED=0 HF_REPO=<rand-s0> OUT_DIR=... python -u randomized/autointerp_collect.py
    PASS=sample  HF_REPO=<trained-20m> python -u randomized/autointerp_collect.py    # read the plan!
    PASS=collect (per arm, as freq)
    PASS=finalize HF_REPO=<trained-20m> python -u randomized/autointerp_collect.py

Mac (`OPENAI_API_KEY` in the gitignored `.env` — key NEVER on the pod):

    python autointerp_explain.py estimate|pilot|submit|collect
    python autointerp_score.py  estimate|pilot|submit|collect|analyze

Pre-registered predictions: trained cells high AUROC; random/plain high (reproduces Heap
et al. on Gemma); random/skip-embed → ~0.5 if Tier-1's "nothing left" is right.

Protocol follows Paulo et al. (2024) / EleutherAI delphi; deliberate deviations (OpenAI
judge for rubric-validation consistency, fuzz negatives on distractor windows, ≤2 peak
examples per document) are documented in the code.
