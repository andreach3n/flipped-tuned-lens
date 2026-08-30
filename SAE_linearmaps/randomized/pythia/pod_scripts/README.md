# Pod-side launchers, kept for provenance

These two ran on the RunPod box (`/dev/shm/`) and were never in the repo, so they would have
died with the pod. They are recorded verbatim, not cleaned up — the point is to show exactly
what produced the `pythia1b_{trained,rand}_base_L8` cells (the non-temporal control), including
every `TSAE_*` environment variable, which is the part that is impossible to reconstruct from
the results alone.

`queue_base_trained.sh` — waits for the smoke run to exit, then scores the non-temporal control
with `TSAE_SELECT=500,500` (the Matryoshka column permutation: 500 high-level latents to ids
0..499, 500 low-level to 500..999) and writes the permutation to `TSAE_MAP_OUT`.

`auto_v2.sh` — the recovery path. delphi's constructor asserts when a latent is too dense to find
non-activating windows; if the first pass tripped it, this relaunches with
`TSAE_MAX_DENSITY=0.05` and `TSAE_FIRING_COUNTS` pointed at the firing counts the first pass
already wrote, so the density-aware selection can skip the offending latents. It also checks the
counts file exists first, because a missing one means caching never finished and a rerun would be
pointless.

CAVEAT: the hardcoded pids (16819, 15893) in `auto_v2.sh` were that session's processes. The
script is a record, not something to rerun as-is.
