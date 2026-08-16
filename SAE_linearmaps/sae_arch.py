"""Which sae_lens SAE architecture a run uses, and how to rebuild one from a saved checkpoint.

Two architectures are in play and they are NOT interchangeable:

  batchtopk  (our default, everything through 2026-08)  BatchTopK keeps the top k*n_tokens
             pre-activations across the WHOLE batch, so a token's firing set depends on what else
             is in the batch. Verified: sae_lens's BatchTopK.forward has no self.training branch,
             so this holds at eval too.
  topk       per-token top-k. This is what Heap et al. arXiv:2501.17727 used (via EleutherAI
             sparsify), and it is the only sparsity rule sparsify -- hence delphi -- can express.
             Train with this when the point is to meet their pipeline on its own terms.

`load_sae` infers the class from the SAVED cfg rather than trusting an env var, because a
checkpoint loaded with the wrong class would mis-apply the sparsity rule and produce plausible,
wrong activations rather than an error.
"""
import sae_lens

ARCHS = ("batchtopk", "topk")
_CFG = {"batchtopk": "BatchTopKTrainingSAEConfig", "topk": "TopKTrainingSAEConfig"}


def make_sae(arch, **cfg_kwargs):
    """Fresh training SAE of the named architecture. cfg_kwargs are the usual d_in/d_sae/k/..."""
    if arch not in ARCHS:
        raise ValueError(f"SAE_ARCH={arch!r} not in {ARCHS}")
    cfg_cls = getattr(sae_lens, _CFG[arch])
    sae_cls = getattr(sae_lens, _CFG[arch].replace("Config", ""))
    return sae_cls(cfg_cls(**cfg_kwargs))


def load_sae(ckpt):
    """Rebuild a training SAE from a checkpoint dict, using the class its own cfg names."""
    cfg = ckpt["cfg"]
    sae_cls = getattr(sae_lens, type(cfg).__name__.replace("Config", ""))
    sae = sae_cls(cfg)
    sae.load_state_dict(ckpt["sae"])
    return sae


def arch_of(ckpt):
    """'batchtopk' | 'topk' for a loaded checkpoint, for logging and provenance."""
    name = type(ckpt["cfg"]).__name__
    for a, c in _CFG.items():
        if c == name:
            return a
    return name
