import torch as th


def randn_inputs(n, seq_len, *, seed=0, device="cpu", dtype=th.float32):
    """Standard normal inputs, matching the training distribution."""
    gen = th.Generator(device=device).manual_seed(seed)
    return th.randn(n, seq_len, generator=gen, device=device, dtype=dtype)
