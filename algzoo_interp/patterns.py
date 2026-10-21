"""Activation-pattern analysis.

With ReLU and no biases, fixing which neurons are active at each step makes
the model exactly linear in x. Each pattern (a [seq_len, hidden] boolean mask)
defines a polyhedral cone of inputs and one linear map from x to the logits.
"""

from dataclasses import dataclass

import torch as th

from .hooked import RNNCache, get_weights


@dataclass
class PatternCounts:
    patterns: th.Tensor  # [n_patterns, seq_len, hidden] bool, most common first
    counts: th.Tensor  # [n_patterns]
    inverse: th.Tensor  # [batch], index into patterns for each input

    @property
    def coverage(self) -> th.Tensor:
        """Cumulative fraction of inputs covered by the top-k patterns."""
        return self.counts.cumsum(0) / self.counts.sum()


def count_patterns(cache: RNNCache, steps=None) -> PatternCounts:
    """Group inputs by activation pattern, optionally restricted to some steps."""
    mask = cache.mask.cpu()
    if steps is not None:
        mask = mask[:, list(steps)]
    flat = mask.flatten(1)
    patterns, inverse, counts = th.unique(flat, dim=0, return_inverse=True, return_counts=True)
    order = counts.argsort(descending=True)
    rank = th.empty_like(order)
    rank[order] = th.arange(len(order))
    return PatternCounts(
        patterns=patterns[order].reshape(-1, *mask.shape[1:]),
        counts=counts[order],
        inverse=rank[inverse],
    )


def linear_map(model, pattern) -> th.Tensor:
    """Exact linear map M [n_out, seq_len] with logits = M @ x for inputs in this pattern.

    pattern: [seq_len, hidden] bool mask of active neurons.
    Uses h_t = A_t x with A_t = D_t (W_hh A_{t-1} + W_ih e_t^T), D_t = diag(pattern[t]).
    """
    w = get_weights(model)
    pattern = pattern.to(device=w.W_ih.device, dtype=w.W_ih.dtype)
    seq_len, hidden_size = pattern.shape
    A = w.W_ih.new_zeros(hidden_size, seq_len)
    for t in range(seq_len):
        A = w.W_hh @ A
        A[:, t] += w.W_ih
        A = pattern[t, :, None] * A
    return w.W_out @ A


def activity(cache: RNNCache) -> th.Tensor:
    """Fraction of inputs where each neuron is active at each step: [seq_len, hidden]."""
    return cache.mask.float().mean(0)
