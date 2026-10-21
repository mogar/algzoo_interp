import torch as th
from alg_zoo.tasks import task_2nd_argmax


@th.no_grad()
def predict(model, x, batch_size=2**16):
    """Model's predicted position (argmax logit) for each input."""
    preds = []
    for batch in x.split(batch_size):
        batch = batch.to(device=model.device, dtype=model.dtype)
        preds.append(model(batch).argmax(-1))
    return th.cat(preds).to(x.device)


def accuracy(model, x, task=task_2nd_argmax, batch_size=2**16):
    """Fraction of inputs where the model's argmax logit matches the task target."""
    return (predict(model, x, batch_size) == task(x)).float().mean().item()


def accuracy_by_top2_positions(model, x):
    """Accuracy split by (argmax position, 2nd-argmax position).

    Returns (acc, counts), each [seq_len, seq_len] indexed [argmax_pos, second_pos];
    acc is nan where there are no inputs (including the diagonal).
    """
    seq_len = x.shape[1]
    order = x.argsort(-1, descending=True)
    cell = (order[:, 0] * seq_len + order[:, 1]).cpu()
    correct = (predict(model, x) == order[:, 1]).float().cpu()
    counts = th.bincount(cell, minlength=seq_len**2).float()
    hits = th.bincount(cell, weights=correct, minlength=seq_len**2)
    acc = hits / counts
    return acc.reshape(seq_len, seq_len), counts.reshape(seq_len, seq_len)


def accuracy_by_top2_gap(model, x, edges):
    """Accuracy binned by the gap between the largest and 2nd-largest values.

    edges: increasing bin edges [n_bins + 1]. Returns (acc, counts), each [n_bins];
    inputs outside the edges are dropped.
    """
    top2 = x.topk(2, dim=-1).values
    gap = (top2[:, 0] - top2[:, 1]).cpu()
    correct = (predict(model, x) == task_2nd_argmax(x)).float().cpu()
    edges = th.as_tensor(edges, dtype=gap.dtype)
    bins = th.bucketize(gap, edges, right=True) - 1
    keep = (bins >= 0) & (bins < len(edges) - 1)
    counts = th.bincount(bins[keep], minlength=len(edges) - 1).float()
    hits = th.bincount(bins[keep], weights=correct[keep], minlength=len(edges) - 1)
    return hits / counts, counts
