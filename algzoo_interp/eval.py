import torch as th
from alg_zoo.tasks import task_2nd_argmax


@th.no_grad()
def accuracy(model, x, task=task_2nd_argmax, batch_size=2**16):
    """Fraction of inputs where the model's argmax logit matches the task target."""
    correct = 0
    for batch in x.split(batch_size):
        batch = batch.to(device=model.device, dtype=model.dtype)
        correct += (model(batch).argmax(-1) == task(batch)).sum().item()
    return correct / len(x)
