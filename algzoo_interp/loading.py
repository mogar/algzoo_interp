"""Load AlgZoo models, caching checkpoints locally under weights/.

alg_zoo fetches every checkpoint from GCS on each call; these wrappers download
each file once and then load it from disk.
"""

import os
from pathlib import Path

import blobfile as bf
from alg_zoo import handcrafted_2nd_argmax
from alg_zoo.architectures import DistRNN
from alg_zoo.loading import load_dist_rnn
from alg_zoo.zoo import ZOO_BASEPATH, best_of_5_seed_2nd_argmax

WEIGHTS_DIR = Path(
    os.getenv("ALGZOO_WEIGHTS_DIR", Path(__file__).resolve().parent.parent / "weights")
)

EXAMPLE_2ND_ARGMAX_PATH = "gs://arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth"

# (hidden_size, seq_len) pairs available in the zoo, from best_of_5_seed_2nd_argmax
ZOO_2ND_ARGMAX_GRID = [
    (h, n) for h in [2, 3, 4, 5, 6, 8, 16, 32] for n in [2, 3, 4, 5, 6, 8, 10]
]


def cached_path(remote_path: str) -> str:
    """Return a local copy of a gs:// checkpoint, downloading it if needed."""
    local_path = WEIGHTS_DIR / remote_path.removeprefix("gs://")
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        prev_auth = os.getenv("BLOBFILE_FORCE_GOOGLE_ANONYMOUS_AUTH")
        try:
            os.environ["BLOBFILE_FORCE_GOOGLE_ANONYMOUS_AUTH"] = "1"
            bf.copy(remote_path, str(tmp_path), overwrite=True)
        finally:
            if prev_auth is None:
                os.environ.pop("BLOBFILE_FORCE_GOOGLE_ANONYMOUS_AUTH", None)
            else:
                os.environ["BLOBFILE_FORCE_GOOGLE_ANONYMOUS_AUTH"] = prev_auth
        tmp_path.rename(local_path)
    return str(local_path)


def load_2nd_argmax(hidden_size, seq_len, *, n_seqs=None, seed=None) -> DistRNN:
    """Cached equivalent of alg_zoo.zoo_2nd_argmax."""
    if (hidden_size, seq_len) not in ZOO_2ND_ARGMAX_GRID:
        raise ValueError(f"No zoo 2nd argmax model with shape {(hidden_size, seq_len)}")
    if n_seqs is None:
        n_seqs = 2**30
    if seed is None:
        seed = best_of_5_seed_2nd_argmax(hidden_size=hidden_size, seq_len=seq_len)
    remote = f"{ZOO_BASEPATH}/2nd_argmax_{hidden_size}_{seq_len}_{n_seqs}_{seed}.pth"
    model = load_dist_rnn(cached_path(remote), bias=False)
    assert (model.hidden_size, model.seq_len) == (hidden_size, seq_len)
    return model.eval()


def load_example_2nd_argmax() -> DistRNN:
    """Cached equivalent of alg_zoo.example_2nd_argmax (hidden 16, seq_len 10)."""
    return load_dist_rnn(cached_path(EXAMPLE_2ND_ARGMAX_PATH), bias=False).eval()


def add_model_args(parser):
    """Add the standard model-selection flags to an argparse parser."""
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--seed", type=int, help="training seed 0-4 (default: best of 5)")
    parser.add_argument("--n-seqs", type=int, help="checkpoint (default: fully trained)")
    parser.add_argument("--example", action="store_true", help="blog post 16x10 model")
    parser.add_argument("--handcrafted", type=int, choices=[2, 3, 10])


def load_from_args(args) -> tuple[DistRNN, str]:
    """Load the model selected by add_model_args flags; returns (model, results name)."""
    if args.example:
        return load_example_2nd_argmax(), "example_16x10"
    if args.handcrafted:
        return handcrafted_2nd_argmax(args.handcrafted).eval(), f"handcrafted_{args.handcrafted}"
    if args.hidden is None or args.seq_len is None:
        raise SystemExit("pass --hidden and --seq-len, or --example, or --handcrafted")
    name = f"2nd_argmax_{args.hidden}x{args.seq_len}"
    if args.seed is not None:
        name += f"_seed{args.seed}"
    if args.n_seqs is not None:
        name += f"_nseqs{args.n_seqs}"
    return load_2nd_argmax(args.hidden, args.seq_len, n_seqs=args.n_seqs, seed=args.seed), name
