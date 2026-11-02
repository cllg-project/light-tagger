"""Train / evaluate the CRF (variant A). Run with the scratchpad venv (python-crfsuite).
Usage: bio_crf.py train OUT.crfsuite TRAIN.tsv [more.tsv ...] | eval MODEL.crfsuite TEST.tsv | write..."""
import sys
import time

import pycrfsuite

from . import common as bc
from . import features as bf


def load(paths):
    X, Y = [], []
    for p in paths:
        for _, toks in bc.read_samples(p):
            X.append(bf.sent_feats([t for t, _ in toks]))
            Y.append([l for _, l in toks])
    return X, Y


def train(out, paths, c1=0.1, c2=0.01, iters=150):
    tr = pycrfsuite.Trainer(verbose=False)
    n = 0
    for p in paths:  # stream: do not keep the python feature lists in memory
        for _, toks in bc.read_samples(p):
            tr.append(bf.sent_feats([t for t, _ in toks]), [l for _, l in toks])
            n += 1
    tr.set_params({"c1": c1, "c2": c2, "max_iterations": iters, "feature.possible_transitions": True})
    t = time.time()
    tr.train(out)
    print(f"trained on {n} samples in {time.time() - t:.0f}s -> {out}")


def predict(model, X):
    tg = pycrfsuite.Tagger()
    tg.open(model)
    return [tg.tag(x) for x in X]


def evaluate(model, path, label=""):
    X, Y = load([path])
    res = bc.score(Y, predict(model, X))
    print(f"== {label or path}")
    for c, r in res.items():
        print(f"{c:9s}" + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(sys.argv[2], sys.argv[3:])
    elif cmd == "eval":
        for p in sys.argv[3:]:
            evaluate(sys.argv[2], p)
