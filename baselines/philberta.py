"""Frozen PhilBERTa + one linear head per annotation layer (layer 1: NOTE; layer 2: BIBL / REF), BIO-constrained Viterbi.
The encoder is never updated: hidden states of the first sub-token of every word are cached once (CPU, fp16) and only the heads are trained.
Needs torch and transformers.  Usage, from this directory:  python philberta.py OUT_DIR [DATA_REPO]  (default ../data)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from cllg_tagging import common as bc

MODEL = "bowphs/PhilBerta"
MAXSUB = 500
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(tok, words):
    """Greedy word chunks whose sub-token count fits MAXSUB: [(start, end)]."""
    enc = [len(tok(w, add_special_tokens=False)["input_ids"]) or 1 for w in words]
    out, a, n = [], 0, 0
    for i, c in enumerate(enc):
        if n + c > MAXSUB and i > a:
            out.append((a, i))
            a, n = i, 0
        n += c
    out.append((a, len(words)))
    return out


@torch.no_grad()
def encode(tok, model, samples):
    """-> list of (n_words, 768) fp16 arrays (one per sample)."""
    out = []
    for si, (_, toks) in enumerate(samples):
        words = [t for t, _ in toks]
        parts = []
        for a, b in chunks(tok, words):
            e = tok(words[a:b], is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512).to(DEV)
            h = model(**e).last_hidden_state[0]
            wid = e.word_ids(0)
            first, seen = [], set()
            for k, w in enumerate(wid):
                if w is not None and w not in seen:
                    seen.add(w)
                    first.append(k)
            v = h[first].half().cpu().numpy()
            if len(v) < b - a:  # truncated: pad with zeros (predicted O)
                v = np.concatenate([v, np.zeros((b - a - len(v), v.shape[1]), np.float16)])
            parts.append(v)
        out.append(np.concatenate(parts))
        if si % 2000 == 0:
            print("encoded", si, "/", len(samples), flush=True)
    return out


LAY = {1: ["O", "B-NOTE", "I-NOTE"], 2: ["O", "B-BIBL", "I-BIBL", "B-REF", "I-REF"]}


def load(paths):
    out = []
    for p in paths:
        out += bc.read_layers(p)
    return out


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    F = (sys.argv[2] if len(sys.argv) > 2 else "../data") + "/final/"
    data = {"train": load([F + "train.layers.tsv"]), "dev": load([F + "dev.layers.tsv"]), "devplus": load([F + "devplus.layers.tsv"]),
            "test-md": load([F + "test-md.layers.tsv"]), "test-aph": load([F + "test-aph.layers.tsv"])}
    tok = AutoTokenizer.from_pretrained(MODEL, add_prefix_space=True)
    model = AutoModel.from_pretrained(MODEL).to(DEV).eval().half()
    H = {n: encode(tok, model, [(h, [(t[0], "O") for t in T]) for h, T in s]) for n, s in data.items()}
    del model
    torch.cuda.empty_cache()
    X = torch.from_numpy(np.concatenate(H["train"]))  # CPU fp16
    H["train"] = None  # free the per-sample copies
    report = {}
    for lay, labels in LAY.items():
        l2i = {l: i for i, l in enumerate(labels)}
        y = torch.from_numpy(np.concatenate([np.array([l2i[t[lay]] for t in T]) for _, T in data["train"]]))
        # layer 2: tokens of samples flagged unknown=1 (a note holds a bibl/ref of unknown extent) carry no label
        keep = torch.from_numpy(np.concatenate([np.full(len(T), not (lay == 2 and h.get("unknown") == "1")) for h, T in data["train"]]))
        freq = torch.bincount(y[keep], minlength=len(labels)).float() + 1
        w = (freq.sum() / freq).sqrt()
        w = (w / w.min()).to(DEV)
        head = torch.nn.Linear(768, len(labels)).to(DEV)
        opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
        lossf = torch.nn.CrossEntropyLoss(weight=w, reduction="none")
        idx_all = torch.nonzero(keep).squeeze(1)
        t0 = time.time()
        for ep in range(20):
            perm = idx_all[torch.randperm(len(idx_all))]
            for i in range(0, len(perm), 8192):
                b = perm[i:i + 8192]
                opt.zero_grad()
                lossf(head(X[b].to(DEV).float()), y[b].to(DEV)).mean().backward()
                opt.step()
        print(f"layer {lay} head trained in {time.time() - t0:.0f}s", flush=True)
        m = np.zeros((len(labels), len(labels)), bool)
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                m[i, j] = not b.startswith("I-") or (a != "O" and a[2:] == b[2:])
        start = np.array([not l.startswith("I-") for l in labels])
        for name in ("dev", "devplus", "test-md", "test-aph"):
            preds = []
            with torch.no_grad():
                for h in H[name]:
                    lp = torch.log_softmax(head(torch.tensor(h, dtype=torch.float32, device=DEV)), -1).cpu().numpy()
                    preds.append([labels[i] for i in _viterbi(lp, m, start)])
            gold = [[t[lay] for t in T] for _, T in data[name]]
            r = bc.score(gold, preds)
            report[f"layer{lay}/{name}"] = r
            print("==", f"layer{lay}", name)
            for c, v in r.items():
                print(f"{c:9s}" + "  ".join(f"{k}={x:.3f}" if isinstance(x, float) else f"{k}={x}" for k, x in v.items()), flush=True)
        json.dump({"labels": labels, "W": head.weight.detach().cpu().tolist(), "b": head.bias.detach().cpu().tolist()}, open(out / f"head{lay}.json", "w"))
    json.dump(report, open(out / "report.json", "w"), indent=1)


def _viterbi(logp, m, start):
    n, k = logp.shape
    neg = -1e9
    trans = np.where(m, 0.0, neg)
    score = logp[0] + np.where(start, 0.0, neg)
    back = np.zeros((n, k), int)
    for i in range(1, n):
        cand = score[:, None] + trans
        back[i] = cand.argmax(0)
        score = cand.max(0) + logp[i]
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    return path[::-1]


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "philberta_out")
