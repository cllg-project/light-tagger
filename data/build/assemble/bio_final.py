"""Build the FINAL datasets (run from cllg-encoding-tagger/scripts):
  1. apply the review-app corrections (out/bio/flags2.jsonl) -> *.final.tsv
  2. APh_Corpus IOB -> layered samples (REFSCOPE and REFAUWORK -> BIBL, adjacent ones merged; everything else O)
  3. assemble ../final/{train,dev,devplus,test-md,test-aph}.layers.tsv and the per-layer flat files {split}.l1.tsv / .l2.tsv
Originals are never overwritten."""
import glob
import re
import shutil
import sys
from pathlib import Path

from cllg_tagging import common as bc
from cllg_tagging import review as rv

ROOT = Path(__file__).resolve().parents[2]  # the data/ folder
SRC = ROOT / "intermediate"  # intermediate datasets of the original working tree (not shipped, see README)
FLAGS = ROOT / "annotations" / "flags2.jsonl"
FINAL = ROOT / "final"
APH = ROOT / "APh_Corpus"
MAP = {"REFSCOPE": "BIBL", "REFAUWORK": "BIBL"}


def finals():
    srcs = {"bio-layers-md-test.tsv": SRC / "bio-layers-md-test.final.tsv", "bio-devplus.tsv": SRC / "bio-devplus.final.tsv",
            "bio-devplus2.tsv": SRC / "bio-devplus2.final.tsv", "bio-layers-d-dev.fixed.tsv": SRC / "bio-layers-d-dev.final.tsv",
            "bio-layers-train.fixed.tsv": SRC / "bio-layers-train.final.tsv", "out/bio/pasted.tsv": SRC / "bio-pasted.final.tsv"}
    for name, dst in srcs.items():
        src = SRC / name if not name.startswith("out/") else Path(__file__).parent / name
        rv.apply_flags(str(src), str(FLAGS), str(dst))


def aph_samples(files, split, maxn=150):
    out = []
    for f in sorted(files):
        rows = [l.rstrip("\n").split("\t") for l in open(f, encoding="utf-8", errors="replace") if l.strip()]
        if not rows or any(len(r) != 3 for r in rows):
            continue  # unannotated (token, POS) file
        toks, prev = [], None
        for tok, pos, tag in rows:
            raw = tag[2:] if tag != "O" else None
            cls = MAP.get(raw)
            # author/work followed directly by its scope (or scope fragments in a row) = ONE bibl; a REFAUWORK after a scope starts a new one
            cont = cls and (tag.startswith("I-") or (raw == "REFSCOPE" and prev in ("REFAUWORK", "REFSCOPE")))
            lab = "O" if not cls else ("I-" if cont else "B-") + cls
            prev = raw
            pieces = bc.TOKEN.findall(tok) or [tok]
            toks += [(p, lab if k == 0 else ("I-" + lab[2:] if lab != "O" else "O"), pos) for k, p in enumerate(pieces)]
        sents, cur = [], []
        for t in toks:
            cur.append(t)
            if t[2] == "SENT" or t[0] in (".", "?", "!"):
                sents.append(cur)
                cur = []
        if cur:
            sents.append(cur)
        win, k = [], 0
        for s in sents + [None]:
            if s is None or (win and len(win) + len(s) > maxn):
                while len(win) > maxn:  # one very long sentence: cut it, never inside a span
                    c = maxn
                    while c < len(win) and win[c][1].startswith("I-"):
                        c += 1
                    out.append((f"id=aph-{Path(f).stem}-{k} src=aph split={split}", [(t, "O", l) for t, l, _ in win[:c]]))
                    win, k = win[c:], k + 1
                if win:
                    out.append((f"id=aph-{Path(f).stem}-{k} src=aph split={split}", [(t, "O", l) for t, l, _ in win]))
                    k += 1
                win = []
            if s is not None:
                win += s
    for _, T in out:  # an I- that follows nothing of its class (window opened inside a span, or a source quirk) starts a span
        for i, (tok, l1, l2) in enumerate(T):
            if l2.startswith("I-") and (i == 0 or T[i - 1][2][2:] != l2[2:] or T[i - 1][2] == "O"):
                T[i] = (tok, l1, "B-" + l2[2:])
    return out


def write_layers_file(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for h, T in rows:
            bc.write_layers(fh, h if isinstance(h, str) else " ".join(f"{k}={v}" for k, v in h.items()), T)


def assemble():
    FINAL.mkdir(exist_ok=True)
    tr = [str(APH / "goldset/iob/*.txt"), str(APH / "devset/iob/*.txt")]
    aph_train = aph_samples([f for g in tr for f in glob.glob(g)], "train")
    aph_test = aph_samples(glob.glob(str(APH / "testset/iob/*.txt")), "test")
    nb = lambda S: sum(t[2] == "B-BIBL" for _, T in S for t in T)  # noqa: E731
    print(f"APh train: {len(aph_train)} samples, {nb(aph_train)} BIBL spans | APh test: {len(aph_test)} samples, {nb(aph_test)} BIBL spans")
    L = bc.read_layers
    train = L(SRC / "bio-layers-train.final.tsv") + L(SRC / "bio-layers-refmid-train.tsv") + L(SRC / "bio-devplus2.final.tsv") + L(SRC / "bio-pasted.final.tsv")
    train = [(h, T) for h, T in train] + aph_train
    sets = {"train": train, "dev": L(SRC / "bio-layers-d-dev.final.tsv"), "devplus": L(SRC / "bio-devplus.final.tsv"),
            "test-md": L(SRC / "bio-layers-md-test.final.tsv"), "test-aph": aph_test}
    for name, rows in sets.items():
        write_layers_file(FINAL / f"{name}.layers.tsv", rows)
        for lay, tag in ((1, "l1"), (2, "l2")):
            n = 0
            with open(FINAL / f"{name}.{tag}.tsv", "w", encoding="utf-8") as fh:
                for h, T in rows:
                    hd = h if isinstance(h, dict) else dict(kv.split("=", 1) for kv in h[2:].split(" ") if "=" in kv) if False else h
                    if lay == 2 and name == "train" and isinstance(h, dict) and h.get("unknown") == "1":
                        continue  # a note holds a bibl/ref of unknown extent: layer 2 unlabelled there
                    hs = h if isinstance(h, str) else " ".join(f"{k}={v}" for k, v in h.items())
                    bc.write_sample(fh, hs, [(t[0], t[lay]) for t in T])
                    n += 1
        print(f"{name}: {len(rows)} samples, NOTE {sum(t[1] == 'B-NOTE' for _, T in rows for t in T)}, "
              f"BIBL {sum(t[2] == 'B-BIBL' for _, T in rows for t in T)}, REF {sum(t[2] == 'B-REF' for _, T in rows for t in T)}")


if __name__ == "__main__":
    {"finals": finals, "assemble": assemble}[sys.argv[1]]()
