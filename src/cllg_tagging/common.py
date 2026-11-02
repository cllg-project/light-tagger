"""Shared BIO helpers: tokenise a labelled character string, write/read TSV, extract entities, score."""
import re
from collections import Counter

TOKEN = re.compile(r"\w+|[^\w\s]")
CLASSES = ("NOTE", "BIBL", "NOTEBIBL", "NOTEREF", "REF")


def tokenize(text, marks):
    """text: str; marks: per-char None or (class, span id). Tokens never cross a mark boundary.
    Returns [(token, tag)] with B-/I-class tags (a new span id starts a new B-)."""
    out, prev = [], None
    for m in TOKEN.finditer(text):
        start = m.start()
        cuts = [i for i in range(start + 1, m.end()) if marks[i] != marks[i - 1]]
        for a, b in zip([start] + cuts, cuts + [m.end()]):
            mk = marks[a]
            if mk is None:
                out.append((text[a:b], "O"))
                prev = None
            else:
                out.append((text[a:b], ("I-" if mk == prev else "B-") + mk[0]))
                prev = mk
    return out


def write_sample(f, header, toks):
    f.write(f"# {header}\n# text={' '.join(t for t, _ in toks)}\n")
    f.writelines(f"{t}\t{l}\n" for t, l in toks)
    f.write("\n")


def read_samples(path):
    """-> [(header dict, [(token, tag)])]"""
    out, hdr, toks = [], {}, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# text="):
            continue
        if line.startswith("# "):
            hdr = dict(kv.split("=", 1) for kv in line[2:].split(" ") if "=" in kv)
        elif line:
            toks.append(tuple(line.split("\t")))
        elif toks:
            out.append((hdr, toks))
            hdr, toks = {}, []
    if toks:
        out.append((hdr, toks))
    return out


def entities(tags):
    """Strict BIO -> set of (start, end, class); a stray I- starts a span only if `lenient`."""
    ents, cur = set(), None
    for i, t in enumerate(list(tags) + ["O"]):
        if cur and not (t.startswith("I-") and t[2:] == cur[0]):
            ents.add((cur[1], i, cur[0]))
            cur = None
        if t.startswith("B-"):
            cur = (t[2:], i)
    return ents


def score(gold_seqs, pred_seqs):
    """Entity-level P/R/F1 per class + micro/macro over classes present in gold."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold_seqs, pred_seqs):
        G = {(i, e) for i, e in enumerate(entities(g))} and entities(g)
        P = entities(p)
        for e in G & P:
            tp[e[2]] += 1
        for e in P - G:
            fp[e[2]] += 1
        for e in G - P:
            fn[e[2]] += 1
    res, classes = {}, sorted(set(tp) | set(fp) | set(fn))
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        res[c] = dict(p=p, r=r, f=2 * p * r / (p + r) if p + r else 0.0, n=tp[c] + fn[c])
    T, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p, r = T / (T + FP) if T + FP else 0.0, T / (T + FN) if T + FN else 0.0
    res["micro"] = dict(p=p, r=r, f=2 * p * r / (p + r) if p + r else 0.0, n=T + FN)
    present = [c for c in classes if res[c]["n"]]
    res["macro"] = dict(f=sum(res[c]["f"] for c in present) / max(1, len(present)))
    return res


# ---- layered (nested) annotation: layer 1 = NOTE spans, layer 2 = BIBL / REF spans; the two may overlap -------------
def tokenize2(text, marks):
    """marks: per-char None or (m1, m2) with m = None or (class, span id). Returns [(token, tag1, tag2)]."""
    out, prev = [], (None, None)
    for m in TOKEN.finditer(text):
        start = m.start()
        cuts = [i for i in range(start + 1, m.end()) if marks[i] != marks[i - 1]]
        for a, b in zip([start] + cuts, cuts + [m.end()]):
            cur = marks[a] or (None, None)
            tags = []
            for k in (0, 1):
                mk = cur[k]
                tags.append("O" if mk is None else ("I-" if mk == prev[k] else "B-") + mk[0])
            out.append((text[a:b], tags[0], tags[1]))
            prev = cur
    return out


def write_layers(f, header, toks):
    f.write(f"# {header}\n# text={' '.join(t[0] for t in toks)}\n")
    f.writelines(f"{t[0]}\t{t[1]}\t{t[2]}\n" for t in toks)
    f.write("\n")


def read_layers(path):
    """-> [(header dict, [(token, tag1, tag2)])]"""
    out, hdr, toks = [], {}, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# text="):
            continue
        if line.startswith("# "):
            hdr = dict(kv.split("=", 1) for kv in line[2:].split(" ") if "=" in kv)
        elif line:
            toks.append(tuple(line.split("\t")))
        elif toks:
            out.append((hdr, toks))
            hdr, toks = {}, []
    if toks:
        out.append((hdr, toks))
    return out
