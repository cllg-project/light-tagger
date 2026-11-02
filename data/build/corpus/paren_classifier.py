"""Note vs. bibl (vs. other) for parentheticals, from a small embedder + logistic regression.

Training data is the corpus itself: every <note> is a "note", every <bibl> a
"bibl", and the parentheticals read as text or speaker labels in the paren_notes
pass are "other". The input is the inner text (parentheses stripped) with some
flattened context around it; multilingual-e5-small embeds it (Greek, Latin,
modern abbreviations), and a few shape features are added (digits, Greek share,
sigla, apparatus words, length): they lift held-out accuracy from 0.81 to 0.92. Examples are deduplicated and capped per work, and the
evaluation holds out whole works.

Targets are the parentheticals still bare in <body> (not in a note or bibl).

Runs with any Python that has numpy, scikit-learn and sentence-transformers
(here the unsloth studio env):
  python scripts/paren_classifier.py train      -> note/bibl only, from the corpus (baseline)
  python scripts/paren_classifier.py train3     -> note/bibl/text with the labelled target sample
                                                   -> scripts/out/paren_clf/{model.pkl,eval.txt}
  python scripts/paren_classifier.py predict    -> scripts/out/paren_clf/predictions.jsonl
"""
import glob
import html
import json
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts/out/paren_clf"
import os
MODEL = os.environ.get("PAREN_EMBEDDER", "intfloat/multilingual-e5-small")
PAR = re.compile(r'\((?:[^()<]|<hi rend="(?:superscript|italic)">[^<]{1,40}</hi>){1,160}\)')
ELEM = re.compile(r"<(note|bibl)\b[^>]*>(.*?)</\1>", re.S)
ENUM = re.compile(r"\d{1,3}[a-z]?|[IVXLC]{1,6}|[a-zα-ω]")
CAP = 40  # examples per work and class


def flat(s):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s)).split())


def inner(s):
    s = flat(s)
    return s[1:-1].strip() if s.startswith("(") and s.endswith(")") else s


def context(raw, s, e):
    return flat(raw[max(0, s - 200):s])[-60:], flat(raw[e:e + 120])[:30]


def text_of(left, mid, right):
    return f"query: {left} ⟦{mid}⟧ {right}"


def masked(raw):
    """Spans of <note>/<bibl> elements (targets inside them are skipped)."""
    return [(m.start(), m.end()) for m in ELEM.finditer(raw)]


def corpus():
    for f in sorted(glob.glob(str(ROOT / "data/*/*/*.cllg-grc1.xml"))):
        raw = Path(f).read_text(encoding="utf-8")
        b = raw.find("<body")
        yield Path(f).name[:14], f, raw, b


def training():
    ex = []
    for key, f, raw, b in corpus():
        for m in ELEM.finditer(raw, b):
            if "<note" in m.group(2) or "<bibl" in m.group(2):
                continue
            mid = inner(m.group(2))
            if not mid or len(mid) > 200:
                continue
            l, r = context(raw, m.start(), m.end())
            ex.append((m.group(1), key, text_of(l, mid, r), mid))
    items = {i["id"]: i for i in json.loads((ROOT / "scripts/out/paren_notes/items.json").read_text())}
    for line in (ROOT / "scripts/out/paren_notes/labels.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["label"] in ("text", "speaker") and False:  # too few (32 after dedupe): enumerations are skipped by rule instead
            it = items[d["id"]]
            ctx = it["ctx"]
            left, rest = ctx.split("⟦", 1)
            mid, right = rest.split("⟧", 1)
            ex.append(("other", Path(it["file"]).name[:14], text_of(left.strip()[-60:], inner(mid), right.strip()[:30]),
                       inner(mid)))
    # dedupe on (label, inner text), cap per work and class
    seen, per, out = set(), Counter(), []
    random.Random(0).shuffle(ex)
    for lab, key, t, mid in ex:
        if (lab, mid) in seen or per[(key, lab)] >= CAP:
            continue
        seen.add((lab, mid))
        per[(key, lab)] += 1
        out.append((lab, key, t))
    return out


APP = re.compile(r"\b(om|add|cod|codd|sic|corr|del|mg|marg|fort|leg|l|γρ|cf|vgl|v|sc|scil)\b\.?")
POINTER = re.compile(r"^\s*(cf|vgl|v|vid|vide|s|siehe|unde|aus|ad|immo)\b\.?|^\s*=|^\s*→")
INTERNAL = re.compile(r"\b(supra|infra|oben|unten|sch|schol|adn|n)\b\.?|\bad\s")
REFNUM = re.compile(r"\b(fr|p|F|T|v|vv|Il|Od|s)\.?\s*\d")


def feats(t):
    """Cheap shape features next to the embedding: digits, Greek, sigla, apparatus words, length."""
    mid = t.split("⟦", 1)[1].split("⟧", 1)[0]
    n, al = max(1, len(mid)), max(1, sum(c.isalpha() for c in mid))
    gr = sum(1 for c in mid if "\u0370" <= c <= "\u03ff" or "\u1f00" <= c <= "\u1fff")
    return [3 * x for x in (sum(c.isdigit() for c in mid) / n, gr / al,
                            len(re.findall(r"\b[A-Z][a-z]?\b", mid)) / (1 + len(mid.split())),
                            int(bool(APP.search(mid))), min(len(mid), 120) / 120, int(bool(REFNUM.search(mid))),
                            int(bool(POINTER.search(mid))), int(bool(INTERNAL.search(mid))))]


def vectors(texts):
    import numpy as np
    return np.c_[embed(texts), np.array([feats(t) for t in texts])]


def embed(texts):
    """Embeddings, cached per model. e5 models expect a "query: " prefix, the others don't."""
    import hashlib
    import numpy as np
    from sentence_transformers import SentenceTransformer
    if "e5" not in MODEL:
        texts = [t[7:] if t.startswith("query: ") else t for t in texts]
    key = hashlib.sha1((MODEL + "\x00" + "\x01".join(texts)).encode()).hexdigest()[:16]
    cache = OUT / "emb" / f"{key}.npy"
    if cache.exists():
        return np.load(cache)
    m = SentenceTransformer(MODEL, device="cuda")
    m.max_seq_length = 128
    X = m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    return X


def train():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GroupKFold
    data = training()
    y = np.array([d[0] for d in data])
    groups = np.array([d[1] for d in data])
    X = embed([d[2] for d in data])
    report = [f"examples: {Counter(y.tolist())}"]
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0).fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    report.append("5-fold, works held out:\n" + classification_report(y, pred, digits=3))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0).fit(X, y)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(clf, open(OUT / "model.pkl", "wb"))
    (OUT / "eval.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


def sample_labels(with_pos=False):
    """The labelled target sample (subagents, 2026-09-29): speaker counts as text."""
    smp = {r["id"]: r for r in json.loads((OUT / "sample.json").read_text(encoding="utf-8"))}
    rows = []
    for f in sorted(OUT.glob("answers_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            r = smp[d["id"]]
            row = ("text" if d["label"] == "speaker" else d["label"], Path(r["file"]).name[:14], r["text"])
            rows.append(row + ((r["file"], r["off"]),) if with_pos else row)
    return rows


def train3():
    """note / bibl / text. Evaluation on the labelled target sample, works held out; the
    corpus notes and bibls are always in the training part."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupKFold
    base = training()
    smp = sample_labels()
    Xb, Xs = vectors([d[2] for d in base]), vectors([d[2] for d in smp])
    yb, ys = np.array([d[0] for d in base]), np.array([d[0] for d in smp])
    gs = np.array([d[1] for d in smp])
    w = lambda n_base, n_s: np.r_[np.full(n_base, 0.05), np.full(n_s, 1.0)]  # noqa: E731 the target sample leads
    pred, prob = np.empty_like(ys), np.zeros(len(ys))
    for tr, te in GroupKFold(n_splits=5).split(Xs, ys, gs):
        X, y = np.r_[Xb, Xs[tr]], np.r_[yb, ys[tr]]
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0).fit(X, y, sample_weight=w(len(yb), len(tr)))
        pred[te] = clf.predict(Xs[te])
        prob[te] = clf.predict_proba(Xs[te]).max(1)
    labels = ["note", "bibl", "text"]
    report = [f"sample: {Counter(ys.tolist())}",
              "target sample, 5-fold, works held out:\n" + classification_report(ys, pred, labels=labels, digits=3),
              "confusion (rows=truth, cols=pred) " + str(labels) + "\n" + str(confusion_matrix(ys, pred, labels=labels))]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        keep = prob >= t
        per = ", ".join(f"{c} precision {(ys[keep & (pred == c)] == c).mean():.3f} (n={int((keep & (pred == c)).sum())})"
                        for c in labels)
        report.append(f"p>={t}: keeps {keep.mean():.0%}, accuracy {(pred[keep] == ys[keep]).mean():.3f}; {per}")
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0).fit(
        np.r_[Xb, Xs], np.r_[yb, ys], sample_weight=w(len(yb), len(ys)))
    pickle.dump(clf, open(OUT / "model.pkl", "wb"))
    (OUT / "eval.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


def sample5():
    """Five labels: the 1,600 target sample (its notes re-annotated) + 1,000 annotated items
    + ~950 items the model tagged note/noteBibl/noteRef (Claude subagents, 2026-09-29)."""
    ann = {}
    # ann2/ann3: the model's note/pointer tags and uncertain items; ann4: items left bare (step 1 p<0.8),
    # spread over every work that has some (answers in two files); ann5: 3,000 more undecided items
    for stem in ("ann", "ann2", "ann3", "ann4", "ann5"):
        answers = sorted(OUT.glob(f"{stem}_answers*.jsonl"))
        if not answers:
            continue
        ann_items = {i["id"]: i for i in json.loads((OUT / f"{stem}_items.json").read_text(encoding="utf-8"))}
        for path in answers:
            for line in path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                ann[(ann_items[d["id"]]["file"], ann_items[d["id"]]["off"])] = (d, ann_items[d["id"]])
    rows, seen = [], set()
    for lab, key, t, fo in sample_labels(with_pos=True):
        if fo in ann:
            continue  # re-annotated below
        rows.append((lab, key, t, None))
    for (f, o), (d, it) in ann.items():
        rows.append((d["label"], Path(f).name[:14], it["text"], d.get("targets")))
    return rows


def train5():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupKFold
    base = [d for d in training()]
    smp = sample5()
    Xb, Xs = vectors([d[2] for d in base]), vectors([d[2] for d in smp])
    yb, ys = np.array([d[0] for d in base]), np.array([d[0] for d in smp])
    gs = np.array([d[1] for d in smp])
    w = lambda n_base, n_s: np.r_[np.full(n_base, 0.05), np.full(n_s, 1.0)]  # noqa: E731
    pred, prob = np.empty_like(ys), np.zeros(len(ys))
    for tr, te in GroupKFold(n_splits=5).split(Xs, ys, gs):
        clf = LogisticRegression(max_iter=4000, class_weight="balanced", C=4.0).fit(
            np.r_[Xb, Xs[tr]], np.r_[yb, ys[tr]], sample_weight=w(len(yb), len(tr)))
        pred[te], prob[te] = clf.predict(Xs[te]), clf.predict_proba(Xs[te]).max(1)
    labels = ["note", "bibl", "text", "noteBibl", "noteRef"]
    report = [f"sample: {Counter(ys.tolist())}",
              "5-fold, works held out:\n" + classification_report(ys, pred, labels=labels, digits=3),
              "confusion (rows=truth, cols=pred) " + str(labels) + "\n" + str(confusion_matrix(ys, pred, labels=labels))]
    for t in (0.6, 0.7, 0.8, 0.9):
        keep = prob >= t
        per = ", ".join(f"{c} {(ys[keep & (pred == c)] == c).mean():.3f} (n={int((keep & (pred == c)).sum())})" for c in labels)
        report.append(f"p>={t}: keeps {keep.mean():.0%}, acc {(pred[keep] == ys[keep]).mean():.3f}; precision {per}")
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", C=4.0).fit(
        np.r_[Xb, Xs], np.r_[yb, ys], sample_weight=w(len(yb), len(ys)))
    pickle.dump(clf, open(OUT / "model5.pkl", "wb"))
    (OUT / "eval5.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


INTERNAL_RULE = re.compile(r"\b(supra|infra|oben|unten|adn|ibid|sch|schol|scholion)\b|\bn\.\s*\d|→"
                           r"|\bad\s+(tit|[A-ΩΑ-Ω]\s+\d)|\b(?:s\.|vgl\.|cf\.)\s*[FT]\s*\d|\bcf\.\s*\d+[a-z]\b")


def rule_feats(t):
    """Step-2 extras: the internal-pointer rule, and which pointer word opens the note."""
    mid = t.split("⟦", 1)[1].split("⟧", 1)[0]
    m = POINTER.search(mid)
    word = (m.group(1) or m.group().strip()).lower() if m else ""
    return [3 * int(bool(INTERNAL_RULE.search(mid)))] + [int(word == w) for w in ("cf", "v", "s", "vgl", "ad", "unde", "aus", "=")]


def two_stage(t2=0.7):
    """Step 1: note / bibl / text / noteWithRef. Step 2 (on noteWithRef): noteBibl vs noteRef;
    below `t2` the pointer note stays a plain note. Evaluated with works held out."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GroupKFold
    base = training()
    smp = sample5()
    Xb, Xs = vectors([d[2] for d in base]), vectors([d[2] for d in smp])
    Rs = np.array([rule_feats(d[2]) for d in smp])
    yb = np.array([d[0] for d in base])
    y5 = np.array([d[0] for d in smp])
    y4 = np.where(np.isin(y5, ["noteBibl", "noteRef"]), "noteWithRef", y5)
    gs = np.array([d[1] for d in smp])
    w = lambda n_base, n_s: np.r_[np.full(n_base, 0.05), np.full(n_s, 1.0)]  # noqa: E731
    p1, pr1 = np.empty_like(y4), np.zeros(len(y4))
    p2, pr2 = np.full(len(y5), "", dtype=object), np.zeros(len(y5))
    ptr = np.isin(y5, ["noteBibl", "noteRef"])
    for tr, te in GroupKFold(n_splits=5).split(Xs, y4, gs):
        c1 = LogisticRegression(max_iter=4000, class_weight="balanced", C=4.0).fit(
            np.r_[Xb, Xs[tr]], np.r_[yb, y4[tr]], sample_weight=w(len(yb), len(tr)))
        p1[te], pr1[te] = c1.predict(Xs[te]), c1.predict_proba(Xs[te]).max(1)
        trp = tr[ptr[tr]]
        c2 = LogisticRegression(max_iter=4000, class_weight="balanced", C=2.0).fit(
            np.c_[Xs[trp], Rs[trp]], y5[trp])
        P = c2.predict_proba(np.c_[Xs[te], Rs[te]])
        p2[te], pr2[te] = c2.classes_[P.argmax(1)], P.max(1)
    L4 = ["note", "bibl", "text", "noteWithRef"]
    rep = ["STEP 1 (works held out):\n" + classification_report(y4, p1, labels=L4, digits=3)]
    for t in (0.6, 0.7, 0.8):
        k = pr1 >= t
        rep.append(f"  step 1 p>={t}: keeps {k.mean():.0%}; precision " + ", ".join(
            f"{c} {(y4[k & (p1 == c)] == c).mean():.3f} (n={int((k & (p1 == c)).sum())})" for c in L4))
    rep.append("STEP 2 on true pointer notes (works held out), abstain -> plain note:")
    rule = np.array(["noteRef" if INTERNAL_RULE.search(d[2].split("⟦", 1)[1].split("⟧", 1)[0]) else "noteBibl" for d in smp])
    rep.append(f"  rule baseline: acc {(rule[ptr] == y5[ptr]).mean():.3f}; ref precision "
               f"{(y5[ptr & (rule == 'noteRef')] == 'noteRef').mean():.3f}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        k = ptr & (pr2 >= t)
        rep.append(f"  p>={t}: decides {k.sum() / ptr.sum():.0%} (rest plain note), acc {(p2[k] == y5[k]).mean():.3f}; "
                   + ", ".join(f"{c} precision {(y5[k & (p2 == c)] == c).mean():.3f} (n={int((k & (p2 == c)).sum())})"
                               for c in ("noteBibl", "noteRef")))
    # end to end, step-1 threshold 0.8, step-2 threshold t2
    final = np.where(pr1 >= 0.8, p1, "abstain").astype(object)
    wr = final == "noteWithRef"
    final[wr] = np.where(pr2[wr] >= t2, p2[wr], "note")
    out = Counter()
    for f, t in zip(final, y5):
        if f == "abstain":
            out["left bare (p1<0.8)"] += 1
        elif f == t:
            out["correct"] += 1
        elif f == "note" and t in ("noteBibl", "noteRef"):
            out["pointer kept as plain note (safe)"] += 1
        else:
            out["wrong"] += 1
    rep.append(f"END TO END (step 1 p>=0.8, step 2 p>={t2}): " + ", ".join(f"{k} {v} ({v / len(y5):.0%})" for k, v in out.most_common()))
    c1 = LogisticRegression(max_iter=4000, class_weight="balanced", C=4.0).fit(
        np.r_[Xb, Xs], np.r_[yb, y4], sample_weight=w(len(yb), len(y4)))
    c2 = LogisticRegression(max_iter=4000, class_weight="balanced", C=2.0).fit(np.c_[Xs[ptr], Rs[ptr]], y5[ptr])
    pickle.dump({"step1": c1, "step2": c2, "t1": 0.8, "t2": t2}, open(OUT / "model_2step.pkl", "wb"))
    (OUT / "eval_2step.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


def predict2():
    """Two-step predictions for every bare parenthetical -> predictions_2step.jsonl
    (label1/p1 from step 1; for noteWithRef, label2/p2 from step 2)."""
    import numpy as np
    m = pickle.load(open(OUT / "model_2step.pkl", "rb"))
    rows = [json.loads(line) for line in (OUT / "predictions5.jsonl").read_text(encoding="utf-8").splitlines()]
    X = vectors([r["text"] for r in rows])
    P1 = m["step1"].predict_proba(X)
    R = np.array([rule_feats(r["text"]) for r in rows])
    P2 = m["step2"].predict_proba(np.c_[X, R])
    with open(OUT / "predictions_2step.jsonl", "w", encoding="utf-8") as fh:
        for r, a, b in zip(rows, P1, P2):
            r = {k: r[k] for k in ("file", "off", "raw", "digits", "text")}
            r["label1"], r["p1"] = str(m["step1"].classes_[a.argmax()]), round(float(a.max()), 3)
            if r["label1"] == "noteWithRef":
                r["label2"], r["p2"] = str(m["step2"].classes_[b.argmax()]), round(float(b.max()), 3)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "rows")


def predict(model="model5.pkl", out="predictions5.jsonl"):
    clf = pickle.load(open(OUT / model, "rb"))
    rows = []
    for key, f, raw, b in corpus():
        spans = masked(raw)
        for m in PAR.finditer(raw, b):
            if raw.rfind("<", 0, m.start()) > raw.rfind(">", 0, m.start()):
                continue  # inside a tag
            if any(s <= m.start() < e for s, e in spans):
                continue
            mid = inner(m.group())
            if not mid or ENUM.fullmatch(mid):
                continue  # an enumeration `(9)`, `(IV)`, `(b)`: text, not a note or a reference
            l, r = context(raw, m.start(), m.end())
            rows.append({"file": str(Path(f).relative_to(ROOT)), "off": m.start(), "raw": m.group(),
                         "digits": bool(re.search(r"\d", mid)), "text": text_of(l, mid, r)})
    X = vectors([r["text"] for r in rows])
    P = clf.predict_proba(X)
    for r, p in zip(rows, P):
        k = int(p.argmax())
        r["label"], r["p"] = clf.classes_[k], round(float(p[k]), 3)
    with open(OUT / out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter((r["label"], r["digits"]) for r in rows)
    print(len(rows), "bare parentheticals:", dict(c))


if __name__ == "__main__":
    {"train": train, "train3": train3, "train5": train5, "two_stage": two_stage, "predict": predict, "predict2": predict2}[sys.argv[1]]()
