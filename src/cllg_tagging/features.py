"""CRF feature spec (variant A: no embeddings). Kept to str operations, one regex per test and per-char lowercasing
so that js/src/features.js can reproduce it exactly. NFC-normalised prefix/suffix character n-grams (1-4)."""
import json
import os
import re
import unicodedata

LEX = set("cf sch schol fr fort supra infra v vv vgl ed p pp l ll lin cod codd ms om add del sc ad loc cit ib ibid op s sq sqq "
          "vol cap lib col ann fol app adn n not nota ep or rec corr ex etc cett gloss lemma".split())
ROMAN = re.compile(r"^[ivxlcdm]+$")
GREEK_NUM = re.compile(r"^[α-ω]+ʹ$")
OPEN, CLOSE = "([«‘“⟨", ")]»’”⟩"


CL = {}
if os.environ.get("BIO_CLUSTERS"):  # variant B: fasttext k-means cluster ids of the lowercased NFC token
    CL = json.load(open(os.environ["BIO_CLUSTERS"]))


def lower(w):
    return "".join(c.lower() for c in w)


def shape(w):
    out = []
    for c in w:
        if c.isdigit():
            k = "d"
        elif c.isalpha():
            k = ("G" if c.isupper() else "g") if "Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿" else ("A" if c.isupper() else "a")
        else:
            k = c
        if not out or out[-1] != k:
            out.append(k)
    return "".join(out)


def token_feats(w):
    n = unicodedata.normalize("NFC", w)
    lw = lower(n)
    f = ["w=" + lw, "sh=" + shape(n), "len=%d" % min(len(n), 8)]
    if CL:
        f.append("cl=%s" % CL.get(lw, "U"))
    for k in (1, 2, 3, 4):
        if len(lw) >= k:
            f += [f"p{k}=" + lw[:k], f"s{k}=" + lw[-k:]]
    if n.isdigit():
        f.append("digit")
    if ROMAN.match(lw):
        f.append("roman")
    if GREEK_NUM.match(lw):
        f.append("gnum")
    if n[:1].isupper():
        f.append("cap")
    if lw in LEX:
        f.append("lex")
    return f


def sent_feats(tokens):
    """-> one feature list per token."""
    base = [token_feats(t) for t in tokens]
    depth, since, rows = 0, 99, []
    ctx = []
    for t in tokens:
        if t in OPEN:
            depth += 1
            since = 0
        ctx.append((min(depth, 3), min(since, 6)))
        if t in CLOSE and depth:
            depth -= 1
        since += 1
    lexpos = [i for i, t in enumerate(tokens) if lower(t) in LEX]
    for i, t in enumerate(tokens):
        f = ["bias"] + base[i]
        f += ["depth=%d" % ctx[i][0], "since=%d" % ctx[i][1]]
        if any(0 < i - j <= 6 for j in lexpos):
            f.append("lex6")
        for d in (-3, -2, -1, 1, 2, 3):
            j = i + d
            if j < 0 or j >= len(tokens):
                f.append("%+d:EDGE" % d)
                continue
            b = base[j]
            f += ["%+d:%s" % (d, x) for x in b if x[:2] in ("sh", "w=", "p2", "s2", "le", "di", "ro", "gn", "ca", "cl") and
                  (abs(d) == 1 or x[:2] in ("sh", "p2", "s2", "cl"))]
        rows.append(f)
    return rows
