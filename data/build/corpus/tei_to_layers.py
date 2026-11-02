"""Corpus TEI -> BIO windows (classes NOTE BIBL NOTEBIBL NOTEREF REF; everything else O).

Gold = the encoding itself: <note> (NOTEBIBL if it holds a <bibl>, NOTEREF if a <ref>), <bibl> -> BIBL, <ref> -> REF,
and pb/cb/milestone[@n] rendered as plain-text markers ([12], |12|, "PG 12", ...; one style per file) tagged REF.
Plus the 7.9k gold-labelled bare parentheticals (paren_classifier ann*/sample answers; 'text' -> O).
The unit is a block (p, lg, head, item, label, sp...) cut to windows of <= ~200 tokens, split by work (urn) into
train/dev (the md pages are the test set). Reads data/, never writes it.
Usage: tei_to_bio.py OUT_PREFIX  -> OUT_PREFIX-{train,dev}.tsv
"""
import glob
import json
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from lxml import etree

from cllg_tagging import common as bc
import make_bio_samples as mb
import paren_classifier as pc

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = {"p", "lg", "head", "item", "label", "sp", "ab", "quote", "cit", "list"}
SPACE_AFTER = {"l", "lb", "head", "label", "item", "speaker", "cell", "row", "p", "sp", "lg", "stage"}
MARKERS = {"pb", "cb", "milestone"}
MAX_TOK = 200
CAPS = {"gold": 10**9, "rich": 8, "refonly": 3, "neg": 2}
CAPS_TRAIN = {"gold": 10**9, "rich": 30, "refonly": 10, "neg": 4}  # --train-only: bigger train, dev file untouched


def gold_parens():
    """{file: [(off, raw, left context, class)]} for the labelled bare parentheticals ('text' is dropped = plain text).
    Offsets are stale (files were re-indented since), so process() relocates each one by its raw text + left context."""
    got = {}
    smp = {r["id"]: r for r in json.loads((pc.OUT / "sample.json").read_text(encoding="utf-8"))}
    for p in sorted(pc.OUT.glob("answers_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            r = smp[d["id"]]
            got[(r["file"], r["off"])] = (r["raw"], r["text"], "text" if d["label"] == "speaker" else d["label"])
    for stem in ("ann", "ann2", "ann3", "ann4", "ann5"):
        items = {i["id"]: i for i in json.loads((pc.OUT / f"{stem}_items.json").read_text(encoding="utf-8"))}
        for p in sorted(pc.OUT.glob(f"{stem}_answers*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                i = items[d["id"]]
                got[(i["file"], i["off"])] = (i["raw"], i["text"], d["label"])
    by = defaultdict(list)
    mine = ROOT / "scripts/out/bio/mine_items.json"
    if mine.exists():  # agent-tagged encoded notes: override the class of the <note> with this exact text
        mi = {i["id"]: i for i in json.loads(mine.read_text(encoding="utf-8"))}
        for p in sorted(mine.parent.glob("mine_ans_*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                by[mi[d["id"]]["file"]].append((-1, mi[d["id"]]["text"], "", d["label"].upper()))
    for (f, o), (raw, text, lab) in got.items():
        by[f].append((o, raw, text.removeprefix("query:").split("\u27e6", 1)[0].strip(), lab.upper()))
    return by


def locate(s, off, raw, left):
    """Start of the parenthetical in the current text, or None (moved, converted, or ambiguous)."""
    if s.startswith(raw, off):
        return off
    pat = re.compile(r"\s+".join(re.escape(w) for w in raw.split()))
    ms = list(pat.finditer(s))
    if len(ms) > 1 and left:
        tail = "".join(left.split())[-20:]
        ms = [m for m in ms if "".join(re.sub(r"<[^>]+>", "", s[max(0, m.start() - 200):m.start()]).split()).endswith(tail)]
    return ms[0].start() if len(ms) == 1 else None


def local(el):
    return etree.QName(el).localname if isinstance(el.tag, str) else None


UNIT_LABELS = {"book": ["Lib.", "Liber", "Βιβλίον", "L."], "chapter": ["Cap.", "c.", "Κεφ."], "section": ["§", "Sect."],
               "fragment": ["Fr.", "F", "fr."], "poem": ["Carm.", "Poem"], "play": ["Fab."], "page": ["p.", "pag."],
               "column": ["col."], "volume": ["Vol.", "t."]}
WRAPS = ["{}", "{}", "{}", "{}.", "({})", "[{}]"]


def num_style(urn, unit):
    return random.Random(f"{urn}|{unit}").choices(["arabic", "roman", "greek", "mixed"], [4, 2.5, 2, 1.5])[0]


def render_num(v, style, rng):
    """Arabic / Roman / Greek rendering of a citeStructure value (digits only, 1-999); anything else is kept."""
    if not (v.isdigit() and 0 < int(v) < 1000):
        return v
    if style == "mixed":
        style = rng.choice(["arabic", "roman", "greek"])
    n = int(v)
    if style == "roman":
        r = mb.roman(n)
        return r.lower() if rng.random() < 0.2 else r
    return mb.greek(n) if style == "greek" else v


def cite_map(root):
    """{element: (unit, value, depth)} for the elements matched by the header's citeStructure (lines excluded)."""
    cmap = {}

    def run(cs, parent, depth):
        match, use, unit = cs.get("match"), cs.get("use") or "@n", cs.get("unit") or ""
        try:
            els = (parent.xpath(match) if parent is not None else root.xpath(match))
        except etree.XPathError:
            return
        if not isinstance(els, list):
            return
        for el in els:
            if not hasattr(el, "get") or not use.startswith("@"):
                continue
            v = (el.get(use[1:]) or "").strip()
            if v and unit not in ("l", "lb", "line", "verse") and local(el) in ("div", "pb", "cb", "milestone"):
                cmap.setdefault(el, (unit, v, depth))
            for sub in cs.findall("citeStructure"):
                run(sub, el, depth + 1)

    for cs in root.iter("citeStructure"):
        if local(cs.getparent()) != "citeStructure":
            run(cs, None, 0)
    return cmap


class Renderer:
    def __init__(self, style, override=None, urn=""):
        self.urn = urn
        self.override = override or {}
        self.style, self.text, self.marks, self.n, self.has_gold = style, [], [], 0, False
        self.inner, self.unknown = {}, False  # inner: element name -> layer-2 class inside the current note

    def emit(self, s, mark):
        if s:
            self.text.append(s)
            self.marks.extend([mark] * len(s))

    def marker(self, el):
        n = el.get("n")
        if not n:
            return
        ed = el.get("ed") or ""
        unit = {"pb": "page", "cb": "column"}.get(local(el), el.get("unit") or "section")
        n = render_num(n, num_style(self.urn, unit), random.Random(f"{self.urn}|{unit}|{n}|{self.n}"))
        s = [n, f"[{n}]", f"|{n}|", f"{ed} {n}".strip(), f"({n})"][self.style]
        self.n += 1
        self.emit(" ", None)
        self.emit(s, (None, ("REF", self.n)))
        self.emit(" ", None)

    def chain_marker(self, pieces, rng):
        """pieces: [(unit, value)] outermost first -> one joined REF span ("III 5", "3.2") or one span per level."""
        txt = []
        for unit, v in pieces:
            t = render_num(v, num_style(self.urn, unit), rng)
            if unit in UNIT_LABELS and rng.random() < 0.15:
                t = f"{rng.choice(UNIT_LABELS[unit])} {t}"
            txt.append(t)
        spans = [rng.choice([" ", ".", ", ", ","]).join(txt)] if len(txt) > 1 and rng.random() < 0.5 else txt
        for t in spans:
            self.n += 1
            self.emit(" ", None)
            self.emit(rng.choice(WRAPS).format(t) if len(spans) == 1 or len(txt) == 1 else t, (None, ("REF", self.n)))
        self.emit(" ", None)

    def walk(self, el, cur):
        name = local(el)
        if name is None:
            return
        if name in MARKERS:
            if cur is None:
                self.marker(el)
            return
        if cur is None and name in ("note", "bibl", "ref", "gold"):
            if name == "gold":
                cls, self.has_gold = el.get("cls"), True
            elif name == "note":
                kids = {local(k) for k in el.iter() if k is not el}
                cls = "NOTEBIBL" if "bibl" in kids else "NOTEREF" if "ref" in kids else "NOTE"
            else:
                cls = name.upper()
            if name in ("note", "bibl"):
                new = self.override.get(" ".join("".join(el.itertext()).split()), cls)
                self.has_gold = self.has_gold or new != cls or " ".join("".join(el.itertext()).split()) in self.override
                cls = new
            self.n += 1
            cur = None
            if cls in ("NOTE", "NOTEBIBL", "NOTEREF"):
                cur = (("NOTE", self.n), None)
                kids = {local(k) for k in el.iter() if k is not el}
                self.inner = {"bibl": "BIBL"} if cls == "NOTEBIBL" else {"ref": "REF"} if cls == "NOTEREF" else {}
                if cls != "NOTE" and not (kids & set(self.inner)):
                    self.unknown = True  # class says the note holds a bibl/ref but the encoding does not show where
            elif cls in ("BIBL", "REF"):
                cur = (None, (cls, self.n))
        elif cur is not None and cur[0] is not None and cur[1] is None and name in self.inner:
            self.n += 1
            cur = (cur[0], (self.inner[name], self.n))
        self.emit(el.text, cur)
        for k in el:
            self.walk(k, cur)
            self.emit(k.tail, cur if cur and name in ("note", "bibl", "ref", "gold") else None)
        if name in ("note", "gold"):
            self.inner = {}
        if name in SPACE_AFTER or name == "lb":
            self.emit(" ", None)


def blocks(el):
    for k in el:
        n = local(k)
        if n in BLOCKS:
            yield k
        elif n is not None and n not in ("note", "bibl", "ref", "gold", "teiHeader"):
            yield from blocks(k)


def windows(toks):
    """Cut a token list ((tok, tag1, tag2)) into <= ~MAX_TOK windows, centred on spans, never splitting one."""
    if len(toks) <= MAX_TOK:
        return [toks]
    starts = [i for i, t in enumerate(toks) if t[1].startswith("B-") or t[2].startswith("B-")]
    cont = lambda t: t[1].startswith("I-") or t[2].startswith("I-")  # noqa: E731
    out, end = [], 0
    for s in starts or [len(toks) // 2]:
        if s < end:
            continue
        a, b = max(end, s - 70), min(len(toks), s + 130)
        while a > 0 and cont(toks[a]):
            a -= 1
        while b < len(toks) and cont(toks[b]):
            b += 1
        out.append(toks[a:b])
        end = b
    if not starts:
        out = [toks[:MAX_TOK]]
    return out


def process(args):
    path, golds = args
    urn = Path(path).name.rsplit(".cllg", 1)[0]
    raw = Path(path).read_text(encoding="utf-8")
    spots, votes = {}, defaultdict(Counter)
    for off, rawp, left, lab in golds:
        votes[" ".join(rawp.split())][lab] += 1
        at = locate(raw, off, rawp, left) if off >= 0 else None
        if at is not None and lab not in ("TEXT", "O"):
            spots[at] = lab
    override = {k: c.most_common(1)[0][0] for k, c in votes.items() if len(c) == 1}
    for at, lab in sorted(spots.items(), reverse=True):
        m = pc.PAR.match(raw, at)
        if m:
            raw = f'{raw[:m.start()]}<gold xmlns="http://www.tei-c.org/ns/1.0" cls="{lab}">{m.group()}</gold>{raw[m.end():]}'
    try:
        root = etree.fromstring(raw.encode("utf-8"))
    except etree.XMLSyntaxError:
        return urn, []
    for e in root.iter():
        if isinstance(e.tag, str):
            e.tag = etree.QName(e).localname
    body = next((e for e in root.iter() if local(e) == "body"), None)
    if body is None:
        return urn, []
    cmap = cite_map(root)
    divs_cited = any(local(e) == "div" for e in cmap)
    out = []
    style = zlib.crc32(path.encode()) % 5
    seen = set()
    for i, b in enumerate(blocks(body)):
        r = Renderer(style, override, urn)
        pieces = []
        for a in reversed(list(b.iterancestors())):
            if local(a) != "div" or id(a) in seen:
                continue
            seen.add(id(a))
            if a in cmap:
                pieces.append(cmap[a][:2])
            elif not divs_cited and (a.get("n") or "").strip():
                pieces.append((a.get("type") or "section", a.get("n").strip()))
        if pieces and len(pieces[-1][1]) <= 8 and random.Random(f"{urn}{i}").random() < 0.9:
            r.chain_marker(pieces, random.Random(f"{urn}|{i}"))
        r.walk(b, None)
        text, marks = "".join(r.text), r.marks
        toks = bc.tokenize2(text, marks)
        for j, w in enumerate(windows(toks)):
            tags = {x[2:] for t in w for x in t[1:] if x != "O"}
            kind = "neg" if not tags else "gold" if r.has_gold else "rich" if tags - {"REF"} else "refonly"
            out.append((f"{urn}#{i}.{j}", kind, w, r.unknown))
    return urn, out


def main(prefix, train_only=False):
    golds = gold_parens()
    files = sorted(f for f in glob.glob(str(ROOT / "data/**/*.xml"), recursive=True) if not f.endswith("metadata.xml"))
    rel = lambda f: str(Path(f).relative_to(ROOT))  # noqa: E731
    jobs = [(f, golds.get(rel(f), [])) for f in files]
    outs = {s: open(f"{prefix}-{s}.tsv", "w", encoding="utf-8") for s in (("train",) if train_only else ("train", "dev"))}
    n = Counter()
    with Pool(14) as pool:
        for urn, wins in pool.imap_unordered(process, jobs, chunksize=8):
            split = "dev" if zlib.crc32(urn.encode()) % 10 == 0 else "train"
            if split not in outs:
                continue
            caps = CAPS_TRAIN if train_only else CAPS
            rng = random.Random(urn)
            by = defaultdict(list)
            for w in wins:
                by[w[1]].append(w)
            for kind in sorted(by):
                rng.shuffle(by[kind])
                for wid, k, w, unk in by[kind][:caps[kind]]:
                    bc.write_layers(outs[split], f"id={wid} urn={urn} kind={k}" + (" unknown=1" if unk else ""), w)
                    n[split, k] += 1
    for f in outs.values():
        f.close()
    print(dict(n))


if __name__ == "__main__":
    main(sys.argv[1], "--train-only" in sys.argv)
