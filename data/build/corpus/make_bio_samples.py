"""Generate BIO-tagged samples (REF = citeStructure identifier, CITE = <bibl>) from the corpus.

Each sample is one leaf citeStructure unit. Its identifier chain (outer to inner) is rendered
back into the text as REF spans (Arabic/Roman/Greek numeral when purely numeric, kept as is
otherwise), at the start, in the middle of the text, or dropped. Text inside <bibl> is CITE.
"""
import glob
import random
import re
import sys
from lxml import etree

MAX_TOKENS = 150
TOKEN = re.compile(r"\w+|[^\w\s]")


def strip_ns(root):
    for el in root.iter():
        if isinstance(el.tag, str):
            el.tag = etree.QName(el).localname
    return root


def roman(n):
    out = ""
    for v, s in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                 (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out, n = out + s, n - v
    return out


GREEK = {1: "α", 2: "β", 3: "γ", 4: "δ", 5: "ε", 6: "ϛ", 7: "ζ", 8: "η", 9: "θ",
         10: "ι", 20: "κ", 30: "λ", 40: "μ", 50: "ν", 60: "ξ", 70: "ο", 80: "π", 90: "ϟ",
         100: "ρ", 200: "σ", 300: "τ", 400: "υ", 500: "φ", 600: "χ", 700: "ψ", 800: "ω", 900: "ϡ"}


def greek(n):
    out = ""
    for v in sorted(GREEK, reverse=True):
        while n >= v:
            out, n = out + GREEK[v], n - v
    return out + "ʹ"


def render_id(ident, rng):
    if ident.isdigit() and 0 < int(ident) < 1000:
        n = int(ident)
        return rng.choice([ident, roman(n), greek(n)])
    return ident


def segments(el):
    """Yield (string, is_bibl) pieces in document order."""
    def walk(node, in_bibl):
        in_bibl = in_bibl or node.tag == "bibl"
        if node.text:
            yield node.text, in_bibl
        for child in node:
            if isinstance(child.tag, str):
                yield from walk(child, in_bibl)
            if child.tail:
                yield child.tail, in_bibl
    yield from walk(el, False)


def tag_text(el):
    """Return [(token, tag, bibl_index, space_after)] for the unit."""
    text, flags = "", []
    for s, is_bibl in segments(el):
        text += s
        flags += [is_bibl] * len(s)
    out, last_k, k, prev = [], 0, 0, False
    for m in TOKEN.finditer(text):
        is_bibl = flags[m.start()]
        if is_bibl and not prev:
            k += 1
        prev = is_bibl
        space = m.end() < len(text) and text[m.end()].isspace()
        if is_bibl:
            out.append((m.group(), "I-CITE" if k == last_k else "B-CITE", k, space))
            last_k = k
        else:
            out.append((m.group(), "O", 0, space))
    return out


def truncate(toks):
    if len(toks) <= MAX_TOKENS:
        return toks
    i = MAX_TOKENS
    while i < len(toks) and toks[i][1] == "I-CITE":
        i += 1
    return toks[:i]


def ref_tokens(text):
    ts = TOKEN.findall(text)
    return [(t, "B-REF" if i == 0 else "I-REF", 0, False) for i, t in enumerate(ts)]


def close(toks):
    """Mark the last token of a REF piece as followed by a space."""
    toks[-1] = toks[-1][:3] + (True,)
    return toks


def with_refs(toks, chain, rng):
    """Insert REF spans. Returns (tokens, description)."""
    start, middle, desc = [], [], []
    for ident in chain:
        mode = rng.choices(["start", "middle", "drop"], [0.6, 0.15, 0.25])[0]
        shown = render_id(ident, rng) if mode != "drop" else ""
        desc.append(f"{ident}:{mode}:{shown}")
        (start if mode == "start" else middle if mode == "middle" else []).append(shown)
    out = []
    for i, shown in enumerate(start):
        out += ref_tokens(shown)
        last = i == len(start) - 1
        sep = rng.choice([".", ""]) if last else rng.choice([",", ".", ""])
        if sep:
            out.append((sep, "I-REF", 0, False))
        close(out)
    body = list(toks)
    for shown in middle:
        # boundary not inside a bibl span
        spots = [i for i in range(len(body) + 1)
                 if i == len(body) or body[i][1] != "I-CITE"]
        at = rng.choice(spots)
        ins = ref_tokens(shown)
        if rng.random() < 0.3:
            ins.append((".", "I-REF", 0, False))
        close(ins)
        if at:
            body[at - 1] = body[at - 1][:3] + (True,)
        if at == len(body):
            ins[-1] = ins[-1][:3] + (False,)
        body[at:at] = ins
    return out + body, " ".join(desc)


def units(root, node_cs, ctx, chain):
    """Yield (leaf element, id chain) for each leaf citeStructure unit."""
    for el in ctx.xpath(node_cs.get("match")):
        ident = el.xpath(node_cs.get("use"))
        ident = ident if isinstance(ident, str) else (ident[0] if ident else None)
        ident = getattr(ident, "text", ident)
        if not ident:
            continue
        sub = node_cs.findall("citeStructure")
        if sub:
            for s in sub:
                yield from units(root, s, el, chain + [str(ident)])
        else:
            yield el, chain + [str(ident)]


def file_units(path):
    root = strip_ns(etree.parse(path).getroot())
    top = root.find(".//refsDecl/citeStructure")
    if top is None:
        return []
    try:
        return list(units(root, top, root, []))
    except (etree.XPathError, TypeError):
        return []


def main(n=1000, seed=0, out="bio-samples-1000.tsv"):
    rng = random.Random(seed)
    paths = sorted(glob.glob("data/**/*.cllg-grc1.xml", recursive=True))
    rng.shuffle(paths)
    written, used = 0, 0
    with open(out, "w", encoding="utf-8") as fh:
        for p in paths:
            if written >= n:
                break
            us = file_units(p)
            if not us:
                continue
            used += 1
            with_b = [u for u in us if u[0].find(".//bibl") is not None]
            without = [u for u in us if u[0].find(".//bibl") is None]
            picks = rng.sample(with_b, min(2, len(with_b))) + rng.sample(without, min(2, len(without)))
            for el, chain in picks:
                if written >= n:
                    break
                toks = truncate(tag_text(el))
                if not toks:
                    continue
                toks, desc = with_refs(toks, chain, rng)
                urn = p.split("/")[-1].replace(".cllg-grc1.xml", "")
                fh.write(f"# id={written + 1} urn={urn} ref={'.'.join(chain)} render={desc}\n")
                fh.write("# text=" + "".join(t + (" " if sp else "") for t, _, _, sp in toks).rstrip() + "\n")
                for t, tag, _, sp in toks:
                    fh.write(f"{t}\t{tag}\t{'_' if sp else 'N'}\n")
                fh.write("\n")
                written += 1
    print(f"{written} samples from {used} files -> {out}")


if __name__ == "__main__":
    main(*(int(a) if a.isdigit() else a for a in sys.argv[1:]))
