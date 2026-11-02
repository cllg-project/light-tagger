"""Convert Markdown + pseudo-XML pages to BIO: <ref> -> REF, <note> -> CITE, all else O.

Other XML tags (<tab/>, ...) and Markdown line markers (#, -, >) are dropped. One chunk per file.
Usage: md_to_bio.py INPUT_GLOB OUTPUT.tsv
"""
import glob
import re
import sys

TOKEN = re.compile(r"\w+|[^\w\s]")
TAG = re.compile(r"<(/?)(\w+)[^>]*?(/?)>")
MD_LINE = re.compile(r"^[ \t]*(?:#+[ \t]+|[-*>][ \t]+)", re.M)


def chars(text):
    """Return (clean text, per-char label) with ref/note spans tracked and other tags removed."""
    text = MD_LINE.sub("", text)
    out, labels, pos, label = [], [], 0, "O"
    for m in TAG.finditer(text):
        chunk = text[pos:m.start()]
        out.append(chunk)
        labels += [label] * len(chunk)
        closing, name, selfclose = m.group(1), m.group(2), m.group(3)
        if name in ("ref", "note") and not selfclose:
            label = "O" if closing else name.upper() if name == "ref" else "CITE"
        else:
            out.append(" ")
            labels.append(label)
        pos = m.end()
    out.append(text[pos:])
    labels += [label] * (len(text) - pos)
    return "".join(out), labels


def tokens(clean, labels):
    """Like TOKEN.finditer, but never let a token cross a ref/note boundary."""
    for m in TOKEN.finditer(clean):
        start = m.start()
        for i in range(start + 1, m.end()):
            if labels[i] != labels[i - 1]:
                yield re.compile(r"[\s\S]{%d}" % (i - start)).match(clean, start)
                start = i
        yield re.compile(r"[\s\S]{%d}" % (m.end() - start)).match(clean, start)


def bio(text):
    clean, labels = chars(text)
    toks, prev = [], None
    for m in tokens(clean, labels):
        lab = labels[m.start()]
        # a gap in the labelled span (whitespace-only) is irrelevant; a new span starts when label changes
        tag = "O" if lab == "O" else ("I-" if prev == lab and toks and toks[-1][1] != "O" else "B-") + lab
        space = m.end() < len(clean) and clean[m.end()].isspace()
        toks.append((m.group(), tag, space))
        prev = lab
    return toks


def main(pattern, out):
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for p in sorted(glob.glob(pattern)):
            toks = bio(open(p, encoding="utf-8").read())
            if not toks:
                continue
            fh.write(f"# file={p.split('greek-data/')[-1]}\n")
            fh.write("# text=" + "".join(t + (" " if s else "") for t, _, s in toks).rstrip() + "\n")
            for t, tag, s in toks:
                fh.write(f"{t}\t{tag}\t{'_' if s else 'N'}\n")
            fh.write("\n")
            n += 1
    print(f"{n} chunks -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
