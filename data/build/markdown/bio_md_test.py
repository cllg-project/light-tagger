"""Re-tagged md pages -> bio-md-test.tsv (test set). Span labels come from the agent answers
(scripts/out/bio/md_ans_*): notes by two annotators (a/b; disagreements resolved below), refs by one.
tab, markdown line markers dropped; O spans become plain text. Usage: bio_md_test.py"""
import glob
import json
import os
import re
from collections import Counter

from cllg_tagging import common as bc

ROOT = os.path.expanduser("~/Downloads/greek-data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out/bio")
SPAN = re.compile(r"<(note|ref)>(.*?)</\1>", re.S)
MD_LINE = re.compile(r"^[ \t]*(?:#+[ \t]+|[-*>][ \t]+)", re.M)
# 20 footnote call markers "(1)" in Porphyrius FHG: annotators split REF/NOTE; they are note calls -> NOTE
TIEBREAK = "NOTE"


def answers():
    ans = {}
    for path in sorted(glob.glob(f"{OUT}/md_ans_*.jsonl")):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            ans.setdefault(d["id"], []).append(d["label"])
    return {i: (v[0] if len(set(v)) == 1 else TIEBREAK) for i, v in ans.items()}


def main():
    ans = answers()
    items = json.load(open(f"{OUT}/md_items.json", encoding="utf-8"))
    by_page = {}
    for it in items:
        by_page.setdefault(it["page"], []).append(it)
    n = Counter()
    with open("../bio-md-test.tsv", "w", encoding="utf-8") as f:
        for page, its in by_page.items():
            raw = MD_LINE.sub("", open(f"{ROOT}/{page}", encoding="utf-8").read())
            text, marks, pos = "", [], 0
            for k, m in enumerate(SPAN.finditer(raw)):
                chunk = raw[pos:m.start()].replace("<tab/>", " ")
                text += chunk
                marks += [None] * len(chunk)
                inner = m.group(2)
                lab = ans[its[k]["id"]]
                assert inner == its[k]["text"]
                text += inner
                marks += [None if lab == "O" else (lab, k)] * len(inner)
                n[lab] += 1
                pos = m.end()
            tail = raw[pos:].replace("<tab/>", " ")
            text, marks = text + tail, marks + [None] * len(tail)
            text = re.sub(r"<(?!/)[^>]*>", " ", text) if False else text
            bc.write_sample(f, f"id=md urn={os.path.basename(page)[:-3]} src=md", bc.tokenize(text, marks))
    print(len(by_page), dict(n))


main()
