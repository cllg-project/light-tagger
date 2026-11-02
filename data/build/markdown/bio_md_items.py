"""List the <note>/<ref> spans of the md pages (train+dev) for agent tagging -> scripts/out/bio/md_items.json
and agent input batches md_in_<k>.txt (notes: 3 batches x 70; refs: 4 batches x 203)."""
import glob
import json
import os
import re

ROOT = os.path.expanduser("~/Downloads/greek-data")
OUT = os.path.join(os.path.dirname(__file__), "out/bio")
SPAN = re.compile(r"<(note|ref)>(.*?)</\1>", re.S)


def flat(s):
    return " ".join(re.sub(r"</?(?:note|ref)>|<tab/>", " ", s).split())


def main():
    items = []
    for path in sorted(glob.glob(f"{ROOT}/[12]*_split/*/*.md")):
        raw = open(path, encoding="utf-8").read()
        page = os.path.relpath(path, ROOT)
        for m in SPAN.finditer(raw):
            items.append({"id": f"{m.group(1)[0]}{sum(i['kind'] == m.group(1) for i in items)}", "kind": m.group(1),
                          "page": page, "off": m.start(), "text": m.group(2),
                          "left": flat(raw[max(0, m.start() - 300):m.start()])[-250:],
                          "right": flat(raw[m.end():m.end() + 200])[:150]})
    json.dump(items, open(f"{OUT}/md_items.json", "w", encoding="utf-8"), ensure_ascii=False)
    notes = [i for i in items if i["kind"] == "note"]
    refs = [i for i in items if i["kind"] == "ref"]
    batches = [notes[i:i + 70] for i in range(0, len(notes), 70)] + [refs[i:i + 203] for i in range(0, len(refs), 203)]
    for k, b in enumerate(batches):
        with open(f"{OUT}/md_in_{k}.txt", "w", encoding="utf-8") as f:
            for i in b:
                f.write(f"{i['id']} [{os.path.basename(i['page'])}]: {i['left']} ⟦{i['text']}⟧ {i['right']}\n")
    print(len(notes), len(refs), len(batches))


main()
