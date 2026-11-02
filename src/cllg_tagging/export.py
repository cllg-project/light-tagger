"""Export a python-crfsuite model to the compact JSON read by js/src/tagger.js.

  python -m cllg_tagging.export MODEL.crfsuite OUT.json"""
import json
import sys

import pycrfsuite


def export(model_path, out_path):
    tg = pycrfsuite.Tagger()
    tg.open(str(model_path))
    info = tg.info()
    labels = tg.labels()
    li = {l: i for i, l in enumerate(labels)}
    attrs = {}
    for (a, l), w in info.state_features.items():
        attrs.setdefault(a, []).append([li[l], round(w, 6)])
    trans = [[li[a], li[b], round(w, 6)] for (a, b), w in info.transitions.items()]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"labels": labels, "attrs": attrs, "trans": trans}, fh, ensure_ascii=False, separators=(",", ":"))
    return len(labels), len(attrs), len(trans)


if __name__ == "__main__":
    print("%d labels, %d attributes, %d transitions" % export(sys.argv[1], sys.argv[2]))
