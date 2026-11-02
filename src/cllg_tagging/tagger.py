"""Python reference tagger (same output as js/src/tagger.js): two CRF layers, NOTE over layer 1, BIBL / REF over layer 2."""
import unicodedata
from pathlib import Path

import pycrfsuite

from . import common as bc
from . import features as bf

MODELS = Path(__file__).resolve().parents[2] / "models"


def spans(text, tokens_spans, tags):
    out, cur = [], None
    for (a, b), t in zip(tokens_spans, tags):
        if cur and not (t.startswith("I-") and t[2:] == cur["label"]):
            out.append(cur)
            cur = None
        if t.startswith("B-"):
            cur = {"label": t[2:], "start": a, "end": b}
        elif cur:
            cur["end"] = b
    if cur:
        out.append(cur)
    for s in out:
        s["text"] = text[s["start"]:s["end"]]
    return out


class LayeredTagger:
    def __init__(self, note_model=MODELS / "crf3_l1.crfsuite", cite_model=MODELS / "crf3_l2.crfsuite"):
        self.note, self.cite = pycrfsuite.Tagger(), pycrfsuite.Tagger()
        self.note.open(str(note_model))
        self.cite.open(str(cite_model))

    def tag_tokens(self, tokens):
        x = bf.sent_feats(tokens)
        return {"note": self.note.tag(x), "cite": self.cite.tag(x)} if tokens else {"note": [], "cite": []}

    def annotate(self, text):
        text = unicodedata.normalize("NFC", text)
        ms = list(bc.TOKEN.finditer(text))
        tags = self.tag_tokens([m.group() for m in ms])
        sp = [(m.start(), m.end()) for m in ms]
        notes, cites = spans(text, sp, tags["note"]), spans(text, sp, tags["cite"])
        for n in notes:
            inner = {c["label"] for c in cites if c["start"] < n["end"] and c["end"] > n["start"]}
            n["kind"] = "NOTEBIBL" if "BIBL" in inner else "NOTEREF" if "REF" in inner else "NOTE"
        return {"notes": notes, "cites": cites}
