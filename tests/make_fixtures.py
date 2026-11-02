"""Regenerate the Python reference data of the JS tests (js/test/fixtures/*.json).

  python tests/make_fixtures.py          # CI re-runs this and fails if the committed fixtures differ
"""
import json
import random
import unicodedata
from pathlib import Path

from cllg_tagging import common as bc
from cllg_tagging import features as bf
from cllg_tagging.tagger import LayeredTagger

OUT = Path(__file__).resolve().parents[1] / "js" / "test" / "fixtures"
SENTENCES = [
    "( cf . Aelian . n . a . 5 , 39 ) καὶ [ 95 ] ἀγαθός ( v . l . ἀγαθός )",
    "Ἐν ἀρχῇ ἦν ὁ λόγος ( cf. Hom. Il. 1.1 ) καὶ ὁ λόγος ἦν πρὸς τὸν θεόν . [ 12 ]",
    "Cic. de Orat. 2, 45 ; Plat. Resp. 514a ; cf. supra p. 3 , n. 2 .",
    "(supra p. 12) text sans note ni référence, simplement des mots.",
    "Chap. IV . § 12 . col. b . p. 245 . 5 10 15 20",
    "",
    "α",
    "𝔄 ① ⅷ ² ǅ ﬁ Σ ς ᾼ ﻿   x",
]


def odd_tokens(rng):
    chars = [chr(c) for c in range(0x20, 0x3400, 3) if not 0xD800 <= c < 0xE000 and unicodedata.category(chr(c)) != "Cn"]
    chars += [chr(c) for c in range(0x1F00, 0x2000, 7)] + list("ǅǈǋǲⅠⅣⅻ²³¹⁴⁵₀₇①⑩½ⒶⓐİıßﬁﬂΣςσᾼᾳ")
    odd = [[c] for c in chars]
    for _ in range(300):
        odd.append([rng.choice(chars) * rng.choice((1, 1, 2)) + rng.choice(("", "", "ʹ", "5", "x", "Α", "ῆ")) for _ in range(rng.randint(3, 9))])
    return odd


def tokenizer_texts(rng):
    pool = [chr(c) for c in range(0x20, 0x30000, 97) if not 0xD800 <= c < 0xE000 and unicodedata.category(chr(c)) != "Cn"]
    pool += [chr(c) for c in list(range(0, 0x20)) + [0x85, 0xA0, 0x1680, 0x2000, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF, 0x180E, 0x200B]]
    return ["".join(rng.choice(pool) if rng.random() < .6 else rng.choice("abc αβγ 12 ( ) . ,") for _ in range(rng.randint(5, 40))) for _ in range(400)]


def main():
    rng = random.Random(9)
    OUT.mkdir(parents=True, exist_ok=True)
    texts = tokenizer_texts(rng)
    dump = lambda name, obj: (OUT / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")  # noqa: E731
    dump("tokens.json", [{"text": t, "tokens": bc.TOKEN.findall(t)} for t in texts])
    sents = [bc.TOKEN.findall(unicodedata.normalize("NFC", s)) for s in SENTENCES] + odd_tokens(rng)
    dump("features.json", [{"tokens": s, "feats": bf.sent_feats(s)} for s in sents])
    lt = LayeredTagger()
    tags = []
    for s in SENTENCES:
        toks = bc.TOKEN.findall(unicodedata.normalize("NFC", s))
        r = lt.tag_tokens(toks)
        tags.append({"text": unicodedata.normalize("NFC", s), "tokens": toks, **r, "annotate": lt.annotate(s)})
    dump("tags.json", tags)
    print(len(texts), "texts,", len(sents), "feature sentences,", len(tags), "tagged sentences")


if __name__ == "__main__":
    main()
