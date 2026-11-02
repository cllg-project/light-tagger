import pycrfsuite

from cllg_tagging import common as bc
from cllg_tagging import crf


def _samples(tmp_path):
    p = tmp_path / "toy.tsv"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(30):
            toks = [("text", "O"), ("[", "B-REF"), (str(i), "I-REF"), ("]", "I-REF"), ("more", "O"), ("text", "O")]
            bc.write_sample(f, f"id={i}", toks)
    return p


def test_train_and_tag_a_toy_model(tmp_path):
    p, model = _samples(tmp_path), tmp_path / "m.crfsuite"
    crf.train(str(model), [str(p)], iters=30)
    res = crf.evaluate(str(model), str(p))
    assert res["REF"]["f"] > 0.9
    tg = pycrfsuite.Tagger()
    tg.open(str(model))
    assert set(tg.labels()) == {"O", "B-REF", "I-REF"}
