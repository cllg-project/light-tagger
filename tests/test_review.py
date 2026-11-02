from cllg_tagging import common as bc
from cllg_tagging import review


def test_flat_to_layers(tmp_path):
    p = tmp_path / "flat.tsv"
    with open(p, "w", encoding="utf-8") as f:
        bc.write_sample(f, "id=1", [("(", "B-NOTEBIBL"), ("cf", "I-NOTEBIBL"), (")", "I-NOTEBIBL"), ("[", "B-REF"), ("1", "I-REF"), ("]", "I-REF")])
    (h, T), = review.flat_to_layers(p)
    assert h["unknown"] == "1"  # the inner extent of the bibl is not known
    assert [t[1] for t in T[:3]] == ["B-NOTE", "I-NOTE", "I-NOTE"] and [t[2] for t in T[3:]] == ["B-REF", "I-REF", "I-REF"]
