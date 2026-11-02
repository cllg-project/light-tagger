from cllg_tagging import common as bc


def test_tokenize_never_crosses_a_mark_boundary():
    text = "cf. Hom. 1.1"
    marks = [None] * 4 + [("BIBL", 1)] * 8
    toks = bc.tokenize(text, marks)
    assert [t for t, _ in toks][:3] == ["cf", ".", "Hom"]
    assert [l for _, l in toks][:2] == ["O", "O"]
    assert toks[2][1] == "B-BIBL" and toks[3][1] == "I-BIBL"


def test_entities_strict_bio():
    assert bc.entities(["O", "B-REF", "I-REF", "O", "B-NOTE"]) == {(1, 3, "REF"), (4, 5, "NOTE")}
    assert bc.entities(["I-REF", "I-REF"]) == set()  # a stray I- opens nothing


def test_score_exact_match_only():
    gold = [["B-REF", "I-REF", "O"]]
    assert bc.score(gold, gold)["REF"]["f"] == 1.0
    res = bc.score(gold, [["B-REF", "O", "O"]])  # wrong extent = one FP and one FN
    assert res["REF"]["p"] == 0.0 and res["REF"]["r"] == 0.0


def test_layered_roundtrip(tmp_path):
    rows = [("(", "B-NOTE", "O"), ("cf", "I-NOTE", "O"), ("Hom", "I-NOTE", "B-BIBL"), (")", "I-NOTE", "I-BIBL")]
    p = tmp_path / "x.tsv"
    with open(p, "w", encoding="utf-8") as f:
        bc.write_layers(f, "id=1", rows)
    (hdr, toks), = bc.read_layers(p)
    assert hdr == {"id": "1"} and [tuple(t) for t in toks] == rows
