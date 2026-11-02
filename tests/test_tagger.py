import json

import pytest

from cllg_tagging import export
from cllg_tagging.tagger import MODELS, LayeredTagger


@pytest.fixture(scope="module")
def tagger():
    return LayeredTagger()


def test_annotate_finds_nested_note_and_reference(tagger):
    out = tagger.annotate("( cf . Aelian . n . a . 5 , 39 ) καὶ [ 95 ] ἀγαθός ( v . l . ἀγαθός )")
    kinds = {n["kind"] for n in out["notes"]}
    assert "NOTEBIBL" in kinds
    assert any(c["label"] == "REF" and c["text"] == "[ 95 ]" for c in out["cites"])


def test_offsets_slice_the_text(tagger):
    text = "Ἐν ἀρχῇ ἦν ὁ λόγος ( cf. Hom. Il. 1.1 ) καὶ [ 12 ]"
    out = tagger.annotate(text)
    for s in out["notes"] + out["cites"]:
        assert text[s["start"]:s["end"]] == s["text"]


def test_empty_text(tagger):
    assert tagger.annotate("") == {"notes": [], "cites": []}


def test_export_matches_model(tmp_path):
    n_labels, n_attrs, n_trans = export.export(MODELS / "crf3_l1.crfsuite", tmp_path / "m.json")
    data = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert len(data["labels"]) == n_labels and len(data["attrs"]) == n_attrs and len(data["trans"]) == n_trans
    assert "B-NOTE" in data["labels"] and "O" in data["labels"]


def test_shipped_json_models_are_in_sync_with_the_crfsuite_models(tmp_path):
    for js, crf in (("note.json", "crf3_l1.crfsuite"), ("cite.json", "crf3_l2.crfsuite")):
        export.export(MODELS / crf, tmp_path / js)
        assert (tmp_path / js).read_bytes() == (MODELS / js).read_bytes(), f"{js} is stale: re-run cllg_tagging.export"
