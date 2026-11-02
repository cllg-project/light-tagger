"""Sanity checks of the released datasets in data/final (skipped when the LFS files are not checked out)."""
import re
from pathlib import Path

import pytest

FINAL = Path(__file__).resolve().parents[1] / "data" / "final"


def _present():
    f = FINAL / "dev.layers.tsv"
    return f.exists() and not f.read_bytes().startswith(b"version https://git-lfs")


pytestmark = pytest.mark.skipif(not _present(), reason="data/final not checked out (Git LFS)")
SETS = ["train", "dev", "devplus", "test-md", "test-aph"]
TAG = {1: re.compile(r"^(O|[BI]-NOTE)$"), 2: re.compile(r"^(O|[BI]-(BIBL|REF))$")}
COUNTS = {  # samples
    "train": 91558, "dev": 4012, "devplus": 1000, "test-md": 322, "test-aph": 90}


def read(path):
    out, hdr, toks = [], "", []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# text="):
            continue
        if line.startswith("# "):
            hdr = line[2:]
        elif line:
            toks.append(tuple(line.split("\t")))
        elif toks:
            out.append((hdr, toks))
            hdr, toks = "", []
    if toks:
        out.append((hdr, toks))
    return out


def bio_ok(tags):
    prev = "O"
    for t in tags:
        if t.startswith("I-") and (prev == "O" or prev[2:] != t[2:]):
            return False
        prev = t
    return True


@pytest.mark.parametrize("name", SETS)
def test_layers_are_valid_bio(name):
    rows = read(FINAL / f"{name}.layers.tsv")
    assert len(rows) == COUNTS[name]
    for _, toks in rows:
        assert all(len(t) == 3 for t in toks)
        for lay in (1, 2):
            tags = [t[lay] for t in toks]
            assert all(TAG[lay].match(x) for x in tags)
            assert bio_ok(tags)


@pytest.mark.parametrize("name", ["dev", "test-md", "test-aph"])
def test_per_layer_files_match_the_layered_file(name):
    layered = read(FINAL / f"{name}.layers.tsv")
    for lay in (1, 2):
        flat = read(FINAL / f"{name}.l{lay}.tsv")
        assert len(flat) <= len(layered)
        by_text = {tuple(t[0] for t in toks): [t[lay] for t in toks] for _, toks in layered}
        for _, toks in flat[:200]:
            assert by_text[tuple(t[0] for t in toks)] == [t[1] for t in toks]


@pytest.mark.parametrize("name", ["test-md", "test-aph"])
def test_test_sets_are_not_in_train(name):
    train = {tuple(t[0] for t in toks) for _, toks in read(FINAL / "train.layers.tsv")}
    leaked = [1 for _, toks in read(FINAL / f"{name}.layers.tsv") if tuple(t[0] for t in toks) in train]
    assert not leaked
