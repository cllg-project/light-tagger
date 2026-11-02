import unicodedata

from cllg_tagging import features as bf


def test_one_feature_list_per_token():
    toks = ["(", "cf", ".", "Hom", ".", "1", ".", "1", ")"]
    feats = bf.sent_feats(toks)
    assert len(feats) == len(toks) and all(isinstance(f, list) and f for f in feats)


def test_deterministic():
    toks = ["Ἐν", "ἀρχῇ", "[", "95", "]"]
    assert bf.sent_feats(toks) == bf.sent_feats(list(toks))


def test_nfc_normalised():
    nfd = unicodedata.normalize("NFD", "ἀγαθός")
    nfc = unicodedata.normalize("NFC", "ἀγαθός")
    assert bf.sent_feats([nfd]) == bf.sent_feats([nfc])
