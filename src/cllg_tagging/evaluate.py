"""Per-class precision / recall / F1 of the final CRFs on dev, devplus, test-md and test-aph.

  python -m cllg_tagging.evaluate [DATA_REPO] [OUT.json]       (DATA_REPO defaults to the data/ folder of this repository)

Samples flagged unknown=1 (a note holds a citation of unknown extent) are not scored for layer 2."""
import json
import sys
from pathlib import Path

import pycrfsuite

from . import common as bc
from . import features as bf

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
SETS = ("dev", "devplus", "test-md", "test-aph")


def evaluate(data_repo, sets=SETS):
    taggers = {}
    for lay in (1, 2):
        taggers[lay] = pycrfsuite.Tagger()
        taggers[lay].open(str(MODELS / f"crf3_l{lay}.crfsuite"))
    report = {}
    for name in sets:
        rows = bc.read_layers(f"{data_repo}/final/{name}.layers.tsv")
        for lay in (1, 2):
            gold, pred = [], []
            for hdr, toks in rows:
                if lay == 2 and hdr.get("unknown") == "1":
                    continue
                gold.append([t[lay] for t in toks])
                pred.append(taggers[lay].tag(bf.sent_feats([t[0] for t in toks])))
            report[f"layer{lay}/{name}"] = bc.score(gold, pred)
    return report


def main(argv):
    report = evaluate(argv[0] if argv else ROOT / "data")
    for key, res in report.items():
        print("==", key)
        for cls, v in res.items():
            print(f"  {cls:7s}", "  ".join(f"{k}={x:.3f}" if isinstance(x, float) else f"{k}={x}" for k, x in v.items()))
    if len(argv) > 1:
        Path(argv[1]).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
