"""Synthetic REF-in-text samples (make_bio_samples: identifier chain rendered at start/middle/dropped) -> bio-refmid-train.tsv.
CITE -> BIBL, dev works excluded (same crc32 split as tei_to_bio). Usage: refmid_bio.py N SEED  (run from the repo root)"""
import sys
import zlib

sys.path.insert(0, "scripts")
import make_bio_samples as mb  # noqa: E402

n, seed = int(sys.argv[1]), int(sys.argv[2])
tmp = "scripts/out/bio/_refmid_raw.tsv"
mb.main(n, seed, tmp)
keep = True
with open(tmp, encoding="utf-8") as f, open("bio-refmid-train.tsv", "w", encoding="utf-8") as out:
    for line in f:
        if line.startswith("# id="):
            urn = line.split("urn=")[1].split()[0]
            keep = zlib.crc32(urn.encode()) % 10 != 0
        if not keep or line.startswith("# text="):
            continue
        if "\t" in line:
            t, l, _ = line.rstrip("\n").split("\t")
            line = f"{t}\t{l.replace('CITE', 'BIBL')}\n"
        out.write(line)
