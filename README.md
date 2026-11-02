# cllg-light-tagging

A light tagger that finds **notes** (`NOTE`), **bibliographic citations** (`BIBL`, `CIT` in the article) and **references** (`REF`: section,
chapter, page, column, line locators and page/column/milestone breaks written as text) in running Greek and Latin text.
It is a pair of linear-chain CRFs with hand-written features, so the exact same model runs in Python and in **pure JavaScript**
(no native code, no dependencies, about 4 MB of JSON weights, about 1,900 tokens in 100 ms in Node).

The repository holds the software (`src/`, `js/`), the trained models, the data (`data/`), the tests and the article material.

## Scheme

Two independent BIO layers that may overlap (a note can contain a citation):

| Layer | Tags |
|---|---|
| 1 | `NOTE` |
| 2 | `BIBL`, `REF` |

`NOTEBIBL` / `NOTEREF` are derived (a NOTE span overlapping a BIBL / REF span), not learned.

![scheme](docs/figures/2-annotation-scheme.png)

## Use

JavaScript (Node >= 18 or a bundler; models are plain JSON in `models/`):

```js
const { LayeredTagger } = require("./js/src/tagger.js");
const lt = new LayeredTagger(require("./models/note.json"), require("./models/cite.json"));
const { notes, cites } = lt.annotate("( cf . Aelian . n . a . 5 , 39 ) καὶ [ 95 ]");
// notes: [{start, end, text, label: "NOTE", kind: "NOTEBIBL"}], cites: [{start, end, text, label: "BIBL" | "REF"}]
```

Python (3.10 - 3.12, python-crfsuite):

```bash
pip install -e ".[dev]"
python -c "from cllg_tagging import LayeredTagger as T; print(T().annotate('( cf . Aelian . n . a . 5 , 39 ) [ 95 ]'))"
```

## Layout

```
src/cllg_tagging/   common (tokeniser, BIO, scoring), features, crf (train), tagger, export (crfsuite -> JSON), evaluate (per-class P/R/F1), review (annotation app)
js/                 src/ (tagger, features, exact copy of Python's unicode predicates), test/ (parity tests + fixtures)
data/               final/ datasets (Git LFS), annotations/ (manual corrections, agent rounds), build/ (recipes of the files, grouped by stage)
models/             crf3_l1 / crf3_l2 (.crfsuite) and their JSON exports note.json / cite.json
tests/              pytest, and make_fixtures.py (Python reference data for the JS tests)
baselines/          philberta.py: frozen PhilBERTa (encoder never updated) + linear heads (needs torch and transformers; `cd baselines && python philberta.py OUT`)
docs/               article section (LaTeX), figures (.drawio, .svg, .png) and make_figures.py
```

## Train, evaluate, export

The datasets are in `data/final/` (`*.l1.tsv` for the NOTE layer, `*.l2.tsv` for BIBL / REF).

```bash
python -m cllg_tagging.crf train models/crf3_l1.crfsuite data/final/train.l1.tsv     # about 15 min
python -m cllg_tagging.crf eval  models/crf3_l1.crfsuite data/final/dev.l1.tsv
python -m cllg_tagging.export models/crf3_l1.crfsuite models/note.json
python -m cllg_tagging.evaluate                                # per-class P / R / F1 on dev, devplus, test-md, test-aph
python tests/make_fixtures.py                                                                      # then run the JS tests
```

Features (`features.py`, ported in `js/src/features.js`): word shape and case flags, NFC-normalised character prefixes and suffixes of one to
four characters, a window of three tokens on each side, bracket depth and a small lexicon of apparatus words.

## Data

Token-level BIO annotation on the two layers above; samples carry a header such as `id=... src=... split=...`, and `unknown=1` marks a note known to contain a citation or reference of unknown extent (excluded from layer-2 training and scoring).

### `data/final/`

| File prefix | Samples | NOTE | BIBL | REF | Role |
|---|---|---|---|---|---|
| `train` | 91,558 | 16,234 | 44,618 | 70,552 | training |
| `dev` | 4,012 | 871 | 1,393 | 2,778 | held-out works |
| `devplus` | 1,000 | 228 | 518 | 958 | 1,000 checked samples from the dev works |
| `test-md` | 322 | 24 | 20 | 1,087 | out-of-distribution: external page-level markdown |
| `test-aph` | 90 | – | 90 | – | out-of-distribution: Année Philologique test set |

Each set comes as `NAME.layers.tsv` (`token<TAB>layer1<TAB>layer2`) and as the per-layer files `NAME.l1.tsv` (NOTE) and `NAME.l2.tsv` (BIBL / REF) used for training
(about 545 MB in all, stored with Git LFS).

The training set is: windows of up to 150 tokens from the TEI-XML corpus (works split between train and dev), synthetic `REF` variants (citeStructure numbers in Arabic,
Roman and Greek numerals), 3,000 cascade-annotated samples from works outside train and dev, the Année Philologique gold and dev files (507 samples) and
the corrections made by hand in the review application (including 39 pasted passages). The Année Philologique corpus is the one distributed with HuCit
(Romanello and Pasin, 2017; <https://github.com/mromanello/APh_Corpus>); its tags `REFAUWORK` and `REFSCOPE` both become `BIBL`, an author/work directly
followed by its scope forming one span, and all other tags become `O`. A window that opens inside a span, or an `I-` tag with no `B-` in the source, is
repaired to start a span (4 samples). Samples with `unknown=1` are not scored for layer 2, which is why some counts in the evaluation are smaller (dev: 1,378 `BIBL`).

### `data/annotations/`

| Path | Content |
|---|---|
| `flags2.jsonl` | the 97 manual corrections (md-test 34, new samples 34, dev 1, pasted text 28) applied to the final files |
| `dp/`, `dp2/` | the 1,000 and 3,000 cascade samples: prompt, samples, answers of the first pass (Claude Sonnet 5.5) and of the arbitration (Claude Opus 5.5) |
| `v2/`, `v3/`, `v4/`, `r1/` | earlier verification rounds of unannotated digit / abbreviation candidates |
| `md/` | prompt and answers for the re-tagging of the markdown pages |

Cascade: a CRF pre-annotates, `claude-sonnet-5-5` checks every candidate in a narrow window, labels on which it agrees with the CRF are accepted, disagreements go to
`claude-opus-5-5`, and the result is reviewed by hand. `claude-haiku-5-5` and Qwen3.6-35B-A3B (`qwen/qwen3.6-35b-a3b`, run locally with LM Studio) were compared with
Opus on a sample (label agreement 90% and 58%, with spans 70% and lower) and not used.

### `data/build/`

The recipe of the files, one script per stage (13 files):

| Folder | Scripts |
|---|---|
| `corpus/` | `tei_to_layers.py` (windows from the TEI corpus, with `make_bio_samples.py`, `paren_classifier.py`), `refmid_bio.py` (synthetic `REF`) |
| `cascade/` | `bio_verify_pick.py`, `bio_verify3.py`, `bio_vround.py`, `bio_devplus.py` (pick candidates, merge answers, apply) |
| `markdown/` | `md_to_bio.py`, `bio_md_items.py`, `bio_md_test.py` (external markdown test pages) |
| `assemble/` | `bio_layers.py`, `bio_final.py` (apply corrections, convert Année Philologique, write `final/`) |

They import `cllg_tagging` (`pip install -e .`) and were written for the original working tree: paths such as `../bio-*.tsv` and `intermediate/`
refer to intermediate files that are not shipped, and they need the TEI corpus, which is not part of this repository. Only `data/final/` is needed to train and evaluate.
The markdown scripts give the file before the later manual corrections, which are in `annotations/flags2.jsonl`.

## Review application

`python -m cllg_tagging.review serve FILE.layers.tsv` starts a local app (port 8766, standard library only) to look at and correct the annotation
on both layers: colour-coded spans, range selection, keys `n` / `b` / `r` to label, `z` / `x` / `0` to clear, a *Paste text* button that
annotates any text with the CRFs and an *Export* button. Corrections go to `flags.jsonl` (or `$CLLG_FLAGS`);
`python -m cllg_tagging.review apply IN.tsv flags.jsonl OUT.tsv` writes the corrected file.

## Results (entity-level P / R / F1, exact match)

CRF (`crf3`) against PhilBERTa, same training set. PhilBERTa is **frozen**: hidden states are cached once and only a linear head per layer is trained (20 epochs, AdamW, BIO-constrained Viterbi); the encoder is not fine-tuned. Full table, with PhilBERTa, in `docs/article-section.tex`.

| Set | Class | CRF P | R | F1 | PhilBERTa F1 | n |
|---|---|---|---|---|---|---|
| dev | NOTE | 0.850 | 0.685 | 0.759 | 0.539 | 871 |
| dev | BIBL | 0.726 | 0.740 | 0.733 | 0.523 | 1,378 |
| dev | REF | 0.861 | 0.919 | 0.889 | 0.755 | 2,768 |
| devplus | NOTE | 0.947 | 0.781 | 0.856 | 0.424 | 228 |
| devplus | BIBL | 0.873 | 0.757 | 0.811 | 0.582 | 518 |
| devplus | REF | 0.959 | 0.931 | 0.945 | 0.884 | 958 |
| test-md | NOTE | 0.692 | 0.375 | 0.486 | 0.000 | 24 |
| test-md | BIBL | 0.069 | 0.100 | 0.082 | 0.026 | 20 |
| test-md | REF | 0.914 | 0.663 | 0.769 | 0.652 | 1,087 |
| test-aph | BIBL | 0.230 | 0.189 | 0.207 | 0.155 | 90 |

References are reliable on held-out works but recall drops to 0.66 on external markdown pages, whose citation styles are absent from the TEI encoding;
citations transfer badly. The test-md NOTE and BIBL rows rest on 24 and 20 entities.

## Tests and parity

`pytest` checks every released data file (valid BIO on both layers, sample counts, per-layer files consistent, no test sample in the training set; skipped without the LFS files) and covers tokeniser, scoring, features, training, export and the tagger. `npm test` (in `js/`) compares the JavaScript tokeniser,
features, tags and `annotate()` output with fixtures produced by the Python implementation, including odd Unicode (superscript digits,
Roman-numeral capitals, U+FEFF, whitespace variants); `js/src/unicode_py.js` is generated from Python's own character predicates
(`cllg_tagging.unicode_gen`) because JavaScript's `\p{..}` classes differ. CI (`.github/workflows/tests.yml`) runs both and checks that the
fixtures are current.

## Annotation method

Labels come from the encoding of the corpus, completed by a cascade: a CRF pre-annotates, **Claude Sonnet 5.5** (`claude-sonnet-5-5`) checks
each candidate, labels on which both agree are accepted, disagreements go to **Claude Opus 5.5** (`claude-opus-5-5`), and the result is
reviewed by hand. Compared with Opus on a sample of candidates, Claude Haiku 5.5 agreed on 90% of the labels but only 70% with spans, and
Qwen3.6-35B-A3B (`qwen/qwen3.6-35b-a3b`) run locally in LM Studio on 58%, so neither was used. The cascade is described in the Data section. The Année Philologique gold files are from the HuCit project (Romanello and Pasin, 2017;
<https://github.com/mromanello/APh_Corpus>).
