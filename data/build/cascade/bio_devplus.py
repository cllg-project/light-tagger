"""Extract N new layered samples (<=100 tokens) from the DEV works, stratified over nested-note / note / bibl / ref,
CRF-annotate them (layers), write agent inputs.  Usage: bio_devplus.py extract N | ... """
import glob, json, os, random, sys, zlib
from multiprocessing import Pool
from cllg_tagging import common as bc
MAXT = 100
DIR = os.environ.get('DP_DIR', 'out/bio/dp')
WORKS = os.environ.get('DP_WORKS', 'dev')
PFX = os.environ.get('DP_PREFIX', 'D')
def windows100(toks):
    if len(toks) <= MAXT: return [toks]
    cont = lambda t: t[1].startswith("I-") or t[2].startswith("I-")  # noqa: E731
    starts = [i for i, t in enumerate(toks) if t[1].startswith("B-") or t[2].startswith("B-")]
    out, end = [], 0
    for s in starts or [len(toks) // 2]:
        if s < end: continue
        a, b = max(end, s - 30), min(len(toks), s + 70)
        while a > 0 and cont(toks[a]): a -= 1
        while b < len(toks) and cont(toks[b]): b += 1
        if b - a <= MAXT: out.append(toks[a:b])
        end = b
    return out
def spans(tags):
    out, i, n = [], 0, len(tags)
    while i < n:
        if tags[i] != 'O':
            c = tags[i][2:]; j = i
            while j + 1 < n and tags[j + 1] == 'I-' + c: j += 1
            out.append((i, j, c)); i = j + 1
        else: i += 1
    return out
def stratum(T):
    s1, s2 = spans([t[1] for t in T]), spans([t[2] for t in T])
    if any(x <= b and y >= a for a, b, _ in s1 for x, y, _ in s2): return 'nested'
    if s1: return 'note'
    if any(c == 'BIBL' for _, _, c in s2): return 'bibl'
    if s2: return 'ref'
    return None
def extract(N):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "corpus"))
    import tei_to_layers as tl
    tl.windows = windows100
    dev_blocks = set()
    for p in ('../bio-layers-d-dev.tsv', '../bio-layers-train.tsv', '../bio-devplus.tsv'):
        for h, _ in bc.read_layers(p): dev_blocks.add(h['id'].rsplit('.', 1)[0])
    golds = tl.gold_parens()
    files = sorted(f for f in glob.glob(str(tl.ROOT / "data/**/*.xml"), recursive=True) if not f.endswith("metadata.xml"))
    jobs = [(f, golds.get(str(tl.Path(f).relative_to(tl.ROOT)), [])) for f in files]
    jobs = [j for j in jobs if (zlib.crc32(tl.Path(j[0]).name.rsplit(".cllg", 1)[0].encode()) % 10 == 0) == (WORKS == 'dev')]
    pool = {k: [] for k in ('nested', 'note', 'bibl', 'ref')}
    with Pool(14) as p:
        for urn, wins in p.imap_unordered(tl.process, jobs, chunksize=2):
            for wid, kind, T, unk in wins:
                if wid.rsplit('.', 1)[0] in dev_blocks or len(T) < 8: continue
                s = 'nested' if unk and any(t[1] != 'O' for t in T) else stratum(T)
                if s: pool[s].append((wid, T))
    print({k: len(v) for k, v in pool.items()})
    random.seed(5); pick = []
    for k, v in pool.items():
        random.shuffle(v); pick += [(k, wid, T) for wid, T in v[:N // 4]]
    rest = [(k, wid, T) for k, v in pool.items() for wid, T in v[N // 4:]]; random.shuffle(rest)
    pick += rest[:N - len(pick)]; random.shuffle(pick)
    S = [{"sid": f"{PFX}{k}", "wid": wid, "stratum": st, "tokens": [t[0] for t in T], "gold": [[t[1], t[2]] for t in T], "unknown": bool(st == "nested" and not stratum(T) == "nested")} for k, (st, wid, T) in enumerate(pick)]
    os.makedirs((DIR + ''), exist_ok=True); json.dump(S, open((DIR + '/samples.json'), 'w'), ensure_ascii=False)
    print(len(S), 'samples', {k: sum(s['stratum'] == k for s in S) for k in pool}, 'avg tokens', sum(len(s['tokens']) for s in S) / len(S))
def annotate():
    import pycrfsuite
    from cllg_tagging import features as bf
    t1, t2 = pycrfsuite.Tagger(), pycrfsuite.Tagger(); t1.open('out/bio/crf_l1.crfsuite'); t2.open('out/bio/crf_l2.crfsuite')
    S = json.load(open((DIR + '/samples.json')))
    for s in S:
        x = bf.sent_feats(s['tokens']); s['crf'] = [t1.tag(x), t2.tag(x)]
    json.dump(S, open((DIR + '/samples.json'), 'w'), ensure_ascii=False); print('annotated', len(S))
def render(toks, l1, l2):
    """nested display: {NOTE: ...} for layer 1, [BIBL: ...] / [REF: ...] for layer 2"""
    o1, o2 = {a: c for a, b, c in spans(l1)}, {a: c for a, b, c in spans(l2)}
    e1, e2 = {b for a, b, c in spans(l1)}, {b for a, b, c in spans(l2)}
    out = []
    for i, t in enumerate(toks):
        if i in o1: out.append('{' + o1[i] + ':')
        if i in o2: out.append('[' + o2[i] + ':')
        out.append(t)
        if i in e2: out.append(']')
        if i in e1: out.append('}')
    return ' '.join(out)
def write_inputs(src, outdir, per, ids=None):
    S = json.load(open((DIR + '/samples.json'))); os.makedirs(outdir, exist_ok=True)
    rows = [s for s in S if ids is None or s['sid'] in ids]
    for b in range(0, len(rows), per):
        open(f'{outdir}/in_{b // per}.txt', 'w').write('\n'.join(f'{s["sid"]}\t' + render(s['tokens'], *src(s)) for s in rows[b:b + per]) + '\n')
    print(len(rows), 'samples ->', -(-len(rows) // per), 'files in', outdir)
if __name__ == '__main__':
    if sys.argv[1] == 'extract': extract(int(sys.argv[2]))
    if sys.argv[1] == 'annotate': annotate()
    if sys.argv[1] == 'sonnet_in': write_inputs(lambda s: s['crf'], (DIR + '/sonnet'), int(sys.argv[2]))


# ---- applying agent edits -------------------------------------------------------------------------------------------
def parse_edits(path_glob):
    import re
    E, bad = {}, 0
    for f in sorted(glob.glob(path_glob)):
        for l in open(f, encoding='utf-8'):
            p = l.rstrip('\n').split('\t')
            if len(p) < 4 or p[1] not in '+-' or p[2] not in ('NOTE', 'BIBL', 'REF'): bad += 1 if l.strip() else 0; continue
            text, unsure = p[3], False
            if text.endswith(' ?'): text, unsure = text[:-2], True
            m = re.search(r' #(\d+)$', text); nth = 1
            if m: nth = int(m.group(1)); text = text[:m.start()]
            E.setdefault(p[0], []).append((p[1], p[2], text.split(' '), nth, unsure))
    return E, bad
def apply_edits(tokens, layers, edits):
    L = [list(layers[0]), list(layers[1])]; miss = 0
    for op in ('-', '+'):
        for o, cls, words, nth, _ in edits:
            if o != op: continue
            n, occ = len(words), [i for i in range(len(tokens) - len(words) + 1) if tokens[i:i + len(words)] == words]
            if len(occ) < nth: miss += 1; continue
            a = occ[nth - 1]; b = a + n - 1; lay = L[0] if cls == 'NOTE' else L[1]
            if op == '-':
                for x, y, c in spans(lay):
                    if c == cls and x <= b and y >= a:
                        for i in range(x, y + 1): lay[i] = 'O'
            else:
                for i in range(a, b + 1): lay[i] = ('B-' if i == a else 'I-') + cls
    for lay in L:
        for i, l in enumerate(lay):
            if l.startswith('I-') and (i == 0 or lay[i - 1] == 'O' or lay[i - 1][2:] != l[2:]): lay[i] = 'B-' + l[2:]
    return L, miss
def merge(stage):
    S = json.load(open((DIR + '/samples.json')))
    if stage == 'sonnet':
        E, bad = parse_edits((DIR + '/sonnet/ans_*.txt')); base, key = 'crf', 'sonnet'
    else:
        E, bad = parse_edits((DIR + '/opus/ans_*.txt')); base, key = 'sonnet', 'final'
    miss = ed = 0
    for s in S:
        e = E.get(s['sid'])
        if stage == 'opus' and s['sid'] not in s.get('opus_ids', ()) and not e and False: pass
        if e:
            s[key], m = apply_edits(s['tokens'], s[base], e); miss += m; ed += 1
        else: s[key] = s[base]
    json.dump(S, open((DIR + '/samples.json'), 'w'), ensure_ascii=False)
    print(stage, 'samples edited', ed, 'edit lines', sum(len(v) for v in E.values()), 'unmatched', miss, 'malformed lines', bad)
    if stage == 'sonnet':
        ids = {s['sid'] for s in S if s['sid'] in E}
        write_inputs(lambda s: s['sonnet'], (DIR + '/opus'), 50, ids)
if __name__ == '__main__' and sys.argv[1] in ('merge_sonnet', 'merge_opus'): merge(sys.argv[1].split('_')[1])
