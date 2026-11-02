"""Narrow-window verification batches (+-W tokens around each candidate, merged), resumable.
Files out/bio/v3/in_N.txt; the agent writes ans_N.txt right after each file; meta.json keeps (file, sample, runs) per id."""
import json, random, sys
from cllg_tagging import common as bc
import bio_verify_pick as vp
W = 10
def render(toks, rs):
    n = len(toks); iv = []
    for a, b in rs:
        lo, hi = max(0, a - W), min(n, b + W + 1)
        if iv and lo <= iv[-1][1] + 2: iv[-1][1] = max(iv[-1][1], hi)
        else: iv.append([lo, hi])
    start = {a: k for k, (a, b) in enumerate(rs, 1)}; end = {b for a, b in rs}
    out = []
    for q, (lo, hi) in enumerate(iv):
        s = [] if q == 0 and lo == 0 else ['…']
        cur = None
        for i in range(lo, hi):
            t, l = toks[i]; c = None if l == 'O' else l[2:]
            if c != cur:
                if cur: s.append('}')
                if c: s.append('{' + c + ':')
                cur = c
            if i in start: s.append(f'⟦{start[i]}')
            s.append(t)
            if i in end: s.append('⟧')
        if cur: s.append('}')
        out += s
    if iv and iv[-1][1] < n: out.append('…')
    return ' '.join(out)
if __name__ == '__main__':
    NEW, PER = int(sys.argv[1]), int(sys.argv[2])
    random.seed(23)
    v2 = json.load(open('out/bio/v2/meta.json'))
    used = {(m['file'], m['sample']) for m in v2} | {(i['file'], i['sample']) for i in json.load(open('out/bio/verify/items.json'))}
    cache = {f: bc.read_samples(f) for f in ('../bio-tei-train.tsv', '../bio-tei-dev.tsv')}
    pool = [(f, si) for f in cache for si, (_, toks) in enumerate(cache[f]) if (f, si) not in used and vp.runs(toks)]
    print(len(pool), 'unused samples with candidates')
    meta = [dict(m, sid=f'S{m["sid"]}') for m in v2[:100]]            # calibration: already labelled by Opus
    for k, (f, si) in enumerate(random.sample(pool, NEW)):
        meta.append({"sid": f'T{k}', "file": f, "sample": si, "runs": vp.runs(cache[f][si][1])})
    json.dump(meta, open('out/bio/v3/meta.json', 'w'))
    for b in range(0, len(meta), PER):
        open(f'out/bio/v3/in_{b // PER}.txt', 'w').write('\n'.join(f'{m["sid"]}\t' + render(cache[m['file']][m['sample']][1], [tuple(r) for r in m['runs']]) for m in meta[b:b + PER]) + '\n')
    print(len(meta), 'samples,', (len(meta) + PER - 1) // PER, 'files')
