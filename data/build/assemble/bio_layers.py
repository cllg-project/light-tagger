"""Layered data tools.
 fix      : map the agent corrections (flat positions) onto bio-layers-{train,d-dev}.tsv by sample id -> *.fixed.tsv
 split    : write per-layer flat TSVs  l1 (NOTE) and l2 (BIBL/REF; samples with unknown=1 dropped) for train / dev / md-test / refmid
 """
import collections, json, sys
from cllg_tagging import common as bc
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "cascade"))
import bio_vround as vr
SETS = {'train': ('../bio-tei-train.tsv', '../bio-layers-train.tsv'), 'dev': ('../bio-tei-dev.tsv', '../bio-layers-d-dev.tsv')}
def fix():
    fixes = vr.collect(); st = collections.Counter()
    by = collections.defaultdict(list)
    for r in fixes: by[(r['file'], r['sample'])].append(r)
    for name, (old, new) in SETS.items():
        O = bc.read_samples(old); N = bc.read_layers(new); idx = {n[0]['id']: i for i, n in enumerate(N)}
        S = [(h, [list(t) for t in toks]) for h, toks in N]
        for (f, si), rs in by.items():
            if f != old: continue
            h, toks = O[si]; j = idx.get(h.get('id'))
            if j is None or [t for t, _ in toks] != [t[0] for t in S[j][1]]: st['unmapped ' + name] += len(rs); continue
            T = S[j][1]; n = len(T)
            for r in sorted(rs, key=lambda r: r['b'] - r['a'], reverse=True):
                lay = 1 if r['label'].startswith('NOTE') else 2
                a, b = max(0, r['a']), min(n - 1, r['b'])
                if (a > 0 and T[a][lay].startswith('I-')) or (b + 1 < n and T[b + 1][lay].startswith('I-')): st['skipped (cuts same-layer entity) ' + r['src'][:5]] += 1; continue
                cls = 'NOTE' if lay == 1 else r['label']
                for i in range(a, b + 1): T[i][lay] = ('B-' if i == a else 'I-') + cls
                if r['label'] in ('NOTEBIBL', 'NOTEREF'): S[j][0]['unknown'] = '1'
                st['applied ' + r['label']] += 1
            for lay in (1, 2):  # repair stray I-
                for i in range(n):
                    l = T[i][lay]
                    if l.startswith('I-') and (i == 0 or T[i - 1][lay] == 'O' or T[i - 1][lay][2:] != l[2:]): T[i][lay] = 'B-' + l[2:]
        with open(new.replace('.tsv', '.fixed.tsv'), 'w') as fo:
            for h, T in S: bc.write_layers(fo, ' '.join(f'{k}={v}' for k, v in h.items()), T)
    print(dict(st))
def flat_to_layers(path):
    out = []
    for h, toks in bc.read_samples(path):
        h = dict(h); T = []
        for t, l in toks:
            c = l[2:]
            T.append([t, (l[:2] + 'NOTE') if c.startswith('NOTE') else 'O', l if c in ('BIBL', 'REF') else 'O'])
        if any(l[2:] in ('NOTEBIBL', 'NOTEREF') for _, l in toks): h['unknown'] = '1'
        out.append((h, T))
    return out
def write_split(name, samples):
    for lay, tag in ((1, 'l1'), (2, 'l2')):
        n = 0
        with open(f'../bio-{tag}-{name}.tsv', 'w') as f:
            for h, T in samples:
                if lay == 2 and h.get('unknown') == '1' and name.startswith('train'): continue
                bc.write_sample(f, ' '.join(f'{k}={v}' for k, v in h.items()), [(t[0], t[lay]) for t in T]); n += 1
        print(name, tag, n)
def split():
    write_split('train', [(h, [list(t) for t in T]) for h, T in bc.read_layers('../bio-layers-train.fixed.tsv')])
    write_split('dev', [(h, [list(t) for t in T]) for h, T in bc.read_layers('../bio-layers-d-dev.fixed.tsv')])
    write_split('refmid-train', flat_to_layers('../bio-refmid-train.tsv'))
    write_split('md-test', flat_to_layers('../bio-md-test.tsv'))
def split2():
    for name, path in (('devplus2', '../bio-devplus2.tsv'), ('devplus', '../bio-devplus.tsv')):
        write_split(name, [(h, [list(t) for t in T]) for h, T in bc.read_layers(path)])
    for name, flat in (('md-test', '../bio-md-test.tsv'), ('refmid-train', '../bio-refmid-train.tsv')):
        rows = flat_to_layers(flat)
        with open(f'../bio-layers-{name}.tsv', 'w') as f:
            for h, T in rows: bc.write_layers(f, ' '.join(f'{k}={v}' for k, v in h.items()), T)
        write_split(name, rows)
if __name__ == '__main__':
    {'fix': fix, 'split': split, 'split2': split2}[sys.argv[1]]()
