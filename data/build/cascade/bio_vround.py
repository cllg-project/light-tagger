"""Generic verification round.  pick R N | merge R | apply
pick:  new N samples with candidates (not used by any earlier round) -> out/bio/rR/{meta.json,in_*.txt}; Sonnet writes rR/ans_*.txt
merge: class agreement of Sonnet and the clean CRF -> rR/accepted.json ; disputes -> rR/opus/{meta.json,in_*.txt}; Opus writes rR/opus/ans_*.txt
apply: gather every source (v2 Opus, v4, rounds) -> bio-tei-{train,dev}.fixed.tsv"""
import collections, glob, json, os, random, sys
from cllg_tagging import common as bc
import bio_verify_pick as vp, bio_verify3 as v3
FILES = ('../bio-tei-train.tsv', '../bio-tei-dev.tsv')
def used():
    u = {(i['file'], i['sample']) for i in json.load(open('out/bio/verify/items.json'))}
    for p in ['out/bio/v2/meta.json', 'out/bio/v3/meta.json'] + glob.glob('out/bio/r*/meta.json'):
        u |= {(m['file'], m['sample']) for m in json.load(open(p))}
    return u
def answers(pattern):
    A = {}
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            p = l.split()
            if len(p) >= 5: A[(p[0], int(p[1]))] = (p[2], int(p[3]), int(p[4]))
    return A
def pick(R, N, per=100):
    cache = {f: bc.read_samples(f) for f in FILES}; u = used()
    pool = [(f, si) for f in cache for si, (_, t) in enumerate(cache[f]) if (f, si) not in u and vp.runs(t)]
    random.seed(100 + R); print(len(pool), 'available')
    d = f'out/bio/r{R}'; os.makedirs(d, exist_ok=True)
    meta = [{"sid": f'R{R}_{k}', "file": f, "sample": si, "runs": vp.runs(cache[f][si][1])} for k, (f, si) in enumerate(random.sample(pool, N))]
    json.dump(meta, open(f'{d}/meta.json', 'w'))
    for b in range(0, N, per):
        open(f'{d}/in_{b // per}.txt', 'w').write('\n'.join(f'{m["sid"]}\t' + v3.render(cache[m['file']][m['sample']][1], [tuple(r) for r in m['runs']]) for m in meta[b:b + per]) + '\n')
    print(N, 'samples,', -(-N // per), 'files')
def merge(R, per=100):
    import pycrfsuite, bio_features as bf
    d = f'out/bio/r{R}'; meta = json.load(open(f'{d}/meta.json')); S = answers(f'{d}/ans_*.txt')
    tg = pycrfsuite.Tagger(); tg.open('out/bio/crf_clean.crfsuite'); cache = {f: bc.read_samples(f) for f in FILES}
    def entity(tags, a, b):
        c = collections.Counter(t[2:] for t in tags[a:b + 1] if t != 'O')
        return c.most_common(1)[0][0] if c else 'O'
    acc, by, st = [], collections.defaultdict(list), collections.Counter()
    for m in meta:
        toks = cache[m['file']][m['sample']][1]; tags = tg.tag(bf.sent_feats([t for t, _ in toks]))
        for k, (a, b) in enumerate(m['runs'], 1):
            s = S.get((m['sid'], k), ('O', 0, 0)); c = entity(tags, a, b)
            if s[0] == 'O' and c == 'O': st['both O'] += 1
            elif s[0] == c: st['class agrees'] += 1; acc.append({"file": m['file'], "sample": m['sample'], "a": a - s[1], "b": b + s[2], "label": s[0]})
            else: st['class differs'] += 1; by[m['sid']].append(k)
    json.dump(acc, open(f'{d}/accepted.json', 'w')); print(dict(st))
    os.makedirs(f'{d}/opus', exist_ok=True); om, lines = [], []
    for m in meta:
        if m['sid'] not in by: continue
        toks = [list(x) for x in cache[m['file']][m['sample']][1]]
        for r in acc:
            if (r['file'], r['sample']) == (m['file'], m['sample']):
                for i in range(max(0, r['a']), min(len(toks), r['b'] + 1)): toks[i][1] = ('B-' if i == r['a'] else 'I-') + r['label']
        runs = [tuple(m['runs'][k - 1]) for k in by[m['sid']]]
        om.append({"sid": m['sid'], "file": m['file'], "sample": m['sample'], "runs": runs}); lines.append(f'{m["sid"]}\t' + v3.render(toks, runs))
    json.dump(om, open(f'{d}/opus/meta.json', 'w'))
    for b in range(0, len(lines), per): open(f'{d}/opus/in_{b // per}.txt', 'w').write('\n'.join(lines[b:b + per]) + '\n')
    print(len(lines), 'samples for Opus,', -(-len(lines) // per), 'files')
def collect():
    fixes = []
    def add(meta, A, src):
        for (sid, k), (lab, el, er) in A.items():
            if sid in meta:
                a, b = meta[sid]['runs'][k - 1]; fixes.append({"file": meta[sid]['file'], "sample": meta[sid]['sample'], "a": a - el, "b": b + er, "label": lab, "src": src})
    v2 = {f'S{m["sid"]}': m for m in json.load(open('out/bio/v2/meta.json'))}
    add(v2, answers('out/bio/v2/ans_*.txt'), 'opus-v2')
    for r in json.load(open('out/bio/v4_accepted.json')): fixes.append(dict(r, src='agree-v4'))
    add({m['sid']: m for m in json.load(open('out/bio/v4/meta.json'))}, answers('out/bio/v4/ans_*.txt'), 'opus-v4')
    for d in sorted(glob.glob('out/bio/r[0-9]*')):
        for r in json.load(open(f'{d}/accepted.json')): fixes.append(dict(r, src='agree-' + d[-2:]))
        if os.path.exists(f'{d}/opus/meta.json'): add({m['sid']: m for m in json.load(open(f'{d}/opus/meta.json'))}, answers(f'{d}/opus/ans_*.txt'), 'opus-' + d[-2:])
    return fixes


def apply():
    fixes = collect()
    by = collections.defaultdict(list)
    for r in fixes: by[(r['file'], r['sample'])].append(r)
    st = collections.Counter()
    for f in FILES:
        S = bc.read_samples(f)
        for (ff, si), rs in by.items():
            if ff != f: continue
            toks = [list(x) for x in S[si][1]]; n = len(toks)
            for r in sorted(rs, key=lambda r: r['b'] - r['a'], reverse=True):
                a, b = max(0, r['a']), min(n - 1, r['b'])
                if (a > 0 and toks[a][1].startswith('I-')) or (b + 1 < n and toks[b + 1][1].startswith('I-')): st['skipped (cuts entity) ' + r['src']] += 1; continue
                for i in range(a, b + 1): toks[i][1] = ('B-' if i == a else 'I-') + r['label']
                st['applied ' + r['src']] += 1
            for i in range(n):
                l = toks[i][1]
                if l.startswith('I-') and (i == 0 or toks[i - 1][1] == 'O' or toks[i - 1][1][2:] != l[2:]): toks[i][1] = 'B-' + l[2:]
            S[si] = (S[si][0], [tuple(x) for x in toks])
        with open(f.replace('.tsv', '.fixed.tsv'), 'w') as fo:
            for h, t in S: bc.write_sample(fo, h, t)
    print(dict(st))
if __name__ == '__main__':
    c = sys.argv[1]
    {'pick': lambda: pick(int(sys.argv[2]), int(sys.argv[3])), 'merge': lambda: merge(int(sys.argv[2])), 'apply': apply}[c]()
