"""Pick O-tagged candidate citation fragments (digits, roman numerals, abbreviations) with narrow context."""
import json, random, re, sys
from cllg_tagging import common as bc
ROM = re.compile(r'^[IVXLCDMivxlcdm]+$')
ABBR = {"p","pp","cf","fr","v","vv","vol","gen","sch","fol","col","cap","c","l","ll","lib","liber","n","no","nr","ed","s","ss","sq","sqq","h","hom","ep","or","frg"}
def cand(t):
    return t.isdigit() or (len(t) > 1 and ROM.match(t) and t.isupper()) or t.lower() in ABBR or re.match(r'^\d+[A-Za-z]$', t)
W = 14
def runs(toks):
    n = len(toks); i = 0; out = []
    while i < n:
        t, l = toks[i]
        if l == 'O' and cand(t):
            j = i
            while j + 1 < n and toks[j + 1][1] == 'O' and (cand(toks[j + 1][0]) or toks[j + 1][0] in '.,-–:'):
                j += 1
            while toks[j][0] in '.,-–:' and j > i: j -= 1
            out.append((i, j)); i = j + 1
        else: i += 1
    return out
def render(toks, a, b):
    lo, hi = max(0, a - W), min(len(toks), b + W + 1)
    s = []; cur = None
    for k in range(lo, hi):
        t, l = toks[k]
        c = None if l == 'O' else l[2:]
        if k == a: s.append('⟦')
        if c != cur and not (a <= k <= b):
            if cur: s.append('}')
            if c: s.append('{' + c + ':')
            cur = c
        s.append(t)
        if k == b: s.append('⟧')
    if cur: s.append('}')
    return ' '.join(s)
if __name__ == '__main__':
    random.seed(7); N = int(sys.argv[1]); pool = []
    for f in ('../bio-tei-train.tsv', '../bio-tei-dev.tsv'):
        for si, (_, toks) in enumerate(bc.read_samples(f)):
            for a, b in runs(toks):
                pool.append({"file": f, "sample": si, "a": a, "b": b, "span": ' '.join(t for t, _ in toks[a:b+1]), "ctx": render(toks, a, b)})
    print(len(pool), 'candidate runs')
    pick = random.sample(pool, N)
    for k, p in enumerate(pick): p["id"] = k
    json.dump(pick, open('out/bio/verify/items.json', 'w'), ensure_ascii=False, indent=0)
    open('out/bio/verify/in.txt', 'w').write('\n'.join(f'{p["id"]}\t{p["ctx"]}' for p in pick))
