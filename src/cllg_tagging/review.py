"""Two-layer review app for the layered BIO datasets (token, layer 1 = NOTE, layer 2 = BIBL / REF; the layers overlap).

  python -m cllg_tagging.review serve FILE.tsv [FILE2.tsv ...] [--port 8766]    -> http://127.0.0.1:8766
  python -m cllg_tagging.review apply IN.tsv FLAGS.jsonl OUT.tsv                 -> IN with the corrections applied

Paste text: the "Paste text" button annotates any text with the two layered CRFs (--l1 / --l2 model paths, default models/crf3_*) and adds it to
pasted.tsv (current directory), where you correct it like any other file; "Export" downloads the corrected TSV or a JSON-lines list of spans.
The annotation needs python-crfsuite (pip install -e .).

Layered TSVs (e.g. the data/final/*.layers.tsv files) and flat TSVs (converted on
load) are accepted. Corrections are appended to flags.jsonl (current directory, or $CLLG_FLAGS) (an "unflag" is a tombstone) and re-applied live in the
page, so nothing is lost and the app can be stopped any time. Stdlib only.

Select a range by dragging over tokens (a plain click selects the span under it, NOTE with alt+click), shift+click extends, arrows
move the end (shift+arrows the start). Keys: n = NOTE, b = BIBL, r = REF (each only touches its own layer), z = clear NOTE layer,
x = clear BIBL/REF layer, 0 = clear both, Esc = deselect."""
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import common as bc

FLAGS = Path(os.environ.get("CLLG_FLAGS", "flags.jsonl"))
LOCK = threading.Lock()
FILES = {}  # name -> {"path", "rows": [(header, [(tok, l1, l2)])], "cls": [set]}
LAYER_OF = {"NOTE": 1, "BIBL": 2, "REF": 2}


def read_any(path):
    """layered file -> rows; a flat file (2 columns) is converted: NOTE* -> layer 1, BIBL / REF -> layer 2."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line and not line.startswith("#") and line.strip():
                layered = line.rstrip("\n").count("\t") >= 2
                break
        else:
            layered = True
    if layered:
        return bc.read_layers(path)
    return [(h, [tuple(t) for t in T]) for h, T in flat_to_layers(path)]


def flat_to_layers(path):
    """flat BIO file (NOTE, BIBL, NOTEBIBL, NOTEREF, REF) -> layered rows; compound notes have an unknown inner extent."""
    out = []
    for h, toks in bc.read_samples(path):
        h = dict(h)
        T = []
        for t, l in toks:
            c = l[2:]
            T.append([t, (l[:2] + "NOTE") if c.startswith("NOTE") else "O", l if c in ("BIBL", "REF") else "O"])
        if any(l[2:] in ("NOTEBIBL", "NOTEREF") for _, l in toks):
            h["unknown"] = "1"
        out.append((h, T))
    return out


def load(name):
    f = FILES[name]
    if "rows" not in f:
        f["rows"] = read_any(f["path"])
        f["cls"] = [{x[2:] for t in T for x in t[1:] if x != "O"} | ({"NESTED"} if any(t[1] != "O" and t[2] != "O" for t in T) else set())
                    for _, T in f["rows"]]
    return f


def current_flags():
    flags = {}
    if FLAGS.exists():
        for line in FLAGS.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if "del" in d:
                flags.pop(d["del"], None)
            else:
                flags[d["id"]] = d
    return list(flags.values())


def append_flag(d):
    FLAGS.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, open(FLAGS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def repair(tags):
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out


def apply_to(l1, l2, fl):
    for k in range(fl["a"], fl["b"]):
        lab, lay = fl["label"], fl["layer"]
        if lay in (0, 1) and (lab == "O" or lab == "NOTE"):
            l1[k] = "O" if lab == "O" else ("B-" if k == fl["a"] else "I-") + "NOTE"
        if lay in (0, 2) and (lab == "O" or lab in ("BIBL", "REF")):
            l2[k] = "O" if lab == "O" else ("B-" if k == fl["a"] else "I-") + lab


def apply_flags(src, flags_path, dst):
    flags = {}
    for line in Path(flags_path).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        flags.pop(d["del"], None) if "del" in d else flags.__setitem__(d["id"], d)
    by = {}
    for fl in sorted(flags.values(), key=lambda x: x["ts"]):
        if Path(fl["file"]).resolve() == Path(src).resolve() or fl["file"] == src or Path(fl["file"]).name == Path(src).name:
            by.setdefault(fl["idx"], []).append(fl)
    rows = read_any(src)
    n = 0
    with open(dst, "w", encoding="utf-8") as out:
        for i, (hdr, T) in enumerate(rows):
            l1, l2 = [t[1] for t in T], [t[2] for t in T]
            for fl in by.get(i, []):
                apply_to(l1, l2, fl)
                n += 1
            l1, l2 = repair(l1), repair(l2)
            if i in by:
                hdr = dict(hdr, reviewed="1")
                hdr.pop("unknown", None) if any(f["layer"] != 1 for f in by[i]) else None
            bc.write_layers(out, " ".join(f"{k}={v}" for k, v in hdr.items()), list(zip([t[0] for t in T], l1, l2)))
    print(f"{n} corrections applied -> {dst}")


PASTED = "pasted.tsv"
MODELS = Path(__file__).resolve().parents[2] / "models"
MODEL_PATHS = {"l1": MODELS / "crf3_l1.crfsuite", "l2": MODELS / "crf3_l2.crfsuite"}
_TAGGERS = {}


def taggers():
    if not _TAGGERS:
        import pycrfsuite  # needs the venv python (python-crfsuite)
        for k, p in MODEL_PATHS.items():
            t = pycrfsuite.Tagger()
            t.open(str(p))
            _TAGGERS[k] = t
    return _TAGGERS


def chunk(tokens, maxn=150, minn=40):
    out, i = [], 0
    while i < len(tokens):
        j = min(len(tokens), i + maxn)
        if j < len(tokens):
            k = j
            while k > i + minn and tokens[k - 1] not in (".", ";", "·", "?", "!"):
                k -= 1
            j = k if k > i + minn else j
        out.append(tokens[i:j])
        i = j
    return out


def annotate(text):
    """text -> new layered rows (paragraph = blank-line separated, long ones cut at sentence ends), CRF layers 1 and 2."""
    import unicodedata
    from . import features as bf
    tg = taggers()
    f = load(PASTED)
    stamp, new = int(time.time()), []
    for para in [p for p in unicodedata.normalize("NFC", text).split("\n\n") if p.strip()]:
        toks = bc.TOKEN.findall(para)
        for part in chunk(toks):
            x = bf.sent_feats(part)
            hdr = {"id": f"paste-{stamp}-{len(f['rows']) + len(new)}", "src": "paste"}
            new.append((hdr, list(zip(part, tg["l1"].tag(x), tg["l2"].tag(x)))))
    with LOCK, open(f["path"], "a", encoding="utf-8") as fh:
        for h, T in new:
            bc.write_layers(fh, " ".join(f"{k}={v}" for k, v in h.items()), T)
    first = len(f["rows"])
    for h, T in new:
        f["rows"].append((h, T))
        f["cls"].append({x[2:] for t in T for x in t[1:] if x != "O"} | ({"NESTED"} if any(t[1] != "O" and t[2] != "O" for t in T) else set()))
    return first, len(new)


def export(name, fmt):
    f = load(name)
    flags = {}
    for fl in sorted(current_flags(), key=lambda x: x["ts"]):
        if fl["file"] == name:
            flags.setdefault(fl["idx"], []).append(fl)
    lines = []
    for i, (hdr, T) in enumerate(f["rows"]):
        l1, l2 = [t[1] for t in T], [t[2] for t in T]
        for fl in flags.get(i, []):
            apply_to(l1, l2, fl)
        l1, l2 = repair(l1), repair(l2)
        if fmt == "tsv":
            lines.append(f"# {' '.join(f'{k}={v}' for k, v in hdr.items())}{' reviewed=1' if i in flags else ''}")
            lines += [f"{t[0]}\t{a}\t{b}" for t, a, b in zip(T, l1, l2)] + [""]
        else:
            sp = []
            for lay, tags in ((1, l1), (2, l2)):
                k = 0
                while k < len(tags):
                    if tags[k] != "O":
                        e = k
                        while e + 1 < len(tags) and tags[e + 1] == "I-" + tags[k][2:]:
                            e += 1
                        sp.append({"class": tags[k][2:], "start": k, "end": e, "text": " ".join(t[0] for t in T[k:e + 1])})
                        k = e + 1
                    else:
                        k += 1
            lines.append(json.dumps({"id": hdr.get("id", ""), "text": " ".join(t[0] for t in T), "spans": sorted(sp, key=lambda x: (x["start"], -x["end"]))}, ensure_ascii=False))
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            return self.send(PAGE, "text/html")
        if u.path == "/api/files":
            return self.send(json.dumps([{"name": n, "n": len(load(n)["rows"])} for n in FILES]))
        if u.path == "/api/export":
            return self.send(export(q["file"], q.get("fmt", "tsv")), "text/plain" if q.get("fmt", "tsv") == "tsv" else "application/x-ndjson")
        if u.path == "/api/flags":
            return self.send(json.dumps(current_flags(), ensure_ascii=False))
        if u.path == "/api/samples":
            f = load(q["file"])
            cls = set(filter(None, q.get("classes", "").split(",")))
            text = q.get("q", "").lower()
            fl = current_flags()
            flagged = {x["idx"] for x in fl if x["file"] == q["file"]}
            idx = []
            for i, (hdr, T) in enumerate(f["rows"]):
                if cls and not (f["cls"][i] & cls):
                    continue
                if q.get("flagged") == "1" and i not in flagged:
                    continue
                if text and text not in " ".join(hdr.values()).lower() and text not in " ".join(t[0] for t in T).lower():
                    continue
                idx.append(i)
            off, lim = int(q.get("offset", 0)), int(q.get("limit", 20))
            out = [{"idx": i, "h": " ".join(f"{k}={v}" for k, v in f["rows"][i][0].items()),
                    "t": [t[0] for t in f["rows"][i][1]], "l1": [t[1] for t in f["rows"][i][1]], "l2": [t[2] for t in f["rows"][i][1]],
                    "flags": [x for x in fl if x["file"] == q["file"] and x["idx"] == i]} for i in idx[off:off + lim]]
            return self.send(json.dumps({"total": len(idx), "rows": out}, ensure_ascii=False))
        self.send("not found", "text/plain", 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/flag":
            f = load(body["file"])
            hdr, T = f["rows"][body["idx"]]
            a, b = body["a"], body["b"]
            d = {"id": uuid.uuid4().hex[:10], "ts": int(time.time() * 1000), "file": body["file"], "idx": body["idx"], "sample": hdr.get("id", ""),
                 "a": a, "b": b, "text": " ".join(t[0] for t in T[a:b]), "old1": [t[1] for t in T[a:b]], "old2": [t[2] for t in T[a:b]],
                 "layer": body["layer"], "label": body["label"], "comment": body.get("comment", "")}
            append_flag(d)
            return self.send(json.dumps(d, ensure_ascii=False))
        if self.path == "/api/annotate":
            try:
                first, n = annotate(body["text"])
            except Exception as e:  # noqa: BLE001
                return self.send(json.dumps({"error": f"{type(e).__name__}: {e} - start the app with the python that has python-crfsuite"}), code=500)
            return self.send(json.dumps({"file": PASTED, "first": first, "n": n}))
        if self.path == "/api/unflag":
            append_flag({"del": body["id"], "ts": int(time.time() * 1000)})
            return self.send("{}")
        self.send("not found", "text/plain", 404)


def serve(paths, port, l1=None, l2=None):
    if l1:
        MODEL_PATHS["l1"] = Path(l1)
    if l2:
        MODEL_PATHS["l2"] = Path(l2)
    pasted = Path(PASTED)
    pasted.touch()
    paths = list(paths) + [str(pasted)]
    names = [Path(p).name for p in paths]
    for p in paths:
        FILES[Path(p).name if len(set(names)) == len(names) else p] = {"path": p}
    print(f"two-layer review app on http://127.0.0.1:{port}  (corrections -> {FLAGS})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Two-layer review</title><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e2dc;--sel:#fff3a8;
--BIBL:#c4ebc4;--REF:#ffd0d6;--NOTEt:#e4eefc;--bNOTE:#2f6fd0;--bBIBL:#2e8b3e;--bREF:#d63a52}
@media(prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--mut:#9a9a92;--card:#1f1f1c;--line:#34342f;--sel:#6b5d00;
--BIBL:#1d4a28;--REF:#6a2230;--NOTEt:#1b2c47;--bNOTE:#6ea0ee}}
*{box-sizing:border-box}body{margin:0;padding-bottom:120px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:3}
h1{font-size:13px;font-weight:500;margin:0 0 8px;color:var(--mut)}.row{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}
select,input,button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
button{cursor:pointer}button:disabled{opacity:.4;cursor:default}
button.NOTE{border-color:var(--bNOTE);box-shadow:inset 0 -3px var(--bNOTE)}button.BIBL{background:var(--BIBL)}button.REF{background:var(--REF)}
label.chip{display:inline-flex;gap:5px;align-items:center;padding:2px 8px;border-radius:999px;cursor:pointer;border:1px solid var(--line)}
.stat{color:var(--mut);font-size:13px}main{padding:12px 16px;max-width:1150px;margin:auto}
.s{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:0 0 10px}.s.fl{border-color:#d63a52}
.h{font:12px ui-monospace,monospace;color:var(--mut);margin-bottom:4px;overflow-wrap:anywhere}
.t{font-family:"Gentium Plus","GFS Didot",Palatino,serif;font-size:18px;line-height:2.5;overflow-wrap:anywhere}
.k{cursor:pointer;user-select:none;padding:1px 1px;border-top:3px solid transparent;border-bottom:3px solid transparent;position:relative}
.k:hover{outline:1px dashed var(--mut)}
.k.BIBL{background:var(--BIBL)}.k.REF{background:var(--REF)}
.k.n{background-color:var(--NOTEt);border-top-color:var(--bNOTE);border-bottom-color:var(--bNOTE)}
.k.n.BIBL{background:linear-gradient(var(--BIBL),var(--BIBL))}.k.n.REF{background:linear-gradient(var(--REF),var(--REF))}
.k.nb{border-left:3px solid var(--bNOTE);padding-left:3px}.k.ne{border-right:3px solid var(--bNOTE);padding-right:3px}
.k.cb{box-shadow:inset 3px 0 var(--bBIBL)}.k.cb.REF{box-shadow:inset 3px 0 var(--bREF)}
.k.nb::before,.k.cb::after{position:absolute;font:700 9px system-ui,sans-serif;line-height:1;top:-1px}
.k.nb::before{content:"NOTE";left:2px;color:var(--bNOTE);top:-12px}.k.cb::after{content:attr(data-c);left:2px;top:auto;bottom:-12px;color:var(--bBIBL)}.k.cb.REF::after{color:var(--bREF)}
.k.sel{background:var(--sel)!important;outline:2px solid #caa800}.k.f{text-decoration:underline wavy #d63a52}
.fl-list{margin-top:8px;font-size:13px}.fl-list span{display:inline-block;margin-right:12px}.fl-list a{color:#d63a52;cursor:pointer}
.nav{display:flex;gap:8px;align-items:center;justify-content:center;margin:14px 0}
#bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:2px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;z-index:3}
</style></head><body>
<header><h1>Two layers · NOTE = blue frame, BIBL = green, REF = red, they can overlap · drag = range, click = span, alt+click = note span, shift+click extends, ←/→ end, shift+←/→ start · n b r set a label in its own layer, z clears NOTE, x clears BIBL/REF, 0 clears both, Esc deselects</h1>
<div class="row"><select id="file"></select>
<span class="row"><label class="chip"><input type="checkbox" data-c="NOTE">NOTE</label><label class="chip"><input type="checkbox" data-c="BIBL">BIBL</label><label class="chip"><input type="checkbox" data-c="REF">REF</label><label class="chip"><input type="checkbox" data-c="NESTED">nested</label></span>
<button id="pastebtn">Paste text</button><button id="exp">Export</button>
<input id="q" type="search" placeholder="search text / id…" size="20"><label class="stat"><input id="onlyfl" type="checkbox"> corrected only</label>
<span class="stat" id="stat"></span></div></header>
<div id="paste" hidden style="padding:10px 16px;border-bottom:1px solid var(--line);background:var(--card)">
<textarea id="ptxt" rows="7" style="width:100%;font:16px/1.5 Palatino,serif;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)" placeholder="Paste Greek / Latin text here (blank line = new passage); the CRF annotates it, then you correct it below."></textarea>
<div class="row" style="margin-top:6px"><button id="pgo" class="NOTE">Annotate with CRF</button><button id="pclose">Close</button><span class="stat" id="pmsg"></span></div></div>
<main><div id="list"></div><div class="nav"><button id="prev">‹</button><span id="pg" class="stat"></span><button id="next">›</button></div></main>
<div id="bar"><span id="selinfo" class="stat">drag over tokens to select a range</span>
<button class="NOTE" data-l="1" data-v="NOTE" disabled>NOTE (n)</button><button class="BIBL" data-l="2" data-v="BIBL" disabled>BIBL (b)</button><button class="REF" data-l="2" data-v="REF" disabled>REF (r)</button>
<button data-l="1" data-v="O" disabled>clear NOTE (z)</button><button data-l="2" data-v="O" disabled>clear BIBL/REF (x)</button><button data-l="0" data-v="O" disabled>clear both (0)</button>
<input id="com" placeholder="comment (optional)" size="26"><span class="stat" id="nflags"></span></div>
<script>
const PER=15,KEY={n:[1,"NOTE"],b:[2,"BIBL"],r:[2,"REF"],z:[1,"O"],x:[2,"O"],"0":[0,"O"]};
let page=0,total=0,rows=[],active=new Set(),sel=null,drag=null,moved=false;
const $=id=>document.getElementById(id),esc=s=>s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const j=async(u,o)=>(await fetch(u,o)).json(),post=(u,b)=>j(u,{method:"POST",body:JSON.stringify(b),headers:{"Content-Type":"application/json"}});
function repair(l){let prev="O";return l.map(t=>{if(t.startsWith("I-")&&prev.slice(2)!==t.slice(2))t="B-"+t.slice(2);prev=t;return t})}
function eff(r){const a=r.l1.slice(),b=r.l2.slice();(r.flags||[]).slice().sort((x,y)=>x.ts-y.ts).forEach(f=>{
 for(let k=f.a;k<f.b;k++){
  if((f.layer===0||f.layer===1)&&(f.label==="O"||f.label==="NOTE"))a[k]=f.label==="O"?"O":(k===f.a?"B-":"I-")+"NOTE";
  if((f.layer===0||f.layer===2)&&(f.label==="O"||f.label==="BIBL"||f.label==="REF"))b[k]=f.label==="O"?"O":(k===f.a?"B-":"I-")+f.label}});
 return[repair(a),repair(b)]}
function span(l,i){if(l[i]==="O")return[i,i+1];let a=i;while(a>0&&l[a].startsWith("I-"))a--;let b=i+1;while(b<l.length&&l[b].startsWith("I-")&&l[b].slice(2)===l[i].slice(2))b++;return[a,b]}
function setSel(ri,a,b){const r=rows[ri],[e1,e2]=eff(r);sel={ri,a,b,text:r.t.slice(a,b).join(" "),n:[...new Set(e1.slice(a,b).map(x=>x.slice(2)||"–"))].join("/"),c:[...new Set(e2.slice(a,b).map(x=>x.slice(2)||"–"))].join("/")};draw()}
function draw(){
 $("list").innerHTML=rows.map((r,ri)=>{const fl=r.flags||[],[e1,e2]=eff(r),fk=new Set();fl.forEach(f=>{for(let k=f.a;k<f.b;k++)fk.add(k)});
  const toks=r.t.map((w,i)=>{const n=e1[i]!=="O",c=e2[i]==="O"?"":e2[i].slice(2);
   const nb=n&&e1[i].startsWith("B-"),ne=n&&!(i+1<e1.length&&e1[i+1].startsWith("I-")),cb=c&&e2[i].startsWith("B-");
   const s=sel&&sel.ri===ri&&i>=sel.a&&i<sel.b?" sel":"";
   return `<span class="k ${c}${n?" n":""}${nb?" nb":""}${ne?" ne":""}${cb?" cb":""}${s}${fk.has(i)?" f":""}" data-r="${ri}" data-i="${i}"${cb?` data-c="${c}"`:""} title="${r.l1[i]} | ${r.l2[i]}">${esc(w)}</span>`}).join(" ");
  const fls=fl.map(f=>`<span>🚩 “${esc(f.text)}” → <b>${f.layer===0?"clear both":(f.label==="O"?"clear "+(f.layer===1?"NOTE":"BIBL/REF"):f.label)}</b>${f.comment?" · "+esc(f.comment):""} <a data-un="${f.id}">undo</a></span>`).join("");
  return `<div class="s${fl.length?" fl":""}"><div class="h">#${r.idx} ${esc(r.h)}</div><div class="t">${toks}</div>${fls?`<div class="fl-list">${fls}</div>`:""}</div>`}).join("");
 $("pg").textContent=`page ${page+1} / ${Math.max(1,Math.ceil(total/PER))} · ${total} samples`;
 $("selinfo").textContent=sel?`“${sel.text}” [${sel.b-sel.a} tok] · NOTE layer: ${sel.n} · BIBL/REF layer: ${sel.c}`:"drag over tokens to select a range";
 document.querySelectorAll("#bar button").forEach(b=>b.disabled=!sel)}
async function loadPage(){const q=new URLSearchParams({file:$("file").value,offset:page*PER,limit:PER,classes:[...active].join(","),q:$("q").value,flagged:$("onlyfl").checked?1:0});
 const d=await j("/api/samples?"+q);rows=d.rows;total=d.total;sel=null;draw();const f=await j("/api/flags");$("nflags").textContent=`${f.length} corrections saved`}
async function commit(layer,label){if(!sel)return;const r=rows[sel.ri],ri=sel.ri,{a,b}=sel;
 const f=await post("/api/flag",{file:$("file").value,idx:r.idx,a,b,layer,label,comment:$("com").value});
 (r.flags=r.flags||[]).push(f);$("com").value="";$("nflags").textContent="saved";setSel(ri,a,b)}
$("list").onmousedown=e=>{const k=e.target.closest(".k");if(!k||e.button)return;e.preventDefault();const ri=+k.dataset.r,i=+k.dataset.i;moved=false;
 if(e.shiftKey&&sel&&sel.ri===ri){setSel(ri,Math.min(sel.a,i),Math.max(sel.b,i+1));return}drag={ri,i,alt:e.altKey}};
$("list").onmouseover=e=>{if(!drag)return;const k=e.target.closest(".k");if(!k||+k.dataset.r!==drag.ri)return;const i=+k.dataset.i;if(i!==drag.i)moved=true;if(moved)setSel(drag.ri,Math.min(drag.i,i),Math.max(drag.i,i)+1)};
window.onmouseup=()=>{if(drag&&!moved){const r=rows[drag.ri],[e1,e2]=eff(r),i=drag.i;
  const [a,b]=span(drag.alt||e2[i]==="O"?e1:e2,i);setSel(drag.ri,a,b)}drag=null};
$("list").onclick=async e=>{const un=e.target.dataset.un;if(un){await post("/api/unflag",{id:un});loadPage()}};
document.querySelectorAll("#bar button").forEach(b=>b.onclick=()=>commit(+b.dataset.l,b.dataset.v));
document.onkeydown=e=>{if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT")return;
 if(KEY[e.key]&&sel&&!e.ctrlKey&&!e.metaKey){e.preventDefault();commit(...KEY[e.key]);return}
 if(e.key==="Escape"){sel=null;draw();return}
 if(sel&&(e.key==="ArrowLeft"||e.key==="ArrowRight")){e.preventDefault();const n=rows[sel.ri].t.length,d=e.key==="ArrowRight"?1:-1;let{a,b}=sel;
  if(e.shiftKey)a=Math.max(0,Math.min(b-1,a+d));else b=Math.min(n,Math.max(a+1,b+d));setSel(sel.ri,a,b)}};
$("pastebtn").onclick=()=>{$("paste").hidden=!$("paste").hidden;if(!$("paste").hidden)$("ptxt").focus()};
$("pclose").onclick=()=>{$("paste").hidden=true};
$("pgo").onclick=async()=>{const t=$("ptxt").value;if(!t.trim())return;$("pmsg").textContent="annotating…";
 const d=await post("/api/annotate",{text:t});if(d.error){$("pmsg").textContent=d.error;return}
 $("pmsg").textContent=`${d.n} passage(s) added to ${d.file}`;$("ptxt").value="";$("paste").hidden=true;
 await refreshFiles(d.file);$("file").value=d.file;page=Math.floor(d.first/PER);await loadPage();scrollTo(0,0)};
$("exp").onclick=()=>{const fmt=confirm("OK = corrected TSV (token, NOTE layer, BIBL/REF layer)\nCancel = JSON lines with the spans")?"tsv":"json";
 location.href="/api/export?file="+encodeURIComponent($("file").value)+"&fmt="+fmt};
["q","onlyfl"].forEach(id=>$(id).oninput=()=>{page=0;loadPage()});
$("prev").onclick=()=>{page=Math.max(0,page-1);loadPage();scrollTo(0,0)};$("next").onclick=()=>{if((page+1)*PER<total)page++;loadPage();scrollTo(0,0)};
$("file").onchange=()=>{page=0;loadPage()};
document.querySelectorAll("[data-c]").forEach(c=>{if(c.type==="checkbox")c.onchange=e=>{const v=e.target.dataset.c;e.target.checked?active.add(v):active.delete(v);page=0;loadPage()}});
async function refreshFiles(keep){const fs=await j("/api/files");$("file").innerHTML=fs.map(f=>`<option value="${f.name}">${f.name} (${f.n})</option>`).join("");if(keep)$("file").value=keep}
(async()=>{await refreshFiles();loadPage()})();
</script></body></html>'''

def main(a=None):
    a = sys.argv[1:] if a is None else a
    if a and a[0] == "apply":
        apply_flags(a[1], a[2], a[3])
    elif a and a[0] == "serve":
        port = int(a[a.index("--port") + 1]) if "--port" in a else 8766
        opt = lambda k: a[a.index(k) + 1] if k in a else None  # noqa: E731
        serve([x for x in a[1:] if x.endswith(".tsv")], port, opt("--l1"), opt("--l2"))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
