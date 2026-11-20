"""
Animated pieces of the app, as small self-contained HTML/JavaScript pages (nothing loaded from the internet):
count-up cards, the skill map (prerequisites as arrows, mastery as colour, playable week by week), the class
heat-map replay, and the what-if race of two ways of practising.
"""

from __future__ import annotations

import html as _html
import json

import numpy as np


def _js(obj):
    """JSON for a <script> block (text such as names cannot close the script)."""
    return json.dumps(obj).replace("</", "<\\/")


def _e(text):
    return _html.escape(str(text), quote=True)


LIGHT = dict(bg="#ffffff", panel="#f6f7f9", ink="#1f2328", muted="#6b7280", grid="#e5e7eb", accent="#2a78d6",
             ok="#1baf7a", mid="#eda100", bad="#e34948", none="#c9ced6", violet="#6250d6")
DARK = dict(bg="#0e1117", panel="#161b22", ink="#e6edf3", muted="#9aa4b2", grid="#2d333b", accent="#4c9be8",
            ok="#2fbf8a", mid="#e3a21a", bad="#f0645f", none="#3a414b", violet="#9085e9")
FONT = "'Source Sans Pro','Segoe UI',Helvetica,Arial,sans-serif"


def _pal(dark):
    return DARK if dark else LIGHT


# the colour of a chance: red below 50%, through amber at 70%, green from 85%
_SCALE_JS = """
function mix(a, b, t) { const p = s => [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16));
  const x = p(a), y = p(b); return 'rgb(' + x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(',') + ')'; }
function col(v) { if (v === null || v === undefined || isNaN(v)) return C.none;
  if (v <= 0.5) return C.bad; if (v < 0.7) return mix(C.bad, C.mid, (v - 0.5) / 0.2);
  if (v < 0.85) return mix(C.mid, C.ok, (v - 0.7) / 0.15); return C.ok; }
"""


def _base_css(c):
    return f"""body{{margin:0;font-family:{FONT};color:{c['ink']};background:transparent}}
.bar{{display:flex;gap:10px;align-items:center;margin:2px 0 8px;flex-wrap:wrap}}
button{{font:600 13px {FONT};border:1px solid {c['grid']};background:{c['panel']};color:{c['ink']};
  border-radius:8px;padding:5px 12px;cursor:pointer}}
button.on{{background:{c['accent']};border-color:{c['accent']};color:#fff}}
input[type=range]{{flex:1;min-width:120px;accent-color:{c['accent']}}}
.lab{{font-weight:700;font-variant-numeric:tabular-nums;min-width:120px}}
.tip{{position:fixed;pointer-events:none;background:{c['panel']};border:1px solid {c['grid']};border-radius:8px;
  padding:6px 9px;font-size:12.5px;display:none;box-shadow:0 4px 14px rgba(0,0,0,.18);max-width:280px;z-index:9}}
.leg{{font-size:12px;color:{c['muted']};display:flex;gap:14px;flex-wrap:wrap;margin-top:4px}}
.sw{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}}"""


def kpi_cards(cards, dark=False):
    """cards: [(label, value, prefix, suffix, decimals, note, colour key)] -> HTML with numbers counting up."""
    c = _pal(dark)
    items = []
    for k, (label, value, prefix, suffix, dec, note, col) in enumerate(cards):
        items.append(f"""<div class="card" style="animation-delay:{k * 0.08:.2f}s">
  <div class="lab">{_e(label)}</div>
  <div class="val" style="color:{c.get(col, col)}"><span>{_e(prefix)}</span><span class="num" data-v="{float(value)}"
     data-d="{int(dec)}">0</span><span class="suf">{_e(suffix)}</span></div>
  <div class="note">{_e(note)}</div></div>""")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:transparent;font-family:{FONT};color:{c['ink']}}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.card{{background:{c['panel']};border:1px solid {c['grid']};border-radius:12px;padding:14px 16px;
  animation:rise .6s ease-out both}}
.lab{{font-size:12.5px;color:{c['muted']};letter-spacing:.3px}}
.val{{font-size:30px;font-weight:700;margin:4px 0 2px;font-variant-numeric:tabular-nums}}
.suf{{font-size:16px;font-weight:600;margin-left:2px}}
.note{{font-size:12.5px;color:{c['muted']}}}
@keyframes rise{{from{{opacity:0;transform:translateY(8px)}}}}
@media (prefers-reduced-motion:reduce){{.card{{animation:none}}}}
</style></head><body><div class="grid">{''.join(items)}</div><script>
const ease = t => 1 - Math.pow(1 - t, 3);
document.querySelectorAll('.num').forEach((el, i) => {{
  const v = parseFloat(el.dataset.v), d = parseInt(el.dataset.d), t0 = performance.now() + i * 80, dur = 1100;
  const fmt = x => x.toLocaleString('en-US', {{minimumFractionDigits: d, maximumFractionDigits: d}});
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {{ el.textContent = fmt(v); return; }}
  const step = now => {{ const t = Math.min(Math.max((now - t0) / dur, 0), 1); el.textContent = fmt(v * ease(t));
    if (t < 1) requestAnimationFrame(step); }};
  requestAnimationFrame(step);
}});
</script></body></html>"""


def map_layout(skill_ids, topics, width=1000, row=54, top=34, margin=44):
    """Node positions for the skill map: one column per topic (in syllabus order), skills down each column."""
    cols = list(dict.fromkeys(topics))
    cw = (width - margin) / max(len(cols), 1)
    pos, depth = {}, {t: 0 for t in cols}
    for k, t in zip(skill_ids, topics):
        j = cols.index(t)
        pos[k] = (margin + cw * j + cw / 2, top + 30 + depth[t] * row)
        depth[t] += 1
    height = top + 30 + max(depth.values(), default=1) * row + 4
    return cols, pos, cw, height


def skill_map(skill_ids, names, topics, prereq, frames, title="", dark=False, height=None, autoplay=True):
    """The skills as a map: one column per topic, arrows from each prerequisite, each skill coloured by its mastery.
    frames: [(label, [mastery or None per skill], [answers or None per skill])]; the map plays through the frames
    (weeks). An arrow turns red and flows when a weak prerequisite (below 60%) holds back a skill not yet secure."""
    c = _pal(dark)
    cols, pos, cw, h = map_layout(skill_ids, topics)
    nw = min(cw - 26, 210)
    fit = max(int((nw - 58) / 7.4), 6)                  # characters that fit beside the percentage (12.5px bold)
    shown = [str(n) if len(str(n)) <= fit else str(n)[:fit - 1].rstrip(" ,") + "…" for n in names]
    data = {"ids": list(skill_ids), "names": [str(n) for n in names], "shown": shown, "cols": cols, "cw": cw, "nw": nw,
            "h": h,
            "pos": [pos[k] for k in skill_ids],
            "edges": [[int(p), int(k)] for k, ps in prereq.items() for p in ps],
            "frames": [{"label": lab, "v": [None if v is None or (isinstance(v, float) and np.isnan(v)) else
                                             round(float(v), 3) for v in vals],
                        "n": [None if n is None else int(n) for n in (cnt if cnt is not None else [None] * len(vals))]}
                       for lab, vals, cnt in frames],
            "title": title, "auto": bool(autoplay)}
    height = height or int(h + 92)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{_base_css(c)}
svg text{{font-family:{FONT}}}
.node rect.bg{{transition:fill .5s ease, stroke .5s ease}}
.node .fill{{transition:width .5s ease, fill .5s ease}}
.edge{{fill:none;stroke:{c['muted']};stroke-opacity:.22;stroke-width:1.5;transition:stroke .4s, stroke-opacity .3s}}
.edge.hi{{stroke:{c['accent']};stroke-opacity:.95;stroke-width:2.2}}
.edge.block{{stroke:{c['bad']};stroke-opacity:.95;stroke-width:2.2;stroke-dasharray:6 5;animation:flow 1s linear infinite}}
@keyframes flow{{to{{stroke-dashoffset:-11}}}}
.pulse{{animation:pulse 1.2s ease-out}}
@keyframes pulse{{0%{{stroke-width:6;stroke-opacity:.9}}100%{{stroke-width:1.5;stroke-opacity:1}}}}
@media (prefers-reduced-motion:reduce){{.edge.block{{animation:none}}}}
</style></head><body>
<div class="bar"><button id="pp">▶ Play</button><input id="sl" type="range" min="0" value="0" step="1">
<span class="lab" id="wk"></span></div>
<svg id="sv" width="100%" viewBox="0 0 1000 {h:.0f}" preserveAspectRatio="xMidYMin meet"></svg>
<div class="leg"><span><i class="sw" style="background:{c['bad']}"></i>weak (below 60%)</span>
<span><i class="sw" style="background:{c['mid']}"></i>developing</span>
<span><i class="sw" style="background:{c['ok']}"></i>secure (80% and up)</span>
<span><i class="sw" style="background:{c['none']}"></i>not taught yet</span>
<span>red flowing arrow: a weak prerequisite holding a skill back</span></div>
<div class="tip" id="tip"></div>
<script>
const D = {_js(data)}, C = {_js(c)}, FONT = {_js(FONT)};
{_SCALE_JS}
const NS = 'http://www.w3.org/2000/svg', sv = document.getElementById('sv');
const el = (t, a, p) => {{ const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]);
  (p || sv).appendChild(e); return e; }};
D.cols.forEach((t, j) => {{ const x = 44 + D.cw * j + D.cw / 2;
  const tx = el('text', {{x: x, y: 18, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, fill: C.muted}});
  tx.textContent = t.toUpperCase(); }});
const NW = D.nw, NH = 38;
const edges = D.edges.map(([p, k]) => {{
  const [x1, y1] = D.pos[p], [x2, y2] = D.pos[k]; let d;
  if (Math.abs(x1 - x2) < 1) {{ const xo = x1 - NW / 2; d = `M${{xo}} ${{y1}} C${{xo - 34}} ${{y1}} ${{xo - 34}} ${{y2}} ${{xo}} ${{y2}}`; }}
  else {{ const s = x2 > x1 ? 1 : -1, xa = x1 + s * NW / 2, xb = x2 - s * NW / 2, m = (xa + xb) / 2;
    d = `M${{xa}} ${{y1}} C${{m}} ${{y1}} ${{m}} ${{y2}} ${{xb}} ${{y2}}`; }}
  return [el('path', {{d: d, class: 'edge'}}), p, k]; }});
const nodes = D.ids.map((id, i) => {{
  const [x, y] = D.pos[i], g = el('g', {{class: 'node', transform: `translate(${{x - NW / 2}},${{y - NH / 2}})`}});
  el('rect', {{width: NW, height: NH, rx: 9, fill: C.bg}}, g);
  const bg = el('rect', {{class: 'bg', width: NW, height: NH, rx: 9, fill: C.panel, stroke: C.grid, 'stroke-width': 1.5}}, g);
  el('rect', {{x: 8, y: NH - 7, width: NW - 16, height: 3, rx: 1.5, fill: C.grid}}, g);
  const fl = el('rect', {{class: 'fill', x: 8, y: NH - 7, width: 0, height: 3, rx: 1.5, fill: C.none}}, g);
  const nm = el('text', {{x: 10, y: 17, 'font-size': 12.5, 'font-weight': 600, fill: C.ink}}, g);
  nm.textContent = D.shown[i];
  const vt = el('text', {{x: NW - 9, y: 17, 'text-anchor': 'end', 'font-size': 13, 'font-weight': 700, fill: C.muted}}, g);
  g.addEventListener('mousemove', ev => {{ tipShow(ev, i); edges.forEach(([e, p, k]) => e.classList.toggle('hi', p === i || k === i)); }});
  g.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; edges.forEach(([e]) => e.classList.remove('hi')); }});
  return {{g, bg, fl, vt}}; }});
const tip = document.getElementById('tip'); let cur = 0, prev = null;
function status(v) {{ return v === null ? 'not taught yet' : v >= 0.8 ? 'secure' : v >= 0.6 ? 'developing' : 'weak'; }}
function tipShow(ev, i) {{ const f = D.frames[cur], v = f.v[i], n = f.n[i];
  tip.innerHTML = `<b>${{D.names[i]}}</b> (${{D.ids[i]}})<br>${{v === null ? 'not taught yet' :
    Math.round(v * 100) + '% chance on a typical question · ' + status(v)}}${{n !== null && v !== null ? '<br>' + n + ' answers' : ''}}`;
  tip.style.display = 'block'; tip.style.left = Math.min(ev.clientX + 14, innerWidth - 290) + 'px'; tip.style.top = (ev.clientY + 12) + 'px'; }}
function draw(j) {{ cur = j; const f = D.frames[j];
  nodes.forEach((nd, i) => {{ const v = f.v[i], c = col(v);
    nd.bg.setAttribute('fill', v === null ? C.panel : c + '');
    nd.bg.setAttribute('fill-opacity', v === null ? 1 : 0.22);
    nd.bg.setAttribute('stroke', v === null ? C.grid : c); nd.bg.setAttribute('stroke-dasharray', v === null ? '4 4' : '');
    nd.fl.setAttribute('width', v === null ? 0 : (NW - 16) * v); nd.fl.setAttribute('fill', c);
    nd.vt.textContent = v === null ? '–' : Math.round(v * 100) + '%'; nd.vt.setAttribute('fill', v === null ? C.muted : c);
    if (prev && prev.v[i] !== null && v !== null && (prev.v[i] < 0.8) !== (v < 0.8)) {{
      nd.bg.classList.remove('pulse'); void nd.bg.getBBox(); nd.bg.classList.add('pulse'); }} }});
  edges.forEach(([e, p, k]) => {{ const vp = f.v[p], vk = f.v[k];
    e.classList.toggle('block', vp !== null && vk !== null && vp < 0.6 && vk < 0.8); }});
  document.getElementById('wk').textContent = f.label; sl.value = j; prev = f; }}
const sl = document.getElementById('sl'), pp = document.getElementById('pp'); sl.max = D.frames.length - 1;
let timer = null;
function stop() {{ clearInterval(timer); timer = null; pp.textContent = '▶ Play'; pp.classList.remove('on'); }}
function play() {{ if (cur >= D.frames.length - 1) draw(0); pp.textContent = '❚❚ Pause'; pp.classList.add('on');
  timer = setInterval(() => {{ if (cur >= D.frames.length - 1) return stop(); draw(cur + 1); }}, 650); }}
pp.onclick = () => timer ? stop() : play(); sl.oninput = () => {{ stop(); draw(+sl.value); }};
if (D.frames.length < 2) {{ pp.style.display = 'none'; sl.style.display = 'none'; }}
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if (D.auto && D.frames.length > 1 && !reduce) {{ draw(0); setTimeout(play, 500); }} else draw(D.frames.length - 1);
</script></body></html>"""


def heat_replay(students, skill_ids, skill_names, frames, labels, taught, dark=False, height=None):
    """The class week by week: one row per student, one column per skill, each cell coloured by the chance of a
    right answer; the cells morph from week to week. frames: (weeks, students, skills); taught: (weeks, skills)."""
    c = _pal(dark)
    F = np.asarray(frames, float)
    T = np.asarray(taught, bool)
    data = {"s": [str(s) for s in students], "k": list(skill_ids), "kn": [str(n) for n in skill_names],
            "f": np.where(T[:, None, :], np.round(F, 3), -1).tolist(), "lab": list(labels)}
    height = height or int(70 + 18 * len(students) + 110)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{_base_css(c)}
canvas{{width:100%;display:block}}
</style></head><body>
<div class="bar"><button id="pp">▶ Play</button><input id="sl" type="range" min="0" step="0.01" value="0">
<span class="lab" id="wk"></span></div>
<canvas id="cv"></canvas><div class="tip" id="tip"></div>
<div class="leg"><span><i class="sw" style="background:{c['bad']}"></i>below 50%</span>
<span><i class="sw" style="background:{c['mid']}"></i>70%</span><span><i class="sw" style="background:{c['ok']}"></i>85% and up</span>
<span><i class="sw" style="background:{c['none']}"></i>not taught yet</span><span>bottom: class average per skill · right: each student's average</span></div>
<script>
const D = {_js(data)}, C = {_js(c)}, FONT = {_js(FONT)};
{_SCALE_JS}
const cv = document.getElementById('cv'), g = cv.getContext('2d'), tip = document.getElementById('tip');
const S = D.s.length, K = D.k.length, W = D.f.length;
let t = 0, L = {{}};
function lay() {{ const w = cv.clientWidth || 800, left = Math.min(150, w * 0.2), right = 70, top = 26, bottom = 74;
  const cw = (w - left - right) / K, ch = Math.max(10, Math.min(18, cw * 0.8));
  const h = top + S * ch + bottom; cv.width = w * devicePixelRatio; cv.height = h * devicePixelRatio; cv.style.height = h + 'px';
  L = {{w, left, right, top, bottom, cw, ch, h}}; }}
function val(i, k) {{ const a = Math.floor(t), b = Math.min(a + 1, W - 1), u = t - a;
  const x = D.f[a][i][k], y = D.f[b][i][k]; if (x < 0 && y < 0) return null; if (x < 0) return u > 0.5 ? y : null;
  if (y < 0) return x; return x + (y - x) * u; }}
function draw() {{ const {{left, top, cw, ch}} = L; g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  g.clearRect(0, 0, L.w, L.h); g.font = '11px ' + FONT; g.textBaseline = 'middle';
  g.fillStyle = C.muted; g.textAlign = 'center';
  D.k.forEach((k, j) => g.fillText(k, left + cw * j + cw / 2, top - 10));
  const colAvg = new Array(K).fill(0), colN = new Array(K).fill(0);
  for (let i = 0; i < S; i++) {{ let s = 0, n = 0;
    g.textAlign = 'right'; g.fillStyle = C.ink; g.fillText(D.s[i], left - 8, top + ch * i + ch / 2);
    for (let k = 0; k < K; k++) {{ const v = val(i, k); g.fillStyle = col(v);
      g.fillRect(left + cw * k + 1, top + ch * i + 1, cw - 2, ch - 2);
      if (v !== null) {{ s += v; n++; colAvg[k] += v; colN[k]++; }} }}
    const a = n ? s / n : null, bw = L.right - 22;
    g.fillStyle = C.grid; g.fillRect(left + cw * K + 10, top + ch * i + 3, bw, ch - 6);
    if (a !== null) {{ g.fillStyle = col(a); g.fillRect(left + cw * K + 10, top + ch * i + 3, bw * a, ch - 6); }} }}
  const by = top + S * ch + 12, bh = 44;
  for (let k = 0; k < K; k++) {{ const a = colN[k] ? colAvg[k] / colN[k] : null; g.fillStyle = C.grid;
    g.fillRect(left + cw * k + 2, by, cw - 4, bh);
    if (a !== null) {{ g.fillStyle = col(a); g.fillRect(left + cw * k + 2, by + bh * (1 - a), cw - 4, bh * a); }} }}
  g.textAlign = 'right'; g.fillStyle = C.muted; g.fillText('class', left - 8, by + bh / 2);
  document.getElementById('wk').textContent = D.lab[Math.round(t)]; sl.value = t; }}
const sl = document.getElementById('sl'), pp = document.getElementById('pp'); sl.max = W - 1;
let raf = null, last = 0;
function stop() {{ cancelAnimationFrame(raf); raf = null; pp.textContent = '▶ Play'; pp.classList.remove('on'); }}
function step(now) {{ if (last) t = Math.min(W - 1, t + (now - last) / 700); last = now; draw();
  if (t >= W - 1) return stop(); raf = requestAnimationFrame(step); }}
pp.onclick = () => {{ if (raf) return stop(); if (t >= W - 1) t = 0; last = 0; pp.textContent = '❚❚ Pause';
  pp.classList.add('on'); raf = requestAnimationFrame(step); }};
sl.oninput = () => {{ stop(); t = +sl.value; draw(); }};
cv.addEventListener('mousemove', ev => {{ const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  const k = Math.floor((x - L.left) / L.cw), i = Math.floor((y - L.top) / L.ch);
  if (k < 0 || k >= K || i < 0 || i >= S) {{ tip.style.display = 'none'; return; }}
  const v = val(i, k); tip.innerHTML = `<b>${{D.s[i]}}</b> · ${{D.kn[k]}}<br>${{v === null ? 'not taught yet' : Math.round(v * 100) + '% chance on a typical question'}}`;
  tip.style.display = 'block'; tip.style.left = Math.min(ev.clientX + 14, innerWidth - 290) + 'px'; tip.style.top = (ev.clientY + 12) + 'px'; }});
cv.addEventListener('mouseleave', () => tip.style.display = 'none');
addEventListener('resize', () => {{ lay(); draw(); }});
lay(); draw();
if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) setTimeout(() => pp.click(), 400); else {{ t = W - 1; draw(); }}
</script></body></html>"""


def race(score_a, score_b, weak_a, weak_b, names, labels=("One revision paper for everybody", "A personal set each"),
         dark=False, height=330):
    """The what-if: each student a dot on the expected exam score, two lanes (two ways of practising), moving week by
    week. score_*: (weeks + 1, students) expected score; weak_*: same shape, weak spots per student."""
    c = _pal(dark)
    A, B = np.asarray(score_a, float), np.asarray(score_b, float)
    lo = float(np.floor(min(A.min(), B.min()) * 20) / 20)
    hi = float(min(1.0, np.ceil(max(A.max(), B.max()) * 20) / 20))
    data = {"a": np.round(A, 4).tolist(), "b": np.round(B, 4).tolist(), "wa": np.asarray(weak_a).tolist(),
            "wb": np.asarray(weak_b).tolist(), "names": [str(n) for n in names], "lab": list(labels),
            "lo": lo, "hi": hi}
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{_base_css(c)}
canvas{{width:100%;display:block}} .stat{{font-size:13px;color:{c['muted']}}}
</style></head><body>
<div class="bar"><button id="pp">▶ Play</button><span class="lab" id="wk"></span></div>
<canvas id="cv"></canvas><div class="tip" id="tip"></div>
<div class="leg"><span>each dot: one student's expected score on a paper with one question per skill taught</span>
<span><i class="sw" style="background:{c['bad']}"></i>6+ weak spots</span><span><i class="sw" style="background:{c['mid']}"></i>3–5</span>
<span><i class="sw" style="background:{c['ok']}"></i>0–2</span></div>
<script>
const D = {_js(data)}, C = {_js(c)}, FONT = {_js(FONT)};
const cv = document.getElementById('cv'), g = cv.getContext('2d'), tip = document.getElementById('tip');
const W = D.a.length, S = D.names.length; let t = 0, L = {{}}, pts = [];
const jit = i => ((Math.sin(i * 12.9898) * 43758.5453) % 1 + 1) % 1;
function lay() {{ const w = cv.clientWidth || 800, h = 270; cv.width = w * devicePixelRatio; cv.height = h * devicePixelRatio;
  cv.style.height = h + 'px'; L = {{w, h, l: 20, r: 20, lane: 105, top: 24}}; }}
const X = v => L.l + (L.w - L.l - L.r) * (v - D.lo) / (D.hi - D.lo);
function at(arr, i) {{ const a = Math.floor(t), b = Math.min(a + 1, W - 1), u = t - a; return arr[a][i] + (arr[b][i] - arr[a][i]) * u; }}
function colW(n) {{ return n >= 6 ? C.bad : n >= 3 ? C.mid : C.ok; }}
function draw() {{ g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); g.clearRect(0, 0, L.w, L.h);
  g.font = '11px ' + FONT; g.textBaseline = 'middle'; pts = [];
  for (let v = Math.ceil(D.lo * 20) / 20; v <= D.hi + 1e-9; v += 0.05) {{ g.strokeStyle = C.grid; g.beginPath();
    g.moveTo(X(v), L.top); g.lineTo(X(v), L.top + 2 * L.lane); g.stroke(); g.fillStyle = C.muted; g.textAlign = 'center';
    g.fillText(Math.round(v * 100) + '%', X(v), L.top + 2 * L.lane + 12); }}
  [[D.a, D.wa], [D.b, D.wb]].forEach(([sc, wk], j) => {{ const y0 = L.top + j * L.lane;
    g.fillStyle = C.ink; g.textAlign = 'left'; g.font = '600 12.5px ' + FONT; g.fillText(D.lab[j], L.l, y0 + 10); g.font = '11px ' + FONT;
    let m = 0, wsum = 0;
    for (let i = 0; i < S; i++) {{ const v = at(sc, i), w = at(wk, i), x = X(v), y = y0 + 30 + jit(i + j * 7) * (L.lane - 44);
      m += v / S; wsum += w / S; g.fillStyle = colW(Math.round(w)); g.globalAlpha = 0.85; g.beginPath();
      g.arc(x, y, 5.5, 0, 7); g.fill(); g.globalAlpha = 1; pts.push([x, y, i, v, w]); }}
    g.strokeStyle = C.accent; g.lineWidth = 2; g.beginPath(); g.moveTo(X(m), y0 + 22); g.lineTo(X(m), y0 + L.lane - 8); g.stroke();
    g.lineWidth = 1; g.fillStyle = C.accent; g.textAlign = 'right'; g.font = '700 12.5px ' + FONT;
    g.fillText(`mean ${{(m * 100).toFixed(1)}}% · ${{wsum.toFixed(1)}} weak spots per student`, L.w - L.r, y0 + 10); g.font = '11px ' + FONT; }});
  document.getElementById('wk').textContent = Math.round(t) === 0 ? 'today' : 'after week ' + Math.round(t); }}
const pp = document.getElementById('pp'); let raf = null, last = 0;
function stop() {{ cancelAnimationFrame(raf); raf = null; pp.textContent = '▶ Play'; pp.classList.remove('on'); }}
function step(now) {{ if (last) t = Math.min(W - 1, t + (now - last) / 1100); last = now; draw();
  if (t >= W - 1) return stop(); raf = requestAnimationFrame(step); }}
pp.onclick = () => {{ if (raf) return stop(); if (t >= W - 1) t = 0; last = 0; pp.textContent = '❚❚ Pause'; pp.classList.add('on');
  raf = requestAnimationFrame(step); }};
cv.addEventListener('mousemove', ev => {{ const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  let best = null, bd = 64; pts.forEach(p => {{ const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < bd) {{ bd = d; best = p; }} }});
  if (!best) {{ tip.style.display = 'none'; return; }}
  tip.innerHTML = `<b>${{D.names[best[2]]}}</b><br>expected score ${{(best[3] * 100).toFixed(0)}}% · ${{Math.round(best[4])}} weak spots`;
  tip.style.display = 'block'; tip.style.left = Math.min(ev.clientX + 14, innerWidth - 290) + 'px'; tip.style.top = (ev.clientY + 12) + 'px'; }});
cv.addEventListener('mouseleave', () => tip.style.display = 'none');
addEventListener('resize', () => {{ lay(); draw(); }});
lay(); draw(); if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) setTimeout(() => pp.click(), 400);
</script></body></html>"""
