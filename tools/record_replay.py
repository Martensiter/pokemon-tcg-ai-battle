"""Record a local game and emit a self-contained interactive HTML replay.

The engine's visualize_data() returns the whole game as a JSON array of board
snapshots. We run a match between any two agents, grab that array, attach
human-readable card names + a per-step event log, and bake everything into one
replay.html you can open in a browser. Step through the match move by move with
no Kaggle and no server.

  python tools/record_replay.py --a deck_crustle_v2.csv --b deck_meta_nonex.csv \
                                --a-agent greedy --b-agent greedy --out replay.html
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cg.game import battle_start, battle_select, battle_finish, visualize_data  # noqa: E402
from cg.api import all_attack, LogType, AreaType, EnergyType  # noqa: E402
from agent.cards import get_db  # noqa: E402
from selfplay.baselines import read_deck, RandomAgent, GreedyAgent, random_legal  # noqa: E402

ENERGY = {  # type id -> (short label, css color)
    0: ("C", "#d8d8d8"), 1: ("G", "#5fbf6f"), 2: ("R", "#e8643c"), 3: ("W", "#4aa3e0"),
    4: ("L", "#f2c233"), 5: ("P", "#b061c4"), 6: ("F", "#c8772e"), 7: ("D", "#5a5560"),
    8: ("M", "#9aa4b0"), 9: ("N", "#c9a227"), 10: ("*", "#cfa6e0"), 11: ("TR", "#b04a4a"),
}
AREA = {a.value: a.name.title() for a in AreaType}


def make_agent(kind, deck, seed):
    if kind == "mcts":
        from agent.agent import MctsAgent
        return MctsAgent(deck=deck, seed=seed)
    if kind == "greedy":
        return GreedyAgent(deck=deck, seed=seed)
    return RandomAgent(deck=deck, seed=seed)


def log_to_text(log, db, attacks):
    """Render one engine log entry as a readable line."""
    t = log.get("type")
    p = log.get("playerIndex")
    who = f"P{(p + 1) if p is not None else '?'}"
    nm = lambda cid: db.name(cid) if cid else "?"
    if t == LogType.TURN_START.value:
        return f"——— {who} turn start ———"
    if t == LogType.TURN_END.value:
        return f"{who} ends turn"
    if t == LogType.DRAW.value:
        return f"{who} draws {nm(log.get('cardId'))}"
    if t == LogType.DRAW_REVERSE.value:
        return f"{who} draws a card"
    if t == LogType.PLAY.value:
        return f"{who} plays {nm(log.get('cardId'))}"
    if t == LogType.ATTACH.value:
        return f"{who} attaches {nm(log.get('cardId'))} to {nm(log.get('cardIdTarget'))}"
    if t == LogType.EVOLVE.value:
        return f"{who} evolves {nm(log.get('cardIdTarget'))} → {nm(log.get('cardId'))}"
    if t == LogType.SWITCH.value:
        return f"{who} switches {nm(log.get('cardIdBench'))} ↔ {nm(log.get('cardIdActive'))}"
    if t == LogType.ATTACK.value:
        a = attacks.get(log.get("attackId"))
        return f"{who} {nm(log.get('cardId'))} attacks: {a.name if a else 'attack'}"
    if t == LogType.HP_CHANGE.value:
        v = log.get("value", 0)
        return f"   {nm(log.get('cardId'))} HP {'+' if v > 0 else ''}{v}"
    if t == LogType.MOVE_CARD.value:
        return f"{who} {nm(log.get('cardId'))}: {AREA.get(log.get('fromArea'),'?')} → {AREA.get(log.get('toArea'),'?')}"
    for lt, word in [(LogType.POISONED, "poisoned"), (LogType.BURNED, "burned"),
                     (LogType.ASLEEP, "asleep"), (LogType.PARALYZED, "paralyzed"),
                     (LogType.CONFUSED, "confused")]:
        if t == lt.value:
            return f"   {nm(log.get('cardId'))} {'recovers from ' if log.get('isRecover') else ''}{word}"
    if t == LogType.COIN.value:
        return f"   coin: {'heads' if log.get('head') else 'tails'}"
    if t == LogType.RESULT.value:
        r = log.get("result")
        reason = {1: "0 prizes", 2: "deck out", 3: "no active", 4: "card effect"}.get(log.get("reason"), "")
        winner = "Draw" if r == 2 else f"P{r + 1} wins"
        return f"★ RESULT: {winner} ({reason})"
    return f"{who} [{LogType(t).name if t in [e.value for e in LogType] else t}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="deck_crustle_v2.csv")
    ap.add_argument("--b", default="deck_meta_nonex.csv")
    ap.add_argument("--a-agent", default="greedy")
    ap.add_argument("--b-agent", default="greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "replay.html"))
    args = ap.parse_args()

    db = get_db()
    attacks = {a.attackId: a for a in all_attack()}
    deck_a = read_deck(os.path.join(ROOT, args.a))
    deck_b = read_deck(os.path.join(ROOT, args.b))
    agents = [make_agent(args.a_agent, deck_a, args.seed),
              make_agent(args.b_agent, deck_b, args.seed + 1)]

    obs, start = battle_start(deck_a, deck_b)
    if obs is None:
        raise SystemExit(f"battle failed: {start.errorPlayer}/{start.errorType}")
    steps, winner = 0, 2
    while True:
        s = obs.get("current")
        if s and s.get("result", -1) != -1:
            winner = s["result"]
            break
        sel = obs.get("select")
        if sel is None:
            break
        obs = battle_select(agents[s["yourIndex"]](obs))
        steps += 1
        if steps > 30000:
            break

    replay = json.loads(visualize_data())
    battle_finish()

    # attach readable logs + collect the card ids that appear
    ids = set()
    for frame in replay:
        frame["logText"] = [log_to_text(lg, db, attacks) for lg in (frame.get("logs") or [])]
        cur = frame.get("current") or {}
        for pl in cur.get("players", []):
            for zone in ("active", "bench"):
                for pk in (pl.get(zone) or []):
                    if pk:
                        ids.add(pk["id"])
                        for grp in ("energyCards", "tools"):
                            for c in (pk.get(grp) or []):
                                ids.add(c["id"])
            for c in (pl.get("hand") or []):
                if c:
                    ids.add(c["id"])

    cards = {}
    for cid in ids:
        c = db.card(cid)
        if c:
            cards[cid] = {"n": c.name, "hp": c.hp, "t": c.energyType, "ex": bool(c.ex or c.megaEx)}

    meta = {
        "title": f"{os.path.basename(args.a)} ({args.a_agent}) vs {os.path.basename(args.b)} ({args.b_agent})",
        "winner": winner, "frames": len(replay),
    }
    html = HTML_TEMPLATE.replace("/*REPLAY*/", json.dumps(replay)) \
                        .replace("/*CARDS*/", json.dumps(cards)) \
                        .replace("/*ENERGY*/", json.dumps(ENERGY)) \
                        .replace("/*META*/", json.dumps(meta))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"recorded {len(replay)} frames, winner=P{winner + 1 if winner != 2 else '(draw)'}")
    print(f"open in a browser:  {args.out}")


HTML_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>PTCG Replay</title><style>
:root{--bg:#0f1419;--panel:#1b2330;--panel2:#222c3c;--line:#33415a;--txt:#dfe7f2;--mut:#8aa;--acc:#6cf;}
*{box-sizing:border-box;font-family:Segoe UI,Roboto,sans-serif}
body{margin:0;background:var(--bg);color:var(--txt);font-size:13px}
#top{position:sticky;top:0;background:#0c1118;border-bottom:1px solid var(--line);padding:8px 12px;z-index:5}
#top h1{font-size:14px;margin:0 0 6px}
.ctrl{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
button{background:var(--panel2);color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover{background:#2c3a50}
input[type=range]{flex:1;min-width:160px}
#wrap{max-width:1080px;margin:0 auto;padding:10px}
.player{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;margin:8px 0}
.phead{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.tag{background:var(--panel2);border:1px solid var(--line);border-radius:20px;padding:2px 9px;font-size:12px}
.turnnow{color:#7ee787;font-weight:bold}
.prizes{display:inline-flex;gap:3px}
.pz{width:11px;height:15px;border-radius:2px;background:#3b4a63;border:1px solid #54688c}
.pz.taken{background:#1a2230;border-style:dashed}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start}
.card{width:118px;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:7px;position:relative}
.card.active{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}
.card.empty{opacity:.35;min-height:60px}
.cn{font-weight:600;font-size:12px;line-height:1.15;margin-bottom:4px}
.ex{color:#ffd479}
.hpbar{height:6px;border-radius:3px;background:#33203a;overflow:hidden;margin:3px 0}
.hpfill{height:100%}
.small{color:var(--mut);font-size:11px}
.en{display:inline-flex;gap:2px;flex-wrap:wrap;margin-top:4px}
.e{width:15px;height:15px;border-radius:50%;font-size:9px;color:#111;display:flex;align-items:center;justify-content:center;font-weight:700}
.cond{color:#ff9a9a;font-size:11px}
.bench-label{color:var(--mut);font-size:11px;margin:6px 0 2px}
#center{text-align:center;color:var(--mut);margin:4px 0}
#log{background:#0c1118;border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin-top:8px;max-height:200px;overflow:auto;font-family:Consolas,monospace;font-size:12px;line-height:1.5}
.win{color:#7ee787;font-weight:bold}
.hand{display:flex;gap:4px;flex-wrap:wrap;margin-top:6px}
.hc{background:#17202c;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px;color:var(--mut)}
</style></head><body>
<div id="top"><h1 id="title"></h1>
<div class="ctrl">
<button onclick="go(0)">⏮</button><button onclick="step(-10)">-10</button>
<button onclick="step(-1)">◀</button><button id="play" onclick="toggle()">▶ play</button>
<button onclick="step(1)">▶</button><button onclick="step(10)">+10</button>
<button onclick="go(R.length-1)">⏭</button>
<input type="range" id="slider" min="0" value="0" oninput="go(+this.value)">
<span id="counter" class="tag"></span></div></div>
<div id="wrap">
<div id="p1" class="player"></div>
<div id="center"></div>
<div id="p0" class="player"></div>
<div id="log"></div>
</div>
<script>
const R=/*REPLAY*/, C=/*CARDS*/, EN=/*ENERGY*/, M=/*META*/;
let idx=0, timer=null;
document.getElementById('title').textContent=M.title;
document.getElementById('slider').max=R.length-1;

function energyDots(es){return (es||[]).map(t=>{const e=EN[t]||["?","#888"];
  return `<span class="e" style="background:${e[1]}">${e[0]}</span>`}).join('')}
function conds(pl){return ['poisoned','burned','asleep','paralyzed','confused']
  .filter(k=>pl[k]).map(k=>k[0].toUpperCase()).join(' ')}
function cardHTML(pk,active){
  if(!pk) return `<div class="card empty">—</div>`;
  const c=C[pk.id]||{n:'#'+pk.id,hp:pk.maxHp,ex:false};
  const pct=Math.max(0,Math.min(100,100*pk.hp/(pk.maxHp||pk.hp||1)));
  const col=pct>50?'#54c163':pct>25?'#e0b13a':'#e0563a';
  return `<div class="card ${active?'active':''}">
    <div class="cn ${c.ex?'ex':''}">${c.n}${c.ex?' ✦':''}</div>
    <div class="hpbar"><div class="hpfill" style="width:${pct}%;background:${col}"></div></div>
    <div class="small">${pk.hp}/${pk.maxHp} HP</div>
    <div class="en">${energyDots(pk.energies)}</div>
    ${(pk.tools&&pk.tools.length)?`<div class="small">🛠 ${pk.tools.map(t=>(C[t.id]||{n:'tool'}).n).join(', ')}</div>`:''}
  </div>`}
function handHTML(pl){
  if(pl.hand&&pl.hand.length)return `<div class="hand">`+pl.hand.map(c=>`<span class="hc">${(C[c.id]||{n:'#'+c.id}).n}</span>`).join('')+`</div>`;
  return '';}
function playerHTML(pl,label,isTurn){
  const taken=6-pl.prize.length;
  const pz=Array.from({length:6},(_,i)=>`<span class="pz ${i<taken?'taken':''}"></span>`).join('');
  const act=(pl.active&&pl.active[0])?pl.active[0]:null;
  const bench=(pl.bench||[]).map(b=>cardHTML(b,false)).join('')||'<div class="small">(empty bench)</div>';
  const cd=conds(pl);
  return `<div class="phead">
     <b>${label}</b> ${isTurn?'<span class="turnnow">● to move</span>':''}
     <span class="tag">Prizes <span class="prizes">${pz}</span></span>
     <span class="tag">Hand ${pl.handCount}</span>
     <span class="tag">Deck ${pl.deckCount}</span>
     <span class="tag">Discard ${pl.discard.length}</span>
     ${cd?`<span class="tag cond">⚠ ${cd}</span>`:''}</div>
   <div class="row"><div><div class="bench-label">Active</div>${cardHTML(act,true)}</div>
   <div style="flex:1"><div class="bench-label">Bench</div><div class="row">${bench}</div>${handHTML(pl)}</div></div>`}
function render(){
  const f=R[idx], s=f.current||{}, pls=s.players||[{},{}];
  const turn=s.turn||0, mover=s.yourIndex;
  document.getElementById('p1').innerHTML=playerHTML(pls[1]||{},'Player 2',mover===1);
  document.getElementById('p0').innerHTML=playerHTML(pls[0]||{},'Player 1',mover===0);
  const sel=f.select?(f.select.context):'';
  document.getElementById('center').innerHTML=
    `Turn ${turn} · action ${s.turnActionCount||0} · decision: <b>${sel||'—'}</b>`+
    (s.result!=null&&s.result!=-1?` · <span class="win">${s.result==2?'DRAW':'Player '+(s.result+1)+' WINS'}</span>`:'');
  // accumulate logs up to this frame for context (show last ~40 lines)
  let lines=[];
  for(let i=0;i<=idx;i++)for(const t of (R[i].logText||[]))lines.push(t);
  const L=document.getElementById('log');
  L.innerHTML=lines.slice(-60).map(t=>t.startsWith('★')?`<div class="win">${t}</div>`:`<div>${t}</div>`).join('');
  L.scrollTop=L.scrollHeight;
  document.getElementById('slider').value=idx;
  document.getElementById('counter').textContent=`${idx+1} / ${R.length}`;
}
function go(i){idx=Math.max(0,Math.min(R.length-1,i));render()}
function step(d){go(idx+d)}
function toggle(){const b=document.getElementById('play');
  if(timer){clearInterval(timer);timer=null;b.textContent='▶ play'}
  else{b.textContent='⏸ pause';timer=setInterval(()=>{if(idx>=R.length-1){toggle()}else step(1)},450)}}
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')step(1);if(e.key==='ArrowLeft')step(-1)});
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
