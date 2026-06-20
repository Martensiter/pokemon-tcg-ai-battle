"""Human-vs-AI: play the Pokemon TCG against our MCTS agent in a browser.

Architecture borrowed from charlielockyer-rice/cabt-viewer (MIT): the human drives
player 0 through the UI; our MctsAgent auto-plays player 1. Self-contained Python
app (stdlib http.server + an embedded single-page UI) — no Node, no Vite. Real
card art is loaded from public hosts (falls back to text tiles offline); a deck
picker lets you choose both decks before a match.

  python tools/play_vs_ai.py                 # human=deck.csv, AI=deck_crustle_v2.csv
  python tools/play_vs_ai.py --human deck_crustle_v2.csv --ai deck_meta_dragapult.csv
  python tools/play_vs_ai.py --selftest      # no browser; drives a few moves to verify

Then open http://127.0.0.1:8080
"""
from __future__ import annotations

import os
import sys
import glob
import json
import random
import argparse
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from cg.api import all_attack, OptionType, SelectContext, LogType, AreaType  # noqa: E402
from agent.cards import get_db  # noqa: E402
from agent.card_images import image_url  # noqa: E402
from agent.agent import MctsAgent  # noqa: E402
from agent.base import read_deck  # noqa: E402

DB = get_db()
ATTACKS = {a.attackId: a for a in all_attack()}
ENERGY = {0: ("C", "#d8d8d8"), 1: ("G", "#5fbf6f"), 2: ("R", "#e8643c"), 3: ("W", "#4aa3e0"),
          4: ("L", "#f2c233"), 5: ("P", "#b061c4"), 6: ("F", "#c8772e"), 7: ("D", "#5a5560"),
          8: ("M", "#9aa4b0"), 9: ("N", "#c9a227"), 10: ("*", "#cfa6e0"), 11: ("TR", "#b04a4a")}
_AREA_KEY = {AreaType.HAND.value: "hand", AreaType.DISCARD.value: "discard",
             AreaType.ACTIVE.value: "active", AreaType.BENCH.value: "bench",
             AreaType.PRIZE.value: "prize"}

HUMAN, AI = 0, 1


def _card_at(state, area, idx, pi):
    try:
        pl = state["players"][pi if pi is not None else state["yourIndex"]]
        key = _AREA_KEY.get(area)
        if not key:
            return None
        arr = pl.get(key) or []
        c = arr[idx]
        return c[0] if (key == "active" and isinstance(c, list)) else c
    except (KeyError, IndexError, TypeError):
        return None


def _inplay_name(state, area, idx, pi):
    c = _card_at(state, area, idx, pi)
    return DB.name(c["id"]) if c else "?"


def option_label(o: dict, state: dict, ctx: int) -> str:
    t = o.get("type")
    try:
        if t == OptionType.END.value:
            return "End turn"
        if t == OptionType.RETREAT.value:
            return "Retreat active"
        if t == OptionType.YES.value:
            return "Yes"
        if t == OptionType.NO.value:
            return "No"
        if t == OptionType.NUMBER.value:
            return f"{o.get('number')}"
        if t == OptionType.ATTACK.value:
            a = ATTACKS.get(o.get("attackId"))
            return f"Attack — {a.name} ({a.damage} dmg)" if a else "Attack"
        if t == OptionType.PLAY.value:
            hand = state["players"][state["yourIndex"]].get("hand") or []
            i = o.get("index")
            nm = DB.name(hand[i]["id"]) if (i is not None and i < len(hand) and hand[i]) else "card"
            return f"Play {nm}"
        if t == OptionType.ATTACH.value:
            src = _card_at(state, o.get("area"), o.get("index"), o.get("playerIndex"))
            tgt = _inplay_name(state, o.get("inPlayArea"), o.get("inPlayIndex"), o.get("playerIndex"))
            return f"Attach {DB.name(src['id']) if src else 'energy'} → {tgt}"
        if t == OptionType.EVOLVE.value:
            src = _card_at(state, o.get("area"), o.get("index"), o.get("playerIndex"))
            tgt = _inplay_name(state, o.get("inPlayArea"), o.get("inPlayIndex"), o.get("playerIndex"))
            return f"Evolve {tgt} → {DB.name(src['id']) if src else '?'}"
        if t == OptionType.ABILITY.value:
            c = _card_at(state, o.get("area"), o.get("index"), o.get("playerIndex"))
            return f"Ability — {DB.name(c['id']) if c else '?'}"
        if t in (OptionType.CARD.value, OptionType.TOOL_CARD.value,
                 OptionType.ENERGY_CARD.value, OptionType.DISCARD.value):
            c = _card_at(state, o.get("area"), o.get("index"), o.get("playerIndex"))
            verb = {SelectContext.SETUP_ACTIVE_POKEMON.value: "Make active",
                    SelectContext.SETUP_BENCH_POKEMON.value: "To bench",
                    SelectContext.TO_BENCH.value: "To bench",
                    SelectContext.SWITCH.value: "Switch to",
                    SelectContext.DISCARD.value: "Discard",
                    SelectContext.TO_HAND.value: "To hand"}.get(ctx, "Choose")
            return f"{verb} {DB.name(c['id']) if c else 'card'}"
        if t == OptionType.ENERGY.value:
            return "Energy"
    except Exception:
        pass
    try:
        return OptionType(t).name.title()
    except Exception:
        return f"Option {t}"


def log_line(lg: dict) -> str | None:
    t = lg.get("type")
    p = lg.get("playerIndex")
    who = "You" if p == HUMAN else ("AI" if p == AI else "")
    n = lambda cid: DB.name(cid) if cid else "?"
    if t == LogType.TURN_START.value:
        return f"— {who} turn —"
    if t == LogType.DRAW.value and p == HUMAN:
        return f"You draw {n(lg.get('cardId'))}"
    if t == LogType.DRAW_REVERSE.value or (t == LogType.DRAW.value and p == AI):
        return f"{who} draws a card"
    if t == LogType.PLAY.value:
        return f"{who} plays {n(lg.get('cardId'))}"
    if t == LogType.ATTACH.value:
        return f"{who} attaches {n(lg.get('cardId'))} to {n(lg.get('cardIdTarget'))}"
    if t == LogType.EVOLVE.value:
        return f"{who} evolves into {n(lg.get('cardId'))}"
    if t == LogType.ATTACK.value:
        a = ATTACKS.get(lg.get("attackId"))
        return f"{who} attacks with {n(lg.get('cardId'))}: {a.name if a else ''}"
    if t == LogType.HP_CHANGE.value:
        v = lg.get("value", 0)
        return f"   {n(lg.get('cardId'))} HP {'+' if v > 0 else ''}{v}"
    if t == LogType.SWITCH.value:
        return f"{who} switches Pokemon"
    if t == LogType.RESULT.value:
        r = lg.get("result")
        return "★ " + ("Draw" if r == 2 else ("You win!" if r == HUMAN else "AI wins"))
    for lt, w in [(LogType.POISONED, "poisoned"), (LogType.BURNED, "burned"),
                  (LogType.ASLEEP, "asleep"), (LogType.PARALYZED, "paralyzed"),
                  (LogType.CONFUSED, "confused")]:
        if t == lt.value:
            return f"   {n(lg.get('cardId'))} {'cured of ' if lg.get('isRecover') else ''}{w}"
    return None


def list_decks() -> list[str]:
    names = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "deck*.csv"))}
    return sorted(names)


class Session:
    def __init__(self, human_deck, ai_deck, human_name, ai_name, seed=0):
        self.human_deck = human_deck
        self.ai = MctsAgent(deck=ai_deck, seed=seed)
        self.human_name = human_name
        self.ai_name = ai_name
        self.lock = threading.Lock()
        self.log: list[str] = []
        self.obs = None
        self.start()

    def start(self):
        self.log = []
        self.obs, sd = battle_start(self.human_deck, self.ai.deck)
        if self.obs is None:
            raise RuntimeError(f"battle_start failed: {sd.errorPlayer}/{sd.errorType}")
        self._advance()

    def _collect_logs(self):
        for lg in (self.obs.get("logs") or []):
            line = log_line(lg)
            if line:
                self.log.append(line)

    def _advance(self):
        for _ in range(2000):
            self._collect_logs()
            cur = self.obs.get("current")
            if not cur or cur.get("result", -1) != -1:
                return
            sel = self.obs.get("select")
            if sel is None:
                return
            who = cur["yourIndex"]
            if who == AI:
                self.obs = battle_select(self.ai.decide(self.obs))
                continue
            if len(sel["option"]) == 1 and sel["minCount"] >= 1:
                self.obs = battle_select([0])
                continue
            return

    def select(self, indices):
        cur = self.obs.get("current")
        if not cur or cur.get("result", -1) != -1 or self.obs.get("select") is None:
            return
        if cur["yourIndex"] != HUMAN:
            return
        self.obs = battle_select(list(indices))
        self._advance()

    def view(self):
        cur = self.obs.get("current") or {}
        sel = self.obs.get("select")
        result = cur.get("result", -1)
        your_turn = bool(sel and cur.get("yourIndex") == HUMAN and result == -1)
        decision = None
        if your_turn:
            ctx = sel["context"]
            decision = {"context": SelectContext(ctx).name.replace("_", " ").title(),
                        "min": sel["minCount"], "max": sel["maxCount"],
                        "options": [{"idx": i, "label": option_label(o, cur, ctx)}
                                    for i, o in enumerate(sel["option"])]}
        ids = set()
        for pl in cur.get("players", [{}, {}]):
            for z in ("active", "bench"):
                for pk in (pl.get(z) or []):
                    if pk:
                        ids.add(pk["id"])
            for c in (pl.get("hand") or []):
                if c:
                    ids.add(c["id"])
        cards = {cid: {"n": DB.name(cid),
                       "ex": bool(DB.card(cid) and (DB.card(cid).ex or DB.card(cid).megaEx)),
                       "img": image_url(cid)} for cid in ids}
        return {"ok": True, "turn": cur.get("turn", 0), "yourTurn": your_turn,
                "result": result, "current": cur, "decision": decision,
                "cards": cards, "energy": ENERGY, "log": self.log[-80:],
                "decks": list_decks(), "human": self.human_name, "ai": self.ai_name}


SESSION: Session | None = None
SEED = 0


def set_session(human_name=None, ai_name=None):
    global SESSION
    h = human_name or SESSION.human_name
    a = ai_name or SESSION.ai_name
    available = set(list_decks())
    if h not in available or a not in available:
        raise ValueError("unknown deck")
    SESSION = Session(read_deck(os.path.join(ROOT, h)), read_deck(os.path.join(ROOT, a)), h, a, seed=SEED)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            with SESSION.lock:
                self._send(200, json.dumps(SESSION.view()))
        else:
            self._send(404, json.dumps({"ok": False}))

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(ln) if ln else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            body = {}
        with (SESSION.lock if self.path == "/api/select" else threading.Lock()):
            if self.path == "/api/select":
                SESSION.select(body.get("indices", []))
            elif self.path == "/api/new":
                if body.get("human") or body.get("ai"):
                    try:
                        set_session(body.get("human"), body.get("ai"))
                    except Exception:
                        SESSION.start()
                else:
                    SESSION.start()
            self._send(200, json.dumps(SESSION.view()))


def selftest():
    global SESSION
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    SESSION = Session(read_deck(os.path.join(ROOT, "deck.csv")),
                      read_deck(os.path.join(ROOT, "deck_crustle_v2.csv")), "deck.csv", "deck_crustle_v2.csv", seed=1)
    rng = random.Random(0)
    for _ in range(40):
        v = SESSION.view()
        if v["result"] != -1:
            print("game ended, result:", v["result"]); break
        if v["yourTurn"]:
            d = v["decision"]
            k = min(max(d["min"], 1), d["max"], len(d["options"]))
            SESSION.select(rng.sample(range(len(d["options"])), k))
    print("selftest OK; decks found:", v.get("decks")); print("last log:", SESSION.log[-3:])


def main():
    global SESSION, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="deck.csv")
    ap.add_argument("--ai", default="deck_crustle_v2.csv")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    SEED = args.seed
    os.environ.setdefault("PTCG_MOVE_BUDGET", "0.25")
    if args.selftest:
        selftest()
        return
    SESSION = Session(read_deck(os.path.join(ROOT, args.human)),
                      read_deck(os.path.join(ROOT, args.ai)), args.human, args.ai, seed=args.seed)
    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"You ({args.human}) vs AI ({args.ai}).  Open  http://127.0.0.1:{args.port}")
    print("Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            battle_finish()
        except Exception:
            pass


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>You vs AI — Pokemon TCG</title>
<style>
:root{--bg:#0f1419;--panel:#1b2330;--p2:#222c3c;--line:#33415a;--txt:#dfe7f2;--mut:#8aa0bd;--acc:#6cf}
*{box-sizing:border-box;font-family:Segoe UI,Roboto,sans-serif}
body{margin:0;background:var(--bg);color:var(--txt);font-size:13px}
#wrap{max-width:1040px;margin:0 auto;padding:12px}
h1{font-size:15px;margin:0 0 8px}
#setup{display:flex;gap:8px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 10px;margin-bottom:8px}
select{background:var(--p2);color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:5px 8px}
.player{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;margin:8px 0}
.phead{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.tag{background:var(--p2);border:1px solid var(--line);border-radius:18px;padding:2px 9px}
.turn{color:#7ee787;font-weight:bold}
.row{display:flex;gap:8px;flex-wrap:wrap}
.card{width:116px;background:var(--p2);border:1px solid var(--line);border-radius:8px;padding:6px}
.card.active{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}
.cimg{width:100%;height:92px;object-fit:cover;object-position:top center;border-radius:5px;margin-bottom:4px;display:block}
.cn{font-weight:600;font-size:12px;line-height:1.15;margin-bottom:4px}.ex{color:#ffd479}
.hpbar{height:6px;border-radius:3px;background:#3a2733;overflow:hidden;margin:3px 0}.hpfill{height:100%}
.small{color:var(--mut);font-size:11px}.en{display:inline-flex;gap:2px;flex-wrap:wrap;margin-top:4px}
.e{width:15px;height:15px;border-radius:50%;font-size:9px;color:#111;display:flex;align-items:center;justify-content:center;font-weight:700}
#decision{background:#0c1118;border:1px solid var(--acc);border-radius:10px;padding:10px;margin:8px 0}
#decision.wait{border-color:var(--line)}
.opt{display:inline-block;background:var(--p2);border:1px solid var(--line);border-radius:7px;padding:7px 11px;margin:4px 4px 0 0;cursor:pointer}
.opt:hover{background:#30425c}.opt.sel{background:#1d6e4f;border-color:#2ea36f}
button{background:var(--p2);color:var(--txt);border:1px solid var(--line);border-radius:7px;padding:7px 13px;cursor:pointer}
button.primary{background:#1d6e4f;border-color:#2ea36f}
#log{background:#0c1118;border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin-top:8px;max-height:160px;overflow:auto;font-family:Consolas,monospace;font-size:12px;line-height:1.5}
.win{color:#7ee787;font-weight:bold}.hand{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}
.hc{background:#172230;border:1px solid var(--line);border-radius:5px;padding:3px 7px;font-size:11px}
.prz{display:inline-flex;gap:3px}.pz{width:11px;height:15px;border-radius:2px;background:#3b4a63;border:1px solid #54688c}.pz.t{background:#1a2230;border-style:dashed}
</style></head><body><div id="wrap">
<h1>You vs AI — Pokemon TCG</h1>
<div id="setup">
  <label>You play <select id="humanSel"></select></label>
  <label>AI plays <select id="aiSel"></select></label>
  <button class="primary" onclick="newGame()">Start new match</button>
</div>
<div id="opp" class="player"></div>
<div id="decision" class="wait">loading…</div>
<div id="me" class="player"></div>
<div id="log"></div>
</div>
<script>
let S=null, picked=new Set(), setupDone=false;
function dots(es){return (es||[]).map(t=>{const e=S.energy[t]||["?","#888"];return `<span class="e" style="background:${e[1]}">${e[0]}</span>`}).join('')}
function nmeOf(id){return (S.cards[id]||{n:'#'+id}).n}
function exOf(id){return (S.cards[id]||{}).ex}
function imgOf(id){return (S.cards[id]||{}).img}
function cardHTML(pk,act){if(!pk)return `<div class="card" style="opacity:.4">—</div>`;
 const pct=Math.max(0,Math.min(100,100*pk.hp/(pk.maxHp||pk.hp||1)));
 const col=pct>50?'#54c163':pct>25?'#e0b13a':'#e0563a';
 const img=imgOf(pk.id)?`<img class="cimg" loading="lazy" src="${imgOf(pk.id)}" onerror="this.remove()">`:'';
 return `<div class="card ${act?'active':''}">${img}<div class="cn ${exOf(pk.id)?'ex':''}">${nmeOf(pk.id)}</div>
 <div class="hpbar"><div class="hpfill" style="width:${pct}%;background:${col}"></div></div>
 <div class="small">${pk.hp}/${pk.maxHp} HP</div><div class="en">${dots(pk.energies)}</div></div>`}
function prizes(pl){const tk=6-pl.prize.length;let h='';for(let i=0;i<6;i++)h+=`<span class="pz ${i<tk?'t':''}"></span>`;return h}
function playerHTML(pl,label,isTurn,showHand){
 const act=(pl.active&&pl.active[0])?pl.active[0]:null;
 const bench=(pl.bench||[]).map(b=>cardHTML(b,false)).join('')||'<span class="small">(empty bench)</span>';
 let hand='';if(showHand&&pl.hand)hand=`<div class="hand">`+pl.hand.map(c=>`<span class="hc">${nmeOf(c.id)}</span>`).join('')+`</div>`;
 return `<div class="phead"><b>${label}</b> ${isTurn?'<span class="turn">● to move</span>':''}
 <span class="tag">Prizes <span class="prz">${prizes(pl)}</span></span>
 <span class="tag">Hand ${pl.handCount}</span><span class="tag">Deck ${pl.deckCount}</span>
 <span class="tag">Discard ${pl.discard.length}</span></div>
 <div class="row"><div><div class="small">Active</div>${cardHTML(act,true)}</div>
 <div style="flex:1"><div class="small">Bench</div><div class="row">${bench}</div>${hand}</div></div>`}
function fillSetup(){
 const hs=document.getElementById('humanSel'),as=document.getElementById('aiSel');
 hs.innerHTML='';as.innerHTML='';
 for(const d of S.decks){
   hs.insertAdjacentHTML('beforeend',`<option ${d===S.human?'selected':''}>${d}</option>`);
   as.insertAdjacentHTML('beforeend',`<option ${d===S.ai?'selected':''}>${d}</option>`);
 }
 setupDone=true;
}
function render(){
 if(!setupDone)fillSetup();
 const cur=S.current, pls=cur.players;
 document.getElementById('opp').innerHTML=playerHTML(pls[1],'AI — '+S.ai,cur.yourIndex===1,false);
 document.getElementById('me').innerHTML=playerHTML(pls[0],'You — '+S.human,cur.yourIndex===0,true);
 const D=document.getElementById('decision');
 if(S.result!==-1){D.className='wait';D.innerHTML=`<b class="win">${S.result===0?'You win!':S.result===2?'Draw':'AI wins'}</b> — press “Start new match”.`}
 else if(S.yourTurn){D.className='';const d=S.decision;
   const need=d.min===d.max?`choose ${d.min}`:`choose ${d.min}–${d.max}`;
   const opts=d.options.map(o=>`<span class="opt" data-i="${o.idx}" onclick="pick(${o.idx},${d.max})">${o.label}</span>`).join('');
   const confirm=d.max>1?`<div style="margin-top:8px"><button class="primary" onclick="submit()">Confirm (<span id="cnt">0</span>)</button> <span class="small">${need}</span></div>`:`<div class="small" style="margin-top:6px">${need} — click an option</div>`;
   D.innerHTML=`<div style="margin-bottom:6px"><b>Your decision:</b> ${d.context}</div>${opts}${confirm}`;picked=new Set();
 } else {D.className='wait';D.innerHTML='<span class="small">AI is thinking…</span>'}
 const L=document.getElementById('log');L.innerHTML=(S.log||[]).map(t=>t.startsWith('★')?`<div class="win">${t}</div>`:`<div>${t}</div>`).join('');L.scrollTop=L.scrollHeight;
}
function pick(i,max){if(max<=1){submitNow([i]);return}
 const el=document.querySelector(`.opt[data-i="${i}"]`);
 if(picked.has(i)){picked.delete(i);el.classList.remove('sel')}else{picked.add(i);el.classList.add('sel')}
 document.getElementById('cnt').textContent=picked.size;}
function submit(){submitNow([...picked])}
async function submitNow(indices){document.getElementById('decision').innerHTML='<span class="small">…</span>';
 const r=await fetch('/api/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({indices})});S=await r.json();render();}
async function newGame(){const human=document.getElementById('humanSel').value,ai=document.getElementById('aiSel').value;
 const r=await fetch('/api/new',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({human,ai})});S=await r.json();render();}
async function load(){const r=await fetch('/api/state');S=await r.json();render();}
load();
</script></body></html>"""


if __name__ == "__main__":
    main()
