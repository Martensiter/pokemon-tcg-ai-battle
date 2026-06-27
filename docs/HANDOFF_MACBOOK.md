# 引き継ぎ書 — MacBook を engine 機化 & 検証パイプライン試走

`docs/HANDOFF.md` (本体) の **§4「engine機（検証・提出側）= MacBook」** を実機で踏破した記録と、その過程で判明した **HANDOFF より新しい事実** をまとめる。次に同じ作業をやる人は、本書を **HANDOFF.md より優先**して読むこと。

実施: 2026-06-27、ホスト: Apple Silicon MacBook (macOS 26.3.1, arm64, Darwin 25.3.0), 作業ディレクトリ: `/Users/ichitaro/Downloads/AI/pokemon-tcg-ai-battle/`

---

## 0. TL;DR
- ✅ MacBook で `from cg.sim import lib` が通る (**`engine OK (native arm64)`**)。Docker amd64 経由は**不要**だった。
- ✅ Kaggle dataset DL → merge → train → verify の **全段を Mac native で完走**。`new 7 - old 13, win rate 35% FAIL` (= 期待値、データ少量のため candidate は昇格せず)。
- ✅ `agent/weights.npz` は未更新 (md5 `ba2c1e2b797841676cae9a3c58ed10f1` のまま) ＝ promote されていないことを確認。
- 🔁 残作業: データ蓄積を待つ → 数万局面溜まったら本番 verify (`--games 30`) → 良ければ提出。

---

## 1. HANDOFF.md より更新すべき事実 (重要)

### 1-1. macOS は **Docker を経由しなくてよい**
HANDOFF §4 は「macOS は `.dylib` 無し → x86_64 Linux Docker で動かす」と書いているが、**実際は `sample_submission.zip` に 3 アーキ全部同梱されている**:
```
sample_submission/cg/
├── libcg.so         # Linux x86_64
├── libcg-arm64.so   # Linux arm64
└── libcg.dylib      # macOS (Apple Silicon arm64 Mach-O)
```
`cg/sim.py` は OS が `nt` でなければ常に `libcg.so` をロードしようとするが、**macOS の `dlopen` は拡張子非依存** (中身が Mach-O なら OK)。よって以下で Mac ネイティブ実行できる:
```bash
cp sample_submission/cg/libcg.dylib cg/libcg.so
xattr -d com.apple.quarantine cg/libcg.so   # ★これを忘れると下記エラー
python3 -c "from cg.sim import lib; print('engine OK')"
```
- エミュレーション無しで動くので Docker amd64 より遥かに高速。
- `.gitignore` に `cg/libcg.so` と `*.so` があるので誤コミットの心配は無い。

### 1-2. quarantine 属性で **library load disallowed** が出る
Downloads 経由のファイルには `com.apple.quarantine` xattr が付き、未署名 dylib のロードを Gatekeeper が拒否する:
```
OSError: dlopen(...): library load disallowed by system policy
```
解決: `xattr -d com.apple.quarantine cg/libcg.so` (もしくは `xattr -c`)。`com.apple.provenance` は残ってもよい。

### 1-3. Kaggle CLI と認証の仕様変更
- **kaggle 2.x** (server 2.2.2 同期): `competitions ...` と `datasets get/download` の両 API が **OAuth `access_token` 必須** に変わっており、legacy `kaggle.json` (username + key) を **403/401 で拒否**する。
- **kaggle 1.6.17** (legacy): `datasets list/download` は legacy auth で動く。`competitions ...` は server 側で 401。
- → **データ取得作業のみなら `pip install "kaggle<1.7"`** で十分。`competitions submit` だけは `kaggle auth login` (OAuth) かブラウザ手動 DL。
- 余談: コンペスラグは **`pokemon-tcg-ai-battle`** (短い方)。WebFetch で長いスラグ `pokemon-tcg-ai-battle-challenge-simulation` を返されたが、それは別物 (404)。

### 1-4. Hub の kaggle.json と MacBook の kaggle.json で **key がずれる**
両方とも `username=ichitaro3` でも、Kaggle で "Generate New Token" を押すたびに **旧 key は invalidate** される。今回:
- 手元の `~/.kaggle/kaggle.json` の key では `datasets list --mine` が `No datasets found`、`datasets download ichitaro3/ptcg-ladder-replays` が `403 Permission 'datasets.get' was denied`
- Hub `/home/node/.openclaw/extensions/ptcg-collector/.env` から `KAGGLE_KEY` を移したら通った
- → **データセット所有権限が無いと出たら、Hub の `.env` から KEY を移すのが第一手**

### 1-5. HANDOFF.md は **branch `claude/replay-collector-production-7ylifb` 前提**
`main` には以下が **無い**:
- `tools/verify_candidate.py`
- `pyproject.toml`
- `tools/hub/run.sh` (Hub の永続 venv ランチャ)

→ MacBook 側作業は必ず先に `git checkout claude/replay-collector-production-7ylifb`。`cg/libcg.so` は gitignored なので branch 切り替えで失われない。

---

## 2. ローカル環境 (再現手順)

### 2-1. 前提
- Apple Silicon Mac (arm64), macOS 26.x。
- Homebrew 入り、`brew install uv` で uv 0.11+ 入る。
- Kaggle アカウント `ichitaro3` でコンペ「Pokemon TCG AI Battle」の Rules accept 済み。

### 2-2. 一度きりの初期化
```bash
# repo
cd /Users/ichitaro/Downloads/AI/pokemon-tcg-ai-battle
git fetch origin claude/replay-collector-production-7ylifb
git checkout claude/replay-collector-production-7ylifb

# engine binary (ブラウザで sample_submission.zip を kaggle.com/competitions/pokemon-tcg-ai-battle/data から DL し ~/Downloads/ に置いた前提)
unzip -q ~/Downloads/sample_submission.zip -d /tmp/ptcg_extract
cp /tmp/ptcg_extract/sample_submission/cg/libcg.dylib cg/libcg.so
xattr -d com.apple.quarantine cg/libcg.so
rm -rf /tmp/ptcg_extract

# kaggle.json (Hub の .env から KAGGLE_USERNAME/KAGGLE_KEY を抜いて JSON 化)
#   {"username":"ichitaro3","key":"<Hub と同じ key>"}
mkdir -p ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json

# venv
brew install uv
uv venv .venv -p 3.11
uv pip install -e ".[kaggle]"
uv pip install "kaggle<1.7" --reinstall    # ★ datasets API を通すため legacy に落とす

# 動作確認
python3 -c "from cg.sim import lib; print('engine OK')"
```

### 2-3. 状態スナップショット (このセッション終了時点)
| パス | 状態 |
|---|---|
| `cg/libcg.so` | Mach-O 64-bit arm64, 1,245,544 bytes, `xattr` から `quarantine` 除去済み |
| `.venv/bin/python` | CPython 3.11.15 |
| 主要パッケージ | numpy 2.4.6, kaggle 1.6.17, pytest 7.4+ (dev extra), tqdm 4.68 |
| `agent/weights.npz` | 6/24 配置の baseline (md5 `ba2c1e2b797841676cae9a3c58ed10f1`) — promote されず未変更 |
| `~/.kaggle/kaggle.json` | username `ichitaro3` + key length 32, 600 perm。 **Hub と同期済み** |
| ブランチ | `claude/replay-collector-production-7ylifb` |

---

## 3. 検証パイプライン実行 (Mac native)

HANDOFF §4 の Docker ワンライナーを Mac native で展開したもの:
```bash
cd /Users/ichitaro/Downloads/AI/pokemon-tcg-ai-battle

# env から legacy key を渡す
export KAGGLE_USERNAME=$(python3 -c "import json; print(json.load(open('~/.kaggle/kaggle.json'.replace('~', __import__('os').path.expanduser('~'))))['username'])")
export KAGGLE_KEY=$(python3 -c "import json; print(json.load(open('~/.kaggle/kaggle.json'.replace('~', __import__('os').path.expanduser('~'))))['key'])")

# 1) dataset DL
rm -rf ds && mkdir ds
.venv/bin/kaggle datasets download -d ichitaro3/ptcg-ladder-replays -p ds --unzip

# 2) merge
.venv/bin/python tools/merge_collected.py --src ds/value --out selfplay/data_all.npz

# 3) train candidate
.venv/bin/python selfplay/train_value_np.py --data selfplay/data_all.npz --out /tmp/cand.npz

# 4) verify (-- 規模は下記表参照)
.venv/bin/python -u tools/verify_candidate.py --new /tmp/cand.npz --games 2 --threshold 0.53 --promote
```

### 3-1. 規模と実時間 (Apple Silicon native での実測)
- 1 試合 ≈ **76 秒**
- pool は **10 デッキ** (`selfplay/gen_data.resolve_decks('pool')`)
- 計算式: 実時間 ≈ 76s × 10 × `--games`

| `--games` | 試合数 | 実時間 | 用途 |
|---:|---:|---:|---|
| 1 | 10 | ~13 分 | smoke test |
| **2** | **20** | **~25 分** | 動作確認 (今回採用) |
| 8 | 80 | ~1.7 時間 | 軽い本番 (HANDOFF の default) |
| 30 | 300 | **~6.3 時間** | HANDOFF 例示の本番 (夜間 cron 向き) |

### 3-2. 今回の実測結果
```
training on 5763 states (dim=32, hidden=[64, 64])
epoch  60: val_loss=0.4512 val_acc=0.549
saved /tmp/cand.npz (dims=[32, 64, 64, 1])

deck pool (10): deck.csv, deck_crustle_v2.csv, deck_meta_dragapult.csv,
                deck_meta_lucario.csv, deck_meta_abomasnow.csv, deck_meta_nonex.csv,
                deck_meta_mixed.csv, deck_meta_fire_ex.csv, deck_meta_alakazam.csv,
                deck_meta_rocket_spidops.csv
verifying /tmp/cand.npz vs agent/weights.npz (10 decks x 2 games, threshold 53%)
  deck 1/10: new 1 - old 1 (draws 0)
  deck 2/10: new 1 - old 1 (draws 0)
  deck 3/10: new 0 - old 2 (draws 0)
  deck 4/10: new 2 - old 0 (draws 0)
  deck 5/10: new 0 - old 2 (draws 0)
  deck 6/10: new 1 - old 1 (draws 0)
  deck 7/10: new 0 - old 2 (draws 0)
  deck 8/10: new 1 - old 1 (draws 0)
  deck 9/10: new 0 - old 2 (draws 0)
  deck 10/10: new 1 - old 1 (draws 0)
RESULT: new 7 - old 13 - draws 0 | candidate win rate 35.0% | FAIL
exit 1 (= candidate did not pass, agent/weights.npz preserved)
```
HANDOFF §5 が明記する通り、`5,763 states` 程度では candidate が baseline を超えないのは正常。**FAIL = システム健全**。

### 3-3. データセットの中身 (取得時点)
```
ds/value/   4 npz  (data_collected_*.npz; 計 5,763 states, dim=32)
ds/meta/    4 jsonl (episodes_*.jsonl)
ds/raw/    13 json  (生 replay; 行動クローン用に温存)
ds/weights/ 1 npz  (weights_candidate.npz — Hub 直近の候補)
```

---

## 4. 次の TODO

1. **データ蓄積を待つ** (Hub の cron 30分毎収集 + 4:00 UTC 日次 publish。`docs/HANDOFF.md` §3 参照)
2. 数万局面 (目安 ≥30,000 states) 溜まったら、上記 §3-1 の `--games 30` で **本番 verify** を回す
3. PASS したら `--promote` で `agent/weights.npz` を書き換え、`tools/make_submission.py` で zip 作成 → ブラウザで `kaggle.com/competitions/pokemon-tcg-ai-battle/submit` から手動提出 (1日5回制限のため `kaggle competitions submit` 自動化は不推奨)
4. (任意) Mac の `xattr` ステップを自動化したいなら `tools/hub/run.sh` 相当の Mac native ランチャ (`tools/mac/run_engine.sh` 的なもの) を書く
5. (任意) HANDOFF.md §4 を上記知見で改訂する PR を出す (Docker 必須記載を外し、`xattr` ステップを追加、`kaggle<1.7` の理由を明記)

---

## 5. ハマりどころ (今回の学び・再発防止)

- **WebFetch のコンペスラグ推測を信用しない**: 似た名前 `pokemon-tcg-ai-battle-challenge-simulation` を返されて 404 ループ。リポジトリ内の `README.md` / `src/collector/config.py` の `COLLECTOR_COMPETITION` を grep するのが正しい一次情報。
- **Docker `-v "$PWD/pokemon-tcg-ai-battle"` 事故**: cwd が既に repo ルートだったため、ホストに空の `pokemon-tcg-ai-battle/pokemon-tcg-ai-battle/` ディレクトリが auto-create された。**Docker `-v` のホスト側パスは絶対値で書く**こと。
- **dlopen 失敗時のファイル消失 (謎)**: 一度 `library load disallowed` で失敗した直後、当該 `cg/libcg.so` が `ls: No such file` 状態になった。原因未解明だが、再 cp + 事前 `xattr -d quarantine` で安定動作。**最初に必ず `xattr` 除去**してからロードを試すこと。
- **`uv pip install --reinstall` で kaggle 1.6.17 に固定**: 単に `pip install "kaggle<1.7"` だと既に入っている 2.x が残るケースあり。
- **HANDOFF.md は branch 専属**: main をクローンしただけでは `verify_candidate.py` / `pyproject.toml` が無い。HANDOFF を読む前に `git branch -a` で branch を確認。

---

## 6. クイック診断

```bash
# engine が生きてる？
python3 -c "from cg.sim import lib; print('engine OK')"

# 認証通る？
.venv/bin/kaggle datasets list --mine

# データセット最新？
.venv/bin/kaggle datasets files ichitaro3/ptcg-ladder-replays

# baseline 触られてない？ (md5 不変なら未 promote)
md5 agent/weights.npz   # 期待値: ba2c1e2b797841676cae9a3c58ed10f1 (2026-06-27時点)

# tests (collector 単体, mock のみ・engine不要)
.venv/bin/pytest tests/collector -q
```

---

## 7. 関連ドキュメント
- `docs/HANDOFF.md` — 全体運用 (Hub + engine機 2段構成、cron、認証、Dataset slug 等)
- `docs/HUB.md` — Hub (aarch64) デプロイ + cron 手順
- `docs/DEPLOY.md` / `docs/UAT.md` / `docs/ONBOARDING.md` — 配備 / 受入 / 共有
- 本書 = HANDOFF.md §4 (engine機) の **Mac native 版・実測詳細**。HANDOFF §4 と本書が矛盾する箇所は **本書を優先**。
