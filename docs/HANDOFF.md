# 引き継ぎ書 — PTCG リプレイ収集システム

Kaggle「Pokémon TCG AI Battle」用の、**実ラダー対戦リプレイを24h自動収集→学習データ化→
日次で候補モデル再学習**する本番システム。これを読めば運用・継続・提出までできる。

> 対象読者は2種類：**(A) 運用を引き継ぐ人**（§0–§6）と **(B) コードを触る新規開発者**（§10）。
> まず §0 → 該当章へ。

---

## 0. TL;DR（現状）
- ✅ **収集は本番稼働中**。SwitchBot AI Hub 上で OpenClaw cron が30分毎に収集、4:00 UTC に日次再学習。
- ✅ 実データ **51エピソード / 5,642学習局面 / 生リプレイ51件** を取得済み。
- ✅ データ＋候補weights＋**manifest** を **プライベート Kaggle Dataset** に自動公開（デバイス全損でも復元可）。
- 🔜 残：①cronのdelivery=none化（掃除）②データ蓄積待ち ③検証/提出（=MacBookをengine機に）④外形監視の常設。

---

## 1. 全体アーキテクチャ（2段構成）
```
[Hub: aarch64 / numpyのみ / エンジン無し]
   収集(--once) → 変換 → 再学習(train_value_np) → 候補weights公開
                                   │  ← ここまで全自動（OpenClaw cron）
                                   ▼
                        Kaggle Dataset (正準保管・共有・manifest同梱)
                                   │
[engine機: エンジンが載る実機 = Apple Silicon Mac(native) など]
   Dataset取得 → verify_candidate(自己対戦A/B勝率ゲート) → 昇格 → make_submission → submit
                                                                       ↑ 最終GOは人推奨
```
**なぜ2段か**：エンジンは **Pokemon配布のネイティブ共有ライブラリ**（再配布不可）。Hubの
**収集側**は aarch64 + numpyのみで回るが、エンジンを使う**検証・提出側**は実機が要る。
SDK (`sample_submission/cg/`) には `libcg.so`(Linux x86_64) / `libcg-arm64.so`(Linux arm64) /
`libcg.dylib`(macOS arm64) が同梱で、**MacBook(Apple Silicon)なら native で動く**（§4・実証済み）。
→「収集はARM常駐 / 検証・提出はエンジンが載る別機」で分離。

---

## 2. どこに何があるか
| もの | 場所 |
|---|---|
| コード | GitHub `Martensiter/pokemon-tcg-ai-battle`（branch `main`、**public**） |
| データ正準保管 | Kaggle **private** dataset `ichitaro3/ptcg-ladder-replays` |
| Hub上の本拠地（永続） | `/home/node/.openclaw/extensions/ptcg-collector/` |
| └ repo / venv | `…/ptcg-collector/repo`（`.venv` 込み・永続） |
| └ データ | `…/ptcg-collector/data/{value,meta,raw,weights,state}` |
| └ 状態(manifest/log) | `…/ptcg-collector/state/`（manifest.jsonl・status.json・各種log） |
| └ 認証/設定(.env) | `…/ptcg-collector/.env`（repo外＝再cloneでも残る） |
| cron定義(永続) | `~/.openclaw/cron/jobs.json` |

> Hubのコンテナは揮発するが、`/home/node/.openclaw` は永続マウント。リセットされても
> データ・manifest・venv・cron全部残り、cronが自動で収集を再開する。

### 2a. リポジトリに入っていない外部素材（新規参加者の必読）
クローンしただけでは**揃わない**もの。入手元と置き場所：

| 素材 | 何 | 入手元 | 置き場所 | 無いと困る所 |
|---|---|---|---|---|
| `cg/libcg.so`（+ `cg/cg.dll`） | エンジン本体 | コンペpage `sample_submission/cg/`（`libcg.so`/`libcg-arm64.so`/`libcg.dylib` 同梱）| repoの `cg/`（Macは `libcg.dylib`→`cg/libcg.so` にコピー＋`xattr -d com.apple.quarantine`）| engine機の検証・対戦・提出すべて |
| `EN_Card_Data.csv` / `JP_Card_Data.csv` | カードデータ | コンペpage | repo root | デッキ構築・分析ツール |
| `deck_sample.csv` | 入力サンプルデッキ | リポジトリ同梱の任意デッキで代用 → `cp deck.csv deck_sample.csv` | repo root | `tests/smoke_test.py`・`tests/search_test.py`・`selfplay/gen_data.py` |
| Kaggle 認証（`KAGGLE_USERNAME`/`KAGGLE_KEY`） | API鍵 | Kaggle → Account → Create New Token | env または `.env` | 収集・Dataset公開・提出 |
| Dataset アクセス | `ichitaro3/ptcg-ladder-replays` | オーナーが collaborator 追加 | — | 候補weights/データ取得 |

いずれも **.gitignore 済み＝コミット厳禁**（鍵・生データ・エンジン・カードCSV・`deck_sample.csv`）。

---

## 3. Hub運用（収集側）
### 動いてるか確認
```bash
R=/home/node/.openclaw/extensions/ptcg-collector
tail -n 15 "$R/state/collector.log"          # 新しい pass_complete があれば稼働中
wc -l "$R/state/manifest.jsonl"; ls "$R/data/value/"
cat ~/.openclaw/cron/jobs.json | python3 -m json.tool | grep -iE '"name"|every|run.sh'
```
### cronジョブ（OpenClaw Control UI = 192.168.1.108:18789 の「Cron ジョブ」）
- `ptcg-collect` … every 30m → `run.sh -m collector --once --rps 0.5`
- `ptcg-daily` … `0 4 * * *` UTC → `run.sh tools/daily_pipeline.py --publish`
- どちらも isolated / tools=exec(elevated)。**結果配信はDiscord宛てでエラーになるので delivery=none に変更すること（未対応・要掃除）**。

### リセット後の復帰
基本「何もしなくていい」。cronが永続→自動で `run.sh` 実行→`run.sh` が永続venvで動く。
万一 venv が壊れたら `run.sh` が自動で再作成（uvを一時導入）。
**デバイス全損**（永続マウントごと消えた）時だけ §4a の復元手順。

### 認証
`$R/.env` に `KAGGLE_USERNAME` / `KAGGLE_KEY`（＋ `DATASET_SLUG`, `COLLECTOR_SINK=local`,
`COLLECTOR_KEEP_RAW=true`, `COLLECTOR_CHUNK_SIZE=20`, `COLLECTOR_DATA_DIR/STATE_DIR`）。
`run.sh` がこれを読んで venv/bin を PATH に通す（kaggle CLIサブプロセス用）。

### 3a. 監視・アラート（外形監視 ＝ 放置でも壊れたら気づく仕組み）
収集は無人で回るので「**いつの間にか死んでいた**」を防ぐ監視が要る。判定材料は
`state/status.json` の **`ts`（最終成功時刻=unix秒）** と **`age_seconds`**（`/status` API側で算出）。

```bash
# 最終成功からの経過秒（cronは30分毎なので、5400秒=90分を超えたら異常）
R=/home/node/.openclaw/extensions/ptcg-collector
now=$(date +%s); ts=$(python3 -c "import json;print(json.load(open('$R/state/status.json'))['ts'])")
age=$(( now - ts )); echo "last success ${age}s ago"; [ "$age" -gt 5400 ] && echo "ALERT: collector stale"
# --serve を立てている場合は age_seconds が直接返る
curl -s localhost:8765/status | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('age_seconds'))"
```

**健全性ルール（しきい値）**
- `status.json` の `ts` が **90分以上**更新されない → 収集が止まっている（cron停止/venv破損/鍵切れ）。
- `state/collector.log` に `pass_complete` が出続けているか。`publish_failed` / `replay_failed` が連発していないか。

**主な故障モードと兆候**
- **Kaggle鍵失効/失効間近**：`401/403` → log に `*_failed`。`ts` が更新されなくなる。→ `.env` の `KAGGLE_KEY` を再発行・差し替え（鍵運用はオーナーが別途管理）。
- **OpenClaw 依存**：cron はOpenClawの常駐とそのLLMモデル設定に依存。OpenClawが落ちる/モデル未設定だとcronが発火しない。Control UI でジョブが enabled か・直近実行時刻を確認。
- **配信エラー**：cron結果のDiscord配信が失敗し続けてもログが汚れるだけで収集自体は無影響。**ただしこれを「アラート」と混同しないこと**。配信=none にした上で、上の `ts` ベースの外形監視を別途用意するのが正解（例：engine機やGitHub Actions schedule等から `kaggle datasets files` の更新時刻を見る、`/status` をcurlする等）。

> アラートの常設は未実装（残TODO §6-4）。最小構成は「engine機の cron が毎時 `kaggle datasets files ichitaro3/ptcg-ladder-replays` を見て、更新が止まってたら通知」。

---

## 4. engine機（検証・提出側）= MacBook
> **実機検証済み（2026-06-27, Apple Silicon Mac）。詳細な実測記録・ハマりどころは
> [`docs/HANDOFF_MACBOOK.md`](HANDOFF_MACBOOK.md) を必ず参照（矛盾時はそちらが正）。**

**訂正：macOS は Docker 不要・native で動く。** SDK の `sample_submission/cg/` には
`libcg.dylib`(macOS arm64) が同梱で、`cg/sim.py` の `dlopen` は拡張子非依存。
`libcg.dylib` を `cg/libcg.so` にコピーすれば Apple Silicon で**ネイティブ実行**できる（Dockerエミュより遥かに速い）。

```bash
# 1) 準備（ブランチ必須：main には verify_candidate.py / pyproject.toml が無い）
git clone https://github.com/Martensiter/pokemon-tcg-ai-battle.git && cd pokemon-tcg-ai-battle
git checkout claude/replay-collector-production-7ylifb
cp /path/to/sample_submission/cg/libcg.dylib cg/libcg.so   # Mac: dylib を .so 名で置く
xattr -d com.apple.quarantine cg/libcg.so                  # ★忘れると dlopen 拒否
uv venv .venv -p 3.11 && uv pip install -e ".[kaggle]"
uv pip install "kaggle<1.7" --reinstall                    # ★datasets API は legacy auth 必須（2.x は403）
# 2) エンジンが載るか（決定的テスト）
.venv/bin/python -c "from cg.sim import lib; print('engine OK (native)')"
# 3) 検証（データ蓄積後に意味が出る。1試合≈76s × 10デッキ × --games）
export KAGGLE_USERNAME=... KAGGLE_KEY=...                   # Hubの .env と同じ実キー
.venv/bin/kaggle datasets download -d ichitaro3/ptcg-ladder-replays -p ds --unzip
.venv/bin/python tools/merge_collected.py --src ds/value --out selfplay/data_all.npz
.venv/bin/python selfplay/train_value_np.py --data selfplay/data_all.npz --out /tmp/cand.npz
.venv/bin/python tools/verify_candidate.py --new /tmp/cand.npz --games 30 --threshold 0.53 --promote
# 通れば §5a の提出へ（最終GOは人推奨：1日5回制限）
```
- **非Mac（x86_64 Linux等）でエンジンが `.so` しか無い場合**のフォールバックは Docker：
  `docker run --rm --platform linux/amd64 -v "$PWD":/app -w /app python:3.11-slim …`（Apple Siliconでは遅いエミュ）。
- **鍵ズレ**：`datasets.get denied / 403` が出たら、Hub の `.env` の `KAGGLE_KEY` を移すのが第一手
  （Generate New Token で旧キーは失効するため Hub と Mac で同じキーに揃える）。

### 4a. ディザスタリカバリ（デバイス全損 → 重複なしで復元）
manifest（処理済みエピソードID台帳）は idempotency と再開の**唯一の根拠**。日次pipelineが
`data/state/manifest.jsonl` として **Dataset に同梱**するので、Hubの永続マウントごと消えても
Dataset から戻せる。戻さず再収集すると**同じ局面が二重に**データへ入る（学習が歪む）。

```bash
# 新しいHub/別マシンで、収集を再開する前に：
R=/home/node/.openclaw/extensions/ptcg-collector
kaggle datasets download -d ichitaro3/ptcg-ladder-replays -p /tmp/ds --unzip
mkdir -p "$R/state" "$R/data"
cp /tmp/ds/state/manifest.jsonl "$R/state/manifest.jsonl"   # ← これが肝
cp -r /tmp/ds/value "$R/data/"                              # 既存チャンクも戻す（任意）
# 以後 cron/run.sh が回ると、manifest にある episode は skip され重複しない
```
> 仮に manifest を戻し損ねても、merge側に**チャンク単位の内容重複除去**があるので
> 「バイト同一のチャンクが2回入る」事故は吸収できる（行単位ではなく**チャンク単位**。
> 同一局面の正当な再出現＝ラベル頻度の信号は保持する）。ただし第一防衛線は manifest 復元。

---

## 5. 改善ループ（本来の目的）
```
収集(Hub,自動) → 再学習候補(Hub,自動) → 検証(engine機) → 昇格 → 提出(人) → 強いAgentがラダーで戦う
```
- **Dataset = 素材＋候補モデルの倉庫**（毎日更新）。**Agent = 提出して順位がつく本体**。
- 検証は `verify_candidate.py`（新vs現行weightsを自己対戦、勝率≥閾値で `--promote`）。
- データが少ない今は verify は「改善なし＝FAIL(昇格せず)」が正常。数万局面貯まってから効く。

### 5a. 提出手順（候補weights → 実際の submit）
**前提**：検証(§4)が PASS して `--promote` 済み＝`agent/weights.npz` が新しい候補に置き換わっている。
提出は engine機（=libcg.so のある x86_64 Docker）で行う。

```bash
# ① 候補weightsを現行に昇格（§4の verify が PASS したら）
python tools/verify_candidate.py --new /tmp/cand.npz --promote
#    → 内部で  cp /tmp/cand.npz agent/weights.npz  相当（PASS時のみ）

# ② 提出物をパッケージ（agent/ 一式 = weights.npz 込み + 選んだデッキ を同梱）
python tools/make_submission.py --deck deck_cand_hops_hybrid_v2.csv --name hops_hybrid_v2
#    → submission_hops_hybrid_v2.tar.gz と .zip を生成（deck は中で deck.csv にリネームされる）

# ③ 健全性チェック（任意・推奨）
python tests/submission_test.py submission_hops_hybrid_v2

# ④ 提出（Simulation Category）。-c はコンペslug、-f は ② の tar.gz
kaggle competitions submit -c pokemon-tcg-ai-battle \
  -f submission_hops_hybrid_v2.tar.gz \
  -m "hops_hybrid_v2 + retrained value-net $(date +%F)"
```
> **認証の注意（実測）**：`kaggle 2.x` では `competitions submit` が **OAuth `access_token` 必須**で、
> legacy `kaggle.json`（username+key）は 401/403 で弾かれる。提出時は `kaggle auth login`（OAuth）か、
> ブラウザで `kaggle.com/competitions/pokemon-tcg-ai-battle/submit` から `submission_*.tar.gz` を**手動アップロード**。
> （`datasets download` 側は逆に `kaggle<1.7` の legacy auth で通る — §4 参照。）

- **どのデッキ？** 既定の本命は `deck_cand_hops_hybrid_v2.csv`（README「Final primary deck」/ 現提出 `submission_hops_hybrid_v2`）。
  `make_submission.py` の `--deck` 既定は `deck.csv`。どのデッキを選んでも、バンドル内では必ず `deck.csv` として載る。
- **回数制限**：1日 **5回**（Simulation）。**しかも Team では5回はチーム共有**（§7参照）。だから自動提出はせず**人がGO**を出す運用。
- **締切**：本コンペは現在 **オープン**。正確な締切日時は Kaggle コンペページの "Timeline" を参照（`<締切日: 要記入>`）。締切超過後は submit が弾かれる。
- **Strategy Category** は別物（コード提出ではなくテキスト = `WRITEUP.md`）。混同しない。

---

## 6. 残TODO（優先順）
1. **cron delivery=none**（Control UIで両ジョブ編集）— エラーログ止め。収集自体は無影響。
2. **データ蓄積を待つ**（放置でOK、新着エピソードが増える）。
3. **MacBookをengine機化**（§4：Docker amd64 + libcg.so 配置 → `engine OK` 確認）。
4. **外形監視の常設**（§3a）— `ts` ベースの staleness 通知を Hub外（engine機 or GitHub Actions schedule）に。
5. データが貯まったら **初の実検証→（良ければ）初提出**（§5a）。
6. 任意：OpenClaw に collector制御APIのツール登録（`--serve` 済み・`docs/DEPLOY.md`）。
7. 任意：収集メタからデッキ/メタ勝率を出す分析ツール（未実装）。
8. 任意：行動クローン用に生replayから (state,action) 抽出（生は keep_raw で保存済み＝将来可能）。

---

## 7. Kaggleアカウント / Team での提出（共有運用）
このアカウントは **Team** 設定。引き継ぎ・共同運用の要点：
- **提出は「1人だけ」ではない**。チームの各メンバーが**自分のKaggleアカウント**で `kaggle competitions submit`
  すれば、その提出は**チームの共有エントリ**としてカウントされる（＝誰が出してもよい）。
- ただし **1日5回の提出枠はチーム共有**。複数人が出すと枠を食い合うので、提出は調整（§5aで人がGO）。
- **友人に渡す**には：①Kaggleコンペで友人を Team に招待 → 友人は自分のアカウントで上記コマンドを実行。
  ②データ取得が要るなら **Dataset の collaborator** にも別途追加（Teamメンバー≠Dataset共有は自動連動しない）。
- 鍵（`KAGGLE_KEY`）は**個人ごと**。共有せず各自が自分のトークンを使う。

---

## 8. ハマりどころ（学習済み・再発防止）
- **Hubコンテナ揮発**：永続は `/home/node/.openclaw` のみ。ここ以外（/app, ~/.local, ~/.kaggle）は消える。
- **OpenClaw承認壁**：CLIは `operator.read` のみ→cron操作は Control UI(=フル権限) かチャットから。初回は **BOOTSTRAP**（ペルソナ設定）を1回答えないとエージェントが動かない。
- **エンジンは x86_64 Linux**：ARM(Hub)/macOS では不可。Docker amd64 で。
- **episode_id は int 必須**／**replayはファイルDL**（stdoutでない）／**run.shはvenv/binをPATHに**（kaggle CLI）／**.envのKEYは実値**（日本語プレースホルダで latin-1 エラー）。
- 実replayの形は **Kaggle env形式**：盤面は `steps[i][seat].observation.current/select`（`visualize`配列ではない）。
- **manifestを戻す前に再収集しない**（§4a）。戻し損ね時の保険はチャンク内容dedupのみ。

---

## 9. 主要ファイル早見
| ファイル | 役割 |
|---|---|
| `src/collector/` | 収集本体（client/parse/convert/sink/manifest/collector/server） |
| `tools/hub/run.sh` | Hub用ランチャ（永続venv直叩き＋.env読み込み＋PATH） |
| `docs/HUB.md` | Hubデプロイ＆cron手順 |
| `tools/merge_collected.py` | 収集npz→学習用1ファイル（チャンク内容dedup付き） |
| `selfplay/train_value_np.py` | torch無しnumpy学習（weights.npz出力） |
| `tools/daily_pipeline.py` | 日次 merge→再学習→manifest同梱→Dataset公開 |
| `tools/verify_candidate.py` | 自己対戦A/B勝率ゲート（engine機・`--promote`で昇格） |
| `tools/make_submission.py` | 提出バンドル生成（agent/+deck→`submission_<name>.tar.gz`） |
| `src/collector/server.py` | `--serve` 制御API（OpenClaw用 status/collect・`age_seconds`） |
| `docs/HANDOFF_MACBOOK.md` | engine機（Mac native）の実測詳細・**§4矛盾時はこちらが正** |
| `docs/DEPLOY.md` / `docs/UAT.md` / `docs/ONBOARDING.md` | 配備/受入/共有 |

テスト：`uv run --extra dev pytest tests/collector`（91件、ネット/エンジン/torch不要）。CIあり（§10）。

---

## 10. 新規開発者クイックスタート（コードを触る人向け）
### クローンして動かす（収集側だけなら鍵・エンジン不要）
```bash
git clone https://github.com/Martensiter/pokemon-tcg-ai-battle.git
cd pokemon-tcg-ai-battle
uv venv && uv pip install -e ".[dev,kaggle]"     # collector: uv + Python3.11+ + numpyのみ
uv run python -m collector --self-test           # 鍵もネットもエンジンも不要のUAT
uv run --extra dev pytest tests/collector        # 91件 mock-only（CIと同じ）
```

### ⚠ 2つのツールチェーンが同居している（混同しない）
| 領域 | ランタイム | 入れ方 | 依存 |
|---|---|---|---|
| **collector**（`src/collector/`, `tools/*pipeline*`, `merge_collected`, `train_value_np`） | uv / **Python 3.11+** | `uv pip install -e ".[dev,kaggle]"` | **numpyのみ**（torch・エンジン無し）。ARM Hubで常駐 |
| **agent / 学習 / 検証 / 提出**（`agent/`, `selfplay/`, `tools/make_submission`, `tools/verify_candidate`, `cg/`） | pip / **Python 3.12**（READMEのテスト基準） | `pip install numpy torch` | **torch + エンジン `libcg.so`**。x86_64必須 |

> 迷ったら：「numpyだけで動く＝collector側」「torch か engine が要る＝agent側」。
> collector は agent/value_net を**壊さない**前提（features 32次元・weights.npz 形式は不変）。

### ブランチ / PR 規約
- 直接 `main` に push しない。作業は **feature ブランチ**（例 `claude/<topic>`）。
- 1論点1PR、説明と「どう検証したか」を書く。**`uv run --extra dev pytest tests/collector` が緑**になってから出す。
- **コミット厳禁**：鍵(`.env`/`kaggle.json`)・生replay・`cg/*.so|*.dll`・カードCSV・`deck_sample.csv`・`collector_data/`・`*.npz`系（.gitignore済み）。
- MIT ライセンスと著作権表記は保持。

### CI が見ている範囲（重要な落とし穴）
`.github/workflows/ci.yml` は **collector だけ**を対象にする：
- トリガpaths＝`src/collector/**`, `tests/collector/**`, `pyproject.toml`, `.github/workflows/ci.yml`。
- 実行は `pytest tests/collector` ＋ `collector --self-test`（Python 3.11/3.12 matrix）。
- **`agent/` や `selfplay/` や engine系テスト（`tests/smoke_test.py` 等）は CI 対象外**。
  これらは `libcg.so` が要る＝CIに置けないため。agent側を変えたら**手元（engine機）で**テストすること。

---

## 11. クイック診断（困ったら）
```bash
# Hub: 収集生きてる？（最終成功からの経過）
R=/home/node/.openclaw/extensions/ptcg-collector
tail -n 5 "$R/state/collector.log"
python3 -c "import json,time;d=json.load(open('$R/state/status.json'));print('age', int(time.time())-d['ts'],'s')"
# Dataset 無事？（更新時刻＝外形監視の代替指標）
kaggle datasets files ichitaro3/ptcg-ladder-replays
# テスト全green？（任意マシン）
uv run --extra dev pytest tests/collector -q
```
