# 引き継ぎ書 — PTCG リプレイ収集システム

Kaggle「Pokémon TCG AI Battle」用の、**実ラダー対戦リプレイを24h自動収集→学習データ化→
日次で候補モデル再学習**する本番システム。これを読めば運用・継続ができる。

---

## 0. TL;DR（現状）
- ✅ **収集は本番稼働中**。SwitchBot AI Hub 上で OpenClaw cron が30分毎に収集、4:00 UTC に日次再学習。
- ✅ 実データ **51エピソード / 5,642学習局面 / 生リプレイ51件** を取得済み。
- ✅ データは **プライベート Kaggle Dataset** に自動公開（消えても安全）。
- 🔜 残：①cronのdelivery=none化（掃除）②データ蓄積待ち ③検証/提出（=MacBookをengine機に）。

---

## 1. 全体アーキテクチャ（2段構成）
```
[Hub: aarch64 / numpyのみ / エンジン無し]
   収集(--once) → 変換 → 再学習(train_value_np) → 候補weights公開
                                   │  ← ここまで全自動（OpenClaw cron）
                                   ▼
                        Kaggle Dataset (正準保管・共有)
                                   │
[engine機: x86_64 / cg/libcg.so あり]   ← MacBookをDocker(amd64)で
   Dataset取得 → verify_candidate(自己対戦A/B勝率ゲート) → 昇格
                                   → (任意) make_submission → submit  ← 提出は人推奨
```
**なぜ2段か**：エンジン `cg/libcg.so` は **Linux x86_64 ネイティブ**（Pokemon配布・再配布不可）。
Hubは aarch64 なのでロード不可 → 「収集はARM常駐 / 検証・提出は x86_64 別機」で分離。

---

## 2. どこに何があるか
| もの | 場所 |
|---|---|
| コード | GitHub `Martensiter/pokemon-tcg-ai-battle`（branch `main`） |
| データ正準保管 | Kaggle **private** dataset `ichitaro3/ptcg-ladder-replays` |
| Hub上の本拠地（永続） | `/home/node/.openclaw/extensions/ptcg-collector/` |
| └ repo / venv | `…/ptcg-collector/repo`（`.venv` 込み・永続） |
| └ データ | `…/ptcg-collector/data/{value,meta,raw,weights}` |
| └ 状態(manifest/log) | `…/ptcg-collector/state/` |
| └ 認証/設定(.env) | `…/ptcg-collector/.env`（repo外＝再cloneでも残る） |
| cron定義(永続) | `~/.openclaw/cron/jobs.json` |

> Hubのコンテナは揮発するが、`/home/node/.openclaw` は永続マウント。リセットされても
> データ・manifest・venv・cron全部残り、cronが自動で収集を再開する。

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

### 認証
`$R/.env` に `KAGGLE_USERNAME` / `KAGGLE_KEY`（＋ `DATASET_SLUG`, `COLLECTOR_SINK=local`,
`COLLECTOR_KEEP_RAW=true`, `COLLECTOR_CHUNK_SIZE=20`, `COLLECTOR_DATA_DIR/STATE_DIR`）。
`run.sh` がこれを読んで venv/bin を PATH に通す（kaggle CLIサブプロセス用）。

---

## 4. engine機（検証・提出側）= MacBook
macOSは直接エンジンを動かせない（.dylib無し）→ **x86_64 Linux Docker** で動かす。
```bash
# 1) 準備
git clone https://github.com/Martensiter/pokemon-tcg-ai-battle.git && cd pokemon-tcg-ai-battle
cp /path/to/libcg.so cg/libcg.so      # Kaggle sample_submission/cg/ の Linux .so
# 2) エンジンが載るか（決定的テスト）
docker run --rm --platform linux/amd64 -v "$PWD":/app -w /app python:3.11-slim \
  bash -lc 'pip install -q numpy && python -c "from cg.sim import lib; print(\"engine OK\")"'
# 3) 検証（データ蓄積後に意味が出る）
docker run --rm --platform linux/amd64 -e KAGGLE_USERNAME -e KAGGLE_KEY \
  -v "$PWD":/app -w /app python:3.11-slim bash -lc '
    pip install -q -e ".[kaggle]"
    kaggle datasets download -d ichitaro3/ptcg-ladder-replays -p ds --unzip
    python tools/merge_collected.py --src ds/value --out selfplay/data_all.npz
    python selfplay/train_value_np.py --data selfplay/data_all.npz --out /tmp/cand.npz
    python tools/verify_candidate.py --new /tmp/cand.npz --games 30 --threshold 0.53 --promote'
# 通れば: make_submission → kaggle competitions submit（最終GOは人推奨：1日5回制限）
```
Apple Siliconは `--platform linux/amd64`（エミュ）必須・遅い。Intelなら省略可。

---

## 5. 改善ループ（本来の目的）
```
収集(Hub,自動) → 再学習候補(Hub,自動) → 検証(engine機) → 昇格 → 提出(人) → 強いAgentがラダーで戦う
```
- **Dataset = 素材＋候補モデルの倉庫**（毎日更新）。**Agent = 提出して順位がつく本体**。
- 検証は `verify_candidate.py`（新vs現行weightsを自己対戦、勝率≥閾値で `--promote`）。
- データが少ない今は verify は「改善なし＝FAIL(昇格せず)」が正常。数万局面貯まってから効く。

---

## 6. 残TODO（優先順）
1. **cron delivery=none**（Control UIで両ジョブ編集）— エラーログ止め。収集自体は無影響。
2. **データ蓄積を待つ**（放置でOK、新着エピソードが増える）。
3. **MacBookをengine機化**（§4：Docker amd64 + libcg.so 配置 → `engine OK` 確認）。
4. データが貯まったら **初の実検証→（良ければ）初提出**。
5. 任意：OpenClaw に collector制御APIのツール登録（`--serve` 済み・`docs/DEPLOY.md`）。
6. 任意：収集メタからデッキ/メタ勝率を出す分析ツール（gap #2・未実装）。
7. 任意：行動クローン用に生replayから (state,action) 抽出（生は keep_raw で保存済み＝将来可能）。

---

## 7. ハマりどころ（学習済み・再発防止）
- **Hubコンテナ揮発**：永続は `/home/node/.openclaw` のみ。ここ以外（/app, ~/.local, ~/.kaggle）は消える。
- **OpenClaw承認壁**：CLIは `operator.read` のみ→cron操作は Control UI(=フル権限) かチャットから。初回は **BOOTSTRAP**（ペルソナ設定）を1回答えないとエージェントが動かない。
- **エンジンは x86_64 Linux**：ARM(Hub)/macOS では不可。Docker amd64 で。
- **episode_id は int 必須**／**replayはファイルDL**（stdoutでない）／**run.shはvenv/binをPATHに**（kaggle CLI）／**.envのKEYは実値**（日本語プレースホルダで latin-1 エラー）。
- 実replayの形は **Kaggle env形式**：盤面は `steps[i][seat].observation.current/select`（`visualize`配列ではない）。

---

## 8. 主要ファイル早見
| ファイル | 役割 |
|---|---|
| `src/collector/` | 収集本体（client/parse/convert/sink/manifest/collector/server） |
| `tools/hub/run.sh` | Hub用ランチャ（永続venv直叩き＋.env読み込み＋PATH） |
| `docs/HUB.md` | Hubデプロイ＆cron手順 |
| `tools/merge_collected.py` | 収集npz→学習用1ファイル |
| `selfplay/train_value_np.py` | torch無しnumpy学習（weights.npz出力） |
| `tools/daily_pipeline.py` | 日次 merge→再学習→Dataset公開 |
| `tools/verify_candidate.py` | 自己対戦A/B勝率ゲート（engine機） |
| `src/collector/server.py` | `--serve` 制御API（OpenClaw用 status/collect） |
| `docs/DEPLOY.md` / `docs/UAT.md` / `docs/ONBOARDING.md` | 配備/受入/共有 |

テスト：`uv run pytest tests/collector`（84件、ネット/エンジン/torch不要）。CIあり。

---

## 9. クイック診断（困ったら）
```bash
# Hub: 収集生きてる？
tail -n 5 /home/node/.openclaw/extensions/ptcg-collector/state/collector.log
# Dataset 無事？
kaggle datasets files ichitaro3/ptcg-ladder-replays
# テスト全green？（任意マシン）
uv run pytest tests/collector -q
```
