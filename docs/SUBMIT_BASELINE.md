# ベースライン提出 手順（今すぐボードに載せる）

現行ベースライン（既存の `agent/weights.npz` ＋ `deck_cand_hops_hybrid_v2.csv`）を
**1回だけ提出**して、Kaggle のリーダーボードに結果を出すための最短手順。
**エンジンも強さ検証も不要** — パッケージしてアップロードするだけ。

> なぜ今これをやるか：まだ誰も提出しておらず「在席」する価値が高い。実績 ~948 の既知bundleで低リスク。
> 再学習した候補は **まだ baseline を超えない**（データ少量で `verify_candidate` が FAIL）ので、今は出さない。
> 詳細な提出方針は [`HANDOFF.md`](HANDOFF.md) §5a。

実行機：**MacBook**（engine機セットアップ済み）を想定。実際にはエンジン不要なので、
`agent/weights.npz` が入った repo クローンがあるマシンならどこでも可。

---

## 0) 事前チェック
```bash
cd /Users/ichitaro/Downloads/AI/pokemon-tcg-ai-battle
git checkout claude/replay-collector-production-7ylifb && git pull
md5 agent/weights.npz   # 期待: ba2c1e2b797841676cae9a3c58ed10f1（baseline＝昇格で書き換わってない確認）
```
- Kaggle で **コンペの Rules を Accept 済み**であること（未acceptだと提出が弾かれる）。
- コンペ slug は **`pokemon-tcg-ai-battle`**（長い方 `…-challenge-simulation` は別物・404）。

## 1) 提出物をパッケージ
> **★最重要★ エンジン本体 `cg/libcg.so` を bundle に必ず同梱する。**
> Kaggleの採点機はエンジンを用意してくれない。エージェントは起動時に `cg/libcg.so` を
> `LoadLibrary` するので、無いと **`cannot open shared object file` で即クラッシュ＝Validation Episode failed**。
> しかも採点機は **Linux x86_64** なので、Mac native 用に置いた `libcg.dylib`（＝Mach-O）を
> そのまま固めると **arch違いでまた落ちる**。必ず **`sample_submission/cg/libcg.so`（Linux x86_64）** を渡す。

```bash
# sample_submission.zip の Linux x86_64 libcg.so を --engine-so で明示（dylibではない！）
python tools/make_submission.py --deck deck_cand_hops_hybrid_v2.csv --name hops_hybrid_v2 \
  --engine-so /path/to/sample_submission/cg/libcg.so
# zip を展開済みでなければ: unzip -o ~/Downloads/sample_submission.zip -d /tmp/ss
#   --engine-so /tmp/ss/sample_submission/cg/libcg.so
```
- ログに **`weights=yes engine=elf-x86_64`** が出れば正解。
  - `engine=missing` → ビルドが**中断**される（同梱忘れを防ぐ安全装置）。`--engine-so` を渡す。
  - `engine=mach-o-macos` / `elf-arm64` → **警告**。Kaggleで落ちる。Linux x86_64 の `.so` を渡し直す。
- 念のため中身確認：`tar tzf submission_hops_hybrid_v2.tar.gz | grep cg/libcg.so` が1行出ること。
- どのデッキを選んでも、bundle 内では必ず `deck.csv` という名前で載る（仕様）。

## 2)（任意・推奨）健全性チェック
```bash
python tests/submission_test.py submission_hops_hybrid_v2
```

## 3) 提出 ―― どちらか
**A. ブラウザ（最短・推奨）**
1. <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/submit> を開く
2. `submission_hops_hybrid_v2.tar.gz` をアップロード
3. メッセージ欄に `baseline hops_hybrid_v2` 等を入れて Submit

**B. CLI（OAuthを通す場合）**
```bash
kaggle auth login    # ブラウザでOAuth認証。※保存済み kaggle.json(key) では competitions submit は 401/403
kaggle competitions submit -c pokemon-tcg-ai-battle \
  -f submission_hops_hybrid_v2.tar.gz -m "baseline hops_hybrid_v2"
```

## 4) 確認
- 提出履歴：<https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/submissions>
- しばらくすると TrueSkill / 順位が付く ＝ **「Agent結果が Kaggle に載った」状態**。

---

## 注意
- **1日5回・Team 共有枠**。今回は1回で十分（複数人で出すと枠を食い合う）。
- これは現行ベースラインの提出。**自動提出はしない**（`competitions submit` は OAuth 必須で無人化不可＋回帰を自動で出さないため）。
- 次の提出は、データが数万局面貯まったあと [`HANDOFF.md`](HANDOFF.md) §4 の `verify_candidate` が PASS したものを出す（継続改善）。
