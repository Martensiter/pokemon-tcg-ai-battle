# GitHub setup — ⚠️ OBSOLETE / historical

> **このファイルは初期化用の歴史的メモです。もう実行しないでください。**
> リポジトリは既に存在します：**https://github.com/Martensiter/pokemon-tcg-ai-battle**
> （**public** な fork）。新しい public リポジトリを作る／初回 push する手順は不要です。
>
> - クローンして動かす：[README.md](README.md) の *Setup* / [docs/HANDOFF.md](docs/HANDOFF.md) §10。
> - ブランチ・PR 規約：[docs/HANDOFF.md](docs/HANDOFF.md) §10「ブランチ / PR 規約」。
> - 運用・提出：[docs/HANDOFF.md](docs/HANDOFF.md) §3–§5a。
>
> 下の旧手順（git identity・"create public repository"・初回 push）は、リポジトリが
> 既に公開済みの現状では**誤り**です。特に新規リポジトリ作成や `--public` での再作成は
> しないこと。

---

## まだ有効な唯一の中身：公開前の機密チェック
public リポジトリなので、push する前に**コミットしてはいけないもの**が紛れていないか必ず確認する
（すべて `.gitignore` 済みだが、人手で足す時の保険）：

```bash
git status            # 何が commit されようとしているか確認
```

リポジトリに**入っていてはいけない**もの：
- ❌ `EN_Card_Data.csv` / `JP_Card_Data.csv`（Pokemon配布のカードデータ）
- ❌ `cg/cg.dll` / `cg/libcg.so`（エンジンバイナリ — Pokemon配布・再配布不可）
- ❌ `episodes/` や生 replay、`collector_data/`、`*.npz` 系の大きなデータ
- ❌ `.env` / `kaggle.json`（`KAGGLE_KEY` などの認証情報）
- ❌ `deck_sample.csv`（入力サンプル）

リポジトリに**入っているべき**もの：✅ `README.md` / `WRITEUP.md` / `LICENSE`(MIT) /
`agent/` `selfplay/` `tools/` `src/collector/` `tests/`。

---

<details>
<summary>旧・初期化手順（実行不要・記録のみ）</summary>

当時 Windows のローカルから初回 push するための手順だった。リポジトリ公開済みの今は
すべて不要。識別情報・パス・"create repository" の記述は現状に合致しないため参照しないこと。
正しい手順は上記リンク（README / HANDOFF）を見る。

</details>
