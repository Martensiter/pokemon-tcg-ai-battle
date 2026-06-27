# 運用 早見（RUNBOOK）

これ1枚で「人が何をすればいいか」が分かる。詳細は各リンク先へ。
**普段は放置でOK**（収集・再学習・Dataset公開は Hub で全自動）。人の操作は実質3つだけ。

---

## 普段の確認（30秒）
- **Kaggle Dataset の "updated" 日付が毎日進んでいるか**を見るだけ＝外形監視。
  止まっていたら異常 → [HANDOFF §3a](HANDOFF.md#3a-監視アラート外形監視--放置でも壊れたら気づく仕組み) / [§11](HANDOFF.md)。
- スマホからは OpenClaw に「収集どう？」→ `/status` が `age_seconds` を返す。

---

## 人がやる操作は3つ

### ① ベースラインを1回提出（今すぐ・ボードに載せる）
Mac（engine機）で：
```bash
python tools/make_submission.py --deck deck_cand_hops_hybrid_v2.csv --name hops_hybrid_v2
# → ブラウザ https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/submit に
#   submission_hops_hybrid_v2.tar.gz をアップロード（CLIは kaggle auth login でOAuth）
```
詳細・事前チェック → **[docs/SUBMIT_BASELINE.md](SUBMIT_BASELINE.md)**

### ② Mac自動検証を仕込む（1回だけ）
Mac で：
```bash
cd ~/Downloads/AI/pokemon-tcg-ai-battle
git checkout claude/replay-collector-production-7ylifb && git pull
MIN_STATES=30000 ./tools/mac/verify_and_promote.sh        # 空振り確認（数秒でskip=配線OK）
( crontab -l 2>/dev/null; \
  echo '0 3 * * 0 cd ~/Downloads/AI/pokemon-tcg-ai-battle && caffeinate -i ./tools/mac/verify_and_promote.sh' \
) | crontab -                                             # 週1自動。寝てても確実に→launchd版はdoc参照
```
以後、データが3万局面を超えた週末に自動で検証→昇格（**提出はしない**）。
詳細・launchd版・調整 → **[docs/MAC_AUTOMATION.md](MAC_AUTOMATION.md)**

### ③ データが育ったら次を提出
②の自動検証が PASS して `agent/weights.npz` が昇格したら、①の手順で再パッケージして提出。
判断基準・回数制限(5/日・Team共有) → **[HANDOFF §5a](HANDOFF.md)**

---

## 友人に渡す
- **Kaggle**：コンペで Team 招待（各自が自分のアカウントで提出＝共有エントリ）＋ Dataset の collaborator 追加。→ [HANDOFF §7](HANDOFF.md)
- **コード**：`git clone` → branch `claude/replay-collector-production-7ylifb` → [HANDOFF §10](HANDOFF.md) のクイックスタート。
- **外部素材**（エンジン/カードCSV/鍵）は同梱されない → [HANDOFF §2a](HANDOFF.md) の表。

## 困ったら
| 症状 | 見る場所 |
|---|---|
| 収集が止まった？ | [HANDOFF §3a・§11](HANDOFF.md) |
| デバイス全損から復元 | [HANDOFF §4a](HANDOFF.md)（manifest復元＝重複なし） |
| Mac でエンジンが動かない | [HANDOFF §4](HANDOFF.md) / [HANDOFF_MACBOOK.md](HANDOFF_MACBOOK.md)（`xattr -d quarantine`） |
| 提出が 401/403 | `kaggle auth login`（submitはOAuth必須）→ [SUBMIT_BASELINE.md](SUBMIT_BASELINE.md) |

---

## いま自動化されているもの / いないもの
| 工程 | 状態 |
|---|---|
| 収集 → 変換 → 再学習 → Dataset公開 | ✅ 全自動（Hub） |
| 検証 → 昇格 | ✅ 自動化可（Mac・②で有効化／自己ゲート） |
| **提出** | ❌ 手動（OAuth必須でここだけ無人化不可） |

> 全体像・背景は [HANDOFF.md](HANDOFF.md)。この RUNBOOK はその操作だけを抜き出した早見。
