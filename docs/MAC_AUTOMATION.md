# MacBook 自動検証（verify→promote）の配線

改善ループの **検証ステップ**だけ、engine機（MacBook native）で自動化する手順。
収集・再学習・Dataset公開は Hub で全自動。検証はエンジンが要るので Hub では不可 →
ここを MacBook の定期ジョブにする。**提出はしない**（OAuth必須＋回帰を自動で出さない。
提出は人手 = [`SUBMIT_BASELINE.md`](SUBMIT_BASELINE.md)）。

スクリプト本体：[`tools/mac/verify_and_promote.sh`](../tools/mac/verify_and_promote.sh)

## これが何をするか
1. 最新 Dataset を取得（data＋公開済み候補weights）
2. **自己ゲート**：収集局面数が `MIN_STATES`(既定30,000)未満なら数秒で skip（CPUを焼かない）
3. 十分なら `verify_candidate.py`（候補 vs 現行champ の自己対戦A/B）を実行
4. 勝率 ≥ `THRESHOLD`(既定0.53) なら **`--promote` で `agent/weights.npz` を更新**
5. 提出はしない（PASS時にログで人手提出を促すだけ）

> **なぜ自己ゲート付きか**：1回の本番verifyは `--games 30` で **約6時間**（1試合≈76s×10デッキ）。
> 今のデータ量（~5,600局面）では候補は必ず FAIL する。ゲートにより、データが育つまでは
> 毎回数秒で skip し、**閾値を超えた時だけ**重いverifyが自動で走る。だから今から仕込んで無害。

## 前提（一度きり）
[`HANDOFF.md`](HANDOFF.md) §4 のセットアップ済みであること：
- `cg/libcg.so`（＝`libcg.dylib` を配置）＋ `xattr -d com.apple.quarantine`
- `.venv`（`uv venv -p 3.11` ＋ `uv pip install -e ".[kaggle]"` ＋ `kaggle<1.7`）
- `.env` に `KAGGLE_USERNAME`/`KAGGLE_KEY`（Hubと同じ実キー）/ `DATASET_SLUG`
- `chmod +x tools/mac/verify_and_promote.sh`

## 手動で1回試す（推奨：まず空振りを確認）
```bash
cd ~/Downloads/AI/pokemon-tcg-ai-battle
MIN_STATES=30000 ./tools/mac/verify_and_promote.sh
tail -n 20 verify_cron.log     # "insufficient data ... -> skip" が出れば配線OK
```

## 定期実行の仕込み（どちらか）

### A. crontab（簡単）。週1・日曜3:00 例
```bash
( crontab -l 2>/dev/null; \
  echo '0 3 * * 0 cd ~/Downloads/AI/pokemon-tcg-ai-battle && caffeinate -i ./tools/mac/verify_and_promote.sh' \
) | crontab -
```
- `caffeinate -i` で実行中スリープを抑止（6時間走る場合に重要）。
- 欠点：実行時刻にMacが**スリープ/電源オフ**だと飛ぶ（次回まで走らない）。

### B. launchd（ノートPC向き・寝てても起床後に取り戻す）
`~/Library/LaunchAgents/com.ptcg.verify.plist` を作成：
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.ptcg.verify</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd ~/Downloads/AI/pokemon-tcg-ai-battle &amp;&amp; caffeinate -i ./tools/mac/verify_and_promote.sh</string>
  </array>
  <key>StartCalendarInterval</key><dict><key>Weekday</key><integer>0</integer><key>Hour</key><integer>3</integer></dict>
  <key>StandardOutPath</key><string>/tmp/ptcg-verify.out</string>
  <key>StandardErrorPath</key><string>/tmp/ptcg-verify.err</string>
</dict></plist>
```
```bash
launchctl load ~/Library/LaunchAgents/com.ptcg.verify.plist
launchctl list | grep ptcg          # 確認
# 外すとき: launchctl unload ~/Library/LaunchAgents/com.ptcg.verify.plist
```

## 調整（env で上書き）
| 変数 | 既定 | 意味 |
|---|---|---|
| `MIN_STATES` | 30000 | これ未満は skip |
| `GAMES` | 30 | デッキ毎ゲーム数（×10デッキ。30で約6h、8で約1.7h） |
| `THRESHOLD` | 0.53 | 昇格に必要な候補勝率 |

## 確認・停止
- ログ：`tail -f verify_cron.log`
- 昇格された？：`md5 agent/weights.npz`（baseline `ba2c1e2b797841676cae9a3c58ed10f1` から変われば昇格済み）
- **人が運用に入ったら自動化を止める**：crontab なら該当行削除／launchd なら `unload`。

> 自動化されるのは「検証→昇格」まで。**最終提出は引き続き人手**（OAuth）。
> 昇格後の提出は [`SUBMIT_BASELINE.md`](SUBMIT_BASELINE.md)（デッキを最新champで作り直して submit）。
