# One-time GitHub setup

Everything in this repo is already staged and ready. Run these commands from
`F:\Claude\pokemon-tcg-agent` in PowerShell.

## 1. Set your git identity (only needed once per machine)

Replace with your actual name and the email tied to your GitHub account:

```powershell
git config --global user.name "Henrique Takahiro Ito"
git config --global user.email "henriquetakahiroito@live.com"
```

## 2. Make the first commit

```powershell
git commit -m "Initial commit — MCTS + value net + episode-mined deck pivots"
```

## 3. Create the GitHub repository

Option A (easier): on github.com, click **New repository**, name it
`pokemon-tcg-ai-battle`, **public**, leave everything else blank, click Create.

Option B (CLI, if you have `gh` installed):

```powershell
gh repo create pokemon-tcg-ai-battle --public --source=. --remote=origin --push
```

## 4. Push (only needed if you used Option A)

GitHub will show you the exact commands; they look like this:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/pokemon-tcg-ai-battle.git
git branch -M main
git push -u origin main
```

## 5. Verify before sharing in the writeup

Open `https://github.com/YOUR_USERNAME/pokemon-tcg-ai-battle` in your browser and confirm:

- ✅ `README.md` and `WRITEUP.md` are visible
- ✅ `LICENSE` shows MIT
- ✅ `agent/`, `selfplay/`, `tools/`, `tests/` directories are present
- ✅ **`EN_Card_Data.csv` is NOT present** (Pokemon's data — must stay private)
- ✅ **`cg/cg.dll` and `cg/libcg.so` are NOT present** (engine binaries — Pokemon's)
- ✅ **`episodes/` is NOT present** (replay data — competition rules)

## 6. Update WRITEUP.md with your repo URL

In `WRITEUP.md`, replace the placeholder line:

```
https://github.com/[username]/pokemon-tcg-ai-battle
```

with your real URL, e.g.:

```
https://github.com/htakahiroito/pokemon-tcg-ai-battle
```

Commit and push the update:

```powershell
git add WRITEUP.md
git commit -m "Update repo URL in writeup"
git push
```

## Subsequent updates (as the project evolves)

Whenever you want to push changes:

```powershell
git add .
git status              # confirm what's about to be committed
git commit -m "short description of what changed"
git push
```
