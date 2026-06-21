# Collaborator onboarding

Two places grant a member access for UAT: the **GitHub repo** (code + CI) and the
**private Kaggle Dataset** (collected data, read from Kaggle notebooks). Add both.

## 1. GitHub repository collaborator

Requires admin on the repo and the [`gh`](https://cli.github.com/) CLI (or use the
web UI). `push` permission is enough for UAT (clone, branch, run, open PRs).

```bash
# helper script (this repo):
scripts/add_collaborators.sh GITHUB_USERNAME [GITHUB_USERNAME ...]
```

Or directly:

```bash
gh api -X PUT \
  /repos/Martensiter/pokemon-tcg-ai-battle/collaborators/GITHUB_USERNAME \
  -f permission=push
```

Web UI: **Settings → Collaborators → Add people**. The invitee accepts via email /
the repo banner. Verify with:

```bash
gh api /repos/Martensiter/pokemon-tcg-ai-battle/collaborators --jq '.[].login'
```

Permission levels: `pull` (read), `triage`, `push` (read/write — recommended for
UAT), `maintain`, `admin`.

## 2. Kaggle Dataset collaborator

Kaggle has **no public API** for managing dataset collaborators — it is done in the
web UI (and the collaborator needs a Kaggle account / their Kaggle username):

1. Open the dataset: `https://www.kaggle.com/datasets/<owner>/<dataset-slug>`
   (the `DATASET_SLUG` from your `.env`).
2. **Settings → Sharing / Collaborators → Add collaborator** → enter the member's
   Kaggle username → choose **Can view** (read for UAT) or **Can edit**.
3. They can now read it from a Kaggle notebook with zero egress:
   ```python
   import numpy as np, glob
   # add the dataset to the notebook, then:
   d = np.load(sorted(glob.glob('/kaggle/input/<dataset-slug>/data_collected_*.npz'))[-1])
   print(d['X'].shape, d['y'].shape)
   ```

If the dataset does not exist yet, create it once from the metadata template:

```bash
# set "id" in collector/dataset-metadata.json to <owner>/<slug> first
cp collector/dataset-metadata.json collector_data/dataset-metadata.json
kaggle datasets create -p collector_data --dir-mode zip
```

## 3. Point the member at UAT

Send them [docs/UAT.md](UAT.md). Steps 1–3 there need no credentials; step 4
needs their own `KAGGLE_USERNAME` / `KAGGLE_KEY`.
