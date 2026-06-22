# Deploying the collector as a 24/7 daemon

There is **no web server** to stand up — the collector is a long-running CLI
process. You only need a machine that can keep it running. It is tiny
(pure-Python + numpy + the Kaggle CLI; no torch, no engine binary), so almost
anything works. Two supported paths: **Docker** (recommended, portable) and
**native uv** (Raspberry Pi / Termux).

Data and progress live in two volumes/dirs so a restart resumes where it left off
(the manifest skips already-fetched episode ids):

- `/data` (`collector_data/`) — npz chunks + per-episode metadata (the output)
- `/state` (`collector_state/`) — manifest + logs

## Option A — Docker (any host: NAS, mini-PC, Pi, …)

The image is multi-arch, so the same `Dockerfile` builds on aarch64 and x86_64.

```bash
git clone https://github.com/Martensiter/pokemon-tcg-ai-battle.git
cd pokemon-tcg-ai-battle
cp .env.example .env            # fill KAGGLE_USERNAME / KAGGLE_KEY (+ DATASET_SLUG for kaggle sink)

# smoke test the image with no credentials / network:
docker compose run --rm collector --self-test     # -> self_test_ok ... feature_dim=32

# run the 24/7 loop:
docker compose up -d --build
docker compose logs -f                              # watch pass_complete / chunk_written
```

One-off pass instead of the loop: `docker compose run --rm collector --once`.

Outputs persist in the `collector_data` / `collector_state` Docker volumes. Copy
chunks out for offline training:

```bash
docker cp ptcg-collector:/data/value ./collector_data_value
python tools/merge_collected.py --src ./collector_data_value --out selfplay/data_collected_all.npz
```

(Or set `COLLECTOR_SINK=kaggle` + `DATASET_SLUG` in `.env` and the collector
publishes a private Kaggle Dataset version each pass — pull that on the training
box instead of copying files.)

## Option B — SwitchBot AI Hub (feasibility first!)

The AI Hub is hardware-capable (≈8 GB RAM / 32 GB, container-based Linux), **but
running your own long-lived code on it is not an officially documented path** —
its supported extensibility is Home Assistant custom components / Matter, not
arbitrary containers or SSH. Before relying on it, confirm on **your** unit:

1. Can you get a shell? (e.g. Home Assistant's *Advanced SSH & Web Terminal*
   add-on, if the Hub's HA exposes the add-on store.)
2. Can you run an arbitrary container (`docker run ...`) or a persistent process?
3. Does it **survive reboots and firmware updates**? (Appliance firmware can wipe
   user changes.)

- **If all three are yes** and Docker is available → use **Option A** as-is.
- **If you only have a shell** (no Docker) but can install Python → use
  **Option C** inside that shell, pointing `COLLECTOR_DATA_DIR` /
  `COLLECTOR_STATE_DIR` at persistent storage (e.g. an attached SSD / the
  MicroSD), and re-arm on reboot.
- **If none** → don't fight the appliance; use a Raspberry Pi / old phone / NAS /
  small VM. The collector is light enough for any of them, and you avoid
  interfering with the Hub's smart-home duties.

> Recommendation: keep the always-on collector on a dedicated box (Pi/NAS/VM) and
> leave the AI Hub to its smart-home job. It removes the firmware-wipe and
> resource-contention risks entirely.

## Option C — Native (Raspberry Pi / Termux / any shell), no Docker

Pure-Python, root-less, aarch64-friendly:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # user-space uv
git clone https://github.com/Martensiter/pokemon-tcg-ai-battle.git
cd pokemon-tcg-ai-battle
uv venv && uv pip install -e ".[kaggle]"
cp .env.example .env                                    # fill credentials
uv run python -m collector --self-test                  # offline acceptance check
```

### Keep it running across reboots

**systemd (Raspberry Pi OS / most Linux):** create
`~/.config/systemd/user/ptcg-collector.service`:

```ini
[Unit]
Description=PTCG replay collector
After=network-online.target

[Service]
WorkingDirectory=%h/pokemon-tcg-ai-battle
ExecStart=%h/.local/bin/uv run python -m collector
Restart=always
RestartSec=30
EnvironmentFile=%h/pokemon-tcg-ai-battle/.env

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now ptcg-collector
loginctl enable-linger "$USER"        # run even when logged out
journalctl --user -u ptcg-collector -f
```

**No systemd (e.g. Termux):** `nohup uv run python -m collector >> collector.out 2>&1 &`
and re-launch on boot (Termux:Boot, or a cron `@reboot`).

## Calling from OpenClaw / on-device agents (remote control)

OpenClaw runs on the same Hub as the collector, so a small local HTTP API lets it
report status or trigger a pass — controllable from a chat app while you're out,
without giving the agent raw shell access. Start it:

```bash
export COLLECTOR_API_TOKEN=$(openssl rand -hex 16)   # put this in .env too
uv run python -m collector --serve                    # binds 127.0.0.1:8765
```

Routes (token via `Authorization: Bearer <token>`):

```bash
curl localhost:8765/health                                            # no auth
curl -H "Authorization: Bearer $TOKEN" localhost:8765/status          # counts + last pass
curl -X POST -H "Authorization: Bearer $TOKEN" localhost:8765/collect  # run ONE pass (409 if busy)
```

Register these in OpenClaw as HTTP tools (e.g. "collector_status" → GET /status,
"collect_now" → POST /collect). Then *"how many battles have we collected?"* or
*"collect now"* works from your phone via OpenClaw.

**Two usage models:**

- **On-demand (simplest with OpenClaw):** run only `--serve`; OpenClaw triggers
  passes via `POST /collect` (on a schedule or on request). One process, one
  consistent manifest.
- **Always-on + status:** run the loop daemon (Option A/C) for continuous
  collection and, if you also want the API, run `--serve` as a *separate*
  read-only status endpoint (its `seen` count is a snapshot from its own start;
  trigger passes from one place to avoid two writers racing the manifest).

Security: keep `COLLECTOR_API_HOST=127.0.0.1` (default). Binding a non-loopback
host is refused unless `COLLECTOR_API_TOKEN` is set. The API never needs public
exposure — you reach it indirectly through OpenClaw's own remote channel.

## Updating

```bash
git pull
# Docker:
docker compose up -d --build
# native:
uv pip install -e ".[kaggle]" && systemctl --user restart ptcg-collector
```

The manifest in `/state` (or `collector_state/`) means an update/restart never
re-downloads episodes you already have.
