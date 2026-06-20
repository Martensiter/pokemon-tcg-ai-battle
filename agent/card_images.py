"""Map a card id to a public card-art image URL.

Set-code -> image-host mapping is ported from charlielockyer-rice/cabt-viewer
(MIT). Each card's expansion + collection number come from EN_Card_Data.csv, so
no Pokemon asset is bundled; we just build URLs to public card-image hosts.
Returns None when a card can't be resolved (callers fall back to a text tile).
"""
from __future__ import annotations

import os
import csv
import functools

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV = os.path.join(_ROOT, "EN_Card_Data.csv")

# set code -> host id (str = pokemontcg.io) or (id, source) for other hosts
_SET_MAP: dict[str, object] = {
    "BASE": "base1", "JUNGLE": "base2", "FOSSIL": "base3",
    "SIT": "swsh12", "ASR": "swsh10", "LOR": "swsh11",
    "SVI": "sv1", "SVE": "sve", "PAL": "sv2", "OBF": "sv3", "MEW": "sv3pt5",
    "PAR": "sv4", "PAF": "sv4pt5", "TEF": "sv5", "TWM": "sv6", "SFA": "sv6pt5",
    "SCR": "sv7", "SSP": "sv8", "PRE": "sv8pt5", "JTG": "sv9", "DRI": "sv10",
    "BLK": "zsv10pt5", "WHT": "rsv10pt5",
    "MEP": ("mep", "scrydex"), "MEE": ("mee", "pkmncards"),
    "MEG": "me1", "M1L": "me1", "M1S": "me1",
    "PFL": ("me2", "scrydex"), "ASC": ("me2pt5", "scrydex"),
    "POR": ("me3", "scrydex"), "CRI": ("me4", "scrydex"),
}


def _url(set_code: str, number: str) -> str | None:
    info = _SET_MAP.get(set_code)
    if not info or not number:
        return None
    set_id, source = (info, "pokemontcg") if isinstance(info, str) else info
    if source == "scrydex":
        return f"https://images.scrydex.com/pokemon/{set_id}-{number}/large"
    if source == "pkmncards":
        return f"https://pkmncards.com/wp-content/uploads/{set_id}_en_{number.zfill(3)}_std.png"
    return f"https://images.pokemontcg.io/{set_id}/{number}.png"


@functools.lru_cache(maxsize=1)
def _id_to_url() -> dict[int, str]:
    out: dict[int, str] = {}
    if not os.path.exists(_CSV):
        return out
    with open(_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["Card ID"])
            except (ValueError, KeyError):
                continue
            url = _url((row.get("Expansion") or "").strip(),
                       (row.get("Collection No.") or "").strip())
            if url and cid not in out:
                out[cid] = url
    return out


def image_url(card_id: int) -> str | None:
    return _id_to_url().get(card_id)
