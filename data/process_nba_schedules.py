#!/usr/bin/env python3
"""
Process raw NBA schedule/results CSVs into slim files used by the Elo pipeline.

Raw inputs live in data/nba/raw/ (Basketball-Reference exports or the
official NBA schedule PDF extraction). Processed outputs are written to
data/nba/ as nba_YYYY.csv and contain only the columns required by
elo_lab.workflows.simulate_season.normalize_schedule:

  Date, Start (ET), Visitor/Neutral, PTS, Home/Neutral, PTS, Notes

- Historical seasons keep scores; upcoming seasons leave PTS blank.
- Rows are sorted by date (and start time when available).
- Extra BR columns (Attend., LOG, Arena, box-score placeholders) are dropped.
- Notes are retained (NBA Cup, Play-In, Neutral, etc.) for optional filtering.

Usage:
  python -m data.process_nba_schedules
  # or
  python data/process_nba_schedules.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Script lives at data/process_nba_schedules.py
# Raw inputs:  data/nba/raw/nba_YYYY.csv
# Outputs:     data/nba/nba_YYYY.csv
_DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = _DATA_DIR / "nba" / "raw"
OUT_DIR = _DATA_DIR / "nba"

# Columns the normalizer + loaders actually need / benefit from.
# Order matches historical BR exports so existing code keeps working.
KEEP_COLS = [
    "Date",
    "Start (ET)",
    "Visitor/Neutral",
    "PTS",
    "Home/Neutral",
    "PTS",  # second PTS; pandas will rename to PTS.1 on read
    "Notes",
]

# Canonical header written to disk (two literal "PTS" columns).
HEADER = [
    "Date",
    "Start (ET)",
    "Visitor/Neutral",
    "PTS",
    "Home/Neutral",
    "PTS",
    "Notes",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map various raw headers onto a consistent set of names."""
    # Strip whitespace from column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Historical BR: duplicate "PTS" becomes PTS / PTS.1 after pandas read
    # Raw 2027 extractor may already have two PTS or empty second column
    rename = {}
    cols = list(df.columns)

    # Find visitor / home
    for c in cols:
        cl = c.lower()
        if cl in ("visitor/neutral", "visitor", "away", "away_team"):
            rename[c] = "Visitor/Neutral"
        elif cl in ("home/neutral", "home", "home_team"):
            rename[c] = "Home/Neutral"
        elif cl in ("date",):
            rename[c] = "Date"
        elif cl in ("start (et)", "start", "time", "start_et"):
            rename[c] = "Start (ET)"
        elif cl in ("notes", "note"):
            rename[c] = "Notes"
        elif cl in ("attend.", "attend", "attendance"):
            rename[c] = "Attend."
        elif cl == "log":
            rename[c] = "LOG"
        elif cl == "arena":
            rename[c] = "Arena"

    df = df.rename(columns=rename)

    # Collect score columns (PTS, PTS.1, G, etc.)
    pts_like = [c for c in df.columns if re.match(r"^PTS(\.\d+)?$", str(c), re.I) or str(c).upper() in ("G", "G.1")]
    if len(pts_like) >= 2:
        df = df.rename(columns={pts_like[0]: "_away_pts", pts_like[1]: "_home_pts"})
    elif len(pts_like) == 1:
        df = df.rename(columns={pts_like[0]: "_away_pts"})
        df["_home_pts"] = pd.NA
    else:
        # Empty schedule file
        df["_away_pts"] = pd.NA
        df["_home_pts"] = pd.NA

    if "Notes" not in df.columns:
        df["Notes"] = ""
    if "Start (ET)" not in df.columns:
        df["Start (ET)"] = ""

    # Keep only meaningful notes; drop stray single-letter flags (R, A, B, …)
    def _clean_notes(val: str) -> str:
        if not isinstance(val, str) or not val.strip():
            return ""
        parts = [p.strip() for p in re.split(r"[;|/]", val) if p.strip()]
        keep = []
        for p in parts:
            pl = p.lower()
            if pl in ("nba cup", "cup", "c") or "cup" in pl:
                keep.append("NBA Cup")
            elif pl in ("play-in game", "play-in", "play in"):
                keep.append("Play-In Game")
            elif pl in ("neutral", "vs") or pl.startswith("at "):
                # "at Mexico City Mexico" / "at Paris France" are neutral sites.
                keep.append("Neutral")
            # ignore single-letter TV/flag codes (R, A, B, …)
        # dedupe while preserving order
        seen = set()
        out = []
        for k in keep:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return "; ".join(out)

    df["Notes"] = df["Notes"].map(_clean_notes)
    # Stamp Neutral when the raw arena is a known neutral building
    # (Global Games, NBA Cup knockout at T-Mobile Arena). Same list as
    # elo_lab.neutral_sites.NBA_NEUTRAL_ARENAS. Home-arena NBA Cup games
    # are not stamped.
    if "Arena" in df.columns:
        try:
            from elo_lab.neutral_sites import arena_is_neutral
        except ImportError:
            arena_is_neutral = None
        if arena_is_neutral is not None:
            stamped = []
            for note, arena in zip(df["Notes"], df["Arena"]):
                text = note or ""
                if arena_is_neutral(arena) and "Neutral" not in text.split("; "):
                    text = "Neutral" if not text else f"{text}; Neutral"
                stamped.append(text)
            df["Notes"] = stamped
    return df


def _parse_sort_key(row: pd.Series):
    """Return a sortable (date, time) tuple."""
    date_val = row.get("Date", "")
    dt = pd.to_datetime(date_val, errors="coerce")
    if pd.isna(dt):
        # try MM/DD/YY style
        dt = pd.to_datetime(date_val, format="%m/%d/%y", errors="coerce")
    if pd.isna(dt):
        dt = pd.Timestamp.min

    start = str(row.get("Start (ET)", "") or "").strip().lower()
    # BR style: 7:30p / 10:00p / 12:00p
    hour, minute = 0, 0
    m = re.match(r"(\d{1,2}):(\d{2})\s*([ap])?", start)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        ap = (m.group(3) or "").lower()
        if ap == "p" and hour != 12:
            hour += 12
        elif ap == "a" and hour == 12:
            hour = 0
    return (dt, hour, minute)


def process_one(raw_path: Path, out_path: Path) -> int:
    """Read one raw CSV, slim + sort, write processed file. Returns row count."""
    df = pd.read_csv(raw_path, dtype=str)  # keep everything as str initially
    df = _normalize_columns(df)

    required = ["Visitor/Neutral", "Home/Neutral"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{raw_path.name}: missing required column {col}")

    # Build output frame
    out = pd.DataFrame()
    out["Date"] = df["Date"].fillna("").astype(str).str.strip()
    out["Start (ET)"] = df["Start (ET)"].fillna("").astype(str).str.strip()
    out["Visitor/Neutral"] = df["Visitor/Neutral"].fillna("").astype(str).str.strip()
    out["PTS"] = pd.to_numeric(df["_away_pts"], errors="coerce")
    out["Home/Neutral"] = df["Home/Neutral"].fillna("").astype(str).str.strip()
    out["PTS.1"] = pd.to_numeric(df["_home_pts"], errors="coerce")
    out["Notes"] = df["Notes"].fillna("").astype(str).str.strip()

    # Drop completely empty team rows
    out = out[(out["Visitor/Neutral"] != "") & (out["Home/Neutral"] != "")]

    # Sort by date / time (ignore any original game index)
    sort_keys = out.apply(_parse_sort_key, axis=1)
    out = out.iloc[sort_keys.argsort()].reset_index(drop=True)

    # Write with two literal PTS columns so header matches BR + normalizer
    # (pandas will read them as PTS / PTS.1)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(HEADER) + "\n")
        for _, row in out.iterrows():
            away_pts = "" if pd.isna(row["PTS"]) else str(int(row["PTS"])) if float(row["PTS"]).is_integer() else str(row["PTS"])
            home_pts = "" if pd.isna(row["PTS.1"]) else str(int(row["PTS.1"])) if float(row["PTS.1"]).is_integer() else str(row["PTS.1"])
            notes = row["Notes"].replace(",", ";")  # avoid CSV breakage
            f.write(
                f'{row["Date"]},{row["Start (ET)"]},{row["Visitor/Neutral"]},'
                f'{away_pts},{row["Home/Neutral"]},{home_pts},{notes}\n'
            )

    return len(out)


def main() -> None:
    if not RAW_DIR.is_dir():
        raise SystemExit(f"Raw directory not found: {RAW_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("nba_*.csv"))
    if not files:
        raise SystemExit(f"No nba_*.csv files in {RAW_DIR}")

    print(f"Processing {len(files)} raw file(s) from {RAW_DIR}")
    for raw in files:
        out = OUT_DIR / raw.name
        n = process_one(raw, out)
        print(f"  {raw.name:20s} -> {out.name:20s}  ({n} games)")
    print("Done.")


if __name__ == "__main__":
    main()
