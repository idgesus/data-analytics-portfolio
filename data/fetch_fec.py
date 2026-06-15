"""
fetch_fec.py — pull FEC bulk operating-expenditures + committee-master files
for one or more election cycles, parse the pipe-delimited records, and cache
as cycle/month-partitioned Parquet for the political-ad-spend case study.

Source: https://www.fec.gov/files/bulk-downloads/{cycle}/oppexp{YY}.zip
        https://www.fec.gov/files/bulk-downloads/{cycle}/cm{YY}.zip
        Public domain bulk data published by the FEC.

Note on format:
    The oppexp/cm bulk files are pipe-delimited (`|`), not ASCII-28 delimited.
    Some legacy FEC documentation references ASCII-28 for *other* file types,
    which is the source of the confusion in the case-study prose. Fixed here.

Security posture:
    - HTTPS only, official fec.gov host.
    - Identifying User-Agent.
    - HTTP timeouts; streamed downloads with size sanity caps.
    - Zip extraction validates file paths (no zip-slip).
    - No shelling out.
    - Idempotent: existing cached Parquet skips re-download.

Run:
    python data/fetch_fec.py
    python data/fetch_fec.py --cycles 2022 2024
    python data/fetch_fec.py --force
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fec"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "fec"

BASE_URL = "https://www.fec.gov/files/bulk-downloads"
HEADER_BASE = f"{BASE_URL}/data_dictionaries"
USER_AGENT = "jesse-g-portfolio-fetch/1.0 (idgesus@gmail.com)"
REQUEST_TIMEOUT = 60
DOWNLOAD_TIMEOUT = (15, 600)  # connect, read

# Sanity caps so a redirect to the wrong host can't OOM the machine.
MAX_OPPEXP_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
MAX_CM_BYTES = 100 * 1024 * 1024            # 100 MB

# Columns we want downstream — match the schema the case study uses.
OPPEXP_COLS = ["cmte_id", "name", "transaction_dt", "transaction_amt", "purpose"]
CM_COLS = ["cmte_id", "cmte_nm", "cmte_tp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cycles", nargs="+", default=["2020", "2022", "2024"],
                        type=str, help="Election cycles to fetch (default: 2020 2022 2024)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if cache exists")
    return parser.parse_args()


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/zip,text/csv,*/*",
    })
    return s


def download_with_progress(url: str, dest: Path, max_bytes: int,
                           session: requests.Session) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        if total and total > max_bytes:
            raise RuntimeError(f"{url} content-length {total} exceeds cap {max_bytes}")

        bytes_written = 0
        with dest.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise RuntimeError(f"{dest.name} exceeded byte cap mid-stream")
                f.write(chunk)
                pbar.update(len(chunk))


def safe_zip_extract(zip_path: Path, member: str, dest_dir: Path) -> Path:
    """Extract a single member from a zip, validating the resolved path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        info = zf.getinfo(member)
        # Reject absolute paths or parent traversals (zip-slip).
        if info.filename.startswith(("/", "\\")) or ".." in Path(info.filename).parts:
            raise RuntimeError(f"Refusing unsafe zip member path: {info.filename}")
        target = dest_dir / Path(info.filename).name
        with zf.open(info) as src, target.open("wb") as out:
            while True:
                buf = src.read(1024 * 1024)
                if not buf:
                    break
                out.write(buf)
    return target


def list_zip_data_member(zip_path: Path, expected_prefix: str) -> str:
    """The bulk zips contain a single .txt file; find its name."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith(".txt")]
    if not candidates:
        raise RuntimeError(f"No .txt member found in {zip_path}")
    # Prefer the member whose name starts with the expected prefix (oppexp/cm).
    matched = [n for n in candidates if expected_prefix in n.lower()]
    return matched[0] if matched else candidates[0]


def fetch_header(name: str, session: requests.Session) -> list[str]:
    """Pull the header CSV (one-row file with column names)."""
    url = f"{HEADER_BASE}/{name}_header_file.csv"
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    cols = [c.strip().lower() for c in resp.text.strip().splitlines()[0].split(",")]
    if not cols:
        raise RuntimeError(f"Header file empty: {url}")
    return cols


def read_pipe_delimited(txt_path: Path, columns: list[str],
                        keep_cols: list[str]) -> pd.DataFrame:
    """Read pipe-delimited FEC text file with explicit column names.

    FEC bulk files end every row with a trailing pipe, so each line splits to
    one more field than the header CSV declares. Detect the actual column
    count and pad the names list rather than letting pandas treat column 0
    as an index (which silently shifts every named column left by one).
    """
    with open(txt_path, "rb") as f:
        first_line = f.readline().decode("latin-1")
    actual_cols = first_line.count("|") + 1
    if actual_cols > len(columns):
        columns = list(columns) + [f"_extra_{i}" for i in range(actual_cols - len(columns))]
    df = pd.read_csv(
        txt_path,
        sep="|",
        header=None,
        names=columns,
        dtype=str,
        encoding="latin-1",
        on_bad_lines="skip",
        low_memory=False,
        index_col=False,
    )
    return df[keep_cols].copy()


def fetch_cycle(cycle: str, session: requests.Session) -> pd.DataFrame:
    yy = cycle[-2:]
    oppexp_url = f"{BASE_URL}/{cycle}/oppexp{yy}.zip"
    cm_url = f"{BASE_URL}/{cycle}/cm{yy}.zip"

    raw = RAW_DIR / cycle
    raw.mkdir(parents=True, exist_ok=True)
    oppexp_zip = raw / f"oppexp{yy}.zip"
    cm_zip = raw / f"cm{yy}.zip"

    print(f"[{cycle}] downloading oppexp + cm ...")
    download_with_progress(oppexp_url, oppexp_zip, MAX_OPPEXP_BYTES, session)
    download_with_progress(cm_url, cm_zip, MAX_CM_BYTES, session)

    print(f"[{cycle}] fetching schemas ...")
    oppexp_cols = fetch_header("oppexp", session)
    cm_cols = fetch_header("cm", session)

    print(f"[{cycle}] extracting + parsing ...")
    oppexp_member = list_zip_data_member(oppexp_zip, "oppexp")
    cm_member = list_zip_data_member(cm_zip, "cm")
    oppexp_txt = safe_zip_extract(oppexp_zip, oppexp_member, raw)
    cm_txt = safe_zip_extract(cm_zip, cm_member, raw)

    oppexp = read_pipe_delimited(oppexp_txt, oppexp_cols, OPPEXP_COLS)
    cm = read_pipe_delimited(cm_txt, cm_cols, CM_COLS).drop_duplicates("cmte_id")

    print(f"[{cycle}] joining {len(oppexp):,} oppexp rows × {len(cm):,} committees ...")
    df = oppexp.merge(cm, on="cmte_id", how="left")

    # Coerce types. Some rows have malformed dates / amounts; drop those.
    df["transaction_amt"] = pd.to_numeric(df["transaction_amt"], errors="coerce")
    df["transaction_dt"] = pd.to_datetime(df["transaction_dt"],
                                          format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["transaction_amt", "transaction_dt"])
    df = df[df["transaction_amt"] > 0]

    # Constrain dates to a sensible window for this cycle so erroneous rows
    # (occasionally dated 1900 or 2099) don't corrupt downstream analytics.
    cycle_year = int(cycle)
    df = df[
        (df["transaction_dt"] >= f"{cycle_year - 2}-01-01") &
        (df["transaction_dt"] <= f"{cycle_year}-12-31")
    ].copy()

    df["cycle"] = cycle_year
    df["month"] = df["transaction_dt"].dt.month
    df["cmte_nm"] = df["cmte_nm"].fillna("UNKNOWN COMMITTEE")
    df["cmte_tp"] = df["cmte_tp"].fillna("X")

    print(f"[{cycle}] cleaned {len(df):,} rows, "
          f"total spend ${df['transaction_amt'].sum():,.0f}")
    return df


def main() -> int:
    args = parse_args()

    if not args.force and CACHE_DIR.exists() and any(CACHE_DIR.glob("cycle=*/**/*.parquet")):
        print(f"Cache already present at {CACHE_DIR}. Use --force to refresh.")
        return 0

    session = make_session()
    frames: list[pd.DataFrame] = []
    for cycle in args.cycles:
        try:
            frames.append(fetch_cycle(cycle, session))
        except Exception as e:
            print(f"  ! cycle {cycle} failed: {e}", file=sys.stderr)

    if not frames:
        print("ERROR: no cycles succeeded", file=sys.stderr)
        return 1

    full = pd.concat(frames, ignore_index=True)
    if CACHE_DIR.exists():
        for p in sorted(CACHE_DIR.rglob("*.parquet"), reverse=True):
            p.unlink()
        for d in sorted(CACHE_DIR.rglob("*"), reverse=True):
            if d.is_dir():
                d.rmdir()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"writing {len(full):,} rows to {CACHE_DIR} ...")
    table = pa.Table.from_pandas(full, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(CACHE_DIR),
        partition_cols=["cycle", "month"],
        compression="snappy",
    )
    print(f"done — {len(list(CACHE_DIR.rglob('*.parquet')))} partition files written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
