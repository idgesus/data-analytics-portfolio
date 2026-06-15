"""
fetch_cms.py — pull a CMS Open Payments General Payments file for one
program year, filter to the eight pharma companies the automated-pipeline
case study uses, and cache as Parquet.

Source: https://download.cms.gov/openpayments/<bulk-file>.zip
        Public-domain bulk export. The exact filename includes the publication
        date (e.g. PGYY2023_P012025.zip), which CMS rotates on each refresh —
        if the default URL 404s, browse https://www.cms.gov/openpayments/data
        for the current bulk download and pass it via --url.

Why filter at fetch time:
    The full General Payments file is ~3 GB CSV with 13M+ rows and 90+ columns.
    The case study only needs eight specific manufacturers. We filter on read
    using Polars lazy scan so the script keeps memory bounded and the cached
    Parquet stays under 100 MB.

Security posture:
    - HTTPS only, official cms.gov host.
    - HTTP timeouts; sanity caps on response size.
    - Zip extraction validates member paths (no zip-slip).
    - No shelling out.
    - Idempotent: existing cached Parquet skips re-download.

Run:
    python data/fetch_cms.py
    python data/fetch_cms.py --url https://download.cms.gov/openpayments/PGYY2022_P01_<date>.zip
    python data/fetch_cms.py --force
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cms"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "cms"

USER_AGENT = "jesse-g-portfolio-fetch/1.0 (idgesus@gmail.com)"
DOWNLOAD_TIMEOUT = (15, 1800)  # connect, read

DEFAULT_URL = "https://download.cms.gov/openpayments/PGYY2023_P012025.zip"
MAX_BYTES = 6 * 1024 * 1024 * 1024  # 6 GB sanity cap

# Companies to keep (case study uses these). The General Payments file's
# manufacturer column has minor capitalization variation across years, so we
# match case-insensitively on a substring.
TARGET_COMPANIES = [
    "Pfizer", "Johnson & Johnson", "Janssen",
    "Merck", "AbbVie", "Eli Lilly",
    "Bristol-Myers Squibb", "Novartis", "GlaxoSmithKline",
]

# Columns in the General Payments file we actually use. Real CMS files have
# 90+ columns; selecting cuts memory and disk by an order of magnitude. The
# names below match the published General Payments 2023 schema.
KEEP_COLS = [
    "Record_ID",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
    "Physician_Specialty",
    "Recipient_State",
    "Date_of_Payment",
    "Nature_of_Payment_or_Transfer_of_Value",
    "Total_Amount_of_Payment_USDollars",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"Bulk download URL (default: {DEFAULT_URL})")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if cache exists")
    return parser.parse_args()


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/zip,*/*"})
    return s


def download_with_progress(url: str, dest: Path,
                           session: requests.Session) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
        if resp.status_code == 404:
            raise RuntimeError(
                f"URL not found: {url}\n"
                f"CMS rotates the publication-date filename. Visit "
                f"https://www.cms.gov/openpayments/data and pass the current "
                f"General Payments bulk download via --url."
            )
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        if total and total > MAX_BYTES:
            raise RuntimeError(f"content-length {total} exceeds cap {MAX_BYTES}")

        bytes_written = 0
        with dest.open("wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                bytes_written += len(chunk)
                if bytes_written > MAX_BYTES:
                    raise RuntimeError(f"{dest.name} exceeded byte cap mid-stream")
                f.write(chunk)
                pbar.update(len(chunk))


def find_general_payments_csv(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    # CMS bulk zips typically include OP_DTL_GNRL_*.csv (general payments),
    # OP_DTL_RSRCH_*.csv (research), and OP_DTL_OWNRSHP_*.csv (ownership).
    candidates = [n for n in names if "GNRL" in n.upper() and n.lower().endswith(".csv")]
    if not candidates:
        raise RuntimeError(
            f"Could not find general-payments CSV in {zip_path}. "
            f"Members: {names}"
        )
    return candidates[0]


def safe_extract_member(zip_path: Path, member: str, dest_dir: Path) -> Path:
    """Extract one zip member with zip-slip validation."""
    if member.startswith(("/", "\\")) or ".." in Path(member).parts:
        raise RuntimeError(f"Refusing unsafe zip member path: {member}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / Path(member).name
    if target.exists() and target.stat().st_size > 0:
        return target
    with zipfile.ZipFile(zip_path, "r") as zf, zf.open(member) as src, target.open("wb") as out:
        # Stream extraction — don't load multi-GB CSV into memory.
        while True:
            buf = src.read(8 * 1024 * 1024)
            if not buf:
                break
            out.write(buf)
    return target


def filter_and_write_parquet(csv_path: Path, out_dir: Path) -> int:
    """Polars lazy scan: project columns, filter manufacturers, materialize."""
    out_dir.mkdir(parents=True, exist_ok=True)

    company_col = "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"
    pattern = "(?i)" + "|".join(TARGET_COMPANIES)

    print(f"scanning {csv_path.name} (lazy, filtered) ...")
    lf = (
        pl.scan_csv(
            csv_path,
            infer_schema_length=0,           # treat everything as Utf8 first
            ignore_errors=True,
            try_parse_dates=False,
        )
        .select(KEEP_COLS)
        .filter(pl.col(company_col).str.contains(pattern))
        .with_columns([
            pl.col("Total_Amount_of_Payment_USDollars")
                .cast(pl.Float64, strict=False)
                .alias("amount_usd"),
            pl.col("Date_of_Payment")
                .str.strptime(pl.Date, format="%m/%d/%Y", strict=False)
                .alias("payment_date"),
        ])
        .drop("Total_Amount_of_Payment_USDollars", "Date_of_Payment")
        .rename({
            company_col: "company",
            "Physician_Specialty": "specialty",
            "Recipient_State": "state",
            "Nature_of_Payment_or_Transfer_of_Value": "payment_type",
            "Record_ID": "record_id",
        })
        .filter(pl.col("amount_usd").is_not_null() & pl.col("payment_date").is_not_null())
    )

    df = lf.collect(engine="streaming")
    n = df.height
    if n == 0:
        raise RuntimeError("Filter returned zero rows — check TARGET_COMPANIES "
                           "against the CSV's company column values")

    out = out_dir / "general_payments.parquet"
    df.write_parquet(out, compression="snappy")
    print(f"wrote {n:,} rows to {out}")
    return n


def main() -> int:
    args = parse_args()

    out_file = CACHE_DIR / "general_payments.parquet"
    if out_file.exists() and out_file.stat().st_size > 0 and not args.force:
        print(f"Cache already present at {out_file}. Use --force to refresh.")
        return 0

    session = make_session()
    zip_name = Path(args.url).name
    if not zip_name.lower().endswith(".zip"):
        print(f"ERROR: --url must end in .zip (got {args.url})", file=sys.stderr)
        return 2
    zip_path = RAW_DIR / zip_name

    print(f"downloading {args.url} ...")
    download_with_progress(args.url, zip_path, session)

    member = find_general_payments_csv(zip_path)
    print(f"extracting {member} ...")
    csv_path = safe_extract_member(zip_path, member, RAW_DIR)

    filter_and_write_parquet(csv_path, CACHE_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
