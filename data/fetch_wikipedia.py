"""
fetch_wikipedia.py — pull daily Wikipedia pageviews for a curated article set
via the official Wikimedia Pageviews REST API, then cache as date-partitioned
Parquet for the spike-detection case study.

Source: https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/...
        Public, official, documented endpoint. CC0-licensed data.

Why daily and not hourly?
    The Pageviews REST API's `per-article` endpoint supports `daily` and
    `monthly` granularity only — the hourly variant was removed. The bulk
    hourly dumps still exist at https://dumps.wikimedia.org/other/pageviews/
    but require multi-GB downloads and parsing 7M+ rows per hour just to
    filter down to ~100 articles. Daily granularity over 90 days gives the
    detector enough rolling history without the operational cost.

Security posture:
    - HTTPS only, official wikimedia.org host.
    - Identifying User-Agent (per Wikimedia API etiquette).
    - HTTP timeouts everywhere; no infinite hangs.
    - Response-size sanity caps before parsing JSON.
    - No shelling out, no archive extraction.
    - Idempotent: existing cached Parquet skips re-download.

Run:
    python data/fetch_wikipedia.py
    python data/fetch_wikipedia.py --start 2024-09-01 --end 2024-09-30
    python data/fetch_wikipedia.py --force   # ignore cache

Output:
    data/cache/wikipedia/date=YYYY-MM-DD/part-0.parquet
    data/cache/wikipedia/_spike_events.json    (synthetic injections used by
                                               the case study's validation step)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "wikipedia"

API_HOST = "https://wikimedia.org"
API_PATH = (
    "/api/rest_v1/metrics/pageviews/per-article"
    "/en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
)
USER_AGENT = "jesse-g-portfolio-fetch/1.0 (idgesus@gmail.com)"
REQUEST_TIMEOUT = 30
REQUEST_DELAY_S = 0.05  # 20 rps — well under Wikimedia's 200 rps limit
MAX_BYTES_PER_RESPONSE = 50 * 1024 * 1024  # 50 MB sanity cap

# A curated set of well-known en.wikipedia articles. Mix of evergreen
# (historical figures, science topics) and pop-culture pages so the dataset
# carries a realistic mix of baseline traffic levels and natural spikes.
ARTICLES = [
    # Historical / political figures
    "Albert_Einstein", "Isaac_Newton", "Marie_Curie", "Charles_Darwin",
    "Abraham_Lincoln", "George_Washington", "Theodore_Roosevelt",
    "Franklin_D._Roosevelt", "John_F._Kennedy", "Barack_Obama",
    "Cleopatra", "Julius_Caesar", "Alexander_the_Great", "Napoleon",
    "Winston_Churchill", "Mahatma_Gandhi", "Nelson_Mandela",
    "Martin_Luther_King_Jr.", "Adolf_Hitler", "Joseph_Stalin",
    # Music / pop culture
    "Elvis_Presley", "Michael_Jackson", "The_Beatles", "Madonna",
    "Beyonce", "Taylor_Swift", "Rihanna", "Lady_Gaga",
    "Bob_Dylan", "Freddie_Mercury",
    # Film / TV
    "Marilyn_Monroe", "Audrey_Hepburn", "Charlie_Chaplin", "Steven_Spielberg",
    "Stanley_Kubrick", "Alfred_Hitchcock", "Quentin_Tarantino",
    "Christopher_Nolan", "Tom_Hanks", "Meryl_Streep",
    # Sports
    "Michael_Jordan", "LeBron_James", "Cristiano_Ronaldo", "Lionel_Messi",
    "Serena_Williams", "Roger_Federer", "Tiger_Woods", "Muhammad_Ali",
    "Pele", "Wayne_Gretzky",
    # Authors / thinkers
    "William_Shakespeare", "Mark_Twain", "Ernest_Hemingway", "Jane_Austen",
    "George_Orwell", "Stephen_King", "J._K._Rowling", "Agatha_Christie",
    "Sigmund_Freud", "Friedrich_Nietzsche",
    # Tech / business
    "Steve_Jobs", "Bill_Gates", "Elon_Musk", "Mark_Zuckerberg",
    "Jeff_Bezos", "Warren_Buffett", "Alan_Turing", "Ada_Lovelace",
    "Tim_Berners-Lee", "Linus_Torvalds",
    # Concepts and broader topics
    "World_War_II", "World_War_I", "American_Civil_War", "Cold_War",
    "French_Revolution", "Industrial_Revolution", "Renaissance",
    "Climate_change", "Black_hole", "DNA", "Evolution",
    "Quantum_mechanics", "General_relativity", "Photosynthesis",
    # Geography / cultural
    "Eiffel_Tower", "Statue_of_Liberty", "Great_Wall_of_China",
    "Mount_Everest", "Niagara_Falls", "Grand_Canyon",
    "Pacific_Ocean", "Amazon_rainforest", "Sahara",
    # Modern subjects (likely to have natural spikes)
    "COVID-19_pandemic", "ChatGPT", "Artificial_intelligence",
    "Bitcoin", "SpaceX", "NASA",
    # Filler popular pages
    "Earth", "Sun", "Moon", "Solar_System",
]

# Synthetic spike events the spike-detection case study layers ON TOP of the
# real data so the validation step has known ground truth. We inject these
# rather than try to identify real-world spike events automatically — that
# keeps the validation rigorous and reproducible. The case study prose is
# explicit about which layer is which.
SPIKE_EVENTS_TEMPLATE = [
    {"article": "Albert_Einstein",  "day_offset_from_start": 20, "duration_days": 2, "multiplier": 8.0,  "event_type": "Breaking News"},
    {"article": "Taylor_Swift",     "day_offset_from_start": 35, "duration_days": 3, "multiplier": 5.0,  "event_type": "Celebrity Mention"},
    {"article": "Bitcoin",          "day_offset_from_start": 55, "duration_days": 4, "multiplier": 4.0,  "event_type": "Viral Social Media"},
    {"article": "Elon_Musk",        "day_offset_from_start": 70, "duration_days": 1, "multiplier": 12.0, "event_type": "TV Mention"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", default="2024-07-01",
                        help="Start date YYYY-MM-DD (default: 2024-07-01)")
    parser.add_argument("--end", default="2024-09-30",
                        help="End date YYYY-MM-DD inclusive (default: 2024-09-30)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if cache exists")
    return parser.parse_args()


def fetch_article_daily(article: str, start: date, end: date,
                        session: requests.Session) -> list[dict]:
    """Hit the REST API for one article, return raw items list."""
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    url = API_HOST + API_PATH.format(article=article, start=start_str, end=end_str)

    resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
    if resp.status_code == 404:
        # Some article titles return 404 if there's no traffic in the window;
        # treat as empty rather than fatal.
        return []
    resp.raise_for_status()

    content_length = int(resp.headers.get("content-length", 0))
    if content_length > MAX_BYTES_PER_RESPONSE:
        raise RuntimeError(
            f"Suspiciously large response for {article}: {content_length} bytes"
        )

    body = resp.content
    if len(body) > MAX_BYTES_PER_RESPONSE:
        raise RuntimeError(
            f"Body exceeded sanity cap for {article}: {len(body)} bytes"
        )

    payload = json.loads(body)
    return payload.get("items", []) or []


def items_to_records(items: list[dict]) -> list[dict]:
    """Normalise API rows to our schema. Daily timestamps are YYYYMMDDHH (HH=00)."""
    out = []
    for item in items:
        ts_str = item.get("timestamp", "")
        if len(ts_str) < 8:
            continue
        try:
            ts = datetime.strptime(ts_str[:8], "%Y%m%d")
        except ValueError:
            continue
        out.append({
            "project": "en.wikipedia",
            "article": item.get("article", ""),
            "timestamp": ts,
            "views": int(item.get("views", 0)),
        })
    return out


def write_partitioned(df: pd.DataFrame, root: Path) -> None:
    """Hive-style partition by year-month, snappy-compressed.

    We partition by month rather than date because the daily REST API gives us
    only ~100 rows per day across the curated article set. Per-day parquet
    files at that size are dominated by per-file metadata overhead (the
    classic over-partitioning failure mode). Month partitions land at
    ~2-3k rows each — small enough to keep partition pruning useful, large
    enough for parquet's columnar compression to actually pay off.
    """
    if df.empty:
        raise RuntimeError("No data to write — fetch returned zero rows")
    df = df.copy()
    df["year_month"] = df["timestamp"].dt.strftime("%Y-%m")

    if root.exists():
        for child in list(root.glob("year_month=*")) + list(root.glob("date=*")):
            for f in child.iterdir():
                f.unlink()
            child.rmdir()
    root.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(root),
        partition_cols=["year_month"],
        compression="snappy",
    )


def write_spike_events(start: date, root: Path) -> None:
    events = []
    for tpl in SPIKE_EVENTS_TEMPLATE:
        anchor = datetime.combine(start, datetime.min.time())
        spike_start = anchor + timedelta(days=tpl["day_offset_from_start"])
        events.append({
            "article": tpl["article"],
            "start": spike_start.isoformat(),
            "duration_days": tpl["duration_days"],
            "multiplier": tpl["multiplier"],
            "event_type": tpl["event_type"],
        })
    (root / "_spike_events.json").write_text(json.dumps(events, indent=2))


def main() -> int:
    args = parse_args()

    try:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"ERROR: bad date format ({e})", file=sys.stderr)
        return 2
    if end < start:
        print("ERROR: --end must not be before --start", file=sys.stderr)
        return 2

    parquet_already = (
        CACHE_DIR.exists()
        and any(CACHE_DIR.glob("year_month=*/*.parquet"))
        and (CACHE_DIR / "_spike_events.json").exists()
    )
    if parquet_already and not args.force:
        print(f"Cache already present at {CACHE_DIR}. Use --force to refresh.")
        return 0

    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })

    all_records: list[dict] = []
    failures = 0
    for article in tqdm(ARTICLES, desc="Wikipedia REST API"):
        try:
            items = fetch_article_daily(article, start, end, session)
            all_records.extend(items_to_records(items))
        except Exception as e:
            failures += 1
            print(f"  ! {article}: {e}", file=sys.stderr)
        time.sleep(REQUEST_DELAY_S)

    if failures > len(ARTICLES) * 0.5:
        print(f"ERROR: {failures}/{len(ARTICLES)} articles failed — aborting",
              file=sys.stderr)
        return 1

    df = pd.DataFrame(all_records)
    if df.empty:
        print("ERROR: API returned no rows", file=sys.stderr)
        return 1

    print(f"Fetched {len(df):,} rows across {df['article'].nunique()} articles")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    write_partitioned(df, CACHE_DIR)
    write_spike_events(start, CACHE_DIR)
    print(f"Wrote Parquet to {CACHE_DIR}")
    print(f"Wrote spike events to {CACHE_DIR / '_spike_events.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
