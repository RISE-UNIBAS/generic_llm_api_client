#!/usr/bin/env python3
"""Refresh the bundled pricing snapshot from the benchmark repo.

`ai_client/pricing.json` is a *generated snapshot*, not a hand-edited file. Pricing
data is maintained by hand in the public benchmark repo
(RISE-UNIBAS/humanities_data_benchmark, scripts/data/pricing.json), which is the single
source of truth. This script pulls that file and overwrites the bundled copy.

Run it as a release step, before building the package (see PUBLISHING.md). It is NOT used
at runtime — the library never fetches pricing while serving requests.

Usage:
    python scripts/update_pricing.py            # fetch and overwrite ai_client/pricing.json
    python scripts/update_pricing.py --dry-run  # show what would change, write nothing
    python scripts/update_pricing.py --url <raw-url>   # override the source URL
"""

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://raw.githubusercontent.com/RISE-UNIBAS/"
    "humanities_data_benchmark/main/scripts/data/pricing.json"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "ai_client" / "pricing.json"


def _summary(data: dict) -> str:
    meta = data.get("metadata", {})
    snapshots = data.get("pricing", {})
    return (
        f"version={meta.get('version')} "
        f"last_updated={meta.get('last_updated')} "
        f"snapshots={len(snapshots)}"
    )


def fetch(url: str) -> dict:
    """Download and parse the pricing file, validating its basic shape."""
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)  # raises on invalid JSON
    if "pricing" not in data or not isinstance(data["pricing"], dict):
        raise ValueError("fetched file has no 'pricing' object; refusing to use it")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="source raw URL")
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would change, write nothing"
    )
    args = parser.parse_args()

    print(f"Fetching pricing from:\n  {args.url}")
    try:
        new_data = fetch(args.url)
    except Exception as e:
        print(f"ERROR: could not fetch/parse pricing: {e}", file=sys.stderr)
        return 1

    print(f"Fetched:  {_summary(new_data)}")
    if TARGET.exists():
        current = json.loads(TARGET.read_text(encoding="utf-8"))
        print(f"Current:  {_summary(current)}")
        if current == new_data:
            print("Already up to date; nothing to write.")
            return 0
    else:
        print(f"Current:  (no existing {TARGET.name})")

    if args.dry_run:
        print("[DRY RUN] Would overwrite", TARGET)
        return 0

    # Atomic write: temp file in the same dir, then replace.
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(TARGET.parent), suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, TARGET)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    print(f"Wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
