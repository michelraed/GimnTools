"""Retrieve ranked passages with traceable paper or repository metadata."""

import argparse
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("keywords", help="Title, DOI, or scientific search terms")
    parser.add_argument("--index", type=Path, default=ROOT / ".cache" / "sources.db")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    tokens = re.findall(r"\w+", args.keywords)
    if not tokens or args.limit <= 0:
        parser.error("provide search terms and a positive limit")
    query = " OR ".join(f'"{token}"' for token in tokens)
    with sqlite3.connect(f"{args.index.resolve().as_uri()}?mode=ro", uri=True) as conn:
        for title, url, revision, section, body in conn.execute(
            "SELECT title,url,revision,section,snippet(excerpts,6,'[',']','…',45) "
            "FROM excerpts WHERE excerpts MATCH ? ORDER BY bm25(excerpts,0,2,4,0,0,0,1) LIMIT ?",
            (query, args.limit),
        ):
            print(f"{title} · passage {section}\n{body}\n{url} {revision}\n")


if __name__ == "__main__":
    main()
