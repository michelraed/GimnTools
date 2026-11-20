"""Index locally reviewed text and, optionally, publicly licensed repositories."""

import argparse
import json
from pathlib import Path
import re
import sqlite3
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, default=ROOT / ".cache" / "texts")
    parser.add_argument("--index", type=Path, default=ROOT / ".cache" / "sources.db")
    parser.add_argument("--fetch", action="store_true", help="Fetch the pinned public texts in the catalog")
    args = parser.parse_args()
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.texts.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    with sqlite3.connect(args.index) as conn:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS excerpts USING fts5(id UNINDEXED, title, doi, "
                     "url UNINDEXED, revision UNINDEXED, section UNINDEXED, body)")
        conn.execute("DELETE FROM excerpts")
        for source in sources:
            path = args.texts / (source["id"] + ".txt")
            if args.fetch and source.get("public_text") and not path.is_file():
                request = Request(source["url"], headers={"User-Agent": "GimnTools/research"})
                with urlopen(request, timeout=20) as response:
                    path.write_text(response.read().decode("utf-8"), encoding="utf-8")
            if not path.is_file():
                print(f"Local text unavailable for {source['id']}")
                continue
            paragraphs = re.split(r"\n[ \t]*\n", path.read_text(encoding="utf-8"))
            rows = [(source["id"], source["title"], source.get("doi", ""), source["url"],
                     source.get("revision", ""), number, " ".join(paragraph.split()))
                    for number, paragraph in enumerate(paragraphs, 1) if paragraph.strip()]
            conn.executemany("INSERT INTO excerpts VALUES (?,?,?,?,?,?,?)", rows)
            print(f"{source['id']} · {len(rows)} passages")


if __name__ == "__main__":
    main()
