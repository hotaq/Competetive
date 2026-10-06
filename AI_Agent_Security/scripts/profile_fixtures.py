"""Print a compact, non-sensitive profile of the public competition fixtures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "kaggle"
    / "sdk"
    / "aicomp_sdk"
    / "fixtures"
)

KEYWORDS = {
    "ignore_previous": ("ignore", "disregard", "forget all previous"),
    "secret": ("secret",),
    "http_or_upload": ("http", "upload", "post"),
    "delete": ("delete",),
    "email_or_send": ("email", "send"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def keyword_counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return {
        label: sum(
            any(keyword in row[field].lower() for keyword in keywords)
            for row in rows
        )
        for label, keywords in KEYWORDS.items()
    }


def profile_rows(
    rows: list[dict[str, str]],
    *,
    id_field: str,
    text_field: str,
) -> dict[str, Any]:
    lengths = [len(row[text_field]) for row in rows]
    values = [row[text_field] for row in rows]
    ids = [row[id_field] for row in rows]
    return {
        "rows": len(rows),
        "fields": sorted({key for row in rows for key in row}),
        "unique_ids": len(set(ids)),
        "duplicate_id_rows": len(ids) - len(set(ids)),
        "empty_text_rows": sum(not value for value in values),
        "unique_text_values": len(set(values)),
        "duplicate_text_rows": len(values) - len(set(values)),
        "text_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(mean(lengths), 2),
        },
        "keyword_rows": keyword_counts(rows, text_field),
    }


def main() -> None:
    mail = load_json(FIXTURES / "mail_seed.json")["inbox"]
    pages = load_json(FIXTURES / "web_corpus.json")["pages"]
    files = [path for path in (FIXTURES / "file_seed").iterdir() if path.is_file()]

    senders = Counter(message["from"] for message in mail)
    profile = {
        "email": {
            **profile_rows(mail, id_field="id", text_field="body"),
            "unique_senders": len(senders),
            "unique_subjects": len({message["subject"] for message in mail}),
            "top_sender_counts": dict(senders.most_common(10)),
        },
        "web": {
            **profile_rows(pages, id_field="id", text_field="content"),
            "unique_urls": len({page["url"] for page in pages}),
            "unique_titles": len({page["title"] for page in pages}),
        },
        "file_seed": {
            "files": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "empty_files": sorted(path.name for path in files if path.stat().st_size == 0),
            "filenames": sorted(path.name for path in files),
        },
    }
    print(json.dumps(profile, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

