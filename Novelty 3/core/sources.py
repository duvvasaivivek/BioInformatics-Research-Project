"""Pinned source acquisition. Importing this module performs no downloads."""

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .storage import ROOT, atomic_bytes, write_json

SOURCE_SPECS = (
    ("bbb", "theochem/B3DB", "75dab1cc607bb7a03f3de5c576cffd223e767844",
     "B3DB/B3DB_classification.tsv", "bbb.tsv", "BBB annotation", "CC0-1.0"),
    ("bbb_license", "theochem/B3DB", "75dab1cc607bb7a03f3de5c576cffd223e767844",
     "LICENSE", "bbb_LICENSE.txt", "source notice", "CC0-1.0"),
    ("pgp", "hwk-grp/SF-GA", "16ef1e1acbc617bc0de70e4cb657ab73a78879f8",
     "data/P_gp_subs.csv", "pgp_substrates.csv", "P-gp substrate classification", "not specified in inspected repository"),
    ("pgp_readme", "hwk-grp/SF-GA", "16ef1e1acbc617bc0de70e4cb657ab73a78879f8",
     "README.md", "pgp_README.md", "source documentation", "not specified in inspected repository"),
    ("metabolism_reference", "christinadebruynkops/GLORYx", "a1f4a8246e09eb971aac88d29123f59f2ad33944",
     "datasets/reference_dataset/gloryx_reference_dataset.json", "metabolism_reference.json", "documented metabolism relationships", "see metabolism_NOTICE.txt"),
    ("metabolism_test", "christinadebruynkops/GLORYx", "a1f4a8246e09eb971aac88d29123f59f2ad33944",
     "datasets/test_dataset/gloryx_test_dataset.json", "metabolism_test.json", "literature-curated drug metabolism relationships", "see metabolism source notices"),
    ("metabolism_notice", "christinadebruynkops/GLORYx", "a1f4a8246e09eb971aac88d29123f59f2ad33944",
     "datasets/reference_dataset/NOTICE", "metabolism_NOTICE.txt", "source notice", "source-specific terms"),
    ("metabolism_readme", "christinadebruynkops/GLORYx", "a1f4a8246e09eb971aac88d29123f59f2ad33944",
     "datasets/README.md", "metabolism_README.md", "source documentation", "source-specific terms"),
)


def digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def source_path(source_id: str):
    spec = next(spec for spec in SOURCE_SPECS if spec[0] == source_id)
    return ROOT / "data" / "raw" / spec[4]


def cached_source(path, receipt, url):
    if not path.exists() or not receipt.exists():
        return None
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if isinstance(record, dict) and record.get("url") == url and record.get("sha256") == digest(path.read_bytes()):
            return record
    except (OSError, ValueError):
        return None
    return None


def verified_sources():
    """Require every pinned source and verify its recorded checksum before use."""
    records = []
    for spec in SOURCE_SPECS:
        path = source_path(spec[0])
        receipt = path.with_suffix(path.suffix + ".source.json")
        if not path.exists() or not receipt.exists():
            raise ValueError(f"Missing source {spec[0]}. Run 'data fetch' first.")
        expected_url = f"https://raw.githubusercontent.com/{spec[1]}/{spec[2]}/{spec[3]}"
        record = cached_source(path, receipt, expected_url)
        if record is None:
            raise ValueError(f"Source verification failed for {path.name}. Run 'data fetch' to repair the snapshot.")
        records.append(record)
    return records


def fetch_sources():
    """Download atomically, preserving completed sources across interrupted runs."""
    records = []
    for source_id, repo, revision, upstream_path, filename, endpoint, license_name in SOURCE_SPECS:
        path = ROOT / "data" / "raw" / filename
        receipt = path.with_suffix(path.suffix + ".source.json")
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{upstream_path}"
        record = cached_source(path, receipt, url)
        if record is not None:
            print(f"[cached] {source_id}", flush=True)
            records.append(record)
            continue
        contents = None
        for attempt in range(1, 4):
            try:
                print(f"[fetch {attempt}/3] {source_id}", flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "Novelty3-data-audit/0.1"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = response.read(25 * 1024 * 1024 + 1)
                if not payload or len(payload) > 25 * 1024 * 1024 or payload.lstrip().lower().startswith(b"<!doctype html"):
                    raise ValueError(f"Unexpected download content for {source_id}")
                contents = payload
                break
            except (OSError, urllib.error.URLError) as error:
                if attempt == 3:
                    raise ValueError(f"Download failed for {source_id}: {error}. Rerun data fetch to continue.") from error
                time.sleep(attempt)
        record = {
            "id": source_id, "repository": repo, "revision": revision,
            "url": url, "filename": filename, "endpoint": endpoint,
            "license": license_name, "sha256": digest(contents), "bytes": len(contents),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_bytes(path, contents)
        write_json(receipt, record)
        records.append(record)
    write_json(ROOT / "data" / "raw" / "manifest.json", {"sources": records})
    return records
