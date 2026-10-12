from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


REPO = "foersben/mvtec-ad"
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/main/bottle?recursive=true&expand=false&limit=1000"
RAW_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def fetch(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "SurfaceSight/1.0"})
            with urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after retries: {url}") from last_error


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    destination = project / "data" / "raw" / "mvtec_ad"
    destination.mkdir(parents=True, exist_ok=True)
    entries = json.loads(fetch(TREE_URL))
    paths = [
        item["path"]
        for item in entries
        if item.get("type") == "file"
        and (item["path"].startswith("bottle/train/good/") or item["path"].startswith("bottle/test/"))
        and item["path"].endswith(".png")
    ]
    paths.extend(["bottle/readme.txt", "bottle/license.txt"])
    for number, remote_path in enumerate(sorted(paths), start=1):
        local_path = destination / remote_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and local_path.stat().st_size > 0:
            continue
        local_path.write_bytes(fetch(RAW_BASE + remote_path))
        if number == 1 or number % 25 == 0 or number == len(paths):
            print(f"downloaded {number}/{len(paths)}: {remote_path}", flush=True)
    print(f"MVTec bottle files ready at {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
