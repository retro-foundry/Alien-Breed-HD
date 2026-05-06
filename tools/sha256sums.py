#!/usr/bin/env python3
"""Write a SHA256SUMS.txt file for release assets."""

import argparse
import hashlib
import glob
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHA256SUMS.txt.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("SHA256SUMS.txt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expanded = []
    for item in args.files:
        matches = glob.glob(str(item))
        if matches:
            expanded.extend(Path(match) for match in matches)
        else:
            expanded.append(item)
    files = sorted({p.resolve() for p in expanded if p.is_file()})
    if not files:
        print("No files matched for checksums.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for path in files:
            out.write(f"{sha256(path)}  {path.name}\n")

    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
