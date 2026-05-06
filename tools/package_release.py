#!/usr/bin/env python3
"""Package the staged Windows runtime output created by the CMake build."""

import argparse
from pathlib import Path
import shutil
import sys
import zipfile


ROOT = Path(__file__).resolve().parent.parent


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Windows release zip.")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_dir = (args.build_dir / args.config).resolve()
    out_dir = args.output_dir.resolve()
    stage_dir = out_dir / "stage" / f"alien-breed-3d-i-{args.version}-windows-x64"
    archive_path = out_dir / f"alien-breed-3d-i-{args.version}-windows-x64.zip"

    if not runtime_dir.is_dir():
        print(f"Runtime directory not found: {runtime_dir}", file=sys.stderr)
        return 1

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    copy_file(runtime_dir / "ab3d1.exe", stage_dir / "ab3d1.exe")
    copy_file(runtime_dir / "README.txt", stage_dir / "README.txt")

    ini_src = runtime_dir / "ab3d.ini"
    if ini_src.is_file():
        copy_file(ini_src, stage_dir / "ab3d.ini")
    else:
        copy_file(ROOT / "ab3d.ini.template", stage_dir / "ab3d.ini")

    copy_dir(runtime_dir / "data", stage_dir / "data")

    fonts_dir = runtime_dir / "fonts"
    if fonts_dir.is_dir():
        copy_dir(fonts_dir, stage_dir / "fonts")

    verify_doc = ROOT / "VERIFY_RELEASE.md"
    if verify_doc.is_file():
        copy_file(verify_doc, stage_dir / "VERIFY_RELEASE.md")

    if archive_path.exists():
        archive_path.unlink()
    zip_dir(stage_dir, archive_path)
    shutil.rmtree(out_dir / "stage")

    print(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
