#!/usr/bin/env python3
"""Build the browser ending harness using an existing single-threaded web build.

Serve the build directory and open ending-check.html. Browser automation can
call Module._feedback_finish_game(), wait for _feedback_at_title(), choose
New Game in the menu, and check _feedback_started_new_game().
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emcc", default="emcc", help="emcc executable or emcc.py path")
    parser.add_argument("--build-dir", default="build/emscripten-release")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    build = (root / args.build_dir).resolve()
    cache = build / "CMakeCache.txt"
    if not cache.exists() or "AB3D_NO_THREADS:BOOL=ON" not in cache.read_text():
        parser.error("Build the single-threaded Emscripten target first")
    sources = re.search(r"set\(SOURCES\s+(.*?)\n\)",
                        (root / "cmake/game_sources.cmake").read_text(), re.S).group(1).split()
    objects = [build / "CMakeFiles/ab3d1.dir" / (source + ".o")
               for source in sources if source not in ("src/main.c", "src/control_loop.c")]
    for obj in objects:
        if not obj.exists():
            parser.error(f"Missing web object {obj}; build the Emscripten target first")
    compiler = ([sys.executable, args.emcc] if args.emcc.endswith(".py")
                else [shutil.which(args.emcc) or args.emcc])
    target = build / "ending-check.html"
    subprocess.run(compiler + ["tests/feedback_ending.c"] + [str(obj) for obj in objects] + [
        "-Isrc", "-Ithird_party", "-O2", "-sUSE_SDL=2", "-sFULL_ES3=1",
        "-sMIN_WEBGL_VERSION=2", "-sMAX_WEBGL_VERSION=2", "-sALLOW_MEMORY_GROWTH=1",
        "-sINITIAL_MEMORY=134217728", "-sSHARED_MEMORY=0",
        "--preload-file", (build / "em_preload").as_posix() + "@/",
        "--shell-file", str(build / "emscripten_shell.html"), "-o", str(target)
    ], cwd=root, check=True)
    print(f"Built {target}")


if __name__ == "__main__":
    main()
