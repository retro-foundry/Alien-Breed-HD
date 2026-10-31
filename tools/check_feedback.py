#!/usr/bin/env python3
"""Run movement, settings, shooting, teleport, audio and HUD checks using Emscripten and Node."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emcc", default="emcc", help="emcc executable or emcc.py path")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = root / "build" / "feedback-tests"
    output.mkdir(parents=True, exist_ok=True)
    compiler = ([sys.executable, args.emcc] if args.emcc.endswith(".py")
                else [shutil.which(args.emcc) or args.emcc])
    node = shutil.which("node")
    if not node:
        parser.error("Node is required to execute the WebAssembly regression checks")
    for name, sources, flags in (
        ("pixels", [], ["-DAB3D_NATIVE_RGB565"]),
        ("movement", ["src/math_tables.c", "src/level.c"], []),
        ("settings", ["src/game_state.c", "src/game_data.c", "src/logging.c"], ["-sUSE_SDL=2"]),
        ("audio", ["src/audio_spatial.c"], ["-sUSE_SDL=2"]),
        ("enemy_audio", ["src/audio_spatial.c", "src/math_tables.c", "src/game_data.c", "src/level.c",
                         "src/movement.c", "src/visibility.c", "src/ai.c", "src/logging.c"],
         ["-sUSE_SDL=2"]),
        ("hud", [], ["-sUSE_SDL=2"]),
        ("shooting", ["src/math_tables.c", "src/game_data.c", "src/objects.c",
                      "src/level.c", "src/movement.c", "src/visibility.c"],
         ["-sUSE_SDL=2"]),
        ("shared_ammo", ["src/math_tables.c", "src/game_data.c", "src/game_state.c",
                         "src/objects.c", "src/level.c", "src/movement.c", "src/visibility.c"],
         ["-sUSE_SDL=2"]),
        ("teleport", ["src/math_tables.c", "src/game_data.c", "src/level.c",
                      "src/movement.c", "src/visibility.c", "src/sb_decompress.c", "src/logging.c"],
         ["-sUSE_SDL=2", "-sNODERAWFS=1"]),
        ("stairs", ["src/math_tables.c", "src/game_data.c", "src/level.c",
                    "src/movement.c", "src/visibility.c", "src/objects.c", "src/ai.c", "src/audio_spatial.c",
                    "src/sb_decompress.c", "src/logging.c"],
         ["-sUSE_SDL=2", "-sNODERAWFS=1"]),
        ("doors", ["src/game_data.c", "src/level.c", "src/sb_decompress.c", "src/logging.c"],
         ["-sUSE_SDL=2", "-sNODERAWFS=1", "-sALLOW_MEMORY_GROWTH=1"]),
        ("robot", ["src/math_tables.c", "src/game_data.c", "src/level.c", "src/movement.c",
                   "src/visibility.c", "src/ai.c", "src/audio_spatial.c", "src/sb_decompress.c",
                   "src/logging.c"], ["-sUSE_SDL=2", "-sNODERAWFS=1"]),
        ("corpses", ["src/math_tables.c", "src/game_data.c", "src/level.c", "src/movement.c",
                     "src/visibility.c", "src/ai.c", "src/audio_spatial.c", "src/sb_decompress.c",
                     "src/logging.c"], ["-sUSE_SDL=2", "-sNODERAWFS=1"]),
    ):
        target = output / f"{name}.js"
        subprocess.run(compiler + [f"tests/feedback_{name}.c"] + sources +
                       ["-Isrc", "-Ithird_party", "-O2", "-sENVIRONMENT=node", "-sEXIT_RUNTIME=1"] +
                       flags + ["-o", str(target)], cwd=root, check=True)
        subprocess.run([node, str(target)], cwd=root, check=True)


if __name__ == "__main__":
    main()
