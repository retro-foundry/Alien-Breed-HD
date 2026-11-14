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
    parser.add_argument("--only", nargs="+", help="Run only the named regression checks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = root / "build" / "feedback-tests"
    output.mkdir(parents=True, exist_ok=True)
    compiler = ([sys.executable, args.emcc] if args.emcc.endswith(".py")
                else [shutil.which(args.emcc) or args.emcc])
    node = shutil.which("node")
    if not node:
        parser.error("Node is required to execute the WebAssembly regression checks")
    cases = (
        ("pixels", [], ["-DAB3D_NATIVE_RGB565"]),
        ("movement", ["src/math_tables.c", "src/level.c"], []),
        ("wall_touch", [], []),
        ("points", ["src/math_tables.c", "src/level.c"], []),
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
        ("weapon_switch", ["src/game_data.c", "src/game_state.c", "src/renderer.c"],
         ["-DAB3D_NO_THREADS=1", "-sUSE_SDL=2"]),
        ("teleport", ["src/math_tables.c", "src/game_data.c", "src/level.c",
                      "src/movement.c", "src/visibility.c", "src/sb_decompress.c", "src/logging.c"],
         ["-sUSE_SDL=2", "-sNODERAWFS=1"]),
        ("stairs", ["src/math_tables.c", "src/game_data.c", "src/level.c",
                    "src/movement.c", "src/visibility.c", "src/objects.c", "src/ai.c", "src/audio_spatial.c",
                    "src/sb_decompress.c", "src/logging.c"],
         ["-sUSE_SDL=2", "-sNODERAWFS=1"]),
        ("doors", ["src/game_data.c", "src/level.c", "src/sb_decompress.c", "src/logging.c"],
         ["-sUSE_SDL=2", "-sNODERAWFS=1", "-sALLOW_MEMORY_GROWTH=1"]),
        ("door_render", ["src/objects.c", "src/game_data.c", "src/level.c",
                         "src/sb_decompress.c", "src/logging.c"],
         ["-DAB3D_NO_THREADS=1", "-sUSE_SDL=2", "-sNODERAWFS=1", "-sALLOW_MEMORY_GROWTH=1"]),
        ("water", ["src/renderer_3dobj.c", "src/renderer_objects.c", "src/math_tables.c", "src/level.c", "src/game_data.c",
                   "src/sb_decompress.c", "src/logging.c", "src/sprite_palettes.c"],
         ["-DAB3D_NO_THREADS=1", "-sUSE_SDL=2", "-sNODERAWFS=1", "-sALLOW_MEMORY_GROWTH=1"]),
        ("spill", ["src/renderer_objects.c", "src/io.c", "src/level.c", "src/objects.c",
                   "src/math_tables.c", "src/game_data.c", "src/sb_decompress.c", "src/logging.c"],
         ["-DAB3D_NO_THREADS=1", "-sUSE_SDL=2", "-sNODERAWFS=1", "-sALLOW_MEMORY_GROWTH=1"]),
        ("robot", ["src/math_tables.c", "src/game_data.c", "src/level.c", "src/movement.c",
                   "src/visibility.c", "src/ai.c", "src/audio_spatial.c", "src/sb_decompress.c",
                   "src/logging.c"], ["-sUSE_SDL=2", "-sNODERAWFS=1"]),
        ("corpses", ["src/math_tables.c", "src/game_data.c", "src/level.c", "src/movement.c",
                     "src/visibility.c", "src/ai.c", "src/audio_spatial.c", "src/sb_decompress.c",
                     "src/logging.c"], ["-sUSE_SDL=2", "-sNODERAWFS=1"]),
    )
    if args.only:
        unknown = set(args.only) - {name for name, _, _ in cases}
        if unknown:
            parser.error("Unknown regression check(s): " + ", ".join(sorted(unknown)))
    for name, sources, flags in cases:
        if args.only and name not in args.only:
            continue
        target = output / f"{name}.js"
        subprocess.run(compiler + [f"tests/feedback_{name}.c"] + sources +
                       ["-Isrc", "-Ithird_party", "-O2", "-sENVIRONMENT=node", "-sEXIT_RUNTIME=1"] +
                       flags + ["-o", str(target)], cwd=root, check=True)
        subprocess.run([node, str(target)], cwd=root, check=True)


if __name__ == "__main__":
    main()
