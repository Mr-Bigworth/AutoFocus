# ==============================================================================
# Copyright (c) 2025 Tiange Luo, tiange.cs@gmail.com
#
# This code is licensed under the MIT License.
# ==============================================================================

"""
summarize_results.py – print ScreenSpot-Pro metric rows for one or more result files.

python summarize_results.py results/example.json
python summarize_results.py results/run1.json results/run2.json

"""

import json
import argparse
from pathlib import Path


KEYS = [
    "group:Dev", "group:Creative", "group:CAD",
    "group:Scientific", "group:Office", "group:OS",
]


def summarize_file(path: Path) -> None:
    with path.open() as f:
        data = json.load(f)

    pieces = []
    for key in KEYS:
        grp = data["metrics"]["leaderboard_simple_style"][key]
        pieces += [
            f"{grp['text_acc']   * 100:.1f}",
            f"{grp['icon_acc']   * 100:.1f}",
            f"{grp['action_acc'] * 100:.1f}",
        ]

    overall = data["metrics"]["overall"]
    pieces += [
        f"{overall['text_acc']   * 100:.1f}",
        f"{overall['icon_acc']   * 100:.1f}",
        f"{overall['action_acc'] * 100:.1f}",
    ]

    print(path, " & ".join(pieces), len(data["details"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ScreenSpot-Pro JSON results."
    )
    parser.add_argument(
        "files",
        metavar="FILE",
        nargs="+",
        help="Result JSON file(s), e.g. results/example.json",
    )
    args = parser.parse_args()

    for file_path in map(Path, args.files):
        summarize_file(file_path)


if __name__ == "__main__":
    main()