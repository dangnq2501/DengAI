"""Execute the course notebooks in order and save their outputs in place.

Usage (from the DengAI folder):  python notebooks/run_all.py [--kernel bkk] [01 03 ...]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("only", nargs="*", help="notebook prefixes to run, e.g. 04 05 (default: all)")
    parser.add_argument("--kernel", default="bkk")
    args = parser.parse_args()

    for path in sorted(NOTEBOOK_DIR.glob("0*.ipynb")):
        if args.only and not any(path.name.startswith(prefix) for prefix in args.only):
            continue
        start = time.time()
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook, timeout=3600, kernel_name=args.kernel, resources={"metadata": {"path": str(NOTEBOOK_DIR)}}
        ).execute()
        nbformat.write(notebook, path)
        print(f"{path.name}: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
