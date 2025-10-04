from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

# Default directories to skip when walking the repository
DEFAULT_EXCLUDE_DIRS: Set[str] = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "logs",
    "output",
    "mlruns",
    "catboost_info",
    ".mypy_cache",
    ".pytest_cache",
}

# Reasonable default file extensions considered as code
DEFAULT_EXTENSIONS: Set[str] = {
    ".py",
    ".sql",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
}


def should_skip_dir(dirname: str, exclude_dirs: Set[str]) -> bool:
    return dirname in exclude_dirs


def iter_files(
    root: Path, extensions: Set[str], exclude_dirs: Set[str]
) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # prune directories in-place to avoid walking them
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d, exclude_dirs)]
        for fname in filenames:
            path = Path(dirpath) / fname
            if not extensions:
                yield path
                continue
            if path.suffix.lower() in extensions:
                yield path


def count_file_lines(path: Path) -> int:
    try:
        with path.open("rb") as f:
            # Count by reading bytes to be robust to encoding issues
            return sum(1 for _ in f)
    except (OSError, PermissionError):
        return 0


def count_loc(
    root: Path, extensions: Set[str], exclude_dirs: Set[str]
) -> Tuple[int, Dict[str, int]]:
    total = 0
    by_ext: Dict[str, int] = {}
    for file in iter_files(root, extensions, exclude_dirs):
        n = count_file_lines(file)
        total += n
        ext = file.suffix.lower() or "<noext>"
        by_ext[ext] = by_ext.get(ext, 0) + n
    return total, dict(sorted(by_ext.items(), key=lambda kv: kv[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count lines of code in the project.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1]),
        help="Root path to scan (defaults to project root)",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=sorted(DEFAULT_EXTENSIONS),
        help="File extensions to include (e.g., --ext .py .sql). Empty means include all files.",
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDE_DIRS),
        help="Directory names to exclude from traversal.",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Show per-extension breakdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.path).resolve()
    extensions: Set[str] = (
        set(map(str.lower, args.ext)) if args.ext is not None else set()
    )
    exclude_dirs: Set[str] = set(args.exclude_dirs)

    total, by_ext = count_loc(root, extensions, exclude_dirs)

    print(f"Scanned root: {root}")
    if extensions:
        print(f"Included extensions: {sorted(extensions)}")
    else:
        print("Included extensions: ALL")
    print(f"Excluded directories: {sorted(exclude_dirs)}")
    print("")
    print(f"Total lines: {total}")

    if args.breakdown:
        print("\nBreakdown by extension:")
        for ext, n in by_ext.items():
            print(f"  {ext:8s} {n}")


if __name__ == "__main__":
    main()
