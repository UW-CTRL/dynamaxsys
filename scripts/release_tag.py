#!/usr/bin/env python3
"""Create the next semantic-version Git tag.

Usage:
  python scripts/release_tag.py patch
  python scripts/release_tag.py minor --push
  python scripts/release_tag.py major --dry-run
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass


SEMVER_TAG_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> "Version | None":
        match = SEMVER_TAG_RE.match(value.strip())
        if not match:
            return None
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    def bump(self, part: str) -> "Version":
        if part == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        if part == "minor":
            return Version(self.major, self.minor + 1, 0)
        if part == "major":
            return Version(self.major + 1, 0, 0)
        raise ValueError(f"Unsupported bump part: {part}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def latest_semver_tag(prefix: str) -> tuple[Version, str] | None:
    tags_output = run(["git", "tag", "--list"])
    if not tags_output:
        return None

    candidates: list[tuple[Version, str]] = []
    for tag in tags_output.splitlines():
        if prefix and not tag.startswith(prefix):
            continue
        raw = tag[len(prefix) :] if prefix else tag
        parsed = Version.parse(raw)
        if parsed is None:
            continue
        candidates.append((parsed, tag))

    if not candidates:
        return None

    return max(
        candidates, key=lambda pair: (pair[0].major, pair[0].minor, pair[0].patch)
    )


def ensure_clean_tree() -> None:
    status = run(["git", "status", "--porcelain"])
    if status:
        print(
            "Error: working tree is not clean. Commit or stash changes before tagging.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the next semantic-version tag")
    parser.add_argument(
        "part", choices=["patch", "minor", "major"], help="Version part to bump"
    )
    parser.add_argument("--prefix", default="v", help="Tag prefix (default: v)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print next tag without creating it"
    )
    parser.add_argument(
        "--push", action="store_true", help="Push created tag to origin"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow creating tags with uncommitted changes",
    )
    args = parser.parse_args()

    if not args.allow_dirty:
        ensure_clean_tree()

    latest = latest_semver_tag(args.prefix)
    current_version = latest[0] if latest else Version(0, 0, 0)
    next_version = current_version.bump(args.part)
    next_tag = f"{args.prefix}{next_version}"

    if args.dry_run:
        print(next_tag)
        return 0

    existing_tags = run(["git", "tag", "--list", next_tag])
    if existing_tags:
        print(f"Error: tag {next_tag} already exists.", file=sys.stderr)
        return 1

    subprocess.run(["git", "tag", next_tag], check=True)
    print(f"Created tag: {next_tag}")

    if args.push:
        subprocess.run(["git", "push", "origin", next_tag], check=True)
        print(f"Pushed tag: {next_tag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
