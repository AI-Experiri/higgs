#!/usr/bin/env python3
"""Duplicate diagnostic-code guard for higgs.

Scans every `#[diagnostic(code(X))]` under `src/` and exits non-zero on
any collision, reporting every file:line offender. HG codes are higgs's
own — collisions mean two error variants share the same operator-facing
identifier, which breaks the "distinct code per failure" contract from
CLAUDE.md (task #36 DIAG).

Wired into `scripts/quality.sh`; also runnable standalone:
    python3 scripts/check_diag_codes.py

Line-based match — every `#[diagnostic(...)]` in higgs sits on its own
line, so a per-line regex is both faster and immune to the multi-line
`snafu(display("... \\ line 2 ..."))` string bodies that trip up naïve
full-file string scrubbing. Commented-out lines (`// #[diagnostic(...)`)
are skipped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# `#[diagnostic(code(NAME), …)]` — capture NAME. Tolerates extra args
# like `severity(Error)` after the code(...) pair.
LINE = re.compile(r"^\s*#\[diagnostic\(\s*code\(([A-Za-z][A-Za-z0-9_]*)\)")
COMMENT = re.compile(r"^\s*//")


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    hits: dict[str, list[str]] = {}
    for path in sorted((repo / "src").glob("**/*.rs")):
        rel = path.relative_to(repo)
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if COMMENT.match(line):
                continue
            m = LINE.match(line)
            if m:
                hits.setdefault(m.group(1), []).append(f"{rel}:{lineno}")

    dups = {c: locs for c, locs in sorted(hits.items()) if len(locs) > 1}
    if not dups:
        print(f"[diag-codes] no duplicates ({len(hits)} codes)")
        return 0

    for code, locs in dups.items():
        print(f"error: duplicate diagnostic code `{code}`:", file=sys.stderr)
        for loc in locs:
            print(f"        {loc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
