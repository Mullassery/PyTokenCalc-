#!/usr/bin/env python3
"""Recompute tiktoken encoding fingerprints against the currently installed
tiktoken version and compare them to the ones recorded in
`pytokencalc/tokenizers/encoding_fingerprint.py`.

Run this after upgrading the pinned tiktoken version, or periodically in CI,
to detect real BPE/vocab drift instead of finding out from a user's
suddenly-different token counts. If a fingerprint has genuinely changed on
purpose (a real tiktoken upgrade you're accepting), update
`KNOWN_FINGERPRINTS` with the new value this script prints.

Usage:
    python scripts/verify_encoding_fingerprints.py
"""

import sys

import tiktoken

from pytokencalc.tokenizers.encoding_fingerprint import (
    KNOWN_FINGERPRINTS,
    compute_fingerprint,
)


def main() -> int:
    tiktoken_version = getattr(tiktoken, "__version__", "unknown")
    print(f"tiktoken version: {tiktoken_version}\n")

    drifted = []
    for name in KNOWN_FINGERPRINTS:
        encoding = tiktoken.get_encoding(name)
        actual = compute_fingerprint(encoding)
        expected = KNOWN_FINGERPRINTS[name]
        status = "OK" if actual == expected else "DRIFTED"
        print(f"{name}: {status}")
        print(f"  recorded: {expected}")
        print(f"  actual:   {actual}")
        if actual != expected:
            drifted.append(name)

    if drifted:
        print(f"\n{len(drifted)} encoding(s) drifted: {', '.join(drifted)}")
        print(
            "If this is an intentional tiktoken upgrade, copy the 'actual' values "
            "above into KNOWN_FINGERPRINTS in encoding_fingerprint.py."
        )
        return 1

    print("\nAll encoding fingerprints match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
