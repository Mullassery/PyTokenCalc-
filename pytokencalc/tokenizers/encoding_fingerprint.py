"""Detect when an installed tiktoken encoding's vocabulary/BPE rules have
drifted from what this library was last verified against.

`OpenAITokenCounter` used to hardcode a model->encoding-name map with no way
to tell if the *encoding itself* (the vocab tiktoken loads for that name)
had changed underneath it -- a tiktoken upgrade that altered `cl100k_base`'s
merge rules would silently change token counts with no signal anywhere.

This computes a fingerprint of an encoding (its vocab size plus the token
IDs it produces for a fixed calibration string covering ASCII, punctuation,
whitespace, and multi-byte Unicode) and compares it against the fingerprint
recorded when this module was last verified against a real tiktoken
install. A mismatch means the encoding's actual tokenization behavior has
changed -- not just a version number bump.
"""

from __future__ import annotations

import hashlib

# Deliberately covers plain ASCII, digits, punctuation, whitespace/tabs, and
# multi-byte Unicode (accented Latin + emoji) -- the token boundaries BPE
# merges produce for each of these categories are exactly what tends to
# shift between tiktoken/model updates.
CALIBRATION_TEXT = (
    'The quick brown fox jumps over the lazy dog. 12345 !@#$% "quoted" \n\ttab. '
    "Emoji: \U0001f680\U0001f525 café naïve résumé."
)

# Fingerprints recorded against real tiktoken installs at the time this file
# was written (tiktoken 0.14.0). Recompute and update these when
# intentionally re-verifying against a newer tiktoken release -- see
# `scripts/verify_encoding_fingerprints.py`.
KNOWN_FINGERPRINTS = {
    "cl100k_base": "accb00e8f0aee79305e9011001996aaab8ee4808c7aa0d9b6bbb3e996ef9722c",
    "o200k_base": "5022a0ae99d316f7e84efcb69710b2dfa567f3d84ea3ac47bf1f0d6985a7f9e9",
    "p50k_base": "7c4f162599d7596c0fc83a7d5c6c95801475d11a884b9a0841cf98e9bafec439",
}


def compute_fingerprint(encoding) -> str:
    """Hash of `encoding`'s vocab size plus how it tokenizes CALIBRATION_TEXT.
    Two tiktoken `Encoding` instances with identical BPE rules always
    produce the same fingerprint; any change to merge rules or vocab
    changes it."""
    token_ids = encoding.encode(CALIBRATION_TEXT, disallowed_special=())
    payload = f"{encoding.name}:{encoding.n_vocab}:{token_ids}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_drift(encoding) -> str | None:
    """Return a warning message if `encoding`'s live fingerprint doesn't
    match the recorded one for its name, `None` if it matches or if there's
    no recorded fingerprint to check against (an encoding this module has
    never been verified with -- not itself a drift signal)."""
    expected = KNOWN_FINGERPRINTS.get(encoding.name)
    if expected is None:
        return None
    actual = compute_fingerprint(encoding)
    if actual == expected:
        return None
    return (
        f"tiktoken encoding '{encoding.name}' tokenizes the calibration string "
        f"differently than when this was last verified (fingerprint {actual[:12]}... "
        f"vs recorded {expected[:12]}...) -- token counts for this encoding may no "
        "longer match what this library version was validated against. Re-verify "
        "with scripts/verify_encoding_fingerprints.py and, if this is an intentional "
        "tiktoken upgrade, update KNOWN_FINGERPRINTS."
    )
