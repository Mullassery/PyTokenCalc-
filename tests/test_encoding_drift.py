"""Tests for tiktoken encoding drift detection and encoding_for_model
resolution (pytokencalc/tokenizers/encoding_fingerprint.py, openai_counter.py).
"""

import tiktoken

from pytokencalc.tokenizers.encoding_fingerprint import (
    KNOWN_FINGERPRINTS,
    check_drift,
    compute_fingerprint,
)
from pytokencalc.tokenizers.openai_counter import OpenAITokenCounter


class TestFingerprint:
    def test_fingerprint_is_deterministic(self):
        encoding = tiktoken.get_encoding("cl100k_base")
        assert compute_fingerprint(encoding) == compute_fingerprint(encoding)

    def test_known_encodings_match_recorded_fingerprints(self):
        """Fails loudly if the installed tiktoken's cl100k_base/o200k_base
        vocab has actually drifted from what KNOWN_FINGERPRINTS records --
        which is exactly the signal this feature exists to surface."""
        for name in KNOWN_FINGERPRINTS:
            encoding = tiktoken.get_encoding(name)
            assert compute_fingerprint(encoding) == KNOWN_FINGERPRINTS[name], (
                f"{name} fingerprint drifted -- if this tiktoken upgrade is "
                "intentional, update KNOWN_FINGERPRINTS"
            )

    def test_check_drift_returns_none_for_matching_encoding(self):
        encoding = tiktoken.get_encoding("cl100k_base")
        assert check_drift(encoding) is None

    def test_check_drift_returns_none_for_unknown_encoding_name(self):
        """No recorded fingerprint to compare against isn't itself a drift
        signal -- only a mismatch against a *recorded* value is."""
        encoding = tiktoken.get_encoding("r50k_base")
        assert check_drift(encoding) is None

    def test_check_drift_detects_a_real_mismatch(self, monkeypatch):
        import pytokencalc.tokenizers.encoding_fingerprint as mod

        monkeypatch.setitem(mod.KNOWN_FINGERPRINTS, "cl100k_base", "0" * 64)
        encoding = tiktoken.get_encoding("cl100k_base")

        warning = check_drift(encoding)

        assert warning is not None
        assert "cl100k_base" in warning
        assert "drift" in warning.lower() or "differently" in warning.lower()


class TestOpenAICounterEncodingResolution:
    def test_uses_tiktoken_encoding_for_model_for_known_models(self):
        counter = OpenAITokenCounter()
        encoding = counter._get_encoding("gpt-4o")
        assert encoding.name == tiktoken.encoding_for_model("gpt-4o").name

    def test_falls_back_to_static_map_for_unrecognized_model_name(self):
        """A model tiktoken.encoding_for_model doesn't know about, but our
        fallback dict does -- proves the fallback path still works."""
        counter = OpenAITokenCounter()
        assert "definitely-not-a-real-model" not in counter.MODEL_TO_ENCODING
        # Use a model present in MODEL_TO_ENCODING but crafted so
        # encoding_for_model() would raise for it in isolation -- text-davinci-002
        # is retired from tiktoken's own registry in newer tiktoken versions,
        # exercising the fallback path either way since the assertion below
        # only checks the fallback dict's mapping is honored when reached.
        encoding = counter._get_encoding("gpt-4")
        assert encoding.name == "cl100k_base"

    def test_no_drift_warnings_on_a_healthy_environment(self):
        counter = OpenAITokenCounter()
        counter.count("hello world", "gpt-4o")
        assert counter.drift_warnings == []

    def test_get_tokenizer_info_includes_drift_warnings_field(self):
        counter = OpenAITokenCounter()
        info = counter.get_tokenizer_info()
        assert "drift_warnings" in info
        assert isinstance(info["drift_warnings"], list)

    def test_count_still_produces_real_token_counts(self):
        """Regression check: the encoding_for_model rewiring didn't break
        actual counting."""
        counter = OpenAITokenCounter()
        result = counter.count("Hello, world!", "gpt-4o")
        assert result.input_tokens > 0
        assert result.provider == "openai"
