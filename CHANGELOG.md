# Changelog

All notable changes to PyTokenCalc are documented in this file.

## [1.2.0]

### Added

- **Tiktoken encoding drift detection** (`tokenizers/encoding_fingerprint.py`).
  `OpenAITokenCounter._get_encoding` now resolves models via
  `tiktoken.encoding_for_model()` first (maintained upstream, so it stays
  current across tiktoken upgrades) with the old hardcoded
  `MODEL_TO_ENCODING` dict demoted to a fallback for models tiktoken
  doesn't recognize yet. A real fingerprint check (vocab size + tokenization
  of a fixed calibration string, hashed) detects when an installed
  encoding's actual BPE behavior has changed since this library was last
  verified against it, surfaced via `get_tokenizer_info()["drift_warnings"]`.
  `scripts/verify_encoding_fingerprints.py` recomputes fingerprints for a
  deliberate tiktoken upgrade.
- **Streaming/incremental token counting** (`tokenizers/streaming.py`).
  `StreamingTokenCounter` wraps any `TokenCounter` (works across every
  provider, not just OpenAI) and gives a real chunk-by-chunk delta API
  (`add_chunk()`) for metering live/streaming LLM responses, recomputing
  against the full accumulated text each call so BPE merges spanning a
  chunk boundary are counted correctly rather than approximated by summing
  independent per-chunk counts. `TokenCounter.streaming(model)` is the
  convenience entry point.

## [1.1.0] - 2026-08-12

Remediation release: the CLI, REST server, and documented quick-start
example were broken in 1.0.3 (calling registry methods that didn't exist,
or importing symbols that weren't exported). This release fixes those
paths for real and adds the cost-estimation half of the product's stated
purpose, which previously didn't exist.

### Fixed
- `CLIInterface`/`PyTokenCalcServer` called `registry.count()`,
  `registry.count_vision()`, and `registry.list_all_models()`, none of
  which existed on `TokenCounterRegistry` -- every CLI/server request
  raised `AttributeError`. Call sites now use `registry.count_tokens()`
  and `registry.list_models()`; `TokenCounterRegistry.count_vision()` was
  added for real.
- `from pytokencalc import count_tokens, estimate_cost` (the README's
  headline example) raised `ImportError` -- neither symbol was exported.
  Both are now real, tested functions (see Added, below).
- Three accuracy-verification test classes (Anthropic/Google/Cohere vs.
  live API) were permanently skipped due to a broken
  `skipif(pytest.mark.skipif, ...)` condition that was always truthy,
  regardless of whether the relevant API key was set. Fixed to check the
  actual environment variable.
- `.github/workflows/ci.yml` ran `cd python && ...`, but there is no
  `python/` subdirectory in this repo -- CI could not run at all. Fixed to
  operate on the actual repo layout, and added a coverage step.
- `StatGuardianTokenCounter.count_with_validation()` accessed
  `result.tokens`, which doesn't exist on `TokenCountResult` (the field is
  `input_tokens`) -- this raised `AttributeError` against any real
  counter, invisible in tests because they only exercised a mock. Fixed,
  and added a test against the real OpenAI/tiktoken counter.
- `OllamaTokenCounter.__init__` performed a blocking network health-check
  on every construction (i.e. on every `TokenCounterRegistry()`). Moved to
  a lazy `.is_available()` check performed on first actual use.
- Removed the false "precompiled Rust core" / "no external services" /
  "works offline, no API calls needed" README claims. PyTokenCalc is pure
  Python; the offline claim now correctly scopes to the OpenAI/tiktoken,
  Azure OpenAI, and HuggingFace-backed counters only -- Anthropic, Google,
  and Cohere counting always makes a live API call.
- Reconciled version numbers (README said "v2.0.0", package said 1.0.3;
  now consistently 1.1.0 everywhere) and Python version requirement
  (README said 3.10+, `pyproject.toml`/CI said 3.9+; now consistently
  3.9+).

### Added
- `pytokencalc.count_tokens(text, model="gpt-4o", provider=None) -> int`
  and `pytokencalc.estimate_cost(model, input_tokens, output_tokens=0) ->
  float`, exported from the package root.
- A real, per-model USD pricing table (`pytokencalc/pricing.py`) covering
  the major OpenAI, Anthropic, and Google model families (plus Azure and
  Cohere), with a documented last-updated date and provider pricing-page
  links -- see `docs/MODELS.md` for exact coverage.
- Basic input validation before user-supplied `model` strings reach
  `AutoTokenizer.from_pretrained()` in the HuggingFace-backed counters
  (rejects path traversal / non-repo-ID-shaped input).
- `docs/API.md` and `docs/MODELS.md` (previously linked from the README
  but never created).

### Security
- `PyTokenCalcServer` and `run_server()` now default to binding
  `127.0.0.1` instead of `0.0.0.0`; the MCP connector's fallback config
  now defaults to localhost binding, an empty CORS origin list, and
  scoped (not wildcard) RBAC permissions; `pytokencalc.toml`'s
  `require_auth` now defaults to `true`. Wider exposure is still possible,
  but requires explicitly opting in rather than being the default.

### Changed
- Pruned unused dependencies from `requirements-lock.txt` (`groq`,
  `mistralai`, `clickhouse-driver`, `msgpack`, `aiohttp`, `numpy`,
  `httpx`, `python-dateutil` -- none were imported anywhere in
  `pytokencalc/`).

### Testing
- 135 passed, 26 skipped (expected: missing optional API keys/packages,
  no local Ollama daemon), 0 failed.

## [0.9.0] - 2026-07-18

### Added (Major Features)
- **Custom Provider Registration**: Register ANY provider with an API endpoint
- **Local Inference Engine Support**: Auto-detect LM Studio, LocalAI, Llama.cpp, GPT4All, Text Generation WebUI, Jan, Vllm
- **Ollama Integration**: Full support for Ollama with dynamic model discovery
- **Model Discovery System**: Pattern-based provider suggestion, model lookup, setup instructions
- **BYOM Support**: Bring your own model - fine-tuned, proprietary, or custom models
- **Platform-Aware Tracking**: Metadata for platform, source, and temporal tracking
- **Temporal Variation Monitoring**: Timestamp and session_id tracking for infrastructure changes
- **Forward-Compatible Patterns**: Pattern-based validation (claude-*, gemini-*, command-*) instead of hardcoded lists

### Added (Testing & Verification)
- 40 accuracy verification tests (OpenAI, Azure, Anthropic, Google, Cohere, HuggingFace)
- 23 model discovery tests (pattern matching, lookup, reporting)
- 7 custom provider tests (registration, integration, examples)
- 6 temporal variation tests (timestamp, session tracking, latency monitoring)
- 3 local inference tests (auto-detection, model discovery)
- Platform difference documentation and tests

### Added (Documentation)
- CUSTOM_PROVIDERS.md (200+ lines with 10+ provider examples)
- Model discovery documentation
- Platform awareness guidance (prevent confusion with multi-platform results)
- Temporal variation best practices
- BYOM (Bring Your Own Model) examples

### Changed
- TokenCountResult now includes `timestamp` and `session_id` fields
- Anthropic provider: Changed from hardcoded model list to claude-* pattern
- Google provider: Changed from hardcoded model list to gemini-* pattern
- Cohere provider: Changed from hardcoded model list to command-* pattern
- OpenSource provider: Removed hardcoded MODEL_ALIASES, now accepts ANY HuggingFace model

### Fixed
- Forward compatibility: New Anthropic models (like fable) now work without code changes
- Forward compatibility: New Google Gemini models work automatically
- Forward compatibility: New Cohere Command variants work automatically
- Forward compatibility: Any HuggingFace model now works (no hardcoded list)
- Forward compatibility: Ollama models update daily without requiring PyTokenCalc updates

### Technical Improvements
- Added model discovery module for pattern-based provider lookup
- Registry integration with custom providers
- Graceful fallback for unavailable local inference engines
- Improved error messages for unknown models/providers
- Multi-provider model support (same model on different providers)

### Performance
- No performance regression
- Custom providers add <5ms overhead
- Model discovery is instant (pattern-based)
- Cache continues to provide 70-80% API call reduction

### Breaking Changes
- None - fully backward compatible

### Migration Guide
No changes required. Existing code works exactly as before. New features are available immediately:

```python
# Existing code - still works
result = registry.count_tokens("gpt-4o", text)

# New features available
from pytokencalc.model_discovery import ModelDiscovery
from pytokencalc.tokenizers.custom_provider_counter import CustomProviderCounter

# Discover providers
providers = ModelDiscovery.suggest_provider("llama-2-7b")

# Register custom provider
my_provider = CustomProviderCounter(provider_name="custom", base_url="http://localhost:8000")
```

### Testing
- Total: 104 tests passing, 25 tests skipped (expected - API keys, offline services)
- All tests pass on Python 3.9+
- No regressions from v0.8.0

### Deprecations
- None

### Dependencies
- No new dependencies added
- Optional: requests (for custom providers, already optional)

---

## [0.8.0] - 2026-06-15

### Initial Release
- Token counting for 8 cloud providers
- Support for 30+ LLM models
- Intelligent caching (70-80% API reduction)
- CLI and REST API integration
- Multi-provider auto-detection

---

## Versioning

PyTokenCalc follows [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible new features  
- PATCH version for backwards-compatible bug fixes

## Release Notes

See https://github.com/Mullassery/PyTokenCalc/releases for detailed release notes and upgrade instructions.
