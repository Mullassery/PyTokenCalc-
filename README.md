# PyTokenCalc

**Know your LLM costs before you hit send.**

Stop guessing tokens. PyTokenCalc counts tokens across OpenAI, Anthropic,
Google, Cohere, Azure-hosted OpenAI models, HuggingFace/open-source models, Ollama, and
custom endpoints, then estimates the dollar cost of a request from a
maintained per-model pricing table.

[![PyPI](https://img.shields.io/pypi/v/pytokencalc)](https://pypi.org/project/pytokencalc)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org)
[![CI](https://github.com/Mullassery/PyTokenCalc/actions/workflows/ci.yml/badge.svg)](https://github.com/Mullassery/PyTokenCalc/actions/workflows/ci.yml)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-blue.svg)](./LICENSE)

---

## 30-Second Start

```python
from pytokencalc import count_tokens, estimate_cost

# Count tokens instantly (runs locally via tiktoken -- no API key needed)
tokens = count_tokens("Tell me a story about a robot", model="gpt-4o")
print(f"Tokens: {tokens}")

# Estimate cost from the pricing table
cost = estimate_cost("gpt-4o", input_tokens=tokens)
print(f"Cost: ${cost:.6f}")
```

---

## Why PyTokenCalc?

**The Problem:**
- LLM costs are unpredictable (different models, different tokenizers)
- Manual calculation is error-prone
- No way to estimate before sending requests
- Each provider has different pricing

**The Solution:**
- One API across cloud, local, and custom providers
- Token counting that matches each provider's own tokenizer
- Real cost estimation from a maintained pricing table (see
  [docs/MODELS.md](docs/MODELS.md) for exactly which models are covered)

---

## Key Features

- **Multi-provider:** OpenAI, Anthropic Claude, Google Gemini, Cohere,
  Azure-hosted OpenAI models, HuggingFace/open-source models, Ollama, and
  any custom HTTP endpoint you register
- **Accurate tokenization:** uses each provider's own tokenizer/API
  (`tiktoken` for OpenAI and Azure-hosted OpenAI models, the HuggingFace
  `transformers` tokenizer, the live count-tokens endpoint for
  Anthropic/Google/Cohere)
- **Real cost estimation:** `estimate_cost()` is backed by a per-model USD
  pricing table (input vs. output rates), not a guess -- see
  [pytokencalc/pricing.py](pytokencalc/pricing.py) for sources and the
  last-updated date
- **Fast local counting:** sub-millisecond in our testing for the local
  (tiktoken/HuggingFace) providers on typical hardware -- see
  [Performance](#performance) below
- **Batch processing:** count tokens for many prompts in one call
- **Custom / BYOM models:** register your own provider or fine-tuned model
  (see [CUSTOM_PROVIDERS.md](CUSTOM_PROVIDERS.md))
- **Streaming/incremental counting:** `counter.streaming(model)` gives you a
  running token count as chunks of a live LLM response arrive, correct
  across chunk/BPE-merge boundaries (not a naive per-chunk sum)
- **Encoding drift detection:** OpenAI's tiktoken encoding resolution now
  prefers `tiktoken.encoding_for_model()` (stays current with tiktoken
  upgrades) and flags in `get_tokenizer_info()["drift_warnings"]` if an
  installed encoding's actual tokenization behavior has changed since this
  library was last verified against it

---

## Real-World Use Cases

*The examples below use Claude/GPT-4 interchangeably for illustration.
Anthropic/Google/Cohere models require the relevant package installed and
an API key set (see [docs/MODELS.md](docs/MODELS.md)); OpenAI/GPT-4 models
work offline out of the box.*

**Budget Tracking:**
```python
from pytokencalc import count_tokens, estimate_cost

prompt = "Hello"
reply_tokens_estimate = 50  # however you estimate expected output length

input_tokens = count_tokens(prompt, model="claude-3-opus")
cost = estimate_cost("claude-3-opus", input_tokens, reply_tokens_estimate)
print(f"Estimated cost: ${cost:.4f}")
```

**Prevent Overruns:**
```python
tokens = count_tokens(prompt, model="gpt-4")
if estimate_cost("gpt-4", tokens) > 0.10:
    print("Request too expensive, rejected")
```

**Compare Providers:**
```python
prompt = "Explain quantum computing in one paragraph."
for model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
    tokens = count_tokens(prompt, model=model)
    cost = estimate_cost(model, tokens)
    print(f"{model}: {tokens} tokens, ${cost:.6f}")
```

---

## Performance

Local (tiktoken/HuggingFace-backed) counting is fast -- in informal local
benchmarking, a single uncached `count_tokens()` call against `gpt-4o` on a
few dozen words took well under 1ms. PyTokenCalc itself is pure Python; the
speed comes from `tiktoken` (OpenAI and Azure-hosted OpenAI models) and the HuggingFace `tokenizers`
library, both of which have compiled (Rust) cores under their Python
bindings. There is no compiled/Rust code in PyTokenCalc itself. Anthropic,
Google, and Cohere counting makes a live network call to the provider's
API, so latency there is dominated by that round trip (with response
caching to avoid repeat calls for identical input).

---

## Installation

```bash
pip install pytokencalc
# or with uv
uv pip install pytokencalc

# with local tokenizer support (tiktoken + HuggingFace transformers)
pip install "pytokencalc[tokenizers]"
```

Requires Python 3.9+.

---

## Documentation

- [API Reference](docs/API.md) — top-level functions, the full registry API, CLI, and REST server
- [Supported Models](docs/MODELS.md) — provider list, offline vs. API-backed, pricing table coverage
- [Custom Providers](CUSTOM_PROVIDERS.md) — register your own endpoint or BYOM
- [Examples](examples/) — runnable code samples

---

## Known issues

- The "Performance" numbers above are informal, uncommitted local observations, not a checked-in benchmark result — there is no benchmark script or results file in this repo backing a specific number as fact.
- No open GitHub issues and no `TODO`/`FIXME` markers in `pytokencalc/` at the time of this writing.

## License

Proprietary License - Free to use with explicit attribution. See [LICENSE](LICENSE).

---

**PyTokenCalc v1.1.0** | Python 3.9+
