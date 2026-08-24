"""Incremental (streaming) token counting.

Every `TokenCounter` implementation (openai, anthropic, google, cohere,
azure_openai, huggingface, ollama, opensource, local_inference,
custom_provider) only exposed whole-input `count()` -- there was no way to
get a running token delta as chunks of a streaming LLM response arrive
without either re-summing whole-input counts yourself (which overcounts:
naively counting each chunk in isolation and adding them up ignores that
BPE tokenizers merge across word/whitespace boundaries, so "hello" + " world"
tokenized separately can produce a different total than "hello world"
tokenized together) or waiting for the full response.

`StreamingTokenCounter` wraps any `TokenCounter` and recomputes against the
full accumulated text on each chunk, returning the real delta. This works
for every provider counter uniformly since it only depends on the
`TokenCounter.count()` contract already implemented everywhere -- it isn't
provider-specific.
"""

from typing import TYPE_CHECKING

from .base import TokenCountResult

if TYPE_CHECKING:
    from .base import TokenCounter


class StreamingTokenCounter:
    """Incremental token counter for a single streaming response.

    Example:
        >>> counter = registry.get_counter("openai")
        >>> stream = StreamingTokenCounter(counter, model="gpt-4o")
        >>> for chunk in llm_stream:  # e.g. OpenAI SSE deltas
        ...     delta_tokens = stream.add_chunk(chunk.choices[0].delta.content or "")
        ...     running_cost = stream.total_tokens * price_per_token
        >>> final = stream.result()  # full TokenCountResult for the whole response
    """

    def __init__(self, counter: "TokenCounter", model: str):
        self.counter = counter
        self.model = model
        self.chunk_count = 0
        self._accumulated_text = ""
        self._cumulative_tokens = 0

    def add_chunk(self, chunk: str) -> int:
        """Append a streamed chunk and return the *additional* tokens it
        contributed (not the running total -- use `total_tokens` for that).

        Recomputes the full accumulated text's token count each call rather
        than counting `chunk` in isolation, so BPE merges spanning the
        previous chunk boundary are counted correctly.
        """
        if not chunk:
            return 0
        self._accumulated_text += chunk
        self.chunk_count += 1
        new_total = self.counter.count(self._accumulated_text, self.model).input_tokens
        delta = new_total - self._cumulative_tokens
        self._cumulative_tokens = new_total
        return delta

    @property
    def total_tokens(self) -> int:
        """Real cumulative token count of everything added so far."""
        return self._cumulative_tokens

    @property
    def accumulated_text(self) -> str:
        return self._accumulated_text

    def reset(self) -> None:
        """Start a new streaming response with the same counter/model."""
        self._accumulated_text = ""
        self._cumulative_tokens = 0
        self.chunk_count = 0

    def result(self) -> TokenCountResult:
        """Full `TokenCountResult` for everything accumulated so far
        (metadata like latency reflects this final recount, not a sum of
        per-chunk calls)."""
        return self.counter.count(self._accumulated_text, self.model)
