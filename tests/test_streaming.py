"""Tests for incremental/streaming token counting
(pytokencalc/tokenizers/streaming.py).
"""

from pytokencalc.tokenizers.base import TokenCounter, TokenCountResult
from pytokencalc.tokenizers.openai_counter import OpenAITokenCounter
from pytokencalc.tokenizers.streaming import StreamingTokenCounter


class WordCountCounter(TokenCounter):
    """Minimal fake TokenCounter (1 token per whitespace-split word) so
    streaming-counter tests don't depend on tiktoken's real BPE boundaries
    to make assertions about the wrapper logic itself."""

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def supported_models(self) -> list:
        return ["fake-model"]

    def count(self, text: str, model: str) -> TokenCountResult:
        return TokenCountResult(
            input_tokens=len(text.split()),
            provider="fake",
            model=model,
        )

    def validate_model(self, model: str) -> bool:
        return model == "fake-model"


class TestStreamingTokenCounterWithFakeCounter:
    def test_add_chunk_returns_delta_not_running_total(self):
        stream = StreamingTokenCounter(WordCountCounter(), "fake-model")

        delta1 = stream.add_chunk("hello world")  # 2 words -> 2 tokens
        delta2 = stream.add_chunk(" foo bar baz")  # now 5 words total -> +3

        assert delta1 == 2
        assert delta2 == 3
        assert stream.total_tokens == 5

    def test_empty_chunk_contributes_zero(self):
        stream = StreamingTokenCounter(WordCountCounter(), "fake-model")
        stream.add_chunk("hello")
        assert stream.add_chunk("") == 0
        assert stream.total_tokens == 1

    def test_chunk_count_tracks_number_of_add_chunk_calls(self):
        stream = StreamingTokenCounter(WordCountCounter(), "fake-model")
        stream.add_chunk("a")
        stream.add_chunk("b")
        stream.add_chunk("")  # empty chunks still don't increment (nothing to add)
        assert stream.chunk_count == 2

    def test_reset_clears_state_for_a_new_response(self):
        stream = StreamingTokenCounter(WordCountCounter(), "fake-model")
        stream.add_chunk("hello world")
        stream.reset()

        assert stream.total_tokens == 0
        assert stream.accumulated_text == ""
        assert stream.chunk_count == 0

    def test_result_returns_full_token_count_result(self):
        stream = StreamingTokenCounter(WordCountCounter(), "fake-model")
        stream.add_chunk("one two three")

        result = stream.result()

        assert isinstance(result, TokenCountResult)
        assert result.input_tokens == 3
        assert result.provider == "fake"

    def test_token_counter_streaming_convenience_method(self):
        stream = WordCountCounter().streaming("fake-model")
        assert isinstance(stream, StreamingTokenCounter)
        stream.add_chunk("a b c")
        assert stream.total_tokens == 3


class TestStreamingTokenCounterHandlesBpeBoundaryCorrectly:
    """The whole point of recomputing on the full accumulated text (instead
    of summing independent per-chunk counts) is correctness across BPE merge
    boundaries -- this uses the real tiktoken-backed OpenAI counter to prove
    that actually holds, not just the fake word-count counter above."""

    def test_delta_sum_equals_whole_text_count(self):
        counter = OpenAITokenCounter()
        stream = StreamingTokenCounter(counter, "gpt-4o")

        chunks = ["The quick ", "brown fox ", "jumps over ", "the lazy dog."]
        total_delta = sum(stream.add_chunk(c) for c in chunks)

        whole_text = "".join(chunks)
        expected_total = counter.count(whole_text, "gpt-4o").input_tokens

        assert total_delta == expected_total
        assert stream.total_tokens == expected_total

    def test_splitting_mid_word_still_sums_correctly(self):
        """Chunk boundaries landing mid-word/mid-BPE-merge (as real SSE
        streaming deltas often do) must still sum to the correct total."""
        counter = OpenAITokenCounter()
        stream = StreamingTokenCounter(counter, "gpt-4o")

        # "internationalization" split at an arbitrary, non-word-boundary point.
        chunks = ["intern", "ationaliz", "ation is hard"]
        total_delta = sum(stream.add_chunk(c) for c in chunks)

        whole_text = "".join(chunks)
        expected_total = counter.count(whole_text, "gpt-4o").input_tokens

        assert total_delta == expected_total
