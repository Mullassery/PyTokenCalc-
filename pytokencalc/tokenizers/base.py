"""
Abstract base class for token counters.
Each provider (OpenAI, Llama, Anthropic, etc.) has its own implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .streaming import StreamingTokenCounter


@dataclass
class TokenCountResult:
    """Result from token counting operation

    ⚠️  CRITICAL: Same model name on different platforms may have DIFFERENT token counts

    Root Causes:
    - Different tokenizers/tokenizer versions
    - Different quantization levels
    - Different model variants or versions
    - Different training data or fine-tuning
    - Different inference engines

    Real Example: "llama2-7b"
    ┌─────────────────────┬─────────┬──────────────┐
    │ Platform            │ Tokens  │ Latency      │
    ├─────────────────────┼─────────┼──────────────┤
    │ Ollama (local)      │ 1000    │ 5ms          │
    │ GCP Vertex AI       │ 998     │ 200ms        │
    │ Azure Container     │ 1002    │ 180ms        │
    │ HuggingFace         │ 1000    │ 50ms (cold)  │
    └─────────────────────┴─────────┴──────────────┘

    ⚠️  DON'T aggregate results from different platforms!
    ⚠️  DON'T assume same model = same token count!

    Always specify and track the platform (provider + source).
    Report results separately by platform, never combined.

    Example - CORRECT:
      "gpt-4: 100 tokens (OpenAI API)"
      "llama2: 105 tokens (Ollama local)"

    Example - WRONG:
      "llama2: avg 102 tokens" (mixes platforms!)
      "llama2: 105 tokens" (hides platform)

    Temporal Variations (Session-to-Session Changes):
    - Cloud providers may change infrastructure
    - Latency varies by time of day, load, region
    - Backend model updates can change token patterns
    - Token counts may shift between sessions
    - Same prompt at different times may yield different counts

    Track Over Time:
    - Use timestamp field to identify when count was made
    - Use session_id to group related counts
    - Don't assume consistency across time
    - Monitor latency trends
    - Alert if token counts diverge unexpectedly

    Example of temporal variation:
      Session 1 (July 17): "hello" = 2 tokens, 50ms
      Session 2 (July 18): "hello" = 2 tokens, 120ms (infrastructure change)
      Session 3 (July 19): "hello" = 3 tokens, 80ms (backend update)
    """
    input_tokens: int
    output_tokens: int = 0
    image_tokens: int = 0
    system_tokens: int = 0
    tool_tokens: int = 0

    # Metadata: Platform, source, and temporal tracking
    cached: bool = False  # True if from cache
    source: str = "local"  # "local", "api", "formula"
    latency_ms: float = 0.0
    provider: str = ""  # "openai", "anthropic", "ollama", "gcp", "azure", etc.
    model: str = ""  # Model ID/name
    platform: str = ""  # Platform variant (e.g., "ollama-local", "gcp-vertex", "azure-container")
    timestamp: datetime = field(default_factory=datetime.utcnow)  # When the count was made
    session_id: str = ""  # Session identifier (for tracking changes over time)

    @property
    def total_tokens(self) -> int:
        """Total tokens (all types)"""
        return (
            self.input_tokens +
            self.output_tokens +
            self.image_tokens +
            self.system_tokens +
            self.tool_tokens
        )


class TokenCounter(ABC):
    """Abstract base class for provider-specific token counters"""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic', 'huggingface')"""
        pass

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of supported models (patterns: 'gpt-4*', 'claude-*', 'llama-*')"""
        pass

    @abstractmethod
    def count(self, text: str, model: str) -> TokenCountResult:
        """
        Count tokens for text input.

        Args:
            text: Input text to tokenize
            model: Model ID (e.g., 'gpt-4o', 'claude-3-5-sonnet', 'llama-70b')

        Returns:
            TokenCountResult with input token count
        """
        pass

    def count_vision(
        self,
        text: str,
        model: str,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        num_images: int = 1
    ) -> TokenCountResult:
        """
        Count tokens for text + vision input.

        Override if provider supports vision tokenization.
        Default: return text tokens (no vision support).
        """
        return self.count(text, model)

    def count_system_prompt(self, system_prompt: str, model: str) -> int:
        """Count tokens in system prompt (optimization for repeated prompts)"""
        return self.count(system_prompt, model).input_tokens

    def count_tools(self, tools: List[Dict[str, Any]], model: str) -> int:
        """Count tokens in tool definitions"""
        # Convert tools to string representation
        tools_str = str(tools)  # Simple approach; can be refined
        return self.count(tools_str, model).input_tokens

    def count_batch(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[TokenCountResult]:
        """
        Count tokens for multiple requests (enables batch optimization).

        Each request: {"text": "...", "model": "...", "type": "text|vision"}
        """
        return [
            self.count(req["text"], req["model"])
            for req in requests
        ]

    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """Check if this counter supports the model"""
        pass

    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Return info about this tokenizer (version, vocabulary size, etc.)"""
        return {
            "provider": self.provider_name,
            "supported_models": self.supported_models,
        }

    def streaming(self, model: str) -> "StreamingTokenCounter":
        """Get an incremental counter for a live/streaming response with
        this provider's tokenizer. See `streaming.StreamingTokenCounter`.
        """
        from .streaming import StreamingTokenCounter

        return StreamingTokenCounter(self, model)
