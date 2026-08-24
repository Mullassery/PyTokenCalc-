"""
PyTokenCalc v1.2.0: Universal Token Counting for ANY LLM

One library. One API. Every LLM provider. Custom models. BYOM support.

Features:
- 8 cloud providers (OpenAI, Anthropic, Google, Cohere, Azure, etc.)
- 7 local inference engines (Ollama, LM Studio, LocalAI, Llama.cpp, etc.)
- Custom provider registration (bring your own provider)
- BYOM: Bring your own model (fine-tuned, proprietary, custom)
- Model discovery (automatic provider suggestion)
- Platform-aware token counting (handles variations across platforms)
- Temporal variation tracking (session-based monitoring)
- Forward-compatible (new models work without code updates)
- 99%+ accuracy verified against official counters
- Intelligent caching: 70-80% fewer API calls
- StatGuardian quality validation (v1.0.0+)

Supported Providers:
- Cloud APIs: OpenAI GPT, Anthropic Claude, Google Gemini, Cohere Command, Azure OpenAI
- Cloud Services: RunPod, Together AI, Replicate, HuggingFace Inference, Llama Labs
- Local Inference: Ollama, LM Studio, LocalAI, Llama.cpp, GPT4All, Text Generation WebUI, Jan
- Custom: Any provider with an API endpoint, any model, any framework

Token Counting + Workflow Automation:
- Count tokens for any LLM (cloud, local, custom, BYOM)
- Intelligent routing (local when fast, API when accurate)
- Aggressive caching (70-80% fewer API calls)
- CLI + REST API for workflow automation (n8n, Power Automate, Temporal, Airflow)
- No external services or persistence required
- No configuration needed

Repository: https://github.com/Mullassery/PyTokenCalc
Documentation: https://github.com/Mullassery/PyTokenCalc/blob/main/README.md
Custom Providers: https://github.com/Mullassery/PyTokenCalc/blob/main/CUSTOM_PROVIDERS.md
"""

from ._version import __version__
__author__ = "Georgi Mammen Mullassery"

# Top-level ergonomic API (v1.1.0+): count_tokens() -> int, estimate_cost() -> USD
from .api import count_tokens, estimate_cost
from .pricing import get_model_pricing, PRICING_TABLE, PRICING_LAST_UPDATED

# Token counting (v0.7+: Multi-provider token counter)
try:
    from .tokenizers import (
        TokenCounter,
        TokenCountResult,
        TokenCounterRegistry,
        TokenCounterCache,
        get_global_registry,
    )
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    TokenCounter = None
    TokenCountResult = None
    TokenCounterRegistry = None
    TokenCounterCache = None
    get_global_registry = None

# Model discovery
from .model_discovery import ModelDiscovery

# Exceptions
from .exceptions import (
    PyTokenCalcError,
    ValidationError,
    TokenCountError,
    ModelNotSupportedError,
    APIError,
    CacheError,
)

# StatGuardian integration (quality validation, cost prediction, etc.)
from .statguardian_integration import (
    TokenCountValidation,
    TokenCountContract,
    ModelEfficiencyContract,
    CostPredictionContract,
    ProviderReliabilityContract,
    StatGuardianTokenCounter,
)

# OKF integration (token baselines, optimization)
from .okf_token_baselines import TokenBaseline, OKFTokenBaselines

# Server and CLI
from .server import PyTokenCalcServer
from .cli import CLIInterface
from .quick_cli import QuickCLI

# MCP 2.0 Support (v1.0.3+)
from pytokencalc._mcp_connector import TokenCalculator

__all__ = [
    # Top-level ergonomic API
    "count_tokens",
    "estimate_cost",
    "get_model_pricing",
    "PRICING_TABLE",
    "PRICING_LAST_UPDATED",
    # Token counting interface
    "TokenCounter",
    "TokenCountResult",
    "TokenCounterRegistry",
    "TokenCounterCache",
    "get_global_registry",
    # Model discovery
    "ModelDiscovery",
    # Exceptions
    "PyTokenCalcError",
    "ValidationError",
    "TokenCountError",
    "ModelNotSupportedError",
    "APIError",
    "CacheError",
    # StatGuardian integration
    "TokenCountValidation",
    "TokenCountContract",
    "ModelEfficiencyContract",
    "CostPredictionContract",
    "ProviderReliabilityContract",
    "StatGuardianTokenCounter",
    # OKF integration
    "TokenBaseline",
    "OKFTokenBaselines",
    # Server and CLI
    "PyTokenCalcServer",
    "CLIInterface",
    "QuickCLI",
    # MCP
    "TokenCalculator",
]
