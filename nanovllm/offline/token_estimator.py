"""Token estimation utilities for offline batch planning.

Provides approximate token counting when a tokenizer is not available,
and exact counting when one is.
"""
from typing import Optional


class TokenEstimator:
    """Estimate token counts for batch planning.

    Uses character-based heuristics for fast estimation, or a real
    tokenizer for exact counts.
    """

    # Average characters per token (English text ≈ 4, Chinese ≈ 1.5)
    CHARS_PER_TOKEN_EN = 4.0
    CHARS_PER_TOKEN_ZH = 1.5

    def __init__(self, tokenizer=None, chars_per_token: float = CHARS_PER_TOKEN_EN):
        self.tokenizer = tokenizer
        self.chars_per_token = chars_per_token

    def count_tokens(self, prompt: str | list[int]) -> int:
        """Count or estimate the number of tokens in a prompt."""
        if isinstance(prompt, list):
            return len(prompt)
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(prompt))
        # Heuristic: chars / avg_chars_per_token
        return max(1, int(len(prompt) / self.chars_per_token))

    def estimate_output_tokens(self, max_tokens: int, conservative_factor: float = 0.8) -> int:
        """Estimate actual output tokens (conservative: assume 80% of max)."""
        return max(1, int(max_tokens * conservative_factor))

    def estimate_total_cost(self, prompt: str | list[int], max_tokens: int,
                            conservative_factor: float = 0.8) -> int:
        """Estimate total tokens (prompt + output) for budget planning."""
        prompt_tokens = self.count_tokens(prompt)
        output_tokens = self.estimate_output_tokens(max_tokens, conservative_factor)
        return prompt_tokens + output_tokens
