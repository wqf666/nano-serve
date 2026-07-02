"""Workload generators for online LLM serving benchmarks.

Supported workloads:
  short:         64-256 input tokens,  64-256 output tokens
  medium:        256-1024 input tokens, 128-512 output tokens
  mixed:         70% short + 30% long (tests head-of-line blocking)
  shared-prefix: shared system prompt + variable questions (tests prefix cache)
  bursty:        burst arrival pattern (tests queuing and SLO violations)
"""
import random
from typing import NamedTuple

# Pool of English sentences (~10-20 tokens each) used to build prompts
_SENTENCES = [
    "The rapid advancement of artificial intelligence has transformed many industries.",
    "Machine learning models are becoming increasingly capable and efficient.",
    "Large language models can generate coherent text across many topics.",
    "Optimizing inference latency is critical for real-time applications.",
    "The key challenge in serving is balancing throughput and tail latency.",
    "GPU memory bandwidth often becomes the bottleneck in LLM inference.",
    "Continuous batching allows multiple requests to share GPU compute.",
    "Prefix caching can significantly reduce redundant prefill computation.",
    "Decode-first scheduling prioritizes stability of ongoing generation.",
    "Chunked prefill helps prevent long prompts from blocking short requests.",
    "SLO-aware scheduling aims to minimize violation rates under high load.",
    "Token-by-token streaming improves perceived latency for end users.",
    "Paged attention enables efficient memory management for KV cache.",
    "The trade-off between prefill and decode is fundamental to serving design.",
    "Benchmark harness must accurately measure first-token and inter-token latency.",
    "Online serving systems face dynamic and unpredictable request patterns.",
    "KV cache utilization directly impacts the maximum concurrent batch size.",
    "Preemption strategies help manage memory pressure during peak traffic.",
    "Tail latency metrics like P99 are more meaningful than averages.",
    "Request scheduling decisions happen at every engine step in continuous batching.",
    "Flash attention reduces memory usage from quadratic to linear in sequence length.",
    "Quantization techniques can reduce model memory footprint significantly.",
    "Speculative decoding can accelerate autoregressive generation speed.",
    "Multi-head attention enables parallel processing of different token positions.",
    "Transformer architectures use positional encoding to track token order.",
]

# Long shared system prompt (~600 tokens / ~2400 chars)
# Must fill at least 2 full blocks (block_size=256) for prefix cache to work.
_SHARED_SYSTEM_PROMPT = (
    "You are a knowledgeable AI assistant specialized in computer science, "
    "machine learning, and software engineering. You provide clear, concise, "
    "and accurate answers. You always explain technical concepts with examples "
    "and reference relevant research when appropriate. "
    + " ".join(_SENTENCES)  # ~350 tokens of shared context
    + " " + " ".join(_SENTENCES)  # repeat to reach ~700 tokens
    + " Now please answer the following question carefully."
)

_QUESTIONS = [
    "Explain the difference between prefill and decode in LLM inference.",
    "What is continuous batching and why is it important for serving?",
    "How does paged attention improve memory efficiency?",
    "Describe the KV cache and its role in autoregressive generation.",
    "What are the main causes of tail latency in LLM serving systems?",
    "Compare decode-first and prefill-first scheduling strategies.",
    "How does chunked prefill reduce head-of-line blocking?",
    "What metrics should be tracked for online LLM serving quality?",
    "Explain the concept of SLO-aware scheduling in inference systems.",
    "How does flash attention reduce memory complexity?",
    "What is speculative decoding and how does it work?",
    "Describe the trade-offs between throughput and latency in serving.",
    "How does prefix caching benefit shared-prompt workloads?",
    "What role does GPU memory bandwidth play in LLM inference speed?",
    "Explain how request preemption works under memory pressure.",
]


class WorkloadItem(NamedTuple):
    prompt: str
    max_tokens: int


def _build_prompt(target_tokens: int) -> str:
    """Build a prompt of approximately target_tokens tokens."""
    avg_tokens_per_sentence = 14
    num_sentences = max(1, target_tokens // avg_tokens_per_sentence)
    sentences = random.choices(_SENTENCES, k=num_sentences)
    prompt = " ".join(sentences)
    # Rough check: ~4 chars per token
    target_chars = target_tokens * 4
    while len(prompt) < target_chars:
        prompt += " " + random.choice(_SENTENCES)
    return prompt


def generate_workload(
    workload_type: str,
    num_requests: int,
    seed: int = 42,
) -> list[WorkloadItem]:
    """Generate a list of benchmark workload items."""
    rng = random.Random(seed)
    items = []

    for i in range(num_requests):
        if workload_type == "short":
            input_tokens = rng.randint(64, 256)
            output_tokens = rng.randint(64, 256)
            items.append(WorkloadItem(_build_prompt(input_tokens), output_tokens))

        elif workload_type == "medium":
            input_tokens = rng.randint(256, 1024)
            output_tokens = rng.randint(128, 512)
            items.append(WorkloadItem(_build_prompt(input_tokens), output_tokens))

        elif workload_type == "mixed":
            if rng.random() < 0.7:
                input_tokens = rng.randint(64, 256)
                output_tokens = rng.randint(64, 256)
            else:
                input_tokens = rng.randint(512, 1024)
                output_tokens = rng.randint(256, 512)
            items.append(WorkloadItem(_build_prompt(input_tokens), output_tokens))

        elif workload_type == "shared-prefix":
            question = _QUESTIONS[i % len(_QUESTIONS)]
            # Add variation so requests aren't identical
            extra = _SENTENCES[i % len(_SENTENCES)]
            prompt = f"{_SHARED_SYSTEM_PROMPT} {extra} Question: {question}"
            output_tokens = rng.randint(64, 256)
            items.append(WorkloadItem(prompt, output_tokens))

        elif workload_type == "bursty":
            # Mixed lengths, will be sent in bursty pattern by the bench client
            input_tokens = rng.randint(64, 512)
            output_tokens = rng.randint(64, 256)
            items.append(WorkloadItem(_build_prompt(input_tokens), output_tokens))

        else:
            raise ValueError(f"Unknown workload type: {workload_type}")

    return items
