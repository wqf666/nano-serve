"""Offline workload generators for enterprise batch inference.

Supported workloads:
  short-batch:       Short prompts, short outputs (e.g. classification)
  long-doc:          Long prompts, medium outputs (e.g. summarization)
  shared-prefix:     Shared system prompt + variable questions (e.g. contract extraction)
  rag-batch:         RAG-style prompts with retrieved context
  code-batch:        Code generation / review prompts
  mixed-enterprise:  Mix of all above (realistic enterprise workload)
"""
import random
from typing import NamedTuple, Optional

# Sentence pool for building prompts (~10-20 tokens each)
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
]

# Shared system prompts for enterprise scenarios
_CONTRACT_SYSTEM_PROMPT = (
    "You are a legal document analysis AI. Your task is to extract key clauses, "
    "identify parties, dates, obligations, and penalties from contract documents. "
    "You must provide structured output with clear sections. "
    "Always identify: (1) Contract parties, (2) Effective date, (3) Key obligations, "
    "(4) Termination clauses, (5) Liability limitations, (6) Governing law. "
    + " ".join(_SENTENCES[:10])  # pad to ~250 tokens
    + " " + " ".join(_SENTENCES[:10])  # repeat to ~500 tokens
)

_INVOICE_SYSTEM_PROMPT = (
    "You are an invoice processing AI. Extract the following fields from invoices: "
    "vendor name, invoice number, date, line items, subtotal, tax, and total amount. "
    "Format the output as structured JSON. Validate that subtotal plus tax equals total. "
    "Flag any discrepancies or missing fields. "
    + " ".join(_SENTENCES[10:20])  # pad to ~250 tokens
    + " " + " ".join(_SENTENCES[10:20])  # repeat to ~500 tokens
)

_CODE_REVIEW_SYSTEM_PROMPT = (
    "You are a senior code reviewer. Analyze the provided code for: "
    "(1) Correctness and potential bugs, (2) Performance issues, "
    "(3) Security vulnerabilities, (4) Code style and readability, "
    "(5) Test coverage gaps. Provide specific line-by-line feedback. "
    + " ".join(_SENTENCES)  # pad to ~350 tokens
    + " " + " ".join(_SENTENCES)  # repeat to ~700 tokens
)

_RAG_SYSTEM_PROMPT = (
    "You are a retrieval-augmented generation assistant. Answer the user's question "
    "based ONLY on the provided context. If the context does not contain enough "
    "information, state that clearly. Do not fabricate information. "
    "Cite specific passages when possible. "
    + " ".join(_SENTENCES[:15])  # pad
    + " " + " ".join(_SENTENCES[:15])
)

_QUESTIONS = [
    "What are the termination conditions?",
    "Who are the contracting parties?",
    "What is the total amount due?",
    "List all line items and their costs.",
    "What is the governing law jurisdiction?",
    "Identify any penalty clauses.",
    "What is the effective date of this agreement?",
    "Summarize the key obligations.",
    "Are there any liability limitations?",
    "What are the payment terms?",
]

_CODE_SNIPPETS = [
    "def calculate_total(items):\n    total = 0\n    for item in items:\n        total += item.price * item.qty\n    return total",
    "async def fetch_data(url):\n    resp = await httpx.get(url)\n    return resp.json()",
    "class UserAuth:\n    def verify(self, token):\n        payload = jwt.decode(token, SECRET)\n        return User(payload['id'])",
    "def process_batch(records):\n    results = []\n    for r in records:\n        results.append(transform(r))\n    return results",
]


class OfflineWorkloadItem(NamedTuple):
    id: str
    prompt: str
    max_tokens: int
    prefix_key: Optional[str] = None
    metadata: dict = {}


def _build_prompt(target_tokens: int, rng: random.Random) -> str:
    """Build a prompt of approximately target_tokens tokens."""
    avg_tokens_per_sentence = 14
    num_sentences = max(1, target_tokens // avg_tokens_per_sentence)
    sentences = [rng.choice(_SENTENCES) for _ in range(num_sentences)]
    prompt = " ".join(sentences)
    target_chars = target_tokens * 4
    while len(prompt) < target_chars:
        prompt += " " + rng.choice(_SENTENCES)
    return prompt


def generate_offline_workload(
    workload_type: str,
    num_requests: int,
    seed: int = 42,
) -> list[OfflineWorkloadItem]:
    """Generate offline workload items."""
    rng = random.Random(seed)
    items = []

    for i in range(num_requests):
        req_id = f"{workload_type}_{i:04d}"

        if workload_type == "short-batch":
            input_tokens = rng.randint(32, 128)
            output_tokens = rng.randint(32, 128)
            prompt = _build_prompt(input_tokens, rng)
            items.append(OfflineWorkloadItem(req_id, prompt, output_tokens))

        elif workload_type == "long-doc":
            input_tokens = rng.randint(512, 1024)
            output_tokens = rng.randint(128, 512)
            prompt = _build_prompt(input_tokens, rng)
            items.append(OfflineWorkloadItem(req_id, prompt, output_tokens))

        elif workload_type == "shared-prefix":
            # Cycle through different enterprise prefix types
            prefix_type = i % 4
            if prefix_type == 0:
                system_prompt = _CONTRACT_SYSTEM_PROMPT
                prefix_key = "contract_extract"
                question = _QUESTIONS[i % len(_QUESTIONS)]
            elif prefix_type == 1:
                system_prompt = _INVOICE_SYSTEM_PROMPT
                prefix_key = "invoice_extract"
                question = _QUESTIONS[(i + 3) % len(_QUESTIONS)]
            elif prefix_type == 2:
                system_prompt = _CODE_REVIEW_SYSTEM_PROMPT
                prefix_key = "code_review"
                code = _CODE_SNIPPETS[i % len(_CODE_SNIPPETS)]
                question = f"Review this code:\n{code}"
            else:
                system_prompt = _RAG_SYSTEM_PROMPT
                prefix_key = "rag_answer"
                question = _QUESTIONS[(i + 5) % len(_QUESTIONS)]

            prompt = f"{system_prompt}\n\nQuestion: {question}"
            output_tokens = rng.randint(64, 256)
            items.append(OfflineWorkloadItem(
                req_id, prompt, output_tokens, prefix_key=prefix_key
            ))

        elif workload_type == "rag-batch":
            context = _build_prompt(rng.randint(200, 500), rng)
            question = _QUESTIONS[i % len(_QUESTIONS)]
            prompt = f"{_RAG_SYSTEM_PROMPT}\n\nContext: {context}\n\nQuestion: {question}"
            output_tokens = rng.randint(64, 256)
            items.append(OfflineWorkloadItem(
                req_id, prompt, output_tokens, prefix_key="rag_answer"
            ))

        elif workload_type == "code-batch":
            code = _CODE_SNIPPETS[i % len(_CODE_SNIPPETS)]
            prompt = f"{_CODE_REVIEW_SYSTEM_PROMPT}\n\nCode to review:\n{code}"
            output_tokens = rng.randint(64, 256)
            items.append(OfflineWorkloadItem(
                req_id, prompt, output_tokens, prefix_key="code_review"
            ))

        elif workload_type == "mixed-enterprise":
            category = rng.random()
            if category < 0.3:
                # Short classification-style
                input_tokens = rng.randint(32, 128)
                output_tokens = rng.randint(16, 64)
                prompt = _build_prompt(input_tokens, rng)
                items.append(OfflineWorkloadItem(req_id, prompt, output_tokens))
            elif category < 0.5:
                # Long document
                input_tokens = rng.randint(512, 1024)
                output_tokens = rng.randint(128, 512)
                prompt = _build_prompt(input_tokens, rng)
                items.append(OfflineWorkloadItem(req_id, prompt, output_tokens))
            elif category < 0.7:
                # Contract extraction (shared prefix)
                system_prompt = _CONTRACT_SYSTEM_PROMPT
                question = _QUESTIONS[i % len(_QUESTIONS)]
                prompt = f"{system_prompt}\n\nQuestion: {question}"
                output_tokens = rng.randint(64, 256)
                items.append(OfflineWorkloadItem(
                    req_id, prompt, output_tokens, prefix_key="contract_extract"
                ))
            elif category < 0.85:
                # RAG
                context = _build_prompt(rng.randint(200, 400), rng)
                question = _QUESTIONS[(i + 3) % len(_QUESTIONS)]
                prompt = f"{_RAG_SYSTEM_PROMPT}\n\nContext: {context}\n\nQuestion: {question}"
                output_tokens = rng.randint(64, 256)
                items.append(OfflineWorkloadItem(
                    req_id, prompt, output_tokens, prefix_key="rag_answer"
                ))
            else:
                # Code review
                code = _CODE_SNIPPETS[i % len(_CODE_SNIPPETS)]
                prompt = f"{_CODE_REVIEW_SYSTEM_PROMPT}\n\nCode to review:\n{code}"
                output_tokens = rng.randint(64, 256)
                items.append(OfflineWorkloadItem(
                    req_id, prompt, output_tokens, prefix_key="code_review"
                ))
        else:
            raise ValueError(f"Unknown workload type: {workload_type}")

    return items
