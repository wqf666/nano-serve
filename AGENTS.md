# AGENTS.md

Project: NanoServe - online LLM serving and SLO-aware scheduling optimization based on nano-vLLM.

Hard rules:

1. Runtime environment is GPU server, project root is /root/autodl-tmp/nanoserve/repos/nanoserve.
2. Do NOT commit model weights, benchmark results, logs, .venv, __pycache__ to Git.
3. Do NOT rewrite the entire project at once; complete only the current phase task.
4. Preserve original nano-vLLM offline inference ability; example.py and bench.py must remain functional.
5. New code must have minimal smoke test or runnable command.
6. All metrics must be saved to JSON or CSV with stable field names.
7. All schedulers must integrate through unified BaseScheduler interface.
8. Online server must NOT call llm.generate directly per HTTP request; requests must enter unified AsyncEngineLoop.
9. Output must include: modified files, core design, run commands, verification method, potential risks.

Environment:
- GPU: NVIDIA GeForce RTX 4090D (24GB VRAM)
- Python: 3.10.8, venv at /root/autodl-tmp/nanoserve/envs/nanoserve
- HF_HOME: /root/autodl-tmp/nanoserve/models/hf_cache
- Logs: /root/autodl-tmp/nanoserve/logs/
- Results: /root/autodl-tmp/nanoserve/results/
- All long-running tasks must use tmux.
