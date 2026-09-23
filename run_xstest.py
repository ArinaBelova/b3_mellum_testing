#!/usr/bin/env python3
"""Run XSTest benchmark against Mellum model.

XSTest measures exaggerated safety behaviors (over-refusal) in LLMs.
- safe subset: 250 prompts with sensitive keywords but benign intent → refusal_rate should be LOW
- unsafe subset: 200 genuinely harmful prompts → refusal_rate should be HIGH

Results are saved as .eval logs in logs/ and printed as a summary.

Prerequisites:
  - HuggingFace token with access to walledai/XSTest (gated dataset).
    Either run `huggingface-cli login` or add HF_TOKEN=hf_xxx to .env.
  - OPENAI_API_KEY in .env for the GPT-5.2 judge.
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# Verify HF auth early for a clear error message
try:
    from huggingface_hub import HfApi
    api = HfApi()
    api.dataset_info("walledai/XSTest")
except Exception as e:
    msg = str(e)
    if "gated" in msg.lower() or "401" in msg or "403" in msg or "authentication" in msg.lower():
        print(
            "ERROR: walledai/XSTest is a gated dataset. You need to:\n"
            "  1. Accept the terms at https://huggingface.co/datasets/walledai/XSTest\n"
            "  2. Run: huggingface-cli login\n"
            "     OR add HF_TOKEN=hf_xxx to your .env file",
            file=sys.stderr,
        )
        sys.exit(1)

from inspect_ai import eval  # noqa: E402
from inspect_evals.xstest import xstest  # noqa: E402

model = os.environ.get("INSPECT_EVAL_MODEL", "vllm/JetBrains/Mellum2-12B-A2.5B-Thinking")

# The Mellum 2 report (Section 5.3) uses GPT-5.2 as the judge for all
# LLM-as-a-Judge benchmarks, including XSTest. inspect_ai routes gpt-5.x through
# the Responses API and translates max_tokens -> max_completion_tokens itself.
JUDGE_MODEL = "openai/gpt-5.2"

print(f"Running XSTest with model: {model}")
print("This runs two tasks: safe subset and unsafe subset.")
print(f"Judge model: {JUDGE_MODEL} (requires OPENAI_API_KEY)\n")

results = eval(
    tasks=[
        xstest(subset="safe", scorer_model=JUDGE_MODEL),
        xstest(subset="unsafe", scorer_model=JUDGE_MODEL),
    ],
    model=model,
    log_dir="logs/",
)

for result in results:
    task_name = result.eval.task
    subset = result.eval.task_args.get("subset", "unknown")
    if result.results and result.results.metrics:
        for name, metric in result.results.metrics.items():
            print(f"[{subset}] {name}: {metric.value:.2f}")
    else:
        print(f"[{subset}] No metrics available.")
