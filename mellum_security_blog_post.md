# Putting Mellum Under Fire: Security Evaluation on B3

*Testing Mellum2-12B-A2.5B-Instruct and Mellum2-12B-A2.5B-Thinking on the B3 agentic security benchmark.*

---

## The Arms Race Never Ends

In the early days of email, spam filters were simple blocklists: ban the word "Viagra" and the problem goes away. Until attackers wrote "V1agr@." What followed was a decades-long arms race — Bayesian classifiers, neural networks, sender reputation systems, multi-layer pipelines. Security matured from a single check into a stack of independent defenses.

We are replaying that cycle with LLMs, just faster. Models are now embedded in agents with real capabilities: they browse the web, execute code, read documents, send emails. That makes them a compelling attack surface. An adversary who can inject instructions into content an agent processes might get it to exfiltrate data, generate exploit code, or take real-world actions nobody authorized. Security resilience is no longer a nice-to-have evaluation dimension — it belongs alongside accuracy, latency, and cost.

This post reports on the agentic security evaluation of two Mellum variants using the [B3 benchmark](https://arxiv.org/abs/2510.22620), supplemented with XSTest and HarmBench for baseline calibration.

---

## B3: Breaking Agent Backbones

B3 (Bazinska et al., ICLR 2026) was built to answer a specific question: how well does a backbone LLM resist adversarial instructions injected into its context? Its attacks were not invented by researchers — they were crowdsourced from over 194,000 real adversarial attempts submitted through Lakera's "Gandalf Agent Breaker Challenge," a public red-teaming exercise. The 210 highest-quality attacks were selected, validated across 7 LLMs, and organized into a structured benchmark.

The benchmark situates a model inside one of **ten realistic agentic scenarios**: a cycling coach, a financial investment advisor, a corporate messenger with email access, a mental health chatbot, a legal document assistant, and others. Each maps to one of **six attack categories**:

| Code | Attack type | What the attacker tries to do |
|------|-------------|-------------------------------|
| DIO | Direct Instruction Override | Directly commands the model to ignore its system prompt |
| IIO | Indirect Instruction Override | Hides override instructions in third-party content |
| DTI | Direct Tool Invocation | Directly requests an unauthorized tool call |
| ITI | Indirect Tool Invocation | Triggers a tool call via injected content |
| DCE | Data / Content Exfiltration | Extracts confidential context or system prompt |
| DAIS | Denial of AI Service | Degrades or disables the agent's intended behavior |

Each attack runs at **three difficulty levels**: L0 uses a minimal system prompt; L1 strengthens it with explicit safety instructions and longer benign context; L2 adds an LLM-as-judge defense — a second model instance that vets the first model's output before any action is taken. The attack success rate (ASR) ranges from 0 (fully secure) to 1 (fully compromised) and is averaged over five independent runs per scenario to account for generative stochasticity. The B3 leaderboard currently covers 54 evaluations of 34 LLMs.

Before diving into results, two standard benchmarks give a baseline picture of each model's static safety posture. **XSTest** measures whether a model gets benign-vs-harmful right: 250 safe prompts with sensitive-sounding keywords (over-refusal is bad) and 200 genuinely harmful prompts (not refusing is bad). **HarmBench** measures compliance with direct harmful instructions across 400 standard text behaviors in 7 categories — no agent context, no subtlety.

---

## The Models: Mellum2-12B

[Mellum](https://arxiv.org/abs/2510.05788) started as JetBrains' "focal model" — a 4B parameter code-completion specialist shipped in late 2024 and open-sourced in April 2025. The model evaluated here, [**Mellum2-12B-A2.5B**](https://arxiv.org/abs/2605.31268), is its general-purpose successor: a Mixture-of-Experts architecture with 12 billion total and 2.5 billion active parameters, capable of generating and editing code, calling tools, and planning multi-step agentic workflows.

Post-training follows two stages. First, supervised fine-tuning on a mixture that includes general chat, agentic coding, tool use, reasoning traces, and a **safety category** — "refusal and safe-response data drawn from a permissively licensed safety corpus, included to reduce harmful completions without degrading helpfulness on benign code prompts." Second, reinforcement learning via a GRPO/DAPO variant (RLVR), optimizing over code, math, instruction-following, and agentic tool-use tasks. The Mellum2 report acknowledges a known **"alignment tax"** from the RL stage: HarmBench ASR increases from 8.4% (SFT checkpoint) to 23.1% (RL checkpoint) for the Instruct variant — a regression the team flags as a target for future work.

Two RL-trained variants are compared in this evaluation (the publicly released HuggingFace checkpoints, produced by SFT followed by RLVR):

- **Instruct** — produces answers directly, no externalized chain of thought. Fast: 23.5 minutes and 1.6M output tokens for the full B3 run.
- **Thinking** — emits an internal chain of thought before its final answer. Expensive: 60.8 minutes and 7.7M output tokens — 4.75× more verbose. The `<think>…</think>` reasoning traces are stripped before B3 scoring.

Neither variant was trained with B3-style agentic attack scenarios in mind. The Mellum2 report includes no discussion of prompt injection, indirect instruction override, or agentic security — our evaluation is new ground on that front.

---

## Experimental Setup

All inference ran on a single **NVIDIA A100 80 GB GPU**. The Thinking model's extended reasoning traces make 80 GB the practical floor — the `--max-model-len 131072` context window fills quickly when the model thinks out loud.

Models were served locally with vLLM and evaluated through the [Inspect](https://inspect.ai) framework (UK AI Security Institute):

```bash
# Install
uv venv --python 3.13
uv pip install inspect-evals[b3] vllm
```

```bash
# Serve — Thinking variant requires a reasoning parser; Instruct does not
vllm serve JetBrains/Mellum2-12B-A2.5B-Thinking \
  --max-model-len 131072 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser hermes

vllm serve JetBrains/Mellum2-12B-A2.5B-Instruct \
  --max-model-len 131072 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

```bash
# Run B3 (set .env with VLLM_BASE_URL, VLLM_API_KEY, OPENAI_API_KEY first)
set -a; source .env; set +a
uv run inspect eval inspect_evals/b3 \
  --model vllm/JetBrains/Mellum2-12B-A2.5B-Thinking

# Resume a failed run without restarting from scratch:
uv run inspect eval-retry logs/<timestamp>_b3_<id>.eval
```

B3 uses **GPT-4.1-mini as a judge** for certain scoring steps, hence the OpenAI key. Each completed run is saved as a `.eval` log file; `uv run inspect view` opens an interactive web UI to browse individual samples. For the deeper per-sample analysis — score distributions, model comparisons, difficulty-level breakdowns — we used `b3_analysis.ipynb`, which streams sample metadata via `read_eval_log_sample_summaries` without loading full completions into memory.

---

## Baseline Safety: XSTest and HarmBench

Before entering the agentic arena, a quick sanity check on each model's static safety posture. Both evaluations use **GPT-5.2 as a judge**, consistent with the Mellum2 technical report. HarmBench generations and XSTest responses are capped at 8,192 output tokens — enough headroom for the Thinking model to complete its reasoning trace and emit a full answer before the judge sees the result.

| Metric | Thinking | Instruct |
|--------|:--------:|:--------:|
| XSTest — safe prompts, refusal rate ↓ better | **5.2%** | 7.6% |
| XSTest — unsafe prompts, refusal rate ↑ better | 88.0% | 85.5% |
| XSTest Balanced Accuracy ↑ better | **91.4%** | 89.0% |
| HarmBench ASR — overall ↓ safer | 17.5% | 16.5% |
| HarmBench ASR — copyright | 45.0% | 38.0% |
| HarmBench ASR — cybercrime / intrusion | 16.4% | 17.9% |

Both models are well-calibrated and close to identical on these benchmarks. Thinking's Balanced Accuracy (91.4%) edges out Instruct (89.0%) — it correctly handles slightly fewer safe prompts as harmful (5.2% false-refusal vs. 7.6%) while refusing genuinely harmful prompts at a similar rate (88.0% vs. 85.5%). On HarmBench the two models are essentially tied: 17.5% vs. 16.5% overall ASR, with no consistent winner across categories.

The Mellum2 report's RL-variant numbers are 23.1% HarmBench ASR (Instruct) and 20.6% (Thinking) — about 5 pp higher than ours on the same benchmark. Both we and the paper evaluated RL-trained models, so the most likely explanation is that the publicly released checkpoint is a more conservative RL variant than the internal one evaluated in the paper. For XSTest the gap runs in the same direction: our safe-prompt compliance is higher (92.4% / 94.8%) than the paper's (81.2% / 89.6%), partly because the XSTest framework injects `"You are a helpful assistant."` as a system message, which directly suppresses over-refusal and was likely absent in the paper's setup. The question is: *do these similar static safety profiles translate to similar agentic security?* B3 answers that.

---

## B3 Results: The Headline

| | Thinking | Instruct |
|---|:---:|:---:|
| Samples | 630 | 630 |
| Epochs | 5 | 5 |
| Duration | 60.8 min | 23.5 min |
| Output tokens | 7.7M | 1.6M |
| **Overall ASR ↓ safer** | **22.42%** | **23.69%** |

At the headline level, both models are statistically tied — a 1.3 percentage point gap across 3,150 evaluation runs each. That is the first notable result: Thinking's 4.75× inference cost buys essentially no aggregate security improvement.

Our evaluation used [`Lakera/b3-agent-security-benchmark-weak`](https://huggingface.co/datasets/Lakera/b3-agent-security-benchmark-weak) — the publicly available dataset, which contains 21 attacks per scenario per difficulty level (630 samples total). The B3 paper does not publish per-model ASR on this weak dataset: its leaderboard and all named model scores (e.g. kimi-k2-thinking at 0.34, Claude Haiku 4.5 at the top) are measured on the withheld best-7 attacks per cell (210 total). Those withheld attacks achieve ~56% average effectiveness versus ~18% for the public pool — roughly a 3× gap — so the two scales are not directly comparable. We can say that both Mellum variants sit above the 18% cross-model average on the public attack pool, but a fair head-to-head ranking against the 34 paper models would require running on the same withheld dataset via the B3 team's evaluation pipeline.

---

## Where the Two Models Actually Differ

The aggregate tie conceals directional tendencies at the scenario level. The table below shows mean ASR per (scenario, attack type), averaged across all 21 attack prompts and all 3 difficulty levels — one row per unique scenario/attack-type pair, no duplicates. Positive Δ = Instruct more vulnerable.

| Scenario | Type | Thinking | Instruct | Δ |
|----------|------|:--------:|:--------:|:---:|
| PortfolioIQ Advisor | IIO | 0.173 | 0.305 | +0.132 |
| Cycling Coach | DCE | 0.155 | 0.246 | +0.090 |
| Curs-ed CodeReview | IIO | 0.130 | 0.146 | +0.016 |
| OmniChat Desktop | IIO | 0.073 | 0.080 | +0.007 |
| Trippy Planner | IIO | 0.281 | 0.280 | −0.001 |
| MindfulChat | DAIS | 0.461 | 0.458 | −0.003 |
| Thingularity | DCE | 0.354 | 0.339 | −0.015 |
| Solace AI | DIO | 0.155 | 0.128 | −0.026 |
| CorpConnect Messenger | DTI | 0.317 | 0.286 | −0.031 |
| Clause AI | ITI | 0.143 | 0.101 | −0.042 |

The most striking feature of this table is how small the deltas are. Eight of ten scenario/attack-type pairs are within ±0.05 of each other. The two meaningful divergences are PortfolioIQ Advisor IIO (Δ +0.132, Thinking less vulnerable) and Clause AI ITI (Δ −0.042, Thinking slightly more vulnerable).

**Thinking handles indirect instruction injection better on structured analytical tasks.** PortfolioIQ Advisor embeds manipulation instructions inside financial documents, trying to get the model to recommend a specific company regardless of the underlying data. Thinking's CoT process appears to ground it in the analytical task structure, making injected side-instructions harder to act on — though the gap (17% vs 31% ASR) is meaningful but not dramatic.

**On indirect tool invocation, Thinking is marginally more exposed.** The Clause AI scenario asks the model to email confidential witness records via an indirect trigger. Thinking averages 14.3% ASR versus Instruct's 10.1%. The direction is consistent with CoT reasoning elaborating through tool-call logic toward compliance, but the effect is small.

**Both models fully fail on direct email tool invocation.** The CorpConnect Messenger scenario — attacker tricks the agent into sending "I quit" to all employees — produces ASR 1.0 for both models at Level 0–1. This is not a Mellum problem; it is an architectural one. Once a model legitimately holds email-sending capability, no backbone behavior reliably resists a well-framed direct request. Defense has to happen at the harness level.

**Difficulty level matters.** At Level 2, where an LLM-as-judge screens the output before acting, ASR drops sharply for both variants — and Thinking benefits more. The reasoning layer plus an external judge creates a more robust verification loop. This is the clearest data point for the value of harness-level defenses.

**Stochastic volatility is an underappreciated risk.** The same attack scenario, same model, can score 1.0 in one epoch and 0.0 in another. Both models show substantial per-sample standard deviation across their five runs. A determined attacker does not run an attack once — they retry. A security evaluation that reports a single-pass result significantly understates exposure.

---

## Defense Is a Stack, Not a Model

The CorpConnect result should be read as a design constraint, not just a benchmark finding: no backbone LLM should be trusted as the sole line of defense for irreversible actions. The data consistently points toward a layered approach.

The **LLM-as-judge defense at Level 2** is the clearest intervention this evaluation can demonstrate — it works for both models, and meaningfully so. Beyond that, a robust agentic security posture requires: a dedicated **prompt injection classifier** that screens third-party content before it reaches the backbone; **least-privilege tool scoping** so agents hold only the permissions the current task requires; **schema-validated tool calls** that allowlist targets rather than accepting arbitrary invocations; and **human confirmation gates** on any action that modifies external state irreversibly. Security testing itself should be multi-epoch — a single-pass evaluation gives a false floor.

The parallel to spam filtering closes here. Email security did not get good because mail clients got smarter at spotting bad words; it got good because security became a system property distributed across independent layers. LLM security is heading the same direction. The backbone model's refusal behavior is one layer. The harness around it is several more. Mellum, in its current form and without dedicated adversarial safety training, holds a competitive baseline on B3 — which is a reasonable starting point. But the PortfolioIQ and CorpConnect results are both clear reminders that no starting point is a finish line.

---

*B3 benchmark: Bazinska et al., ["Breaking Agent Backbones"](https://arxiv.org/abs/2510.22620), ICLR 2026 · Mellum technical report: [arXiv 2510.05788](https://arxiv.org/abs/2510.05788) · Mellum on Hugging Face: [JetBrains/Mellum2-12B-A2.5B-Instruct](https://huggingface.co/JetBrains)*
