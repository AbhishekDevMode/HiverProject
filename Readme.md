# AI Customer Support Agent & Evaluation Pipeline

An offline-first, production-oriented take-home implementation for `@AmazonHelp` customer-support conversations. Run the complete benchmark with:

```bash
python -m pip install -r requirements.txt
python main.py
```

To force a credential-free run even when a key is configured, use `SUPPORT_AGENT_OFFLINE=1` (PowerShell: `$env:SUPPORT_AGENT_OFFLINE=1; python main.py`). This is useful for local tests and CI.

It completes in seconds in mock mode and normally well under 15 minutes with an API key. Set `GEMINI_API_KEY` to use Gemini structured JSON classification, retrieval-grounded drafting, and LLM-as-a-judge. Gemini is selected by default when present; set `LLM_PROVIDER=openai` plus `OPENAI_API_KEY` to use OpenAI instead. Without a provider key, all code paths run deterministically with a clearly marked heuristic judge proxy.

## Problem framing

“Good” support is correct, actionable, empathetic, concise, and safe: it must not solicit credentials or claim account changes it cannot perform. The pipeline identifies the request, produces a reply based on approved resolution-pattern templates, and routes uncertain, risky, or account-changing requests to a human.

Intentionally out of scope: live order lookup, refunds, address changes, authentication actions, customer identity verification, multilingual localization, and publishing to a real social channel. These require authenticated backend APIs, consent, audit trails, and operational ownership.

## Repository layout

| Path | Purpose |
|---|---|
| `src/ingest.py` | Filters parent tweets linked to `@AmazonHelp` replies in Kaggle `twcs.csv`, then stably hashes and samples 2,000 records. |
| `data/subsample.csv` | Generated reproducible subset; a synthetic fallback is used when Kaggle data is absent. |
| `src/agent.py` | Pydantic contract, structured classifier, TF-IDF retrieval over linked historical resolutions, safe reply policy, escalation policy. |
| `src/baselines.py` | Majority-class and intentionally weak keyword/zero-shot proxy baselines. |
| `src/eval_harness.py` | Classification metrics, judge scoring, and human/LLM agreement. |
| `data/sample_golden_set.json` | 200 schema-validated labelled examples, including 30 helpfulness calibration ratings. |

To use the actual Kaggle source, download `twcs.csv` from [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter), then run:

```bash
python -m src.ingest --source path/to/twcs.csv --output data/subsample.csv --n 2000
```

The Kaggle format contains both brand replies and parents. Ingestion selects customer parent tweets that elicited an `@AmazonHelp` response, preventing the model from learning to classify agent prose as customer requests. Stable SHA-256 sorting makes the exact subset reproducible across machines.

## Design

The typed output contract is `IntentResult`, `EscalationResult`, and `AgentResponse`. Eight mutually exclusive operational intents are used: Order Tracking, Delivery Delay, Refund/Billing, Account Auth, Technical Issue, General Inquiry, Missing Package, and Cancellation. Gemini mode requests JSON using the Pydantic-derived schema and validates it again locally. OpenAI is an optional compatible provider. An API error transparently falls back to the safe deterministic classifier rather than breaking a production batch.

The pipeline retains the linked historical `@AmazonHelp` reply for each sampled customer message (1,998 of 2,000 in the current sample). A local TF-IDF retriever selects the two closest historical resolutions and, when the live API is available, conditions a concise draft on those examples. Offline mode uses an intent-specific, safe template instead. Historical tweets are style and resolution evidence only: the prompt forbids requests for personal data and unsupported promises. In production, replace public tweets with versioned, policy-approved articles and internal citations.

Escalation is mandatory for confidence below 0.70, likely PII/credentials, frustration or sarcasm, and changes requiring a backend action. Replies direct customers to a secure channel and never request personal data in public.

## Evaluation

`main.py` evaluates every implementation on the same 200-record golden set and prints accuracy, macro precision/recall/F1 (available through the harness), escalation macro F1, and mean helpfulness. Macro F1 prevents a frequent intent from hiding failures in rare intents. The harness evaluates factuality/groundedness, tone/brand alignment, and helpfulness from 1–5 on 30 examples.

### Golden-set annotation protocol

The checked-in fixture is an executable synthetic bootstrap so that first-run and CI are deterministic. **It is not evidence of a completed human-labelled evaluation and must not be used as a take-home headline result.** For a final submission, sample 200 stratified, de-duplicated records from `data/subsample.csv`; remove handles, links, order references, and other identifying details; independently label intent, escalation, reference reply quality, and the three 1–5 judge dimensions. A second reviewer labels the 30-record calibration slice blind to model outputs; adjudicate disagreements and replace `human_helpfulness` with those ratings. Record the sampling seed, date, labeller initials, and disagreement rate in this README.

| System | Intent strategy | Reply strategy | Safety strategy |
|---|---|---|---|
| Trivial baseline | Always General Inquiry (majority proxy) | Generic Help-page reply | Never escalates |
| Simple baseline | Keyword zero-shot proxy | Generic reply | Never escalates |
| Headline system | Structured JSON / typed fallback | Historical-resolution retrieval + safe fallback | Four mandatory routing checks |

### What is misleading about my headline number?

Intent accuracy can look strong while confusing nearby operational states, such as a late package versus a package marked delivered. Macro F1 is still insensitive to the unequal cost of missing credential exposure or an urgent billing dispute. Synthetic golden examples have less ambiguity and messier language than actual social traffic, so they overestimate field performance. Escalation F1 treats false escalation and a dangerous missed escalation as equal even though they plainly are not. LLM judges share training biases with the generator, tend to favor polite and longer answers, and cannot verify a claim against an order backend. Kappa on only 30 samples has wide uncertainty and agreement is not correctness. Report slices, confidence intervals, and human review alongside any headline score.

## Failure analysis

| Input | Observed risk | Expected behavior | Root hypothesis |
|---|---|---|---|
| “Awesome, another ‘delivered’ package that vanished.” | May miss subtle sarcasm | Escalate missing-package claim | Keyword sentiment cannot understand implicit sarcasm. |
| “Can you fix the charge on my other order?” | May classify billing but lacks order context | Securely route to billing | Single-turn input omits the referenced order. |
| “My auth code is 123456.” | Regex catches some credential terms but not all secrets | Escalate and redact | PII detection is pattern-based, not a dedicated DLP model. |
| “Why did the courier leave it?” | Could choose delay instead of missing package | Ask a clarifying question | Taxonomy forces one intent before enough evidence exists. |
| “Cancel it, actually don’t.” | Cancellation pattern wins | Clarify before action | No dialogue-state or contradiction resolver. |

## Decision log

- Selected customer parent tweets, not brand replies, to preserve the incoming-message distribution.
- Used stable hashing rather than random sampling so benchmark deltas are attributable to code changes.
- Kept eight intents, enough to distinguish routing queues without creating fragile micro-labels.
- Used strict Pydantic validation at the API boundary to prevent malformed model output from leaking downstream.
- Made API failure fail safe to deterministic behavior, not fail closed for an offline take-home run.
- Set generation temperature to zero for repeatable classification and judge outcomes.
- Used approved resolution-pattern templates instead of claiming retrieval over untrusted public tweets.
- Escalated backend mutations even at high confidence because classification authority is not transaction authority.
- Treated public social support as a sensitive context and never requested order, card, password, or OTP values.
- Used macro metrics to surface weak minority intents rather than optimizing an aggregate majority score.
- Kept the judge sample at 30 to cap cost, while exposing agreement code so calibration is measurable.
- Labelled the no-key judge as a heuristic proxy to prevent it being mistaken for an LLM quality result.

## One-week roadmap

1. Build a stratified, double-annotated real golden set; measure adjudication rate and per-intent confidence intervals.
2. Add policy-approved knowledge retrieval with source IDs, freshness checks, and offline evaluation for grounded citations.
3. Fine-tune or distill a lightweight intent/triage SLM, calibrated with temperature scaling and abstention thresholds.
4. Add NeMo Guardrails or equivalent PII, jailbreak, toxicity, and promise-making checks before generation and before send.
5. Add conversation state, language identification, clarification questions, and deterministic action handoff payloads.
6. Batch evaluation asynchronously with caching, tracing, cost budgets, dashboards, and drift alerts by intent and language.
7. Run shadow mode with human agents, then A/B test only assistive drafts before any automatic send capability.

## Development notes

The application requires Python 3.10+. Tests can import `SupportAgent` directly; no network call is made unless `GEMINI_API_KEY` or `OPENAI_API_KEY` is present. Copy `.env.example` to `.env`, add `GEMINI_API_KEY`, and never commit a real secret. Do not commit Kaggle source data or secrets. The supplied synthetic golden data is intentionally non-sensitive and exists only to make the assignment runnable immediately; it is not human-labelled production evidence.
