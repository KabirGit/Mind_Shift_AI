# Production GenAI architecture

## Scope and compatibility

This hardening slice extends the existing decision-support path without replacing reflection, safety, FAISS, SQLite, analytics, profiles, or the `/api/chat` contract. It adds no database, migration, runtime dependency, model reranker, MCP integration, LangGraph graph, multi-agent system, fine-tuning, Memory V2, OpenTelemetry stack, or frontend redesign. `Intent.md` remains unchanged and authoritative.

The resulting flow is:

```text
POST /api/chat
  -> deterministic crisis check, emotion/NLP, idempotent persistence, risk gate
  -> reflection: budget-independent legacy retrieval -> one prose model call
  -> safety: bounded safety response, no guidance tools or recommendations
  -> guidance: typed JSON decision call
       -> Pydantic validation or deterministic parse fallback
       -> at most three local read-only context tools
       -> 1,500-token estimated context pack
       -> deterministic option scoring, reversibility, and actions
       -> typed JSON response-draft call
       -> Pydantic validation and deterministic prose rendering or local fallback
  -> one request-local redacted JSONL trace
```

## 1. Typed Hugging Face model boundary

### What is it?

`LLMRequest` describes a logical call: purpose, separate system/user messages, prompt version, temperature, output format, and token limit. `LLMCallResult` reports text, success, requested and actual models, status, attempts, retries, latency, finish reason, token usage, and a sanitized failure category. `HuggingFaceInferenceClient` sends chat-completion requests for `arsoban/ocd-therapist-27b-v0.3` through Hugging Face Inference Providers, pinned to `featherless-ai`. `MistralInferenceClient` remains available behind `LLM_BACKEND=mistral` for compatibility.

### Why was it needed?

The old client hid provider failures behind a generic string. The pipeline could mistake that fallback for a successful completion and had no way to trace status, retry, model, or usage metadata.

### How does it work in this project?

The client fails a call immediately when no API key is configured, sends policy through the system message, uses separate connect/read timeouts, calls `raise_for_status()`, and validates the provider response shape. It retries once only for connection errors, timeouts, `429`, and `500/502/503/504`. Authentication, invalid requests, and other permanent `4xx` responses are not retried. Existing injected objects with only `generate(prompt)` still work through a compatibility adapter.

### Input → Processing → Output

```text
LLMRequest -> Hugging Face router attempt -> optional one transient retry
           -> response-shape validation -> LLMCallResult
```

### What would happen if we removed it?

Provider failures would again be indistinguishable from genuine model output, retry behavior would be unbounded or absent, and production traces could not explain degraded responses.

### What trade-off did we make?

One retry improves transient reliability but can increase worst-case latency. The cap of two attempts and separate timeouts bound that cost.

### Likely interview questions

- Why distinguish a logical LLM call from a provider attempt?
- Which errors are safe to retry, and why should most `4xx` responses fail fast?
- Why keep the previous provider behind a configuration switch?
- Why return failure metadata instead of a fallback string from the client?

## 2. Structured output and application validation

### What is it?

Decision extraction and the final guidance draft use provider JSON mode followed by Pydantic validation. `DecisionState` bounds the parsed problem, options, fears, constraints, goals, emotion, missing information, and uncertainty. `GuidanceResponseDraft` requires validation, recommendation, evidence explanation, uncertainty, and two to four actions.

### Why was it needed?

JSON syntax alone does not guarantee the application schema. The previous final answer check only looked for headings, so malformed or incomplete prose could appear valid.

### How does it work in this project?

Decision parsing uses `temperature=0` and JSON mode. Final generation also returns a compact JSON draft. The application parses and validates both structures, then renders the final headings deterministically. There is no repair-model loop: malformed output activates the existing deterministic parser or renderer.

### Input → Processing → Output

```text
untrusted message -> JSON-mode decision output -> Pydantic DecisionState
guidance analysis -> JSON-mode response draft -> Pydantic GuidanceResponseDraft
                  -> deterministic heading renderer
```

### What would happen if we removed it?

Downstream scoring could receive unbounded fields, and a response missing evidence or actions could pass a superficial prose check.

### What trade-off did we make?

Strict validation rejects some otherwise readable model answers. A deterministic fallback preserves availability without spending a third logical model call.

### Likely interview questions

- Why validate JSON mode output again?
- Why avoid a model-based repair loop?
- How do field limits reduce reliability and security risk?
- Why render prose after validation instead of asking directly for headings?

## 3. Retrieval diversity and prompt context policy

### What is it?

Guidance retrieval can exclude entry hashes and apply a deterministic relevance/diversity penalty. A `ContextPolicy` then packs prompt data under an estimated 1,500-token budget using four characters per token. It caps memories at three, patterns at three, goals at two, related decisions at two, and conversation turns at six.

### Why was it needed?

The current journal entry is stored before retrieval, and exact or near-duplicate memories could consume the small evidence window. Per-source limits alone did not guarantee a global prompt budget.

### How does it work in this project?

The retrieval query is normalized from the problem, user-stated options, and relevant goals. The current content hash is excluded before top-K selection. Exact normalized text and high token-overlap items are deduplicated. Allocation follows this priority: decision/options/constraints, retrieval evidence, matching patterns/goals, allow-listed profile summary, related prior decisions, and six recent turns. `retrieval_evidence` and `conversation_history` remain separate untrusted blocks. The selection report records counts only.

### Input → Processing → Output

```text
DecisionState + current hash -> FAISS candidates -> existing relevance ranking
 -> guidance-only diversity/exclusion -> typed evidence
 -> caps + dedup + global character budget -> PromptContext + count report
```

### What would happen if we removed it?

Self-matches and repeated journal entries could crowd out useful evidence, token usage would be unpredictable, and conversation text could be confused with retrieved evidence.

### What trade-off did we make?

The four-character estimate is conservative but not tokenizer-exact. It avoids a provider tokenizer dependency. Diversity can reorder close candidates, so it is opt-in for guidance and default retrieval behavior is preserved.

### Likely interview questions

- Why exclude the current message before top-K truncation?
- How is deterministic diversity different from claiming full vector MMR?
- Why prioritize decision constraints before conversation history?
- How would you calibrate the token-overlap threshold?

## 4. Bounded read-only agent

### What is it?

The guidance agent is a deterministic selector over four local tools: similar memories, emotional patterns, user profile/goals, and recent decision-like entries. It records selected tools, sanitized observations, accumulated evidence counts, remaining budget, and a termination reason.

### Why was it needed?

Previously tools swallowed failures, so empty failures appeared successful. The profile tool also called `update()`, which made a supposedly read-only tool persist data.

### How does it work in this project?

The agent always starts with similar memories. Fear or negative emotion selects emotional patterns second; otherwise it selects the persisted profile. A third tool is chosen only if evidence remains insufficient. Calls are allow-listed, deduplicated, local, narrowly typed, and capped unconditionally at three. Tools raise failures; the agent converts them to safe empty observations. Profile refresh, when needed, happens once in `RAGService` before the agent, while `get_user_profile` only calls `load()`.

Termination is one of `sufficient_context`, `max_calls`, or `no_additional_tool_needed`.

### Input → Processing → Output

```text
DecisionState -> deterministic tool choice -> read-only observation
 -> evidence-strength check -> optional next tool -> AgentResult
```

### What would happen if we removed it?

Every request would either fetch all context wastefully or lack a bounded, inspectable rule for selecting evidence. Hidden tool failures and accidental writes could return.

### What trade-off did we make?

Deterministic selection is less flexible than an LLM planner, but it is cheaper, easier to test, resistant to model-selected arbitrary tools, and guaranteed to terminate.

### Likely interview questions

- Why call this an agent if tool selection is deterministic?
- How do you enforce least privilege and termination?
- Why should the agent, rather than each tool, own failure degradation?
- What evidence threshold controls early stopping?

## 5. Deterministic scoring, reversibility, and actions

### What is it?

The guidance engine normalizes at most four options and scores goal alignment (30%), historical evidence (25%), feasibility (20%), reversibility (15%), and downside containment (10%). It creates concrete information-gathering or small-trial actions.

### Why was it needed?

Decision support should not let free-form model wording silently determine recommendations or encourage emotion-driven irreversible actions.

### How does it work in this project?

A strong recommendation requires a top score of at least `0.55`, a lead of at least `0.08`, and context evidence of at least `0.35`. Otherwise the result explicitly requests missing information. A negative emotion at confidence `0.70` or above blocks recommendation of an irreversible option and requires cooling-off/review.

### Input → Processing → Output

```text
DecisionState + DecisionContext -> bounded candidate options
 -> normalized factors -> weighted scores -> reversibility gate
 -> recommendation or insufficient evidence -> two to four actions
```

### What would happen if we removed it?

Recommendations would become model-dependent, thresholds would be untestable, and identical evidence could yield inconsistent safety behavior.

### What trade-off did we make?

Heuristic scores are interpretable and stable but not learned utility estimates. They organize evidence; they do not prove that one life choice is objectively best.

### Likely interview questions

- Why use a margin as well as an absolute score threshold?
- How do you avoid presenting heuristic scores as truth?
- Why treat reversibility separately from feasibility?
- What cases intentionally return insufficient evidence?

## 6. Request-local structured tracing and log redaction

### What is it?

One `TraceRecord` is accumulated per request and appended to the existing JSONL file. It includes route, requested/actual models, prompt versions, logical calls, provider attempts/retries, available token usage, tool outcomes, retrieval and context counts, agent termination, stage latencies, failure category, final status, `latency_ms`, and compatibility `elapsed_ms`.

### Why was it needed?

The old latency log could not distinguish transport retry from another logical prompt, explain degraded output, or identify the failing stage. Existing stage logs could include raw journal, prompt, memory, or response content.

### How does it work in this project?

`RAGService` owns a request-local accumulator and records deterministic stage timings. Model and agent results contribute metadata only. Recursive stage sanitization redacts content-bearing keys. JSONL append failure returns false and never breaks the user request.

Example redacted record:

```json
{
  "trace_id": "8c3f...",
  "mode": "guidance",
  "prompt_versions": ["decision-parse-v2", "guidance-response-v2"],
  "logical_llm_calls": 2,
  "provider_attempts": 3,
  "retry_count": 1,
  "token_usage": {"prompt_tokens": 522, "completion_tokens": 181, "total_tokens": 703},
  "tools_called": ["search_similar_memories", "get_emotional_patterns"],
  "memories_retrieved": 2,
  "agent_steps": 2,
  "agent_termination_reason": "sufficient_context",
  "context_selection": {"included": {"memories": 2}, "estimated_tokens": 934, "budget_tokens": 1500},
  "stage_latencies_ms": {"decision_parsing": 410.2, "context_agent": 19.6, "final_generation": 522.1, "total": 1018.4},
  "failure_category": null,
  "status": "success",
  "latency_ms": 1018.4,
  "elapsed_ms": 1018.4
}
```

### What would happen if we removed it?

Operations could see latency but not whether it came from retrieval, the provider, retry, or parsing, and private text could leak into developer logs.

### What trade-off did we make?

JSONL is intentionally simple and local. It does not provide distributed trace correlation, querying, retention policies, or remote dashboards.

### Likely interview questions

- Why is trace state request-local rather than stored on the service?
- How do you prove logs do not contain raw private text?
- What is the difference between `degraded`, `blocked`, and `failed`?
- Why preserve `elapsed_ms`?

## 7. Golden production-pipeline evaluation

### What is it?

A fixed 15-case fixture validates the actual `RAGService.run_pipeline()` path using deterministic injected model and retrieval doubles. It covers the original eight routing cases plus relevant retrieval, duplicate suppression, schema failure, retry recovery, permanent failure, unnecessary-tool prevention, and max-call termination.

### Why was it needed?

The previous evaluator duplicated routing logic, so both production and evaluator could share the same wrong assumption without detecting divergence.

### How does it work in this project?

`evaluate_pipeline()` accepts a case runner and only validates returned production outputs. Each fixture case can specify route, required and forbidden tools, maximum calls, expected and forbidden memory IDs, guidance contract, expected model calls, trace status, provider attempts/retries, and termination reason. Assertions are deterministic; there is no LLM judge or RAGAS dependency.

### Input → Processing → Output

```text
golden case -> RAGService with injected deterministic doubles -> API-shaped output + trace
 -> route/tool/retrieval/contract/safety assertions -> pass/failure list
```

### What would happen if we removed it?

Provider fallback, tool termination, retrieval diversity, and route regressions could pass unit tests while failing in their combined production sequence.

### What trade-off did we make?

Deterministic doubles test orchestration and contracts, not hosted-model response quality in the wild. Live-provider evaluation remains a separate operational concern.

### Likely interview questions

- Why avoid an LLM-as-judge for this slice?
- Which properties belong in golden fixtures versus unit tests?
- How does dependency injection keep pipeline tests deterministic?
- What does the suite not measure?

## End-to-end example: negative feedback, resign versus request feedback

User message:

> I received negative feedback and feel afraid. Should I resign now or ask my manager for clearer feedback? I need stable income.

1. Safety does not identify crisis or a protected high-risk decision. Emotion/NLP detects fear and career context; persistence keeps its existing content-hash/SQLite upsert behavior.
2. Conservative decision cues trigger one JSON-mode parsing call. `DecisionState` contains the two user-stated options, stable-income constraint, fear, and missing facts such as whether the criticism is specific and actionable.
3. The agent searches similar memories, then chooses emotional patterns because fear is present. It stops early if evidence reaches the threshold; otherwise it uses one remaining allowed tool.
4. The context pack excludes the just-stored message, suppresses near duplicates, keeps retrieval separate from conversation turns, and reports only selection counts to tracing.
5. Scoring compares the two options plus only generic reversible alternatives. Immediate resignation is costly to reverse and, with fear confidence at or above `0.70`, cannot receive a strong recommendation. Requesting concrete examples or running a short feedback trial contains downside.
6. The final model receives the deterministic analysis as untrusted structured context and returns a validated JSON draft. The application renders the response headings.

Before the vertical slice, the same message had only a generic reflection shape:

```text
It sounds like the feedback was painful and left you uncertain. What feelings come up when you think about work? Try to be gentle with yourself.
```

After hardening, the response has both grounded prose and machine-readable guidance:

```text
Validation:
Negative feedback can make an immediate exit feel urgent, especially when stable income matters.

Recommendation:
There is not enough verified evidence to justify resigning now. Gather specific feedback and review it after a cooling-off period.

Why:
- Asking for examples is reversible and tests whether the situation can improve.
- Immediate resignation has a higher reversal cost and conflicts with the stated income constraint.

Uncertainty:
- The specificity of the feedback and the manager's willingness to support improvement are unknown.

Next steps:
1. Ask for two concrete examples and the expected standard.
2. Write down the income and timing constraints before making an irreversible move.
3. Set a review date after a short feedback trial.
```

The structured result separately exposes candidate options, factor scores, reversibility assessments, recommendation confidence, evidence, uncertainties, and ordered actions. It never claims that a past decision succeeded because the journal schema does not store verified outcomes.

## Security and reliability boundaries

- User messages, memories, analytics, profiles, and conversation history are untrusted data blocks; policy is carried in the provider system message.
- Model output is advisory text only and does not directly trigger persistence, URL access, filesystem access, code execution, or arbitrary tool selection.
- Guidance tools are local and read-only; persistent profile refresh is explicit pipeline work.
- Safety precedes decision parsing. High-risk requests receive no guidance model calls, scores, options, or tools.
- Content hashes prevent duplicate vector entries, and SQLite primary-key upsert prevents duplicate journal records.
- No raw journal, memory, prompt, response, or retrieved content is written to structured traces.

## Known limitations

- Token estimation uses characters rather than the served model's exact tokenizer.
- The selected model is an OCD-support fine-tune and has not been clinically validated; deterministic safety routing remains authoritative, and the model must not replace professional care.
- Deterministic relevance/diversity is not a learned reranker and can only work with the quality of the underlying embeddings.
- The journal schema does not record verified decision outcomes, so historical evidence must not imply causal success.
- Deterministic eval doubles validate control flow and contracts, not live model quality or provider drift.
- JSONL tracing is local observability, not a distributed telemetry platform.
