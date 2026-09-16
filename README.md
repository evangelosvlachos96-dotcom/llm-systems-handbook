# Applied AI Engineering Handbook

**A practical guide to agents, RAG, evaluation, data quality, and LLM infrastructure.**

**Author:** Evangelos Vlachos

Building an AI demo takes an afternoon. Building an AI system that works reliably, that you can measure, debug, and improve, is a different discipline. This handbook collects the core concepts behind that discipline in one place, with formulas, worked examples, and short Python snippets.

It's written for software engineers, solution architects, and data practitioners moving into applied AI, and for anyone preparing for roles such as AI engineer, forward deployed engineer, or ML platform engineer.

**How to use it.** Each section is self-contained and ends with "Check your understanding" questions. If you can answer them out loud in a couple of minutes, you've got the core ideas.

**Suggested reading order:** Section 0 (foundations) → 2 (RAG) → 1 (agents) → 3 (evaluation) → 4 (data quality) → 5 (post-training) → 6 (infrastructure) → 7 (Python patterns).

> Code examples use the Anthropic Python SDK, but the concepts apply to any LLM provider. Model names, prices, and library versions change quickly, so check current documentation before using snippets in production.

## Contents

0. [Foundations](#0-foundations-you-need-first)
1. [Agentic and Multi-Turn Systems](#1-agentic-and-multi-turn-systems)
2. [RAG (Retrieval-Augmented Generation)](#2-rag-retrieval-augmented-generation)
3. [Evaluation Harnesses](#3-evaluation-harnesses)
4. [Data Taxonomy, Labeling, and Quality](#4-data-taxonomy-labeling-and-quality)
5. [ML Pipelines and Post-Training](#5-ml-pipelines-and-post-training)
6. [Inference and Infrastructure](#6-inference-and-infrastructure)
7. [Production Python Patterns](#7-production-python-patterns)
8. [Key Takeaways](#8-key-takeaways)
9. [Further Reading](#9-further-reading)

---

## 0. Foundations You Need First

**Tokens.** LLMs read and write tokens, not words. A token is roughly ¾ of an English word. Pricing, latency, and context limits are all measured in tokens.

**Context window.** The maximum number of tokens (input + output) the model can process in one call. The model has no memory outside it. Every "memory" feature in an application is really the app deciding what to put back into the context.

**Stateless API calls.** Each API call is independent. For a multi-turn conversation, you resend the entire history every time. This is why long conversations get slow and expensive, and why context management matters.

**Temperature.** Controls randomness in sampling. Low (0–0.3) for extraction, classification, and judging; higher for creative generation. Even at temperature 0, outputs are not guaranteed identical across runs, so evals must run multiple trials.

**Embeddings.** A model that turns text into a vector of numbers (e.g. 768 or 1536 dimensions) such that texts with similar meaning land close together. Similarity is usually measured with **cosine similarity**:

```
cos(a, b) = (a · b) / (‖a‖ ‖b‖)      range: -1 to 1, higher = more similar
```

If vectors are normalized to length 1, cosine similarity equals the dot product, which is faster to compute.

**Structured output.** Asking the model to return JSON matching a schema. In production, always validate it (e.g. with Pydantic) and handle failures, because models occasionally return malformed output.

---

## 1. Agentic and Multi-Turn Systems

### 1.1 What makes something an agent

An **agent** is an LLM that pursues a goal by repeatedly deciding on actions, executing them through tools, observing results, and continuing until done. The model controls the flow. A **workflow**, by contrast, is a fixed sequence of LLM calls where your code controls the flow.

A useful rule from Anthropic's "Building Effective Agents" guidance: **use the simplest thing that works.** Many problems need a workflow, not an agent. Agents trade predictability and cost for flexibility.

### 1.2 The agent loop

```
while not done:
    response = llm(messages, tools)
    if response wants to call a tool:
        result = execute_tool(name, args)
        messages.append(tool call + result)
    else:
        done = True   # model produced a final answer
```

A concrete version with the Anthropic Python SDK:

```python
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5"

tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]

def run_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        return f"Sunny, 24°C in {args['city']}"
    return f"Error: unknown tool {name}"

def agent(user_msg: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": user_msg}]
    for _ in range(max_steps):                      # hard step limit
        resp = client.messages.create(
            model=MODEL, max_tokens=1024, tools=tools, messages=messages
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                try:
                    out = run_tool(block.name, block.input)
                except Exception as e:
                    out = f"Tool error: {e}"         # return errors to the model
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": out,
                })
        messages.append({"role": "user", "content": results})
    return "Stopped: step limit reached"
```

Notice three production habits already in this small example: a step limit, tool errors returned to the model instead of crashing, and the full history passed each turn.

### 1.3 Tool design

Tools are the agent's interface to the world, and bad tools cause most agent failures. Good tools have clear names and descriptions written as if for a new colleague, tight input schemas (enums instead of free strings where possible), helpful error messages that tell the model how to fix its call, and outputs that are concise (don't dump a 50,000-token file when a summary or pagination will do). Fewer, well-designed tools beat many overlapping ones, because overlapping tools confuse the model about which to use.

### 1.4 Workflow and orchestration patterns

| Pattern | How it works | Use when |
|---|---|---|
| Prompt chaining | Output of step 1 feeds step 2, with checks between | Task splits cleanly into fixed steps |
| Routing | Classifier sends input to a specialized prompt or model | Distinct input categories need different handling |
| Parallelization | Run sub-tasks simultaneously, or run the same task several times and vote | Independent subtasks, or you need confidence |
| Orchestrator–workers | A lead LLM breaks down the task and delegates to sub-agents | Subtasks can't be predicted in advance |
| Evaluator–optimizer | One LLM generates, another critiques, loop until good | Clear quality criteria exist |
| Autonomous agent | Single LLM with tools in a loop | Open-ended tasks with unpredictable steps |

**Multi-agent tradeoff:** sub-agents give each worker a clean, focused context, which helps on broad research-style tasks. The costs are more tokens, harder debugging, and coordination failures (agents duplicating work or losing information in handoffs).

### 1.5 State, memory, and context management

Multi-turn agents run into context limits and "context rot" (quality degrades as context fills with irrelevant history). Techniques:

- **Short-term state:** the message history itself, plus structured state your app tracks (e.g. a task list or current file).
- **Compaction/summarization:** when history grows large, summarize older turns and replace them.
- **Tool result clearing:** drop or truncate old, bulky tool outputs that are no longer needed.
- **External memory:** write notes or facts to a file or database and retrieve them when relevant (this is RAG applied to the agent's own history).
- **Sub-agent isolation:** delegate a messy exploration to a sub-agent and only return its conclusion.

Rule of thumb: *"Context is the agent's entire world. Investigate most agent failures first as context problems, meaning missing, stale, or noisy information, before assuming the model can't reason."*

### 1.6 Human-in-the-loop (HITL)

Put humans at points where errors are costly or irreversible: before sending emails, spending money, deleting data, or deploying code. Common designs are approval gates (agent pauses and asks), confidence thresholds (low-confidence outputs routed to a reviewer), and sampled review (a percentage of completed runs audited). HITL outputs also become valuable training and evaluation data.

### 1.7 MCP (Model Context Protocol)

An open protocol for connecting LLM applications to tools and data sources. An **MCP server** exposes tools, resources, and prompts; an **MCP client** (inside the agent application) connects to servers and passes their tools to the model. The value is standardization: build a tool integration once and any MCP-compatible agent can use it. Agent evaluation environments often expose apps like docs, email, and spreadsheets as MCP servers inside sandboxes.

### 1.8 Taking agents to production

The gap between demo and production is reliability. Key practices:

- **Observability:** log every step of every trajectory (prompts, tool calls, results, tokens, latency). Use tracing tools such as Langfuse, LangSmith, or OpenTelemetry-based setups.
- **Guardrails:** validate tool inputs, restrict permissions to the minimum needed, sandbox code execution.
- **Termination:** step limits, token budgets, timeouts, and loop detection (same tool called with same args repeatedly).
- **Idempotency:** tools that can be safely retried without double-charging or double-sending.
- **Evals as regression tests:** every prompt or model change runs against a fixed suite before deploy.

### 1.9 Common agent failure modes

Claiming actions it didn't take; infinite loops or repeated tool calls; drifting from original instructions over long tasks; ignoring tool errors and continuing as if they succeeded; fabricating data when a file or result is missing; stopping early and declaring success; misusing tools due to poor descriptions; prompt injection from content it reads (web pages, documents).

### Check your understanding

1. When would you choose a workflow over an agent?
2. Walk through the agent loop, including how tool errors are handled.
3. Name three ways to manage context in a long-running agent.
4. What are the tradeoffs of a multi-agent architecture?
5. What is MCP and why does it matter?

---

## 2. RAG (Retrieval-Augmented Generation)

### 2.1 Why RAG

LLMs don't know your private or recent data and they hallucinate when they don't know. RAG retrieves relevant documents at query time and places them in the prompt, so the model answers from evidence. It's usually cheaper and easier to update than fine-tuning, and it enables citations.

### 2.2 The pipeline

```
INGESTION (offline)                         QUERY (online)
documents                                   user question
  → parse/clean                               → (optional) rewrite query
  → chunk                                     → embed query + keyword search
  → add metadata                              → retrieve top-k candidates
  → embed chunks                              → rerank → keep top-n
  → store in vector index + keyword index     → build prompt with chunks
                                              → generate answer with citations
```

### 2.3 Parsing

Garbage in, garbage out. PDFs with tables, headers, and multi-column layouts often parse badly. Always inspect parsed output manually before worrying about embeddings. Preserve structure (headings, table rows) where possible, because it helps chunking and adds useful metadata.

### 2.4 Chunking

Chunks are the unit of retrieval. Too small and chunks lose context ("it increased by 3%" with no subject). Too large and retrieval gets noisy and the prompt fills with irrelevant text.

| Strategy | Description | Notes |
|---|---|---|
| Fixed-size | Split every N tokens | Simple baseline; cuts sentences awkwardly |
| Recursive | Split on paragraphs, then sentences, then words, until under size limit | Good general default |
| Structure-aware | Split on document structure (headings, sections, code functions) | Best for docs, markdown, code |
| Semantic | Split where embedding similarity between sentences drops | More compute; sometimes better coherence |

Common starting points: roughly 300–800 tokens per chunk with 10–20% overlap, then tune using retrieval evals. Overlap reduces the chance that a key fact is split across a boundary.

**Contextual chunking:** prepend a short, LLM-generated description of where the chunk sits in its document (e.g. "From ACME's 2025 annual report, section on Q3 revenue") before embedding. Anthropic's "Contextual Retrieval" write-up reported large reductions in failed retrievals from this, especially combined with keyword search and reranking.

**Metadata:** store source, title, section, date, author, permissions. Enables filtering ("only 2026 documents"), citations, and access control.

### 2.5 Embeddings and vector search

Each chunk is embedded once at ingestion. At query time, the question is embedded with the **same model** and compared to all chunk vectors.

Exact search over millions of vectors is slow, so vector databases use **approximate nearest neighbor (ANN)** indexes. The most common is **HNSW** (Hierarchical Navigable Small World), a layered graph that trades a little recall for large speed gains.

Vector store options: FAISS (library, in-process), Chroma (simple, great for prototypes), pgvector (Postgres extension, good when you already use Postgres), Qdrant/Weaviate/Pinecone/Milvus (dedicated databases with filtering and scaling).

**Weakness of pure vector search:** it's poor at exact matches such as product codes, error IDs, names, and rare terms.

### 2.6 Keyword search and hybrid retrieval

**BM25** is the classic keyword ranking algorithm. It scores documents by term frequency, weighted down for very common terms and normalized for document length. It excels exactly where embeddings are weak.

**Hybrid search** runs both and merges the ranked lists. The standard merge is **Reciprocal Rank Fusion (RRF)**, which uses ranks rather than raw scores (scores from BM25 and cosine similarity aren't on comparable scales):

```
RRF_score(doc) = Σ over each result list  1 / (k + rank_in_that_list)     (k ≈ 60)
```

```python
def rrf(result_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, doc_id in enumerate(results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# rrf([bm25_ids, vector_ids])
```

### 2.7 Reranking

Retrieval uses **bi-encoders**: query and documents are embedded separately, which is fast but loses fine-grained interaction. A **cross-encoder reranker** reads the query and each candidate together and outputs a relevance score. It's much more accurate and much slower, so the standard pattern is: retrieve top 50–100 cheaply, rerank, keep the top 5–10. Options include Cohere Rerank, Voyage rerankers, BGE rerankers (open source), or an LLM as a reranker.

### 2.8 Query transformation

- **Query rewriting:** turn a conversational follow-up ("what about last year?") into a standalone query using chat history. Essential for multi-turn RAG.
- **Multi-query:** generate several phrasings, retrieve for each, merge with RRF.
- **HyDE (Hypothetical Document Embeddings):** have the LLM write a hypothetical answer, then embed that and search with it, since answers often resemble documents more than questions do.
- **Decomposition:** split a complex question into sub-questions and retrieve for each.

### 2.9 Generation

A good RAG prompt tells the model to answer only from the provided context, cite sources by ID, and say "I don't know" when the context doesn't contain the answer. Put retrieved chunks in clearly delimited blocks (e.g. XML tags with source IDs). Placing the question after the documents often works well for long contexts.

### 2.10 Agentic RAG

Instead of a single fixed retrieval step, the model gets a `search` tool and decides when to search, what to search for, and whether results are sufficient. Better for complex, multi-hop questions; slower, costlier, and harder to evaluate.

### 2.11 Evaluating RAG (the part most teams skip)

**Evaluate retrieval and generation separately.** If you only score final answers, you can't tell whether a failure came from bad retrieval or bad generation.

**Retrieval metrics** (need a test set of questions with known relevant chunks):

| Metric | Meaning |
|---|---|
| Recall@k | Of all relevant chunks, what fraction appear in the top k? (Most important for RAG: if it isn't retrieved, the model can't use it) |
| Precision@k | Of the top k results, what fraction are relevant? |
| Hit rate@k | Did at least one relevant chunk appear in the top k? |
| MRR | Mean of 1/rank of the first relevant result. Rewards putting a relevant result near the top |
| nDCG@k | Rank-aware score supporting graded relevance (highly vs. somewhat relevant) |

**Generation metrics:**

- **Faithfulness / groundedness:** is every claim in the answer supported by the retrieved context? (Catches hallucination.)
- **Answer relevance:** does it actually answer the question asked?
- **Correctness:** does it match a reference answer?
- **Citation accuracy:** do cited sources actually support the claims attached to them?
- **Abstention:** does it correctly say "I don't know" for unanswerable questions? Include some in your test set.

Tools like RAGAS implement LLM-judged versions of these. Know the concepts even if you use a library.

**Building a test set:** start with 50–100 real or realistic questions. You can bootstrap by having an LLM generate questions from specific chunks (so you know the relevant chunk), then manually review and fix them. Include easy lookups, multi-hop questions, questions needing exact terms, and unanswerable questions.

### 2.12 RAG failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Right doc exists but never retrieved | Bad chunking, vocabulary mismatch, exact-term query | Hybrid search, contextual chunks, query rewriting |
| Retrieved but answer still wrong | Too much noise in context, key chunk ranked low | Reranking, fewer but better chunks |
| Answer includes unsupported claims | Weak grounding instructions | Stricter prompt, citations, faithfulness eval |
| Follow-up questions fail | Query lacks conversation context | Rewrite queries using chat history |
| Stale answers | Index not updated | Incremental ingestion, date metadata |
| Tables and numbers wrong | Poor parsing | Better parser, table-aware chunking |

### Check your understanding

1. Why evaluate retrieval separately from generation?
2. Why does hybrid search beat pure vector search, and how does RRF work?
3. Bi-encoder vs cross-encoder: what's the difference and why use both?
4. How would you build a RAG test set from scratch?
5. A stakeholder says the RAG bot "gives wrong answers." What's your debugging process?

---

## 3. Evaluation Harnesses

### 3.1 Why evals are the core of applied AI

Without evals, every change is a guess. Evals let you compare models, catch regressions, decide when something is good enough to ship, and (in training) provide reward signals. For teams deploying AI with customers, a well-designed eval is often the most valuable deliverable, because it turns "it feels worse" into a measurable problem.

### 3.2 Types of graders

| Grader | Strengths | Weaknesses |
|---|---|---|
| Code / programmatic | Fast, cheap, deterministic, objective | Only works for checkable outputs (exact match, tests pass, JSON valid, cell value correct) |
| LLM-as-judge | Scales to open-ended outputs, nuanced | Biased, can be inconsistent, needs calibration |
| Human | Gold standard for nuance and domain expertise | Slow, expensive, also inconsistent without guidelines |

**Rule:** use programmatic verification wherever possible, LLM judges for what can't be coded, and humans to calibrate the judges and handle hard cases.

### 3.3 Rubric design

Good criteria are:

- **Atomic:** one thing per criterion, so failures are diagnosable.
- **Verifiable:** two reviewers would agree on the verdict.
- **Grounded in expert judgment:** reflects what a real professional would care about.
- **Not gameable:** can't be passed by length, confident tone, or formatting alone.

Weak: *"The analysis is thorough and accurate."*
Strong: *"States Q3 revenue as $4.2M, matching the source spreadsheet."* / *"Flags the change-of-control clause in section 12."*

Binary pass/fail criteria are usually more reliable than 1–10 scales, which judges and humans use inconsistently. If you need a scale, anchor every level with a concrete description.

### 3.4 LLM-as-judge: biases and mitigations

| Bias | Description | Mitigation |
|---|---|---|
| Position bias | Prefers the first (or second) option in pairwise comparisons | Run both orders; count only consistent verdicts |
| Verbosity bias | Prefers longer answers | Rubric explicitly about correctness; length-controlled comparisons |
| Self-preference | Prefers outputs from its own model family | Use a different model family as judge; validate with humans |
| Leniency / sycophancy | Overly generous scores | Binary criteria, require evidence quotes before verdict |

**Best practices:** ask the judge to reason and quote evidence before giving a verdict; grade one criterion per call when accuracy matters; use low temperature; and most importantly, **measure judge–human agreement** on a labeled sample before trusting the judge at scale.

### 3.5 Measuring agent reliability: pass@k and pass^k

Agents are non-deterministic, so run each task multiple times.

**pass@k:** probability that *at least one* of k attempts succeeds. Useful when you can verify and pick the best attempt (e.g. code with tests). Unbiased estimator from n runs with c successes:

```
pass@k = 1 − C(n−c, k) / C(n, k)
```

**pass^k** (introduced in the τ-bench agent benchmark): probability that *all* k attempts succeed. Measures consistency, which is what matters for production agents a customer relies on every time.

```
pass^k ≈ C(c, k) / C(n, k)
```

Example: an agent succeeds 70% of the time per attempt. pass@3 ≈ 97%, but pass^3 ≈ 34%. The same agent looks great or unreliable depending on which question you ask.

### 3.6 Statistics you should be able to discuss

Eval scores are estimates. With n tasks and success rate p, the standard error is approximately `sqrt(p(1−p)/n)`. At p = 0.7 and n = 100, that's about ±4.6 points, so a 95% confidence interval spans roughly ±9 points. A "3-point improvement" on 100 tasks is probably noise.

Practical habits: report confidence intervals; use **bootstrap resampling** when metrics are complex; compare models on the **same tasks** (paired comparisons are more sensitive); and increase task count or trials before claiming small wins.

### 3.7 Outcome vs trajectory evaluation

- **Outcome:** is the final state correct? (File contents, database rows, final answer.) Objective and robust to different valid paths.
- **Trajectory:** was the process sound? Did it use appropriate tools, avoid unsafe actions, verify its work, avoid fabricating steps?

Use outcome grading as the primary signal, and trajectory checks for safety, efficiency, and catching right-for-wrong-reasons successes. Don't over-constrain trajectories: penalizing valid alternative paths makes evals brittle.

### 3.8 Eval hygiene

- **Held-out sets:** never tune prompts on the same set you report results on.
- **Contamination:** public benchmarks may be in training data; prefer private, freshly written tasks.
- **Saturation:** when all models score 95%+, the eval no longer discriminates. Add harder tasks.
- **Regression suites:** a fixed set run automatically on every change, like unit tests.
- **Versioning:** version tasks, rubrics, and judge prompts. A score is meaningless without knowing which version produced it.

### 3.9 Anatomy of an eval harness

```
Task registry (versioned tasks + rubrics + expected outputs)
   → Environment setup (sandbox with files/tools, fresh per trial)
   → Runner (execute model/agent, n trials per task, async + retries)
   → Trace store (full trajectories, tokens, latency, cost)
   → Graders (programmatic → LLM judge → human review queue)
   → Aggregation (per-task, per-category, pass@k, pass^k, CIs)
   → Reporting (dashboards, diffs between runs, failure examples)
```

### 3.10 Validating the eval itself

An eval is a measuring instrument and can be broken. Check that: scores correlate with expert human judgment; it discriminates between models you know differ in quality; failing examples are genuinely failures when a human reads them; and tasks are actually solvable (a reference solution passes).

### Check your understanding

1. When would you use a code grader vs an LLM judge vs a human?
2. Name three LLM-judge biases and how to mitigate each.
3. Explain pass@k vs pass^k with an example. Which matters for a production agent?
4. Model A scores 72% and model B 75% on 100 tasks. Is B better?
5. How do you know your eval is measuring the right thing?

---

## 4. Data Taxonomy, Labeling, and Quality

### 4.1 Why this matters

Model quality is bounded by data quality. Much of modern AI development is fundamentally about producing expert data: tasks, rubrics, preference judgments, and trajectories. Taxonomies and quality systems make that data structured, measurable, and trustworthy.

### 4.2 Taxonomy design

A taxonomy is a structured classification of your data (e.g. task types, failure modes, domains, difficulty levels). Principles:

- **Mutually exclusive, collectively exhaustive (MECE)** at each level, so every item fits exactly one category.
- **Grounded in real data:** start by manually reviewing a sample (100–300 items) and clustering, rather than designing top-down from imagination. LLM-assisted clustering can speed this up.
- **Shallow and usable:** 2–3 levels is usually enough. Annotators can't reliably use 80 leaf categories.
- **Clear definitions with examples and counter-examples** for each category, especially near boundaries.
- **An "other/unclear" bucket** that you monitor. If it grows beyond ~5–10%, the taxonomy needs revision.
- **Versioned:** when categories change, record the mapping so old labels stay interpretable.

**Uses:** measuring coverage (do we have enough legal tasks?), balancing datasets, slicing eval results by category, and prioritizing failure modes.

Example failure taxonomy for agents:
```
Context failure  → missing info / stale info / ignored available info
Tool failure     → wrong tool / malformed call / ignored error
Reasoning failure→ calculation error / logic error / misread instruction
Execution failure→ incomplete / loop / premature stop / fabricated action
Safety failure   → unauthorized action / data exposure / followed injection
```

### 4.3 Labeling guidelines

The guideline document is the single biggest lever on label quality. It should include the task purpose (why labels matter), precise definitions, decision rules for ambiguous cases, many worked examples including hard edge cases, and a changelog. Update it whenever reviewers keep disagreeing on the same kind of item.

### 4.4 Measuring agreement

Raw percent agreement is misleading because some agreement happens by chance. Use chance-corrected metrics.

**Cohen's kappa** (two annotators, categorical labels):

```
κ = (p_o − p_e) / (1 − p_e)
p_o = observed agreement
p_e = agreement expected by chance (from each annotator's label frequencies)
```

Worked example: two annotators label 100 items pass/fail and agree on 80 (p_o = 0.80). Annotator A says pass 60% of the time, B says pass 70%. Chance agreement p_e = 0.6×0.7 + 0.4×0.3 = 0.42 + 0.12 = 0.54. So κ = (0.80 − 0.54)/(1 − 0.54) ≈ 0.57, which is only moderate despite 80% raw agreement.

Rough interpretation (Landis & Koch): < 0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, > 0.80 almost perfect.

```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(labels_a, labels_b)
```

**Fleiss' kappa** handles more than two annotators on categorical labels. **Krippendorff's alpha** is the most general: any number of annotators, missing labels, and nominal, ordinal, or interval data.

**Key insight:** low agreement usually signals ambiguous guidelines or a genuinely subjective task, not bad annotators. Investigate disagreements before blaming people.

### 4.5 Quality control systems

- **Gold questions:** items with known correct labels seeded invisibly into the queue to measure each annotator's accuracy.
- **Qualification tests** before annotators start real work.
- **Overlap:** a percentage of items labeled by multiple annotators to track agreement over time.
- **Adjudication:** disagreements resolved by a senior reviewer, and the resolution fed back into guidelines.
- **Annotator-level metrics:** accuracy on gold, agreement with peers, speed (suspiciously fast often means low effort), drift over time.
- **Calibration sessions:** annotators label the same items and discuss differences.
- **Consensus labels:** majority vote, or weighted by annotator reliability.

### 4.6 Dataset curation

- **Deduplication:** exact duplicates via hashing; near-duplicates via MinHash/LSH or embedding similarity. Duplicates inflate eval scores and cause overfitting.
- **Train/eval leakage:** make sure eval items (or near-copies) don't appear in training data.
- **Filtering:** remove low-quality, malformed, off-topic, or unsafe items; heuristics first, then classifiers or LLM filters.
- **Balance and coverage:** use the taxonomy to check distribution across categories and difficulty.
- **PII handling:** detect and redact personal data.
- **Hard example mining / active learning:** prioritize labeling items where the model is uncertain or failing, since they give the most signal per dollar.
- **Documentation:** dataset cards describing source, collection process, taxonomy version, known limitations, and quality metrics.

### Check your understanding

1. How would you build a taxonomy for a new dataset you've never seen?
2. Compute and explain Cohen's kappa. Why not just use percent agreement?
3. Agreement between expert annotators is low. What do you do?
4. Design a quality control system for 100,000 expert-labeled items.
5. Why does deduplication matter for both training and evaluation?

---

## 5. ML Pipelines and Post-Training

### 5.1 Where data goes in the model lifecycle

**Pretraining** teaches general language and knowledge from huge text corpora. **Post-training** shapes the model into a useful assistant or agent, and this is where expert data companies contribute most.

### 5.2 Post-training methods

**SFT (Supervised Fine-Tuning):** train on high-quality example input–output pairs (demonstrations). Data needed: expert-written responses or trajectories. Teaches format, style, and behaviors.

**RLHF (Reinforcement Learning from Human Feedback):** humans compare pairs of model outputs; a **reward model** learns to predict preferences; the policy is optimized against the reward model (classically with PPO). Data needed: preference comparisons.

**DPO (Direct Preference Optimization):** uses the same preference pairs (chosen vs rejected) but optimizes the model directly, without training a separate reward model. Simpler and more stable than PPO-based RLHF.

**RL with verifiable rewards (RLVR) and RL environments:** the model attempts tasks in an environment, and a verifier (tests, rubric, programmatic checker) produces the reward. Algorithms like GRPO compare a group of attempts on the same task. This is how agents get trained on long-horizon tasks, and it's why realistic environments, well-designed tasks, and rigorous rubrics are so valuable: **the rubric is the reward.** A gameable rubric trains a model to game it (**reward hacking**).

### 5.3 The data flywheel

```
Deploy → collect real usage and failures → categorize with taxonomy
   → create targeted data (demos, preferences, tasks) → train/tune
   → evaluate on regression + new tasks → deploy again
```

### 5.4 Pipeline engineering fundamentals

- **Dataset versioning:** every training or eval run references an immutable dataset version (tools: DVC, Hugging Face datasets with revisions, or versioned files in object storage with manifests).
- **Experiment tracking:** log config, data version, code commit, metrics, and artifacts for every run (MLflow, Weights & Biases).
- **Reproducibility:** pin seeds, library versions, and data versions. You should be able to rerun any reported result.
- **Lineage:** trace any model result back to the exact data and config that produced it.
- **Orchestration:** schedule and chain pipeline steps (Airflow, Prefect, Dagster, Ray for distributed work).
- **Data validation:** schema checks at every stage so bad data fails loudly instead of silently corrupting a run.

### 5.5 Splits and leakage

Train, validation (for tuning decisions), and test (touched only for final reporting). Split by the right unit: if multiple items come from the same document, user, or task, keep them in the same split, or the model effectively sees test data during training.

### Check your understanding

1. Explain SFT, RLHF, and DPO, and what data each needs.
2. Why does rubric quality matter so much in RL environments? What is reward hacking?
3. What would you log to make an experiment reproducible?
4. Describe a data flywheel for an enterprise agent.

---

## 6. Inference and Infrastructure

### 6.1 Key latency and throughput metrics

- **TTFT (time to first token):** perceived responsiveness; dominated by input processing (prefill).
- **Output tokens per second:** generation speed (decode).
- **End-to-end latency:** full request time.
- **Throughput:** requests or tokens handled per second across all users.
- **Cost:** input tokens × input price + output tokens × output price. Track per request, per experiment, per customer.

### 6.2 How serving works (conceptually)

- **KV cache:** stores intermediate attention results for previous tokens so they aren't recomputed each step. It consumes a lot of GPU memory, which limits concurrency.
- **Continuous batching:** new requests join the running batch as others finish, instead of waiting for the whole batch. Big throughput gain.
- **vLLM:** popular open-source serving engine; its PagedAttention manages KV cache memory like virtual memory pages, reducing waste. Alternatives include SGLang and TensorRT-LLM.
- **Quantization:** storing weights in fewer bits (8-bit, 4-bit) to cut memory and cost, with some quality risk that must be evaluated.
- **Prompt caching:** API providers can cache a repeated prompt prefix (system prompt, documents), cutting cost and latency. Put stable content first and variable content last.

### 6.3 Calling APIs at scale in Python

Running thousands of eval rollouts means handling concurrency, rate limits, and failures.

```python
import asyncio, random
from anthropic import AsyncAnthropic, RateLimitError, APIStatusError

client = AsyncAnthropic()
sem = asyncio.Semaphore(20)          # cap concurrent requests

async def call_with_retry(prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await asyncio.wait_for(
                    client.messages.create(
                        model="claude-sonnet-5",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=120,
                )
            return resp.content[0].text
        except (RateLimitError, asyncio.TimeoutError, APIStatusError) as e:
            if isinstance(e, APIStatusError) and e.status_code < 500 \
               and not isinstance(e, RateLimitError):
                raise                      # don't retry client errors like 400
            delay = min(60, 2 ** attempt) + random.uniform(0, 1)  # backoff + jitter
            await asyncio.sleep(delay)
    raise RuntimeError("Max retries exceeded")

async def main(prompts: list[str]) -> list:
    return await asyncio.gather(
        *(call_with_retry(p) for p in prompts), return_exceptions=True
    )
```

Key ideas to explain: a **semaphore** bounds concurrency; **exponential backoff with jitter** prevents all clients retrying in sync; only retry **transient** errors (429, 5xx, timeouts), not client errors; `return_exceptions=True` keeps one failure from killing the batch. For very large offline jobs, provider **batch APIs** are cheaper and handle this for you.

### 6.4 Sandboxed environments for agent rollouts

Agents that run code or modify files need isolation: each trial gets a fresh container (Docker) with the environment's files and tools, torn down afterward. Services like Modal, Daytona, or E2B run many sandboxes in parallel. Requirements: reproducible images, resource limits, network restrictions, full logging, and a way to extract the final state for grading.

### 6.5 Observability

Log structured traces for every request and agent step: inputs, outputs, tool calls, token counts, latency, cost, model version, prompt version. Use trace IDs to connect steps. Without this you can't debug customer issues, compute costs, or build eval sets from production failures.

### 6.6 Deployment basics

Containerize services; expose them through an API (FastAPI is the common choice in Python); configure via environment variables and secrets managers (never hard-code keys); add health checks; roll out changes gradually with the ability to roll back; gate releases on eval results.

### Check your understanding

1. What's TTFT and what affects it?
2. What do KV cache and continuous batching do?
3. Write (from memory) an async LLM caller with bounded concurrency and retries.
4. Which errors should you retry and which shouldn't you?
5. How would you run 10,000 agent rollouts safely in parallel?

---

## 7. Production Python Patterns

These patterns show up in every production AI system.

**Typed data models with validation:**
```python
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str = Field(min_length=1)
    metadata: dict[str, str] = {}
```

**Configuration from environment**, not hard-coded values (`pydantic-settings` or `os.environ`).

**Structured logging** with context (request ID, task ID) instead of `print`.

**Generators for large data** so you don't load huge files into memory:
```python
import json
def read_jsonl(path: str):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
```

**Idempotent pipelines:** each stage writes outputs keyed by input hash or ID, so reruns skip completed work and crashes don't corrupt results.

**Testing:** `pytest` unit tests for pure logic (chunking, RRF, scoring); mock LLM calls in unit tests so tests are fast and deterministic; a small integration test that hits the real API.

**Project hygiene:** clear module structure, dependency management (`uv` or `poetry`), linting and formatting (`ruff`), type checking (`mypy` or `pyright`), a README explaining how to run everything.

---

## 8. Key Takeaways

- **Agents:** Most agent failures are context or tool-design problems before they're model problems.
- **Production:** The demo-to-production gap is reliability: observability, guardrails, termination, and regression evals.
- **RAG:** Evaluate retrieval and generation separately, or you can't tell which half is broken.
- **Hybrid search:** Embeddings handle meaning, BM25 handles exact terms, and RRF merges them by rank.
- **Evals:** Programmatic checks where possible, calibrated LLM judges where not, and humans to validate the judges.
- **Reliability:** For production agents, pass^k matters more than pass@k, because users need it to work every time.
- **Statistics:** On 100 tasks, a 3-point difference is usually within noise.
- **Data:** Low annotator agreement is usually a guideline problem, not a people problem.
- **Post-training:** In an RL environment, the rubric is the reward, so a gameable rubric trains a model to game it.

---

## 9. Further Reading

Primary sources behind the ideas in this handbook:

**Agents**
- Anthropic, *Building Effective Agents* (2024): workflows vs agents and orchestration patterns.
- Anthropic, *Effective Context Engineering for AI Agents*: context management for long-running agents.
- Model Context Protocol documentation: modelcontextprotocol.io
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022).

**RAG**
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020).
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009).
- Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* (2009).
- Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs* (2016).
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE, 2022).
- Anthropic, *Introducing Contextual Retrieval* (2024).
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023).

**Evaluation**
- Chen et al., *Evaluating Large Language Models Trained on Code* (2021): the unbiased pass@k estimator.
- Yao et al., *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains* (2024): pass^k.
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023): judge biases.
- Efron & Tibshirani, *An Introduction to the Bootstrap* (1993).

**Data quality**
- Cohen, *A Coefficient of Agreement for Nominal Scales* (1960).
- Landis & Koch, *The Measurement of Observer Agreement for Categorical Data* (1977).
- Krippendorff, *Content Analysis: An Introduction to Its Methodology*.
- Gebru et al., *Datasheets for Datasets* (2018).

**Post-training and infrastructure**
- Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT, 2022).
- Rafailov et al., *Direct Preference Optimization* (2023).
- Shao et al., *DeepSeekMath* (2024): introduces GRPO.
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, 2023).

---

## Companion Project

The concepts in this handbook are implemented in [agentic-rag-eval-platform](https://github.com/YOUR_USERNAME/agentic-rag-eval-platform): hybrid retrieval, grounded generation with citations, a tool-using agent, and an evaluation harness with calibrated LLM judges and reliability metrics.

## Contributing

Found an error or want to add a topic? Issues and pull requests are welcome.

## Author

**Evangelos Vlachos**
## License

© 2026 Evangelos Vlachos. Text licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code snippets licensed under MIT.

