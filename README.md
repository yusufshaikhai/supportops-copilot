# SupportOps Copilot

Milestones 6 and 7 add grounded response drafting, persistent human decisions,
audit history, and a review dashboard.

For the reasoning and learning history behind each milestone, read
[`docs/milestones.md`](docs/milestones.md). The README describes the current
system; the milestone log preserves how and why it evolved.

The CLI and browser workflow accept ticket text and return:

- one of five categories: `shipping`, `refund`, `damaged_item`, `billing`, or
  `account`;
- `urgent` or `normal` priority;
- a confidence score and the classification method;
- `unknown` when the selected method has insufficient evidence.

The browser workflow additionally returns:

- up to two relevant HarborHome policy records with source citations;
- an explicit no-evidence state when retrieval is unsupported;
- an optional LLM draft that must cite retrieved sources;
- approve, edit, or reject actions that are stored in SQLite;
- an audit history and aggregate review dashboard.

Category routing uses TF-IDF text features and logistic regression. Urgency
remains an explicit keyword policy. Knowledge retrieval uses TF-IDF and cosine
similarity over six short policy records. Drafting uses a configurable
OpenAI-compatible endpoint only after a reviewer explicitly requests it.

## Run it

```sh
uv sync

uv run python supportops.py classify \
  "My order arrived broken and I need help immediately"

uv run python supportops.py classify --method rules \
  "My order arrived broken"

uv run python supportops.py compare
uv run python supportops.py analyze-errors

# Hosted OpenAI-compatible endpoint
export SUPPORTOPS_LLM_API_KEY="..."
export SUPPORTOPS_LLM_MODEL="gpt-4.1-mini"

uv run uvicorn app:app --reload

uv run python -m pytest
uv run ruff check .
```

The default remote base URL is `https://api.openai.com/v1`. Override it with
`SUPPORTOPS_LLM_BASE_URL` for another compatible service. A hosted endpoint
receives the ticket and retrieved evidence; use an approved provider or local
Ollama for sensitive data.

For local drafting on a 16 GB Apple Silicon Mac:

```sh
brew install ollama
ollama serve
ollama pull qwen2.5:7b

export SUPPORTOPS_LLM_BASE_URL="http://localhost:11434/v1"
export SUPPORTOPS_LLM_MODEL="qwen2.5:7b"
unset SUPPORTOPS_LLM_API_KEY

uv run uvicorn app:app --reload
```

Open <http://127.0.0.1:8000> to create a review. The workflow stores routing
and evidence first, then requires an explicit **Generate grounded draft**
action. Drafts are never sent automatically. Reviewers approve, edit, or reject
them; a decision is final and the draft becomes read-only.
<http://127.0.0.1:8000/dashboard> summarizes the current decisions.

SQLite data defaults to `data/supportops.db`. Override it with
`SUPPORTOPS_DATABASE_PATH`, especially in deployed or test environments.

`compare` trains on `data/training.csv` and evaluates both methods on the fixed
examples in `data/evaluation.csv`. `analyze-errors` reports per-category recall,
confusion counts, low-confidence rejections, and each failed ticket.

## Milestones 6–7 architecture

```text
training.csv --> TF-IDF --> logistic regression --+
                                                  |
CLI ticket -----> validation --> category method -+--> prediction
                              |                   |
                              +--> urgency rules -+

knowledge.csv --> TF-IDF + cosine similarity --> cited policy matches

browser --> FastAPI --> classifier + retrieval --> SQLite review
                                                   |
                             explicit draft action |
                                                   v
ticket + route + cited evidence --> OpenAI-compatible LLM
                                                   |
                                      citation validation
                                                   |
                                                   v
                                   persisted human-reviewed draft

approve / edit / reject --> audit events --> dashboard

evaluation.csv --> rules + ML --> comparison metrics
               --> ML error analysis --> recall + confusions + failed tickets
```

### Why this architecture

The current product question is whether grounded drafts and reviewer feedback
can become a complete, auditable support workflow without automatic sending.

- **TF-IDF and logistic regression:** this is a strong and inspectable baseline
  for small labeled text datasets. It trains quickly on a laptop and exposes
  class probabilities. A transformer or hosted LLM router would add cost and
  operational complexity before the routing dataset justifies it.
- **Rules remain available:** deleting the old method would remove the benchmark
  needed to prove the ML model improves routing.
- **Urgency stays rule-based:** urgency encodes business risk rather than only
  linguistic similarity. Keep the policy explicit until enough reviewed
  examples support a separate model.
- **Model trains in process:** the corpus is only 115 rows and training is
  cached for the process. Persisting a model artifact would introduce version
  and deployment concerns without measurable benefit yet.
- **FastAPI after the CLI boundary was stable:** the browser now calls the same
  `Ticket` validation and `classify_ticket` function instead of introducing a
  second prediction path.
- **Server-rendered HTML:** a shared base template covers the form, review
  detail, and dashboard. React, a JSON API, and client-side state add no value
  to the current workflow.
- **Manual decision remains explicit:** known predictions are suggestions and
  `unknown` predictions are marked for manual routing. Drafting requires a
  separate action, and the UI never sends a customer response.
- **TF-IDF retrieval:** the repository already depends on scikit-learn, and the
  knowledge base has only six records. Embedding services and vector databases
  would add infrastructure without improving the current learning goal.
- **Retrieve across every policy:** retrieval does not trust the predicted
  category as a filter. A wrong route should not hide a relevant policy.
- **Evidence threshold and result limit:** only matches scoring at least `0.08`
  are shown, with at most two results. Below that threshold the safe result is
  no knowledge.
- **Stable citations:** every policy row has a source ID such as
  `HH-SHIP-001`, which is displayed with the retrieved text.
- **OpenAI-compatible HTTP boundary:** one small `httpx` client supports hosted
  APIs and local Ollama without a provider SDK. Remote endpoints require an API
  key; localhost does not.
- **Grounding checks after generation:** a draft must contain at least one
  retrieved source ID and cannot cite a source outside the supplied evidence.
  Invalid drafts fail explicitly and are recorded in the audit history.
- **SQLite persistence:** the standard library database is enough for one local
  deployment and makes review state, feedback, and audit events durable.
- **Post-redirect-get workflow:** creating reviews, drafts, and decisions
  redirects to stable review URLs, avoiding accidental form resubmission.
- **Append-only audit events:** review creation, draft success or failure, and
  approve/edit/reject decisions remain inspectable.
- **Server-rendered dashboard:** aggregate counts and acceptance rate require no
  frontend framework or analytics service.
- **Real module boundaries:** routing/retrieval remain in `supportops.py`,
  external model behavior lives in `drafting.py`, persistence lives in
  `storage.py`, and `app.py` orchestrates them.
- **No cached retrieval index yet:** fitting six short records is negligible
  and avoids cache invalidation. Cache or persist an index only when the
  knowledge base or measured latency requires it.
- **Versioned CSV inputs:** training, evaluation, and knowledge data remain
  reviewable in Git. Mutable review and feedback state belongs in SQLite.
- **Pydantic at input boundaries:** invalid or incomplete tickets fail
  explicitly instead of silently entering evaluation results.
- **Confidence threshold:** low-confidence ML predictions become `unknown` and
  require human review rather than forcing a route.
- **Structured error analysis:** failures are grouped into threshold rejections
  and category confusions before data or policy changes are made.
- **Targeted training examples:** 15 distinct examples cover weak language
  patterns across all five categories. The evaluation file and confidence
  threshold remain unchanged.

Alternatives such as a provider SDK, PostgreSQL, ORM, background queue, React,
automatic sending, and tenant configuration are deliberately deferred. The
current local workflow does not require them.

## Evaluation

Each method reports:

- category accuracy;
- category macro F1, so every category matters;
- urgent-ticket recall;
- unknown prediction rate;
- the exact evaluation tickets it misclassified.

Milestone 3 acceptance targets:

| Metric | Target |
| --- | ---: |
| Category macro F1 | `> 0.81` (the Milestone 2 baseline) |
| Urgent-ticket recall | `1.00` |
| Automatic customer replies | `0` |

Current evaluation results:

| Method | Accuracy | Macro F1 | Urgent recall | Unknown rate |
| --- | ---: | ---: | ---: | ---: |
| Keyword rules | `0.24` | `0.34` | `1.00` | `0.72` |
| TF-IDF + logistic regression | `0.78` | `0.827` | `1.00` | `0.12` |

The ML model improves macro F1 over rules by `0.487`. Training rows were written as
general examples rather than copies of evaluation tickets, so the gain is
modest: failures fell from 12 to 11 and low-confidence rejections stayed at 6,
without lowering the confidence threshold.

The included corpus contains 115 training tickets and 50 fixed evaluation
tickets. Because Milestone 3 used evaluation errors to guide improvements, this
file now serves as a regression and validation set rather than an untouched
final test. A fresh blind test set is required before claiming generalization or
business performance.

Milestones 6–7 add no classifier-quality claim. Their checks cover:

- OpenAI-compatible request and response behavior;
- missing and unsupported citation rejection;
- remote API-key requirements and keyless local Ollama;
- SQLite review, draft, decision, and audit persistence;
- approve, edit, and reject dashboard metrics;
- browser workflows and visible configuration failures.

The full suite contains 22 deterministic tests. A real `qwen2.5:7b` Ollama run
on an M3 MacBook Air with 16 GB RAM produced a grounded draft citing
`HH-DMG-001`.

## Data schema

Both data files contain:

| Column | Meaning |
| --- | --- |
| `text` | Ticket submitted by the customer |
| `category` | Expected routing category |
| `urgency` | Expected `normal` or `urgent` priority |

`data/knowledge.csv` contains:

| Column | Meaning |
| --- | --- |
| `source` | Stable citation ID displayed to the reviewer |
| `title` | Short policy title |
| `content` | Policy evidence available for retrieval |

SQLite contains:

| Table | Meaning |
| --- | --- |
| `reviews` | Ticket, prediction, evidence, draft, model, status, and timestamps |
| `audit_events` | Review creation, draft outcomes, and reviewer decisions |

## Change history and next milestone

Per-milestone changes, decisions, deferred alternatives, and the next question
live in [`docs/milestones.md`](docs/milestones.md).
