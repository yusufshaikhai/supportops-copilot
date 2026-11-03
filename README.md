# SupportOps Copilot

Milestone 5 adds cited HarborHome knowledge retrieval to the browser review
workflow.

For the reasoning and learning history behind each milestone, read
[`docs/milestones.md`](docs/milestones.md). The README describes the current
system; the milestone log preserves how and why it evolved.

The CLI and browser workflow accept ticket text and return:

- one of five categories: `shipping`, `refund`, `damaged_item`, `billing`, or
  `account`;
- `urgent` or `normal` priority;
- a confidence score and the classification method;
- `unknown` when the selected method has insufficient evidence;
- up to two relevant HarborHome policy records with source citations;
- an explicit no-evidence state when retrieval is unsupported.

No LLM is used yet. Category routing uses TF-IDF text features and logistic
regression. Urgency remains an explicit keyword policy because missing urgent
cases is safety-sensitive and the current corpus is too small to learn that
policy reliably. Knowledge retrieval uses TF-IDF and cosine similarity over six
short, versioned policy records.

## Run it

```sh
uv sync

uv run python supportops.py classify \
  "My order arrived broken and I need help immediately"

uv run python supportops.py classify --method rules \
  "My order arrived broken"

uv run python supportops.py compare
uv run python supportops.py analyze-errors

uv run uvicorn app:app --reload

uv run python -m pytest
uv run ruff check .
```

Open <http://127.0.0.1:8000> to submit a ticket and review its suggested
category, urgency, confidence, method, and cited policy evidence. `unknown`
predictions are explicitly marked for manual routing. If retrieval has no
evidence, the page says not to answer from memory. It does not automatically
reply to customers or persist ticket data.

`compare` trains on `data/training.csv` and evaluates both methods on the fixed
examples in `data/evaluation.csv`. `analyze-errors` reports per-category recall,
confusion counts, low-confidence rejections, and each failed ticket.

## Milestone 5 architecture

```text
training.csv --> TF-IDF --> logistic regression --+
                                                  |
CLI ticket -----> validation --> category method -+--> prediction
                              |                   |
                              +--> urgency rules -+

knowledge.csv --> TF-IDF + cosine similarity --> cited policy matches

browser --> FastAPI form --> classifier + retrieval --> HTML review page

evaluation.csv --> rules + ML --> comparison metrics
               --> ML error analysis --> recall + confusions + failed tickets
```

### Why this architecture

The current product question is whether a reviewer can see relevant business
evidence without allowing the system to invent an answer. The architecture adds
only a small retrieval boundary.

- **TF-IDF and logistic regression:** this is a strong and inspectable baseline
  for small labeled text datasets. It trains quickly on a laptop and exposes
  class probabilities. A transformer or hosted LLM would add cost and
  operational complexity before the dataset justifies it.
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
- **Server-rendered HTML:** one Jinja template covers the current form and
  result. React, a JSON API, and client-side state would add no value yet.
- **No database:** the page reviews one submitted ticket and returns the result
  immediately. Persistence starts when approve, edit, reject, and audit history
  become product requirements.
- **Manual decision remains explicit:** known predictions are suggestions and
  `unknown` predictions are marked for manual routing. The UI never sends a
  customer response.
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
- **No generated answer:** retrieval supplies evidence only. Grounded response
  drafting remains a separate milestone with its own evaluation and human
  approval boundary.
- **No cached retrieval index yet:** fitting six short records is negligible
  and avoids cache invalidation. Cache or persist an index only when the
  knowledge base or measured latency requires it.
- **Two fixed CSV files:** training and evaluation data are reviewable in Git.
  Evaluation rows are never fitted by the model, though their errors now guide
  iteration. Add a database when users create tickets or feedback in the
  product.
- **One module before package layers:** the schema, classifier, evaluator, and
  CLI remain one small cohesive workflow. Split them when the FastAPI boundary
  introduces a second runtime concern.
- **Pydantic at input boundaries:** invalid or incomplete tickets fail
  explicitly instead of silently entering evaluation results.
- **Confidence threshold:** low-confidence ML predictions become `unknown` and
  require human review rather than forcing a route.
- **Structured error analysis:** failures are grouped into threshold rejections
  and category confusions before data or policy changes are made.
- **Targeted training examples:** 15 distinct examples cover weak language
  patterns across all five categories. The evaluation file and confidence
  threshold remain unchanged.

Alternatives such as an embedding service, vector database, LLM answer
generation, job queue, and tenant configuration are deliberately deferred.
None improves the current six-record retrieval workflow.

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

## Change history and next milestone

Per-milestone changes, decisions, deferred alternatives, and the next question
live in [`docs/milestones.md`](docs/milestones.md).
