# SupportOps Copilot

Milestone 3 adds structured error analysis and targeted data-quality
improvements to the HarborHome Commerce ticket router.

For the reasoning and learning history behind each milestone, read
[`docs/milestones.md`](docs/milestones.md). The README describes the current
system; the milestone log preserves how and why it evolved.

It accepts ticket text and returns:

- one of five categories: `shipping`, `refund`, `damaged_item`, `billing`, or
  `account`;
- `urgent` or `normal` priority;
- a confidence score and the classification method;
- `unknown` when the selected method has insufficient evidence.

No LLM is used yet. Category routing uses TF-IDF text features and logistic
regression. Urgency remains an explicit keyword policy because missing urgent
cases is safety-sensitive and the current corpus is too small to learn that
policy reliably.

## Run it

```sh
uv sync

uv run python supportops.py classify \
  "My order arrived broken and I need help immediately"

uv run python supportops.py classify --method rules \
  "My order arrived broken"

uv run python supportops.py compare
uv run python supportops.py analyze-errors

uv run python -m pytest
uv run ruff check .
```

`compare` trains on `data/training.csv` and evaluates both methods on the fixed
examples in `data/evaluation.csv`. `analyze-errors` reports per-category recall,
confusion counts, low-confidence rejections, and each failed ticket.

## Milestone 3 architecture

```text
training.csv --> TF-IDF --> logistic regression --+
                                                  |
ticket text --> validation --> category method ---+--> prediction
                              |                   |
                              +--> urgency rules -+

evaluation.csv --> rules + ML --> comparison metrics
               --> ML error analysis --> recall + confusions + failed tickets
```

### Why this architecture

The current product question is whether a small trained model handles varied
customer language better than keyword rules. The architecture changes only
what is needed to answer that question.

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
- **CLI before FastAPI:** there is no browser or external client yet. A CLI
  exercises the complete behavior without committing to an HTTP contract.
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

Alternatives such as an LLM router, vector database, job queue, repository
interfaces, and tenant configuration are deliberately deferred. None improves
the current error-analysis workflow.

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
| Category macro F1 | `>= 0.90` |
| Urgent-ticket recall | `1.00` |
| Automatic customer replies | `0` |

Current evaluation results:

| Method | Accuracy | Macro F1 | Urgent recall | Unknown rate |
| --- | ---: | ---: | ---: | ---: |
| Keyword rules | `0.24` | `0.34` | `1.00` | `0.72` |
| TF-IDF + logistic regression | `0.88` | `0.906` | `1.00` | `0.06` |

The ML model improves macro F1 over rules by `0.566`. Targeted training-data
changes reduced failures from 12 to 6 and low-confidence rejections from 6 to
3 without lowering the confidence threshold.

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

## Change history and next milestone

Per-milestone changes, decisions, deferred alternatives, and the next question
live in [`docs/milestones.md`](docs/milestones.md).
