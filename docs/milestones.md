# SupportOps milestone learning log

This is the append-only learning history for SupportOps Copilot.

The README explains the current product. This file preserves what existed at
each milestone, why that architecture was selected, what alternatives were
deferred, what the evidence showed, and what should change next.

## Documentation rule for every milestone

Append a new section containing:

1. **Question:** What product or engineering uncertainty does this milestone
   answer?
2. **Scope:** What is included and explicitly excluded?
3. **Changes:** What observable behavior and supporting code or data were added?
4. **Architecture:** How does information flow through the system?
5. **Decisions:** Why was this design selected?
6. **Alternatives:** What was rejected or deferred, and what evidence would make
   it appropriate later?
7. **Evaluation:** How was the result tested and measured?
8. **Outcome:** What did the evidence prove?
9. **Limitations:** What can the system still not claim or safely do?
10. **Next question:** What should the following milestone answer?

Do not rewrite earlier milestone sections when the product changes. Add a new
section so assumptions and decisions remain reviewable.

---

## Milestone 1: deterministic ticket-triage baseline

### 1. Question

Can we define HarborHome's ticket taxonomy, urgency policy, input contract, and
evaluation workflow before choosing a machine-learning model?

### 2. Scope

Included:

- five routing categories: shipping, refund, damaged item, billing, and account;
- normal or urgent priority;
- validated ticket and prediction schemas;
- deterministic keyword routing;
- a command-line interface;
- a small synthetic evaluation set;
- automated behavior and acceptance tests.

Excluded:

- trained machine learning;
- LLM calls and response drafting;
- FastAPI and HTML pages;
- databases and persistent user feedback;
- external helpdesk or commerce integrations;
- automatic customer replies.

### 3. Changes

- Replaced the generated hello-world program with `supportops.py`.
- Added Pydantic schemas for tickets, predictions, and evaluation output.
- Added explicit category keywords and urgency signals.
- Added `classify` and `evaluate` CLI commands with JSON output.
- Added 30 synthetic evaluation tickets.
- Added tests for damaged-item routing, urgent routing, unknown routing, and
  starter acceptance targets.
- Added architecture and decision documentation to the README.

### 4. Architecture

```text
ticket text
    |
    v
Pydantic validation
    |
    v
keyword category rules ----+
                           +--> prediction --> JSON CLI output
explicit urgency rules ----+

evaluation.csv --> same workflow --> accuracy + macro F1 + urgent recall
```

### 5. Decisions

**Use deterministic rules first.** Rules made the taxonomy visible and easy to
challenge. A trained model would have hidden label or policy mistakes behind a
score before the expected behavior was understood.

**Use Pydantic at the input boundary.** Invalid or empty tickets should fail
explicitly rather than silently corrupt evaluation results.

**Return `unknown` when no rule matches.** Human review is safer than forced
confidence when the system has no evidence.

**Use a CLI instead of FastAPI.** There was no external client or review UI.
The CLI exercised the complete workflow without inventing an HTTP contract.

**Use CSV instead of a database.** The dataset was small, static, reviewable in
Git, and written by one developer. Persistence and concurrency did not exist.

**Keep one module.** The schemas, rules, evaluator, and CLI formed one small
workflow. Package layers or interfaces would have added navigation without
separating real runtime responsibilities.

### 6. Alternatives

| Alternative | Why deferred | Revisit when |
| --- | --- | --- |
| TF-IDF classifier | Labels and evaluation were not yet trustworthy | A rule baseline and fixed train/test data exist |
| Hosted LLM router | Higher cost and less predictable behavior | Simpler models fail on representative data |
| FastAPI | No HTTP consumer existed | A browser or integration needs the workflow |
| Database | No mutable shared state existed | Users create tickets or feedback |
| Vector database | No knowledge retrieval existed | The product adds grounded response drafting |
| Multi-tenant configuration | Only one customer existed | A second customer proves real variation |

### 7. Evaluation

The initial 30-ticket corpus produced:

- category accuracy: `1.00`;
- category macro F1: `1.00`;
- urgent recall: `1.00`.

### 8. Outcome

The milestone proved that the code path, schemas, category vocabulary, urgency
policy, metrics, CLI, and tests worked together.

It did **not** prove strong routing quality. The evaluation tickets used obvious
phrases that overlapped heavily with the rules. The perfect score exposed a
weak dataset rather than a finished classifier.

### 9. Limitations

- The corpus was too small and keyword-friendly.
- Rule confidence was a heuristic, not a calibrated probability.
- Synonyms and indirect customer language frequently became `unknown`.
- The synthetic data could not support a business-performance claim.
- Urgency policy covered only phrases explicitly listed by the developer.

### 10. Next question

Does a small trained text classifier route unfamiliar wording better than the
transparent rules baseline on a fixed held-out dataset?

---

## Milestone 2: trained category classifier

### 1. Question

Can a simple supervised model improve category routing on varied customer
language while preserving explicit safety policy and human review?

### 2. Scope

Included:

- a fixed 100-ticket training corpus;
- a separate 50-ticket held-out corpus;
- TF-IDF unigram and bigram text features;
- logistic-regression category classification;
- cached in-process model training;
- the original rules as a benchmark;
- a low-confidence `unknown` outcome;
- exact misclassification reporting;
- corpus size and train/test overlap tests;
- GitHub Actions for pytest and Ruff.

Excluded:

- learned urgency prediction;
- model serialization and a model registry;
- hyperparameter-search infrastructure;
- LLM routing;
- FastAPI and HTML pages;
- a database;
- production or real-customer quality claims.

### 3. Changes

- Replaced the keyword-friendly dataset with distinct training and evaluation
  corpora.
- Added a scikit-learn pipeline using `TfidfVectorizer` and
  `LogisticRegression`.
- Added `rules` and `ml` classification methods.
- Added `compare` output with both metric sets and their macro-F1 difference.
- Added a confidence threshold that routes weak predictions to `unknown`.
- Preserved explicit urgency rules and expanded them using observed misses.
- Added the exact category errors to evaluation output.
- Added tests for paraphrased language, unrelated text, target metrics, model
  improvement, corpus size, and train/test separation.
- Removed unused FastAPI and pandas dependencies.
- Added pull-request and `main` CI for the locked test and lint commands.

### 4. Architecture

```text
training.csv --> TF-IDF --> logistic regression --+
                                                  |
ticket text --> validation --> category method ---+--> prediction
                              |                   |
                              +--> urgency rules -+

evaluation.csv --> rules + ML --> metrics + exact category errors
```

### 5. Decisions

**Use TF-IDF and logistic regression.** This is a strong small-data baseline:
fast to train, inexpensive, inspectable, and able to produce class
probabilities. It answers whether learned word patterns improve routing without
introducing model hosting.

**Keep rules as the benchmark.** A replacement cannot demonstrate improvement
if the previous behavior disappears.

**Keep urgency deterministic.** Urgency encodes business risk. The dataset is
too small to justify learning a policy where a false negative can matter more
than ordinary classification accuracy.

**Train in process and cache it.** Training 100 rows is cheap. Persisting a model
would introduce artifact versioning, loading, and deployment concerns before
training time creates a measurable problem.

**Use a fixed confidence threshold.** Logistic regression must always choose a
class, but the product should not. Low confidence becomes `unknown` so a human
reviews it.

**Preserve a held-out dataset.** Evaluation data must not be used for fitting.
Without that separation, improvements may only measure memorization.

**Add CI only after a remote repository exists.** Local pytest and Ruff were
already the source of truth. GitHub Actions now executes those same commands
rather than introducing a different validation path.

### 6. Alternatives

| Alternative | Why deferred | Revisit when |
| --- | --- | --- |
| Transformer classifier | The corpus is too small to justify its cost and complexity | TF-IDF plateaus on a larger representative corpus |
| LLM classifier | More expensive and less deterministic for a five-label task | Conventional classifiers fail required quality or generalization |
| Learned urgency model | Insufficient reviewed urgent examples | There is enough data to evaluate false-negative risk per category |
| Saved model artifact | Training is currently fast | Startup or request latency becomes measurable |
| Hyperparameter search | It risks tuning to synthetic data | A larger validation set exists |
| FastAPI and UI | Routing quality was still the open question | The workflow is credible enough for human review |

### 7. Evaluation

Held-out results after the pipeline's duplicate-data check:

| Method | Accuracy | Macro F1 | Urgent recall | Unknown rate |
| --- | ---: | ---: | ---: | ---: |
| Keyword rules | `0.24` | `0.34` | `1.00` | `0.72` |
| TF-IDF + logistic regression | `0.76` | `0.81` | `1.00` | `0.12` |

The model improved macro F1 by `0.47`. It still misclassified or rejected 12 of
50 held-out tickets.

Validation also found and corrected:

- unused FastAPI and pandas dependencies;
- a missing executable contract for the 100/50 corpus sizes and train/test
  separation;
- one duplicate evaluation ticket;
- formatting drift caught by Ruff.

### 8. Outcome

The trained model substantially improved routing on less direct customer
language and passed the starter macro-F1 and urgent-recall targets.

The remaining 12 errors are valuable evidence. They show which categories need
better examples, wording coverage, confidence calibration, or taxonomy review.
The score is sufficient to continue learning, not sufficient to claim
production readiness.

### 9. Limitations

- All examples remain synthetic.
- Some vocabulary appears in both corpora even though exact ticket text does
  not overlap.
- The `0.24` confidence threshold is empirically chosen on a small corpus and is
  not calibrated for business risk.
- Urgency rules still depend on manually enumerated phrases.
- The model trains during process startup.
- There is no human workflow, persistence, knowledge retrieval, or response
  drafting.

### 10. Next question

What do the 12 held-out errors reveal, and can data-quality improvements reduce
them without editing the held-out labels to flatter the score?

After that analysis, the next product boundary is a minimal FastAPI and
server-rendered review page around the credible prediction workflow.
