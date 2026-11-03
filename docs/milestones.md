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

---

## Milestone 3: error analysis and data quality

### 1. Question

What do the 12 evaluation failures reveal, and can targeted data and policy
changes improve routing without editing the evaluation tickets or lowering the
confidence threshold?

### 2. Scope

Included:

- structured ML error analysis;
- per-category recall and confusion counts;
- separation of low-confidence rejections from category confusions;
- confidence and cause fields on exact classification errors;
- targeted additions to the training corpus;
- one evidence-based urgency-policy correction;
- focused regression tests and current-system documentation.

Excluded:

- edits to `data/evaluation.csv`;
- confidence-threshold tuning;
- hyperparameter search;
- transformer or LLM classification;
- automatic error-cause inference beyond observable model behavior;
- FastAPI, HTML pages, persistence, and customer integrations.

### 3. Changes

- Added an `analyze-errors` CLI command.
- Added structured per-category recall, confusion counts, failure causes, and
  confidence values.
- Classified ML failures as either `low_confidence_rejection` or
  `category_confusion`.
- Added 15 distinct training examples, balanced across the five categories.
- Expanded the training corpus from 100 to 115 tickets.
- Added `right now` to the explicit urgency policy after one urgent account
  ticket was missed.
- Preserved the evaluation dataset and `0.24` confidence threshold unchanged.
- Added tests covering the analysis totals, rejection threshold, new urgency
  phrase, improved metric target, corpus size, and train/evaluation separation.

### 4. Architecture

```text
training.csv --> TF-IDF --> logistic regression --+
                                                  |
evaluation.csv --> classify each ticket ----------+--> predictions
                                                        |
                                                        +--> comparison metrics
                                                        |
                                                        +--> error analysis
                                                             |-- category recall
                                                             |-- confusion counts
                                                             |-- threshold rejections
                                                             +-- exact failures

ticket text --> explicit urgency policy --> urgency
```

### 5. Decisions

**Analyze before changing the model.** The 12 failures split evenly between six
low-confidence rejections and six accepted predictions for the wrong category.
That evidence pointed to sparse language coverage rather than a need for a new
model family.

**Add balanced, distinct training examples.** Three examples per category
introduced missing language patterns without copying evaluation sentences.
Balanced additions avoided improving one weak category by distorting the class
distribution.

**Do not lower the confidence threshold.** A lower threshold would convert some
`unknown` results into automatic routes without adding evidence. Improving the
training signal preserves the human-review boundary.

**Keep causes observable.** The report states whether the model rejected a
prediction or confused categories. Claims such as "ambiguous customer intent"
or "bad label" still require human judgment and are not guessed by the code.

**Treat `right now` as an urgency-policy gap.** Urgency remains deterministic,
so the missed phrase belongs in the explicit policy rather than the category
training corpus.

**Reclassify the evaluation set's role.** Once its errors guide training
changes, it is no longer an untouched final test. It remains useful as a fixed
validation and regression set, but a new blind set is required before claiming
generalization.

### 6. Alternatives

| Alternative | Why deferred | Revisit when |
| --- | --- | --- |
| Lower the confidence threshold | It would hide uncertainty rather than improve evidence | A larger calibration set quantifies review cost and routing risk |
| Hyperparameter search | The failure patterns were explainable data gaps | A larger validation corpus shows the current model has plateaued |
| Automatic semantic cause labels | The model cannot reliably distinguish ambiguity, missing vocabulary, and bad labels | Human-reviewed error annotations exist |
| Transformer or LLM router | The simple model still improved substantially from better data | Representative data shows TF-IDF cannot meet the required quality |
| Fresh blind test immediately | Current work is still synthetic and exploratory | Before making a generalization or business-performance claim |

### 7. Evaluation

Before the changes:

| Metric | Result |
| --- | ---: |
| Category accuracy | `0.76` |
| Category macro F1 | `0.810` |
| Urgent recall | `0.90` |
| Unknown rate | `0.12` |
| Category failures | `12 / 50` |
| Low-confidence rejections | `6` |

The initial per-category recall was:

| Category | Recall |
| --- | ---: |
| Shipping | `0.70` |
| Refund | `0.70` |
| Damaged item | `0.90` |
| Billing | `0.70` |
| Account | `0.80` |

After adding 15 generalized training examples (not copies of evaluation
tickets) and the urgency phrase:

| Method | Accuracy | Macro F1 | Urgent recall | Unknown rate |
| --- | ---: | ---: | ---: | ---: |
| Keyword rules | `0.24` | `0.34` | `1.00` | `0.72` |
| TF-IDF + logistic regression | `0.78` | `0.827` | `1.00` | `0.12` |

The ML model now has 11 category failures, including 6 low-confidence
rejections. Its per-category recall is:

| Category | Recall |
| --- | ---: |
| Shipping | `0.50` |
| Refund | `0.80` |
| Damaged item | `1.00` |
| Billing | `0.70` |
| Account | `0.90` |

### 8. Outcome

Generalized, error-driven data improvements reduced category failures from 12
to 11, raised macro F1 from `0.810` to `0.827`, kept the unknown rate at
`0.12`, and restored urgent recall from `0.90` to `1.00`. Avoiding
evaluation-ticket paraphrases deliberately trades a larger headline gain for a
more honest generalization estimate.

The result supports keeping the small conventional classifier. The evidence did
not justify a lower threshold, new model family, or additional infrastructure.

### 9. Limitations

- Training and evaluation examples remain synthetic.
- The fixed evaluation set has influenced training decisions and is no longer a
  blind final test.
- Eleven category failures remain, including five accepted wrong routes.
- Confidence scores are not calibrated probabilities.
- Error causes describe observable outcomes, not semantic root causes.
- The urgency policy still depends on manually reviewed phrases.
- There is no human review UI, persistence, retrieval, or response drafting.

### 10. Next question

Can a minimal FastAPI and server-rendered page make category, urgency,
confidence, and error-aware human review usable without changing the validated
classification behavior?

---

## Milestone 4: server-rendered ticket review

### 1. Question

Can the existing classifier support a usable browser-based human review
workflow without duplicating prediction logic or introducing persistence?

### 2. Scope

Included:

- a FastAPI application;
- a browser form for one customer ticket;
- server-rendered category, urgency, confidence, and method;
- an explicit manual-routing state for `unknown`;
- server-side ticket validation;
- endpoint tests for the form, known routes, unknown routes, and invalid input;
- current architecture and operating instructions.

Excluded:

- ticket persistence and audit history;
- approve, edit, or reject actions;
- authentication and authorization;
- automatic customer replies;
- knowledge retrieval and citations;
- response drafting;
- React or another client-side framework;
- background jobs, queues, and external integrations.

### 3. Changes

- Added `app.py` with `GET /` and `POST /review`.
- Added one Jinja template containing the form, result, manual-review state,
  and minimal responsive styling.
- Reused the existing `Ticket` schema and `classify_ticket` function.
- Returned an explicit `422` page when submitted text violates the existing
  ticket contract.
- Added FastAPI, Uvicorn, Jinja, multipart form parsing, and HTTP endpoint-test
  dependencies.
- Added four endpoint behavior tests.
- Added browser startup and workflow instructions to the README.

### 4. Architecture

```text
browser
   |
   | GET /
   v
FastAPI --> Jinja form
   ^
   | POST /review
   |
ticket text --> Ticket validation --> classify_ticket
                                      |-- TF-IDF category model
                                      +-- explicit urgency policy
                                               |
                                               v
                                      Jinja review result
```

The CLI, evaluation workflow, and browser all call the same classification
boundary.

### 5. Decisions

**Reuse the classifier directly.** FastAPI is only a delivery boundary. A
separate service class or HTTP-specific prediction implementation would risk
behavior drift without separating a real responsibility.

**Use server-rendered HTML.** The page submits one form and displays one result.
Jinja and ordinary HTML provide the complete workflow without client state,
JavaScript build tooling, or a frontend deployment.

**Keep one application module.** Two routes and one template do not justify a
package hierarchy, routers, dependency injection, or repository interfaces.

**Keep requests synchronous.** Classification trains once per process and is
cached; individual predictions are local and fast. A queue would only add
latency and operational state.

**Return the result directly from the POST.** There is no stored review that a
redirect could retrieve. Add POST-redirect-GET when persistence gives the
result a stable identifier.

**Preserve human authority.** The interface calls every known route a
suggestion and marks `unknown` for manual routing. It does not imply that a
model decision was approved or acted upon.

**Keep styling in the template.** One page does not justify a static asset
pipeline. Extract CSS when multiple pages share it.

### 6. Alternatives

| Alternative | Why deferred | Revisit when |
| --- | --- | --- |
| React or another SPA | One form has no meaningful client state | Multiple interactive review views need shared client behavior |
| JSON API plus separate frontend | There is no second consumer | An integration or independent frontend needs a stable API |
| Database | No review state is created or updated | Approve, edit, reject, feedback, or audit history is added |
| Background queue | Classification is local and fast | Work becomes slow, remote, or retryable |
| Authentication | The app is still a local learning workflow | It is deployed to users or exposes customer data |
| Package and router layers | There are only two cohesive routes | A second runtime domain creates a real module boundary |

### 7. Evaluation

The endpoint tests verify:

- the review form loads;
- a damaged urgent ticket displays the classifier result;
- an unrelated ticket displays `unknown` and requires manual routing;
- invalid ticket text returns a visible validation error with status `422`.

The existing classifier, error-analysis, corpus, and metric tests remain
unchanged. Milestone 4 does not claim a model-quality improvement.

### 8. Outcome

The validated classification behavior is now usable from a browser without
duplicating the model path. The page exposes the information a reviewer needs
for one routing decision and keeps uncertainty visible.

The milestone also confirms that the CLI-first design created a reusable domain
boundary: adding HTTP and HTML required no classifier changes.

### 9. Limitations

- Submitted tickets and predictions disappear after the response.
- There is no reviewer identity, decision, feedback, or audit trail.
- Refreshing the result page can resubmit the form.
- The page supports one ticket at a time and has no queue.
- The app has no authentication or production deployment configuration.
- Predictions remain based on synthetic training data.
- There is no knowledge retrieval, citation, or drafted response.

### 10. Next question

Can HarborHome policy and product knowledge be retrieved with visible citations
and presented beside the ticket without allowing unsupported answers?

---

## Milestone 5: cited HarborHome knowledge retrieval

### 1. Question

Can the review workflow find relevant HarborHome policy evidence, display its
source, and return a safe no-evidence state without generating an answer?

### 2. Scope

Included:

- six short HarborHome knowledge records;
- stable source IDs for citations;
- TF-IDF document retrieval with cosine similarity;
- a minimum evidence threshold;
- at most two retrieved records;
- cited evidence on the review page;
- an explicit instruction not to answer from memory when retrieval is empty;
- retrieval and browser integration tests.

Excluded:

- LLM or template-based response drafting;
- automatic customer replies;
- vector databases and embedding APIs;
- semantic reranking;
- policy chunking beyond one row per short record;
- document ingestion pipelines;
- persistence, reviewer decisions, and audit history;
- customer-specific knowledge configuration.

### 3. Changes

- Added `data/knowledge.csv` with shipping, refund, damaged-item, billing, and
  account guidance.
- Added validated `KnowledgeDocument` and `KnowledgeMatch` schemas.
- Added `load_knowledge` with schema, empty-data, and duplicate-source checks.
- Added `retrieve_knowledge` using TF-IDF and cosine similarity.
- Added a `0.08` minimum score and a two-result limit.
- Added cited knowledge or a no-evidence warning to the FastAPI review page.
- Added retrieval tests for a relevant shipping ticket, unrelated text, and
  the fixed unique source set.
- Extended endpoint tests to verify a damaged-item citation and the safe empty
  state.

### 4. Architecture

```text
knowledge.csv
     |
     v
validated knowledge records
     |
     v
TF-IDF document vectors <--- validated ticket text
     |                            |
     +------ cosine similarity ---+
                   |
                   v
        threshold + top-two limit
                   |
                   v
       cited matches or no evidence

browser review = routing prediction + cited knowledge
```

Routing and retrieval are independent consumers of the same validated ticket.

### 5. Decisions

**Use TF-IDF and cosine similarity.** Scikit-learn is already installed, the
knowledge set is tiny, and the goal is to learn and verify the retrieval
boundary. Remote embeddings would add cost, credentials, network failure, and
versioning before lexical retrieval has failed.

**Retrieve across all documents.** Filtering by the predicted category would
make one model error suppress potentially relevant knowledge. Independent
retrieval lets the reviewer compare the routing suggestion with the evidence.

**Use short CSV records.** Six static policies are easy to review in Git and do
not require file parsers, chunking infrastructure, or a database.

**Cite stable source IDs.** IDs such as `HH-DMG-001` give the reviewer a visible
reference and create a future boundary for linking to source documents.

**Require minimum evidence.** The retriever returns no records below `0.08`.
Showing no knowledge is safer than presenting a weak match as policy.

**Limit output to two records.** A reviewer needs focused evidence, not every
document containing a related word.

**Do not generate an answer.** Retrieval quality can be evaluated directly.
Draft generation would combine retrieval and language-model behavior, making
failures harder to attribute.

**Fit the tiny index per request.** Six records train in negligible time and
there is no invalidation concern. Cache or persist the index after measured
latency or knowledge volume justifies it.

### 6. Alternatives

| Alternative | Why deferred | Revisit when |
| --- | --- | --- |
| Embedding model | Adds model and dependency decisions before lexical retrieval is measured | Reviewed queries show important semantic misses |
| Vector database | Six records fit in memory | Knowledge volume or tenant isolation requires indexed storage |
| Category-filtered retrieval | A routing error could hide correct evidence | Category quality is sufficient and cross-category evidence is intentionally excluded |
| LLM response generation | Retrieval must be trustworthy first | Cited retrieval has a stable evaluation set |
| Markdown ingestion and chunking | Policies are already short records | Real documents exceed useful retrieval size |
| Cached or serialized index | Current index creation is negligible | Profiling shows retrieval latency matters |

### 7. Evaluation

The retrieval tests verify:

- a delayed package query returns `HH-SHIP-001` first;
- every returned match includes a positive similarity score;
- unrelated text returns no matches;
- the knowledge set contains six unique source IDs.

The browser tests verify:

- a shattered-item ticket displays `HH-DMG-001`;
- an unrelated ticket displays `No relevant knowledge found. Do not answer
  from memory.`

The classifier corpus, threshold, predictions, and evaluation metrics remain
unchanged.

### 8. Outcome

The reviewer can now see policy evidence beside the routing suggestion and can
identify exactly which HarborHome record supplied it. Unsupported queries
produce no evidence instead of a plausible but unrelated policy.

This milestone establishes the retrieval half of retrieval-augmented
generation without introducing generation. Routing errors and retrieval errors
remain independently observable.

### 9. Limitations

- The knowledge base contains only six synthetic records.
- Retrieval is document-level and may return a broader passage than needed.
- TF-IDF depends on shared vocabulary and does not understand deeper semantic
  equivalence.
- The `0.08` threshold is a starter heuristic, not a calibrated business-risk
  boundary.
- Source IDs are visible but do not yet link to full policy documents.
- The index is rebuilt for every request.
- There is no retrieval evaluation corpus with graded relevance judgments.
- No response is drafted, approved, edited, rejected, or persisted.

### 10. Next question

Can the system draft a response using only the retrieved HarborHome evidence,
include its citations, refuse when evidence is missing, and require a reviewer
to approve, edit, or reject the draft?
