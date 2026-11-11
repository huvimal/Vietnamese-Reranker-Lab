# Vietnamese Reranker Lab

A CPU-only learning-to-rank experiment for Vietnamese document search. BM25 retrieves
20 candidates; a pairwise linear model reranks the **same** candidates. No API key,
external dependency, GPU, model download, or generated answer is required.

This is a feature-based learned ranker, **not a transformer or cross-encoder**.
It is an experimental training/evaluation pipeline, not a production search service.

## Run on Windows

Python 3.11 or newer is recommended. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe lab.py train
.\.venv\Scripts\python.exe lab.py evaluate --split test
.\.venv\Scripts\python.exe lab.py search --query "Sau khi duyệt trả SP001, khi nào tiền về tài khoản?"
Get-Content reports/test.json -Encoding utf8
```

The committed collection is ready to use. `python make_data.py` deterministically
regenerates it. You do not need to install packages or activate the virtual environment.
Commands resolve input/output paths relative to the project, not the working directory.

## Collection and split

`data/collection.json` contains 180 synthetic policy documents and 360 queries:

| Split | Products | Queries |
| --- | ---: | ---: |
| train | 18 | 216 |
| dev | 6 | 72 |
| test | 6 | 72 |

There are six intents: returns, warranty, shipping, refunds, cancellation, and payment.
Each query has one binary-relevant document. The split groups all queries for a
product together. Positive documents must belong to the query's product group.
Input validation rejects duplicate IDs, invalid relevance references, and group leakage.

The full corpus is indexed at all stages, as in a fixed-corpus retrieval experiment;
BM25 corpus statistics therefore include all documents. Only **train query labels**
produce training pairs. This is not an inductive evaluation on an unseen corpus.

## Learning and model selection

1. Normalize Unicode and accents; tokenize into word-like units. This is not a
   dedicated Vietnamese linguistic word segmenter.
2. Retrieve the top 20 BM25 candidates for each training query.
3. Pair retrieved positives with up to five high-scoring nonrelevant candidates.
   A missed positive is counted and skipped, never inserted into the candidate list.
4. Train a linear score `w · features(query, document)` by pairwise logistic loss
   `log(1 + exp(-(score_positive - score_negative)))` and L2 regularization.
5. Select L2 from 0.001, 0.01, and 0.1 using **dev nDCG@5** only, with deterministic
   first-choice tie breaking. Do not tune parameters based on the test report.
6. Independently train a lexical-feature ablation with the same dev-selection rule.
7. Freeze the model and evaluate the held-out product groups.

The full model uses BM25, query/title coverage, bigram and character overlap,
length ratio, product-code matching/conflict, and six manually engineered intent
agreement features. Intent cues come from text, not qrels, metadata, or document IDs.
They encode domain knowledge and are a major limitation of this synthetic experiment.
The lexical ablation uses only the first five features, with its own trained weights.

`models/reranker.json` saves weights, training settings, candidate count, loss curves,
and SHA-256 fingerprints. Evaluation rejects changed collection/feature implementation
until the model is retrained. Fingerprints support reproducibility, not tamper resistance.

## Reports

- `reports/training.json`: six dev experiments, losses, elapsed time, model fingerprint.
- `reports/test.json`: BM25/lexical/full metrics, per-query rankings, failures, fingerprints.
- `models/reranker.json`: selected models and training metadata.

Metrics are mean nDCG@5, truncated MRR@5, Recall@5, and candidate recall at the
configured retrieval depth. A paired **product-group** bootstrap estimates the
nDCG difference between the full ranker and BM25. It resamples groups, not individual
correlated queries. Six test groups and shared templates make uncertainty estimates
weak; identical group deltas can produce a zero-width interval.

Observed default run on the synthetic collection:

| Method | nDCG@5 | MRR@5 | Recall@5 | Candidate recall@20 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.8494 | 0.7986 | 1.0000 | 1.0000 |
| Learned lexical | 0.8801 | 0.8403 | 1.0000 | 1.0000 |
| Learned full | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

These numbers describe this small templated fixture only. Perfect scores do not
demonstrate general Vietnamese retrieval quality or semantic language understanding.
Latency includes retrieval and reranking in one local process; it is not a load test.
Ranking scores are not probabilities and are not calibrated answerability estimates.

## Inspect behavior

```powershell
.\.venv\Scripts\python.exe lab.py search --query "SP005 được đổi trả trong mấy ngày?"
.\.venv\Scripts\python.exe lab.py search --query "Mua SP012 cần bao nhiêu tiền để miễn phí giao hàng?"
.\.venv\Scripts\python.exe lab.py search --query "Tôi có thể trả tiền mặt lúc nhận SP020 không?"
```

Search prints the initial and reranked top five passages. It does not judge arbitrary
free-text answers, call an LLM, or refuse queries outside the corpus.

## Extend with independently labeled data

Replace the collection with permitted public or original data using the same schema:

```json
{
  "version": "my-v1",
  "synthetic": false,
  "documents": [
    {"id": "d1", "group": "product-a", "title": "Title", "text": "Passage"}
  ],
  "queries": [
    {"id": "q1", "group": "product-a", "split": "train", "text": "Question", "qrels": {"d1": 1}}
  ]
}
```

The snippet only illustrates one record: a runnable collection needs all three
disjoint group splits, relevant passages, and sufficient negative documents.
For real data, annotate candidate negatives too: this demo assumes every unlisted
document is nonrelevant. Incomplete judgments can create false negatives.
Use independent wording/authors and independently reviewed labels; include ambiguous,
unanswerable, and difficult cases in a separately defined evaluation protocol.
Adapt domain features using train/dev examples, freeze them, and reserve a new test
set for any further iteration after viewing test results. If you modify the feature
schema, update its version/implementation and retrain.

A neural comparison is a future extension, not an implemented capability here.
Official references:
- [Sentence Transformers reranker training examples](https://www.sbert.net/examples/cross_encoder/training/rerankers/README.html)
- [Stanford IR book: evaluation of ranked retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)

## Continuous integration

The workflow runs unit tests, trains models, evaluates test fixtures, and uploads
reports. A completed experiment does **not** enforce a quality floor or promise
improvement over BM25: metric changes are results to inspect, not automatically failures.

Generated models and reports are gitignored. Keep the collection, training code,
tests, and workflow in version control; retain selected run reports as release
artifacts when publishing findings.
