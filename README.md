# ReconIQ

**A conservative payment-reconciliation engine that matches incoming payments to invoices, and knows when to leave a payment unresolved.**

ReconIQ takes a batch of payments, invoices, bank credits and related records, then decides which payment settles which invoice. Its design rule is simple: **a wrong match costs far more than an unmatched payment**, so anything uncertain is sent to manual review instead of being guessed.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Module Interfaces](#module-interfaces)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Single entry point**: `decide(observation, state)` returns either a query action or a final posting.
- **Asymmetric-risk matching**: wrong links are treated as roughly 5x costlier than unresolved payments (a wrong link costs 5 units, suspense costs 1).
- **Confidence-gated posting**: payments post only above a confidence threshold, with a stricter threshold for ambiguous payer families.
- **Budgeted verification queries**: at most 4 queries per run, one per decision step, never repeated for the same payment, and only when the expected avoided loss exceeds the query cost.
- **Hard safety validator**: every posting is checked for duplicate invoice use, over-capture, cancelled invoices, merchant mismatch, currency mismatch and review capacity before being returned.
- **Fail-safe by default**: any exception falls back to a posting with *zero* allocations and every payment unresolved.
- **Idempotent input handling**: retried deliveries of transactions, bank credits and remittance documents are de-duplicated.
- **JSON-serializable state** that can be persisted between decision steps.

## How It Works

```
raw observation
      │
      ▼
 prepare.py        normalise currencies / strings, de-duplicate retried records
      │
      ▼
 rules.py /        generate and score candidate invoices per payment
 matcher.py /      (fuzzy names, amount, currency, date proximity,
 scorer.py          optimal bipartite assignment)
      │
      ▼
 query.py          pick at most one high-value, ambiguous payment to verify
      │             (expected-value test against the 180-unit query cost)
      ▼
 solution.py       build allocations → validate_posting() → post or fall back
```

Run the checker with `python local_check.py solution.py training.json` (or
`practice.json`), the focused tests with
`python -m unittest -v test_solution.py`, and the evaluator with
`python evaluate.py solution.py training.json`.

**Decision flow in `solution.py`:**

1. Clean the observation and drop duplicate payments (tracked in `duplicate_map`).
2. Build a ranked candidate list for each payment.
3. Ingest any verified answers from earlier queries. Verified answers are authoritative, **including a `null` answer, which leaves the payment unresolved**.
4. If a query is worth its cost, return `{"op": "query", "payment_id": ...}`.
5. Otherwise build complete ledger/suspense postings and return `{"op": "submit", ...}`.

Training-only tuning selected `ACCEPT_THRESHOLD = 0.85`,
`AMBIGUOUS_THRESHOLD = 0.92`, minimum query value `180`, and top-two
closeness `0.15`. With the integrated matcher, the average scores were
**53.7947** on training (60 batches) and **45.5308** on practice (20 batches);
both evaluations reported zero integrity violations. Practice was evaluated
once after tuning.
Verified query answers are authoritative, including a null answer, which
remains unresolved. Every final posting supplies complete ledger rows and
passes duplicate, capture, cancellation, merchant, currency, and conservation
checks.

**Query selection** (`query.py`): a payment is queried only if it has two or more candidates, the top two are within 0.15 confidence (or the hints conflict), and `amount × (1 − top_confidence) × 4 > 180`.

## Project Structure

```
ReconIQ/
├── solution.py              # decide() state machine + final safety validator
├── query.py                 # expected-value query selection
├── prepare.py               # input normalisation and de-duplication
├── rules.py                 # rule-based candidate scoring and balance checks
├── scorer.py                # weighted confidence scoring (stdlib difflib)
├── matcher.py               # candidate generation + optimal assignment (scipy)
├── parser.py                # observation → typed objects (work in progress)
├── models.py                # dataclasses: Invoice, Payment, BankCredit, Refund, Batch
├── evaluate.py              # scores a solution against a JSON dataset
├── interfaces.md            # contracts for parser / rules / matcher modules
├── test_solution.py         # unittest suite for the decision layer
├── test_matcher_scorer.py   # pytest suite for scorer + matcher
├── test_rules.py            # tests for the rules module
└── test_parser.py           # tests for the parser
```

## Getting Started

### Prerequisites

- Python 3.10+ (the code uses `X | None` type syntax)
- The decision layer (`solution.py`, `query.py`) is dependency-free.
- The scoring and matching engine additionally needs `numpy` and `scipy`.
- `pytest` is needed for `test_matcher_scorer.py`.

### Installation

```bash
git clone https://github.com/projectsbytejas/ReconIQ.git
cd ReconIQ

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy scipy pytest
```

## Usage

```python
from solution import decide

state = {}
observation = {
    "payments": [
        {"id": "p1", "amount": 10000, "currency": "USD", "merchant_id": "m1"},
    ],
    "invoices": [
        {"id": "i1", "amount": 10000, "currency": "USD", "merchant_id": "m1"},
    ],
    "candidates": {
        "p1": [{"invoice_id": "i1", "confidence": 0.97}],
    },
}

result = decide(observation, state)
action, state = result["action"], result["state"]

print(action)
# {'type': 'post', 'allocations': [{'payment_id': 'p1', 'invoice_id': 'i1', 'amount': 10000}], 'unresolved': []}
```

If `action["type"] == "query"`, resolve the query externally, add the answer to `observation["query_results"][payment_id]`, and call `decide()` again with the returned `state`.

### State

The `state` object is plain JSON and carries everything needed between steps:

| Key | Purpose |
| --- | --- |
| `queries_used` | Number of queries issued so far (capped at 4) |
| `queried` | Payment IDs already queried (never repeated) |
| `query_results` | Verified answers keyed by payment ID |
| `duplicate_map` | Duplicate payment ID → canonical payment ID |
| `pending` | Payment IDs currently unresolved |
| `phase` | `awaiting_query` or `final` |

## Configuration

| Constant | Default | Location | Meaning |
| --- | --- | --- | --- |
| `ACCEPT_THRESHOLD` | `0.85` | `solution.py` | Minimum confidence to auto-post a clear match |
| `AMBIGUOUS_THRESHOLD` | `0.92` | `solution.py` | Stricter threshold for ambiguous payer families or close runner-ups |
| `REVIEW_CAPACITY` | `4` | `solution.py` | Max unresolved payments the manual review queue can absorb |
| `MAX_QUERIES` | `4` | `query.py` | Hard cap on verification queries |
| `QUERY_COST` | `180` | `query.py` | Cost of one query, in minor currency units |
| `HIGH_CONFIDENCE_THRESHOLD` | `0.85` | `scorer.py` | Minimum score for the matcher to accept a pair |

Scoring weights live in `scorer.WEIGHTS`: amount 0.40, currency 0.15, merchant 0.15, reference 0.15, date 0.10, status 0.05.

## Testing

```bash
# Decision layer
python -m unittest -v test_solution.py

# Scorer and matcher
pytest test_matcher_scorer.py -v

# Everything pytest can discover
pytest -v
```

## Evaluation

`evaluate.py` runs `decide()` over a dataset and reports an average score, total queries and integrity violations.

```bash
python evaluate.py solution.py training.json
python evaluate.py solution.py practice.json
```

The dataset can be a list of observations or an object with a `batches` list; ground truth is read from `expected_allocations` or `truth`. The score per batch is:

```
score = 100 × (1 − (unresolved_value + 4 × wrong_value + 180 × queries_used) / total_value)
```

The script exits non-zero if the query cap is exceeded or a payment is queried twice. Dataset files (`training.json`, `practice.json`) are git-ignored and must be supplied separately.

## Module Interfaces

`solution.py` is built around three replaceable seams, documented in [`interfaces.md`](interfaces.md):

```python
def parse_observation(observation) -> tuple[list[dict], list[dict]]:
    """Return clean payment and invoice records."""

def score_candidates(payment, invoices) -> list[dict]:
    """Return candidate invoices with a confidence in [0, 1]."""

def match_payment(payment, invoices) -> list[dict]:
    """Return candidates ordered from most to least confident."""
```

Optional modules are imported if present; otherwise the system falls back to leaving payments unresolved when evidence is insufficient. The safety validator is always the final authority, regardless of which modules are plugged in.

## Project Status

ReconIQ is under active development. Known gaps in the current snapshot:

- **Integration wiring is incomplete.** `solution.py` imports `match_payment` from `matcher.py`, but `matcher.py` currently exposes `match()` and `generate_candidates()`. `prepare()` returns a cleaned observation dict while `solution.py` expects a `(payments, invoices)` tuple. Until these line up, the fail-safe path leaves payments unresolved unless `candidates` are supplied in the observation.
- **`parser.py` / `models.py`** are early drafts (`load_batch` does not yet populate every `Batch` field).
- **Naming**: some module docstrings still say "EcoRecon", the project's earlier working name.
- `local_check.py`, referenced in older notes, is not part of this repository.

## Contributing

Contributions are welcome.

1. Fork the repository and create a feature branch.
2. Add or update tests for your change.
3. Make sure the test suites above pass.
4. Open a pull request describing what changed and why.

When changing matching or posting logic, keep the core principle intact: **when in doubt, leave it unresolved.**

## License

No license has been specified yet. Add a `LICENSE` file (for example MIT or Apache-2.0) before publishing if you want others to be able to use the code.
