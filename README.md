# ReconIQ

Payment reconciliation integration layer. `solution.py` exposes:

```python
decide(observation, state) -> {"action": {...}, "state": {...}}
```

Run the checker with `python local_check.py solution.py training.json` (or
`practice.json`), the focused tests with
`python -m unittest -v test_solution.py`, and the evaluator with
`python evaluate.py solution.py training.json`.

State is JSON-serializable and contains `queries_used`, `queried`,
`query_results`, `duplicate_map`, `pending`, and `phase`. At most four queries
are issued, never more than one per decision step, and repeated IDs are
skipped. A query is selected only for a high-value payment with multiple close
candidate matches when its expected avoided wrong-link loss exceeds 180 minor
units.

Training-only tuning selected `ACCEPT_THRESHOLD = 0.85`,
`AMBIGUOUS_THRESHOLD = 0.92`, minimum query value `180`, and top-two
closeness `0.15`. The resulting average scores were **65.2270** on training
(60 batches) and **54.8853** on practice (20 batches); both evaluations
reported zero integrity violations. Practice was evaluated once after tuning.
Verified query answers are authoritative, including a null answer, which
remains unresolved. Every final posting supplies complete ledger rows and
passes duplicate, capture, cancellation, merchant, currency, and conservation
checks.

Parser, rule-engine, and matcher integration points are documented in
`interfaces.md`. Optional teammate modules are imported when available;
otherwise the explicit fallback stubs keep all payments safely unresolved
when evidence is insufficient.
