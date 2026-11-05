# ReconIQ

Payment reconciliation integration layer. `solution.py` exposes:

```python
decide(observation, state) -> {"action": {...}, "state": {...}}
```

Run the focused tests with `python -m unittest -v test_solution.py`. If the
challenge fixtures are supplied, run `python local_check.py solution.py`.
Evaluate a dataset with `python evaluate.py solution.py training.json`, then
repeat once for `practice.json`.

State is JSON-serializable and contains `queries_used`, `queried`,
`query_results`, `duplicate_map`, `pending`, and `phase`. At most four queries
are issued, never more than one per decision step, and repeated IDs are
skipped. A query is selected only for a high-value payment with multiple close
candidate matches when its expected avoided wrong-link loss exceeds 180 minor
units.

The tuned starting threshold is `ACCEPT_THRESHOLD = 0.85`; ambiguous payer
families use `0.92`. Verified query answers are authoritative, including a
null answer, which remains unresolved. Every final posting passes duplicate,
capture, cancellation, merchant, currency, and review-capacity validation.

Parser, rule-engine, and matcher integration points are documented in
`interfaces.md`. Optional teammate modules are imported when available;
otherwise the explicit fallback stubs keep all payments safely unresolved
when evidence is insufficient.
