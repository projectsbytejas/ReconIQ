# ReconIQ - hackathon implementation notes

Submit **only `solution.py`** (standard library only, deterministic, no ML).

## Contract
`decide(observation, state)` returns `{"action", "state"}`. Action is either
`{"op":"query","payment_id"}` or `{"op":"submit","matches","ledger","suspense","bank_total"}`
exactly as in SCHEMA.md. State is `{"queried":[...]}`; investigations are read from
`observation["investigations"]`, so a payment is never queried twice.

## Pipeline
1. Dedupe transactions by `transaction_id`, bank credits by `bank_id`; integer money only.
2. Payment group totals from unique transactions (captured/refunded/fees/net).
3. Hard eligibility: merchant, normalised currency, captured>0, captured<=amount,
   refunded<=captured, cancelled invoice only if captured==refunded.
4. Soft evidence (log-odds): source_ref/tax suffix (raw payment + receipt = ONE origin),
   independent active remittance lineages (void/superseded/receipt ignored), document
   sender names, day gap 0-3 (soft only). A payment whose tax suffix matches no eligible
   invoice gets a large "unmatched" boost.
5. Exact one-to-one marginals by enumeration per connected component; matches chosen to
   maximise expected score `v*(5p-4)`, so low-confidence links are left in suspense.
6. Queries: greedy by expected gain `v*P(real) - 180 - current expected gain`; verified
   invoice locks the payment, verified null puts it in suspense.
7. Ledger/suspense/bank_total built from unique transactions; the fail-safe path still
   returns a valid `op:"submit"` (everything in suspense).

Weights are hand-set (rounded) after tabulating evidence patterns on training.json, and
`TEMPERATURE=1.35` was chosen by comparing training and practice scores. Not ML, but it is
tuned on the supplied data, so hidden-set scores may be lower.

## Verified results (actual `local_check.py` output)
- practice.json (20): mean 94.8868, queries 80, wrong_loss 162000, unresolved 75000
- training.json (60): mean 94.2069, queries 240, wrong_loss 470000, unresolved 238000
- No exceptions or assertion failures (integrity failures: 0). `pytest`: 7 passed.

Commands:
```
python -m pytest -q
python local_check.py solution.py practice.json
python local_check.py solution.py training.json
```

## Known limits
Remaining loss is mostly adversarial cases where the raw payment (ref + tax) and a single
independent document point to different invoices; these are genuinely ambiguous and only
the 4 queries can settle them.
