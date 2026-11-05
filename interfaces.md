# Reconciliation interfaces

The solution is intentionally dependency-free. These are the seams that can be
replaced by a parser, rule engine, or matcher without changing the state
machine:

```python
from typing import Any, Mapping, Sequence, TypedDict

def parse_observation(
    observation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return clean payment and invoice records."""

def score_candidates(
    payment: Mapping[str, Any],
    invoices: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return candidate invoice records with a confidence in [0, 1]."""

def match_payment(
    payment: Mapping[str, Any],
    invoices: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return candidates ordered from most to least confident."""
```

`solution.py` provides conservative placeholders for these interfaces until
specialized modules are supplied.
