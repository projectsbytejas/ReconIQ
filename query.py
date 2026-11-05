"""Investigation selection for the v6 reconciliation contract."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

QUERY_COST = 180
MAX_QUERIES = 4
MINIMUM_QUERY_VALUE = QUERY_COST
QUERY_CLOSENESS = 0.15


def _amount(payment_id: str, totals: Mapping[str, int]) -> int:
    return abs(int(totals.get(payment_id, 0)))


def choose_query(
    payments: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    values: Mapping[str, int],
    state: Mapping[str, Any],
    *,
    minimum_value: int = QUERY_COST,
    closeness: float = 0.15,
) -> str | None:
    """Choose one unqueried high-value ambiguous payment."""
    queried = {str(item) for item in state.get("queried", [])}
    if int(state.get("queries_used", 0)) >= MAX_QUERIES:
        return None
    best: tuple[float, str] | None = None
    for payment in payments:
        pid = str(payment.get("payment_id", ""))
        if not pid or pid in queried or _amount(pid, values) <= minimum_value:
            continue
        options = sorted(
            candidates.get(pid, ()),
            key=lambda item: float(item.get("confidence", 0)),
            reverse=True,
        )
        if len(options) < 2:
            continue
        top = max(0.0, min(1.0, float(options[0].get("confidence", 0))))
        second = max(0.0, min(1.0, float(options[1].get("confidence", 0))))
        if not payment.get("conflicting_hints", False) and top - second > closeness:
            continue
        probability_wrong = 0.5 if top - second <= closeness else (1.0 - top)
        gain = _amount(pid, values) * probability_wrong * 4
        if gain > QUERY_COST and (best is None or gain > best[0]):
            best = (gain, pid)
    return best[1] if best else None


def record_query(state: Mapping[str, Any], payment_id: str) -> dict[str, Any]:
    result = dict(state)
    queried = list(result.get("queried", []))
    if payment_id not in queried and len(queried) < MAX_QUERIES:
        queried.append(payment_id)
    result["queried"] = queried
    result["queries_used"] = len(queried)
    result["phase"] = "query"
    return result
