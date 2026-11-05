"""Expected-value query selection for reconciliation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

QUERY_COST = 180
MAX_QUERIES = 4


def _number(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def choose_query(
    payments: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    state: Mapping[str, Any],
) -> str | None:
    """Return the highest-value unqueried payment, at most once per step."""
    used = {str(item) for item in state.get("queried", [])}
    if _number(state.get("queries_used")) >= MAX_QUERIES:
        return None

    best: tuple[float, str] | None = None
    for payment in payments:
        payment_id = str(
            payment.get("payment_id", payment.get("id", payment.get("transaction_id", "")))
        )
        if not payment_id or payment_id in used:
            continue
        options = list(candidates.get(payment_id, ()))
        if len(options) < 2:
            continue
        confidences = sorted(
            (max(0.0, min(1.0, float(item.get("confidence", item.get("score", 0.0)) or 0.0)))
             for item in options),
            reverse=True,
        )
        top, second = confidences[:2]
        conflicting = bool(
            payment.get("conflicting_hints", payment.get("conflict", False))
        )
        if not conflicting and top - second > 0.15:
            continue
        value = abs(
            _number(payment.get("amount", payment.get("gross", payment.get("value", 0))))
        )
        if value <= QUERY_COST:
            continue
        # Querying avoids the five-unit wrong-link loss; suspense is the
        # one-unit alternative, so only the four-unit incremental loss is
        # attributable to the query.
        gain = value * max(0.0, 1.0 - top) * 4
        if gain <= QUERY_COST:
            continue
        if best is None or gain > best[0]:
            best = (gain, payment_id)
    return best[1] if best else None


def record_query(state: Mapping[str, Any], payment_id: str) -> dict[str, Any]:
    """Return state after issuing one query, preserving JSON-compatible data."""
    result = dict(state)
    queried = list(result.get("queried", []))
    if payment_id not in queried and len(queried) < MAX_QUERIES:
        queried.append(payment_id)
        result["queries_used"] = len(queried)
    result["queried"] = queried
    result["phase"] = "awaiting_query"
    return result
