"""Query selection for the reconciliation state machine."""

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
    """Choose one high-value payment/group query, or return ``None``.

    The expected loss of accepting the current best link is approximated as
    ``(1-p) * 5 * value``.  Querying is worthwhile when that avoidable loss
    exceeds the fixed query cost.
    """
    used = {str(item) for item in state.get("queried", [])}
    if _number(state.get("queries_used")) >= MAX_QUERIES:
        return None

    best: tuple[int, str] | None = None
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
            (float(item.get("confidence", item.get("score", 0.0)) or 0.0)
             for item in options),
            reverse=True,
        )
        top, second = confidences[:2]
        hints_conflict = bool(
            payment.get("conflicting_hints", payment.get("conflict", False))
        )
        ambiguous = hints_conflict or (top - second <= 0.15)
        if not ambiguous:
            continue
        value = abs(
            _number(payment.get("amount", payment.get("gross", payment.get("value", 0))))
        )
        expected_gain = value * max(0.0, 1.0 - top)
        # A query prevents a wrong link (four extra units beyond suspense).
        expected_gain *= 5
        if expected_gain <= QUERY_COST:
            continue
        rank = int(expected_gain)
        if best is None or rank > best[0]:
            best = (rank, payment_id)
    return best[1] if best else None


def record_query(state: Mapping[str, Any], payment_id: str) -> dict[str, Any]:
    """Return a JSON-serializable state after issuing one query."""
    result = dict(state)
    queried = list(result.get("queried", []))
    if payment_id not in queried:
        queried.append(payment_id)
    result["queried"] = queried
    result["queries_used"] = int(result.get("queries_used", 0)) + 1
    result["phase"] = "awaiting_query"
    return result
