"""Integration decision function for payment reconciliation.

Teammate parser/rule/matcher modules are optional.  When they are added, expose
the functions documented in ``interfaces.md`` and update the three imports in
``_integrations``; the safety validator remains the final authority.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from query import MAX_QUERIES, choose_query, record_query

ACCEPT_THRESHOLD = 0.85
AMBIGUOUS_THRESHOLD = 0.92
REVIEW_CAPACITY = 4


def _id(record: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return str(value)
    return ""


def _amount(record: Mapping[str, Any]) -> int:
    for name in ("amount", "gross", "value", "minor_units"):
        try:
            return abs(int(record.get(name, 0)))
        except (TypeError, ValueError):
            continue
    return 0


def _currency(record: Mapping[str, Any]) -> str:
    return str(record.get("currency", record.get("currency_code", ""))).upper()


def _merchant(record: Mapping[str, Any]) -> str:
    return str(record.get("merchant_id", record.get("merchant", "")))


def _records(observation: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    value = observation.get(name, [])
    return [dict(item) for item in value] if isinstance(value, list) else []


def _integrations(
    observation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    """Use teammate modules when present, otherwise use explicit TODO stubs."""
    # TODO(teammate-1/2): replace these optional imports with landed modules.
    parser = None
    rule_engine = None
    matcher = None
    try:
        from prepare import prepare as parser # type: ignore
    except ImportError:
        pass
    try:
        from rules import score_candidates as rule_engine  # type: ignore
    except ImportError:
        pass
    try:
        from matcher import match_payment as matcher  # type: ignore
    except ImportError:
        pass
    if parser is not None:
        payments, invoices = parser(observation)
        return [dict(item) for item in payments], [dict(item) for item in invoices], matcher
    return (
        _records(observation, "payments") or _records(observation, "transactions"),
        _records(observation, "invoices"),
        matcher,
    )


def _unique_payments(
    payments: Sequence[Mapping[str, Any]], state: dict[str, Any]
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    duplicate_map = dict(state.get("duplicate_map", {}))
    for payment in payments:
        payment_id = _id(payment, "payment_id", "id", "transaction_id")
        duplicate_of = _id(payment, "duplicate_of", "duplicate_transaction_id")
        key = duplicate_of or payment_id
        if key and key in seen:
            if payment_id:
                duplicate_map[payment_id] = key
            continue
        if key:
            seen.add(key)
        unique.append(dict(payment))
    state["duplicate_map"] = duplicate_map
    return unique


def _candidate_map(
    observation: Mapping[str, Any],
    payments: Sequence[Mapping[str, Any]],
    invoices: Sequence[Mapping[str, Any]],
    matcher: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    supplied = observation.get("candidates", observation.get("matches", {}))
    by_id = {_id(invoice, "invoice_id", "id"): invoice for invoice in invoices}
    result: dict[str, list[dict[str, Any]]] = {}
    for payment in payments:
        pid = _id(payment, "payment_id", "id", "transaction_id")
        raw = supplied.get(pid, []) if isinstance(supplied, dict) else []
        if not raw and matcher is not None:
            raw = matcher(payment, invoices)
        if not raw:
            raw = payment.get("candidates", [])
        options: list[dict[str, Any]] = []
        for item in raw if isinstance(raw, list) else []:
            candidate = {"invoice_id": item} if isinstance(item, str) else dict(item)
            iid = _id(candidate, "invoice_id", "id")
            candidate["invoice_id"] = iid
            candidate["invoice"] = by_id.get(iid, candidate.get("invoice", {}))
            try:
                candidate["confidence"] = max(
                    0.0, min(1.0, float(candidate.get("confidence", candidate.get("score", 0.0))))
                )
            except (TypeError, ValueError):
                candidate["confidence"] = 0.0
            if iid:
                options.append(candidate)
        result[pid] = sorted(options, key=lambda item: item["confidence"], reverse=True)
    return result


def _query_answer(observation: Mapping[str, Any], payment_id: str) -> tuple[bool, Any]:
    answers = observation.get("query_results", observation.get("query_answer", {}))
    if isinstance(answers, dict) and payment_id in answers:
        return True, answers[payment_id]
    return False, None


def _valid_link(payment: Mapping[str, Any], invoice: Mapping[str, Any]) -> bool:
    if invoice.get("cancelled", invoice.get("status") == "cancelled"):
        return False
    return not (
        _currency(payment)
        and _currency(invoice)
        and _currency(payment) != _currency(invoice)
    ) and not (
        _merchant(payment)
        and _merchant(invoice)
        and _merchant(payment) != _merchant(invoice)
    )


def _build_allocations(
    payments: Sequence[Mapping[str, Any]],
    invoices: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    allocations: list[dict[str, Any]] = []
    unresolved: list[str] = []
    used: set[str] = set()
    captured: dict[str, int] = {}
    verified = state.get("query_results", {})
    for payment in payments:
        pid = _id(payment, "payment_id", "id", "transaction_id")
        if pid in verified:
            chosen = verified[pid]
            if chosen is None:
                unresolved.append(pid)
                continue
            iid = _id(chosen if isinstance(chosen, dict) else {"id": chosen}, "invoice_id", "id")
        else:
            options = list(candidates.get(pid, ()))
            if not options:
                unresolved.append(pid)
                continue
            chosen = options[0]
            confidence = float(chosen.get("confidence", 0.0))
            second = float(options[1].get("confidence", 0.0)) if len(options) > 1 else 0.0
            family = bool(payment.get("payer_family", payment.get("ambiguous_payer_family", False)))
            threshold = AMBIGUOUS_THRESHOLD if family or confidence - second <= 0.15 else ACCEPT_THRESHOLD
            iid = _id(chosen, "invoice_id", "id")
            if confidence < threshold:
                unresolved.append(pid)
                continue
        invoice = invoices.get(iid)
        amount = _amount(payment)
        if (
            not iid
            or invoice is None
            or iid in used
            or not _valid_link(payment, invoice)
            or captured.get(iid, 0) + amount > _amount(invoice)
        ):
            unresolved.append(pid)
            continue
        used.add(iid)
        captured[iid] = captured.get(iid, 0) + amount
        allocations.append({"payment_id": pid, "invoice_id": iid, "amount": amount})
    return allocations, unresolved


def _within_capacity(observation: Mapping[str, Any], unresolved: Sequence[str]) -> bool:
    capacity = observation.get(
        "review_capacity",
        observation.get("capacity", observation.get("max_reviews", REVIEW_CAPACITY)),
    )
    try:
        return len(unresolved) <= int(capacity)
    except (TypeError, ValueError):
        return False


def validate_posting(
    observation: Mapping[str, Any],
    payments: Sequence[Mapping[str, Any]],
    invoices: Mapping[str, Mapping[str, Any]],
    allocations: Sequence[Mapping[str, Any]],
    unresolved: Sequence[str],
) -> bool:
    """Final hard safety filter; unresolved payments are always safe."""
    if not _within_capacity(observation, unresolved):
        return False
    seen: set[str] = set()
    captured: dict[str, int] = {}
    payment_map = {_id(item, "payment_id", "id", "transaction_id"): item for item in payments}
    for allocation in allocations:
        pid = str(allocation.get("payment_id", ""))
        iid = str(allocation.get("invoice_id", ""))
        payment = payment_map.get(pid)
        invoice = invoices.get(iid)
        amount = _amount(allocation)
        if not payment or not invoice or iid in seen or amount != _amount(payment):
            return False
        if not _valid_link(payment, invoice):
            return False
        captured[iid] = captured.get(iid, 0) + amount
        if captured[iid] > _amount(invoice):
            return False
        seen.add(iid)
    return True


def _safe_baseline(
    observation: Mapping[str, Any],
    state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    current = {
        "queries_used": min(MAX_QUERIES, int((state or {}).get("queries_used", 0) or 0)),
        "queried": list((state or {}).get("queried", []))[:MAX_QUERIES],
        "query_results": dict((state or {}).get("query_results", {})),
        "duplicate_map": dict((state or {}).get("duplicate_map", {})),
        "pending": [],
        "phase": "final",
    }
    payments = _records(observation, "payments") or _records(observation, "transactions")
    current["pending"] = [_id(item, "payment_id", "id", "transaction_id") for item in payments]
    return {"action": {"type": "post", "allocations": [], "unresolved": current["pending"]}, "state": current}


def decide(
    observation: Mapping[str, Any], state: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return one query action or a validated final posting."""
    try:
        current = dict(state or {})
        current.setdefault("queries_used", 0)
        current.setdefault("queried", [])
        current.setdefault("query_results", {})
        current.setdefault("duplicate_map", {})
        current.setdefault("pending", [])
        current["queries_used"] = min(MAX_QUERIES, int(current["queries_used"]))
        payments, invoice_list, matcher = _integrations(observation)
        payments = _unique_payments(payments, current)
        invoices = {_id(item, "invoice_id", "id"): item for item in invoice_list}
        candidates = _candidate_map(observation, payments, invoice_list, matcher)
        for pid in list(current["queried"]):
            if pid not in current["query_results"]:
                found, answer = _query_answer(observation, pid)
                if found:
                    current["query_results"][pid] = answer
        query_id = choose_query(payments, candidates, current)
        if query_id is not None:
            return {"action": {"type": "query", "payment_id": query_id}, "state": record_query(current, query_id)}
        allocations, unresolved = _build_allocations(payments, invoices, candidates, current)
        if not validate_posting(observation, payments, invoices, allocations, unresolved):
            return _safe_baseline(observation, current)
        current["pending"] = list(unresolved)
        current["phase"] = "final"
        result = {
            "action": {"type": "post", "allocations": allocations, "unresolved": unresolved},
            "state": current,
        }
        json.dumps(result)
        return result
    except Exception:
        return _safe_baseline(observation, state)
