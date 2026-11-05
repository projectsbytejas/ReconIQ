"""Conservative payment-to-invoice reconciliation decision function."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from query import MAX_QUERIES, choose_query, record_query

ACCEPT_THRESHOLD = 0.83
AMBIGUOUS_THRESHOLD = 0.92


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
            pass
    return 0


def _currency(record: Mapping[str, Any]) -> str:
    return str(record.get("currency", record.get("currency_code", ""))).upper()


def _merchant(record: Mapping[str, Any]) -> str:
    return str(record.get("merchant_id", record.get("merchant", "")))


def _records(observation: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    value = observation.get(name, [])
    return [dict(item) for item in value] if isinstance(value, list) else []


def _unique_payments(
    payments: Sequence[Mapping[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collapse repeated deliveries without changing the original payment id."""
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
) -> dict[str, list[dict[str, Any]]]:
    supplied = observation.get("candidates", observation.get("matches", {}))
    result: dict[str, list[dict[str, Any]]] = {}
    by_id = {_id(invoice, "invoice_id", "id"): invoice for invoice in invoices}
    for payment in payments:
        pid = _id(payment, "payment_id", "id", "transaction_id")
        raw = supplied.get(pid, []) if isinstance(supplied, dict) else []
        if not raw:
            raw = payment.get("candidates", [])
        options: list[dict[str, Any]] = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, str):
                item = {"invoice_id": item}
            candidate = dict(item)
            iid = _id(candidate, "invoice_id", "id")
            if iid in by_id:
                candidate.setdefault("invoice", by_id[iid])
            candidate["invoice_id"] = iid
            try:
                candidate["confidence"] = float(
                    candidate.get("confidence", candidate.get("score", 0.0))
                )
            except (TypeError, ValueError):
                candidate["confidence"] = 0.0
            options.append(candidate)
        result[pid] = sorted(
            (item for item in options if item["invoice_id"]),
            key=lambda item: item["confidence"],
            reverse=True,
        )
    return result


def _query_answer(observation: Mapping[str, Any], payment_id: str) -> Any:
    answers = observation.get("query_results", observation.get("query_answer", {}))
    if isinstance(answers, dict):
        return answers.get(payment_id)
    return None


def _valid_link(payment: Mapping[str, Any], invoice: Mapping[str, Any]) -> bool:
    if invoice.get("cancelled", invoice.get("status") == "cancelled"):
        return False
    if _currency(payment) and _currency(invoice) and _currency(payment) != _currency(invoice):
        return False
    if _merchant(payment) and _merchant(invoice) and _merchant(payment) != _merchant(invoice):
        return False
    return True


def _posting(
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
            confidence = 1.0
        else:
            options = list(candidates.get(pid, ()))
            if not options:
                unresolved.append(pid)
                continue
            chosen = options[0]
            confidence = float(chosen.get("confidence", 0.0))
            second = float(options[1].get("confidence", 0.0)) if len(options) > 1 else 0.0
            threshold = AMBIGUOUS_THRESHOLD if confidence - second <= 0.15 else ACCEPT_THRESHOLD
            iid = _id(chosen, "invoice_id", "id")
            if confidence < threshold:
                unresolved.append(pid)
                continue
        invoice = invoices.get(iid)
        if not iid or invoice is None or iid in used or not _valid_link(payment, invoice):
            unresolved.append(pid)
            continue
        amount = _amount(payment)
        if captured.get(iid, 0) + amount > _amount(invoice):
            unresolved.append(pid)
            continue
        used.add(iid)
        captured[iid] = captured.get(iid, 0) + amount
        allocations.append({"payment_id": pid, "invoice_id": iid, "amount": amount})
    return allocations, unresolved


def decide(observation: Mapping[str, Any], state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return one query action or a final posting plus JSON-serializable state."""
    current = dict(state or {})
    current.setdefault("queries_used", 0)
    current.setdefault("queried", [])
    current.setdefault("query_results", {})
    current.setdefault("duplicate_map", {})
    current.setdefault("pending", [])
    payments = _unique_payments(
        _records(observation, "payments") or _records(observation, "transactions"),
        current,
    )
    invoices_list = _records(observation, "invoices")
    invoices = {_id(item, "invoice_id", "id"): item for item in invoices_list}
    candidates = _candidate_map(observation, payments, invoices_list)

    for pid in list(current["queried"]):
        if pid in current["query_results"]:
            continue
        answer = _query_answer(observation, pid)
        if answer is not None or pid in observation.get("query_results", {}):
            current["query_results"][pid] = answer

    query_id = choose_query(payments, candidates, current)
    if query_id is not None and current["queries_used"] < MAX_QUERIES:
        next_state = record_query(current, query_id)
        return {"action": {"type": "query", "payment_id": query_id}, "state": next_state}

    allocations, unresolved = _posting(payments, invoices, candidates, current)
    current["pending"] = unresolved
    current["phase"] = "final"
    return {
        "action": {"type": "post", "allocations": allocations, "unresolved": unresolved},
        "state": current,
    }
