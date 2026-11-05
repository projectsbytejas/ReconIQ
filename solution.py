"""Conservative, stateful payment-to-invoice reconciliation decision engine."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
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
            value = record.get(name)
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _currency(record: Mapping[str, Any]) -> str:
    return str(record.get("currency", record.get("currency_code", "")) or "").strip().upper()


def _merchant(record: Mapping[str, Any]) -> str:
    return str(record.get("merchant_id", record.get("merchant", record.get("merchant_name", ""))) or "").strip().casefold()


def _text(record: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value:
            return str(value).strip().casefold()
    return ""


def _reference(record: Mapping[str, Any]) -> str:
    return _text(record, "processor_reference", "reference", "remittance_reference", "memo", "description")


def _reference_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) >= 3}


def _strong_reference(payment: Mapping[str, Any], invoice: Mapping[str, Any]) -> bool:
    payment_ref = _reference(payment)
    invoice_ref = _text(invoice, "invoice_reference", "reference", "invoice_number", "id", "invoice_id")
    if not payment_ref or not invoice_ref:
        return False
    return payment_ref == invoice_ref or bool(_reference_tokens(payment_ref) & _reference_tokens(invoice_ref))


def _invoice_limit(invoice: Mapping[str, Any]) -> int:
    if "remaining_balance" in invoice:
        return max(0, abs(int(invoice.get("remaining_balance", 0) or 0)))
    return max(0, abs(_amount(invoice)) - abs(int(invoice.get("captured_amount", 0) or 0)))


def _records(observation: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    value = observation.get(name, [])
    return [dict(item) for item in value] if isinstance(value, list) and all(isinstance(item, Mapping) for item in value) else []


def _integrations(observation: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    try:
        from prepare import prepare
        prepared = prepare(observation)
        return _records(prepared, "payments"), _records(prepared, "invoices"), None
    except (ImportError, TypeError, ValueError):
        payments = _records(observation, "payments")
        if not payments:
            payments = _records(observation, "transactions")
        return payments, _records(observation, "invoices"), None


def _unique_payments(payments: Sequence[Mapping[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    duplicate_map = dict(state.get("duplicate_map", {}))
    for payment in payments:
        item = dict(payment)
        pid = _id(item, "payment_id", "id", "transaction_id", "event_id")
        duplicate = _id(item, "duplicate_of", "duplicate_transaction_id", "original_payment_id")
        key = duplicate or pid
        if key and key in seen:
            if pid:
                duplicate_map[pid] = key
            continue
        if key:
            seen.add(key)
        unique.append(item)
    state["duplicate_map"] = duplicate_map
    return unique


def _similar(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _candidate_score(payment: Mapping[str, Any], invoice: Mapping[str, Any]) -> float:
    if not _valid_link(payment, invoice) or _amount(payment) <= 0 or _amount(invoice) <= 0:
        return 0.0
    score = 0.0
    payment_amount = abs(_amount(payment))
    invoice_amount = abs(int(invoice.get("remaining_balance", invoice.get("amount", 0)) or 0))
    if payment_amount == invoice_amount:
        score += 0.50
    elif payment_amount <= invoice_amount and payment_amount >= invoice_amount * 0.5:
        score += 0.30
    else:
        score += max(0.0, 0.20 - abs(payment_amount - invoice_amount) / max(payment_amount, invoice_amount) * 0.20)
    score += 0.15
    payment_ref = _reference(payment)
    invoice_ref = _text(invoice, "invoice_reference", "reference", "invoice_number", "id", "invoice_id")
    p_tokens, i_tokens = _reference_tokens(payment_ref), _reference_tokens(invoice_ref)
    if payment_ref and invoice_ref and (payment_ref == invoice_ref or p_tokens & i_tokens):
        score += 0.25
    elif payment_ref and invoice_ref:
        score += 0.10 * _similar(payment_ref, invoice_ref)
    payer = _text(payment, "customer_name", "sender_name", "payer", "customer")
    customer = _text(invoice, "customer_name", "customer", "customer_id")
    payment_merchant = _merchant(payment)
    invoice_merchant = _merchant(invoice)
    if payment_merchant and invoice_merchant and payment_merchant == invoice_merchant:
        score += 0.15
    if payer and customer:
        score += 0.08 * _similar(payer, customer)
    status = str(invoice.get("status", "open") or "open").casefold()
    if status in {"open", "unpaid", "partial", "issued"}:
        score += 0.02
    return min(1.0, round(score, 4))


def _candidate_map(observation: Mapping[str, Any], payments: Sequence[Mapping[str, Any]], invoices: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    supplied = observation.get("candidates", observation.get("matches", {}))
    by_id = {_id(invoice, "invoice_id", "id"): invoice for invoice in invoices}
    result: dict[str, list[dict[str, Any]]] = {}
    for payment in payments:
        pid = _id(payment, "payment_id", "id", "transaction_id", "event_id")
        raw = supplied.get(pid, []) if isinstance(supplied, Mapping) else []
        options: list[dict[str, Any]] = []
        if not raw:
            raw = [{"invoice_id": _id(invoice, "invoice_id", "id"), "confidence": _candidate_score(payment, invoice)} for invoice in invoices]
        for item in raw if isinstance(raw, list) else []:
            candidate = {"invoice_id": item} if isinstance(item, str) else dict(item)
            iid = _id(candidate, "invoice_id", "id")
            if not iid or iid not in by_id:
                continue
            try:
                confidence = float(candidate.get("confidence", candidate.get("score", 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            candidate["invoice_id"] = iid
            candidate["confidence"] = max(0.0, min(1.0, confidence))
            options.append(candidate)
        result[pid] = sorted(options, key=lambda item: item["confidence"], reverse=True)
    return result


def _valid_link(payment: Mapping[str, Any], invoice: Mapping[str, Any]) -> bool:
    status = str(invoice.get("status", "") or "").casefold()
    if invoice.get("cancelled") or status in {"cancelled", "canceled", "void", "voided"}:
        return False
    if _currency(payment) and _currency(invoice) and _currency(payment) != _currency(invoice):
        return False
    if _merchant(payment) and _merchant(invoice) and _merchant(payment) != _merchant(invoice):
        return False
    return True


def _query_answer(observation: Mapping[str, Any], payment_id: str) -> tuple[bool, Any]:
    answers = observation.get("query_results", observation.get("query_answer", {}))
    return (True, answers[payment_id]) if isinstance(answers, Mapping) and payment_id in answers else (False, None)


def _build_allocations(payments: Sequence[Mapping[str, Any]], invoices: Mapping[str, Mapping[str, Any]], candidates: Mapping[str, Sequence[Mapping[str, Any]]], state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    allocations: list[dict[str, Any]] = []
    unresolved: list[str] = []
    captured = {str(k): int(v) for k, v in dict(state.get("captured", {})).items()}
    processed = {str(k) for k in state.get("processed_payments", [])}
    verified = state.get("query_results", {})
    ranked = sorted(payments, key=lambda p: abs(_amount(p)), reverse=True)
    for payment in ranked:
        pid = _id(payment, "payment_id", "id", "transaction_id", "event_id")
        if not pid or pid in processed or _amount(payment) <= 0:
            continue
        if str(payment.get("status", "")).casefold() in {"refund", "refunded", "cancelled", "canceled"}:
            unresolved.append(pid)
            continue
        options = list(candidates.get(pid, ()))
        if pid in verified:
            answer = verified[pid]
            options = [] if answer is None else [{"invoice_id": _id(answer, "invoice_id", "id") if isinstance(answer, Mapping) else str(answer), "confidence": 1.0}]
        if not options:
            unresolved.append(pid)
            continue
        chosen = options[0]
        confidence = float(chosen.get("confidence", 0.0))
        second = float(options[1].get("confidence", 0.0)) if len(options) > 1 else 0.0
        threshold = AMBIGUOUS_THRESHOLD if payment.get("payer_family", payment.get("ambiguous_payer_family")) or confidence - second <= 0.15 else ACCEPT_THRESHOLD
        iid = _id(chosen, "invoice_id", "id")
        invoice = invoices.get(iid)
        amount = abs(_amount(payment))
        remaining = _invoice_limit(invoice) - captured.get(iid, 0) if invoice else 0
        reference_threshold = 0.65 if _strong_reference(payment, invoice or {}) else threshold
        if confidence < min(threshold, reference_threshold) or not invoice or not _valid_link(payment, invoice) or amount > remaining:
            unresolved.append(pid)
            continue
        allocations.append({"payment_id": pid, "invoice_id": iid, "amount": amount})
        captured[iid] = captured.get(iid, 0) + amount
    return allocations, unresolved


def _within_capacity(observation: Mapping[str, Any], unresolved: Sequence[str]) -> bool:
    try:
        return len(unresolved) <= int(observation.get("review_capacity", observation.get("capacity", observation.get("max_reviews", REVIEW_CAPACITY))))
    except (TypeError, ValueError):
        return False


def validate_posting(observation: Mapping[str, Any], payments: Sequence[Mapping[str, Any]], invoices: Mapping[str, Mapping[str, Any]], allocations: Sequence[Mapping[str, Any]], unresolved: Sequence[str]) -> bool:
    if not _within_capacity(observation, unresolved):
        return False
    payment_map = {_id(item, "payment_id", "id", "transaction_id", "event_id"): item for item in payments}
    captured: dict[str, int] = {}
    used_payments: set[str] = set()
    for allocation in allocations:
        pid, iid = str(allocation.get("payment_id", "")), str(allocation.get("invoice_id", ""))
        payment, invoice = payment_map.get(pid), invoices.get(iid)
        amount = _amount(allocation)
        if not payment or not invoice or pid in used_payments or amount != abs(_amount(payment)) or amount <= 0:
            return False
        if not _valid_link(payment, invoice):
            return False
        captured[iid] = captured.get(iid, 0) + amount
        if captured[iid] > _invoice_limit(invoice):
            return False
        used_payments.add(pid)
    return True


def _safe_baseline(observation: Mapping[str, Any], state: Mapping[str, Any] | None) -> dict[str, Any]:
    prior = dict(state or {})
    payments = _records(observation, "payments") or _records(observation, "transactions")
    pending = [_id(item, "payment_id", "id", "transaction_id", "event_id") for item in payments]
    current = {"queries_used": min(MAX_QUERIES, int(prior.get("queries_used", 0) or 0)), "queried": list(prior.get("queried", []))[:MAX_QUERIES], "query_results": dict(prior.get("query_results", {})), "duplicate_map": dict(prior.get("duplicate_map", {})), "processed_payments": list(prior.get("processed_payments", [])), "captured": dict(prior.get("captured", {})), "pending": pending, "phase": "final"}
    return {"action": {"type": "post", "allocations": [], "unresolved": pending}, "state": current}


def decide(observation: Mapping[str, Any], state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        current = dict(state or {})
        current["queries_used"] = min(MAX_QUERIES, int(current.get("queries_used", 0) or 0))
        current.setdefault("queried", [])
        current.setdefault("query_results", {})
        current.setdefault("duplicate_map", {})
        current.setdefault("processed_payments", [])
        current.setdefault("captured", {})
        payments, invoice_list, _ = _integrations(observation)
        payments = _unique_payments(payments, current)
        invoices = {_id(item, "invoice_id", "id"): item for item in invoice_list if _id(item, "invoice_id", "id")}
        candidates = _candidate_map(observation, payments, invoice_list)
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
        captured = dict(current["captured"])
        processed = list(current["processed_payments"])
        for item in allocations:
            captured[item["invoice_id"]] = int(captured.get(item["invoice_id"], 0)) + int(item["amount"])
            processed.append(item["payment_id"])
        current.update({"captured": captured, "processed_payments": sorted(set(processed)), "pending": list(unresolved), "phase": "final"})
        result = {"action": {"type": "post", "allocations": allocations, "unresolved": unresolved}, "state": current}
        json.dumps(result)
        return result
    except (TypeError, ValueError, KeyError, ImportError):
        return _safe_baseline(observation, state)
