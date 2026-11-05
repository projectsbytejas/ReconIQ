"""Schema-aligned payment reconciliation decision function."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import query as query_policy
from query import MAX_QUERIES, choose_query, record_query

ACCEPT_THRESHOLD = 0.85
AMBIGUOUS_THRESHOLD = 0.92


def _currency(value: Any) -> str:
    return str(value or "").strip().upper()


def _record_id(record: Mapping[str, Any], primary: str, fallback: str) -> str:
    return str(record.get(primary, record.get(fallback, "")))


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _dedupe(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    return [dict(row) for row in {str(row[key]): row for row in rows if key in row}.values()]


def _transactions(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _dedupe(observation.get("transactions", []), "transaction_id")


def _payment_values(transactions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        pid: sum(max(0, int(t.get("gross", 0))) for t in transactions if t.get("payment_id") == pid)
        for pid in {str(t.get("payment_id", "")) for t in transactions}
    }


def _payment_balances(
    transactions: Sequence[Mapping[str, Any]], payment_id: str
) -> dict[str, int]:
    rows = [t for t in transactions if str(t.get("payment_id")) == payment_id]
    return {
        "captured": sum(max(0, int(t.get("gross", 0))) for t in rows),
        "refunded": sum(max(0, -int(t.get("gross", 0))) for t in rows),
        "fees": sum(int(t.get("fee", 0)) for t in rows),
        "net": sum(int(t.get("net", 0)) for t in rows),
    }


def _documents(observation: Mapping[str, Any], payment_id: str) -> list[Mapping[str, Any]]:
    return [
        item for item in observation.get("remittance_documents", [])
        if str(item.get("payment_id")) == payment_id
    ]


def _candidate_map(
    observation: Mapping[str, Any],
    payments: Sequence[Mapping[str, Any]],
    invoices: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for payment in payments:
        pid = str(payment["payment_id"])
        docs = _documents(observation, pid)
        scored: list[dict[str, Any]] = []
        for invoice in invoices:
            if _currency(payment.get("currency")) != _currency(invoice.get("currency")):
                continue
            if str(payment.get("merchant")) != str(invoice.get("merchant")):
                continue
            score = 0.0
            payment_reference = _token(payment.get("source_ref"))
            if payment_reference and payment_reference == _token(invoice.get("reference")):
                score += 0.60
            if _token(payment.get("tax_suffix")) == _token(invoice.get("tax_suffix")):
                score += 0.30
            if _token(payment.get("sender_name")) == _token(invoice.get("customer_name")):
                score += 0.15
            for source in docs:
                reference = source.get("reference")
                if reference and _token(reference) == _token(invoice.get("reference")):
                    score += 0.35
                if source.get("tax_suffix") and _token(source.get("tax_suffix")) == _token(invoice.get("tax_suffix")):
                    score += 0.20
                if source.get("sender_name") and _token(source.get("sender_name")) == _token(invoice.get("customer_name")):
                    score += 0.05
            scored.append({"invoice_id": invoice["invoice_id"], "confidence": min(1.0, score)})
        result[pid] = sorted(scored, key=lambda x: x["confidence"], reverse=True)
    return result


def _verified(observation: Mapping[str, Any], state: dict[str, Any]) -> None:
    for item in observation.get("investigations", []):
        pid = str(item.get("payment_id", ""))
        if pid in state["queried"]:
            state["query_results"][pid] = item.get("verified_invoice_id")


def _best_matches(
    payments: Sequence[Mapping[str, Any]],
    invoices: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    state: Mapping[str, Any],
    balances: Mapping[str, Mapping[str, int]],
) -> dict[str, str]:
    matches: dict[str, str] = {}
    used: set[str] = set()
    for payment in payments:
        pid = str(payment["payment_id"])
        if pid in state["query_results"]:
            iid = state["query_results"][pid]
            if iid is None:
                continue
        else:
            options = list(candidates.get(pid, ()))
            if not options:
                continue
            top = options[0]
            second = float(options[1].get("confidence", 0)) if len(options) > 1 else 0
            gap = float(top["confidence"]) - second
            if gap <= 0.15:
                continue
            threshold = AMBIGUOUS_THRESHOLD if payment.get("ambiguous_payer_family", False) else ACCEPT_THRESHOLD
            if float(top["confidence"]) < threshold:
                continue
            iid = str(top["invoice_id"])
        invoice = invoices.get(str(iid))
        balance = balances[pid]
        if (
            invoice is None
            or str(iid) in used
            or balance["captured"] > int(invoice["amount"])
            or balance["refunded"] > balance["captured"]
            or (invoice["status"] == "cancelled" and balance["captured"] != balance["refunded"])
        ):
            continue
        matches[pid] = str(iid)
        used.add(str(iid))
    return matches


def _posting(
    observation: Mapping[str, Any],
    payments: Sequence[Mapping[str, Any]],
    matches: Mapping[str, str],
    transactions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    invoice_rows = {
        _record_id(i, "invoice_id", "id"): {
            "invoice_id": _record_id(i, "invoice_id", "id"),
            "captured": 0,
            "refunded": 0,
            "fees": 0,
            "net": 0,
            "remaining_due": int(i.get("amount", 0)),
        }
        for i in observation["invoices"]
    }
    suspense = {"captured": 0, "refunded": 0, "fees": 0, "net": 0}
    by_payment = {str(p["payment_id"]): _payment_balances(transactions, str(p["payment_id"])) for p in payments}
    for pid, balance in by_payment.items():
        row = invoice_rows[matches[pid]] if pid in matches else suspense
        for field in ("captured", "refunded", "fees", "net"):
            row[field] += balance[field]
    for invoice in observation["invoices"]:
        row = invoice_rows[_record_id(invoice, "invoice_id", "id")]
        row["remaining_due"] = int(invoice["amount"]) - row["captured"] + row["refunded"]
    bank_total = sum(
        int(row["amount"])
        for row in _dedupe(observation.get("bank_credits", []), "bank_id")
    )
    return {
        "op": "submit",
        "matches": [{"payment_id": pid, "invoice_id": iid} for pid, iid in matches.items()],
        "ledger": list(invoice_rows.values()),
        "suspense": suspense,
        "bank_total": bank_total,
    }


def validate_posting(
    observation: Mapping[str, Any],
    payments: Sequence[Mapping[str, Any]],
    invoices: Mapping[str, Mapping[str, Any]],
    allocations: Sequence[Mapping[str, Any]],
    unresolved: Sequence[str],
) -> bool:
    """Compatibility validator retained for the focused unit tests."""
    seen: set[str] = set()
    captured: dict[str, int] = {}
    payment_map = {str(p.get("payment_id", p.get("id", ""))): p for p in payments}
    for allocation in allocations:
        pid = str(allocation.get("payment_id", ""))
        iid = str(allocation.get("invoice_id", ""))
        payment = payment_map.get(pid)
        invoice = invoices.get(iid)
        amount = allocation.get("amount")
        if not payment or not invoice or iid in seen or not isinstance(amount, int) or isinstance(amount, bool):
            return False
        if _currency(payment.get("currency")) != _currency(invoice.get("currency")):
            return False
        if str(payment.get("merchant")) != str(invoice.get("merchant")):
            return False
        if invoice.get("cancelled", False):
            return False
        captured[iid] = captured.get(iid, 0) + amount
        if captured[iid] > int(invoice.get("amount", 0)):
            return False
        seen.add(iid)
    return True


def _initial_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    current = dict(state or {})
    current.setdefault("queries_used", 0)
    current.setdefault("queried", [])
    current.setdefault("query_results", {})
    current.setdefault("duplicate_map", {})
    current.setdefault("pending", [])
    current.setdefault("phase", "match")
    current["queried"] = list(dict.fromkeys(current["queried"]))[:MAX_QUERIES]
    current["queries_used"] = len(current["queried"])
    return current


def decide(observation: Mapping[str, Any], state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        current = _initial_state(state)
        payments = [
            {**dict(p), "payment_id": _record_id(p, "payment_id", "id")}
            for p in observation["payments"]
        ]
        invoices = [
            {**dict(i), "invoice_id": _record_id(i, "invoice_id", "id")}
            for i in observation["invoices"]
        ]
        transactions = _transactions(observation)
        values = _payment_values(transactions)
        balances = {str(p["payment_id"]): _payment_balances(transactions, str(p["payment_id"])) for p in payments}
        _verified(observation, current)
        candidates = _candidate_map(observation, payments, invoices)
        payment_id = choose_query(
            payments,
            candidates,
            values,
            current,
            minimum_value=query_policy.MINIMUM_QUERY_VALUE,
            closeness=query_policy.QUERY_CLOSENESS,
        )
        if payment_id is not None and int(observation.get("queries_remaining", 0)) > 0:
            return {"action": {"op": "query", "payment_id": payment_id}, "state": record_query(current, payment_id)}
        invoice_map = {str(i["invoice_id"]): i for i in invoices}
        matches = _best_matches(payments, invoice_map, candidates, current, balances)
        action = _posting(observation, payments, matches, transactions)
        current["pending"] = [str(p["payment_id"]) for p in payments if str(p["payment_id"]) not in matches]
        current["phase"] = "submit"
        result = {"action": action, "state": current}
        json.dumps(result)
        return result
    except Exception:
        payments = [
            {**dict(p), "payment_id": _record_id(p, "payment_id", "id")}
            for p in observation.get("payments", [])
        ]
        transactions = _transactions(observation)
        action = _posting(observation, payments, {}, transactions)
        current = _initial_state(state)
        current["pending"] = [str(p.get("payment_id", "")) for p in payments]
        current["phase"] = "submit"
        return {"action": action, "state": current}
