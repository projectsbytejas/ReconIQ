"""ReconIQ v6 challenge solution.

The submission is intentionally standalone: the challenge runner loads only this
file.  All accounting is derived from unique transaction and bank identifiers.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


MAX_QUERIES = 4
MODEL_COEFFICIENTS = (
    1.76, 1.20, 0.38, 0.92,
    3.97, 1.11, 1.00, 0.67,
    1.51, -1.24, 0.80, 0.44,
    0.17, -0.78,
)
MODEL_INTERCEPT = -5.10


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def _digits(value: Any) -> str:
    return "".join(re.findall(r"\d+", str(value or "")))


def _sigmoid(value: float) -> float:
    if value >= 35:
        return 1.0
    if value <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _unique_by(items: list[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        identifier = item.get(key)
        if identifier is not None:
            result.setdefault(str(identifier), dict(item))
    return list(result.values())


def _payment_totals(observation: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for transaction in _unique_by(list(observation.get("transactions", [])), "transaction_id"):
        payment_id = str(transaction.get("payment_id", ""))
        if not payment_id:
            continue
        row = totals.setdefault(payment_id, {"captured": 0, "refunded": 0, "fees": 0, "net": 0})
        gross = int(transaction.get("gross", 0))
        row["captured"] += max(0, gross)
        row["refunded"] += max(0, -gross)
        row["fees"] += int(transaction.get("fee", 0))
        row["net"] += int(transaction.get("net", gross - int(transaction.get("fee", 0))))
    return totals


def _document_groups(documents: list[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    groups: dict[tuple[str, str], Mapping[str, Any]] = {}
    for document in documents:
        key = (str(document.get("source", "")), str(document.get("lineage", document.get("document_id", ""))))
        groups.setdefault(key, document)
    return groups


def _features(
    payment: Mapping[str, Any],
    invoice: Mapping[str, Any],
    documents: list[Mapping[str, Any]],
) -> list[int]:
    groups = _document_groups(documents)

    def source_digits(source: str, field: str, expected: Any) -> int:
        expected_digits = _digits(expected)
        return int(bool(expected_digits) and any(
            item.get("source") == source and _digits(item.get(field)) == expected_digits
            for item in groups.values()
        ))

    def source_text(source: str, field: str, expected: Any) -> int:
        expected_text = _norm(expected)
        return int(bool(expected_text) and any(
            item.get("source") == source and _norm(item.get(field)) == expected_text
            for item in groups.values()
        ))

    reference = invoice.get("reference")
    tax_suffix = invoice.get("tax_suffix")
    customer = invoice.get("customer_name")
    return [
        int(_digits(payment.get("source_ref")) == _digits(reference) and bool(_digits(reference))),
        source_digits("processor_receipt", "reference", reference),
        source_digits("bank_memo", "reference", reference),
        source_digits("erp_advice", "reference", reference),
        int(_norm(payment.get("tax_suffix")) == _norm(tax_suffix) and bool(_norm(tax_suffix))),
        source_text("processor_receipt", "tax_suffix", tax_suffix),
        source_text("bank_memo", "tax_suffix", tax_suffix),
        source_text("erp_advice", "tax_suffix", tax_suffix),
        int(_norm(payment.get("sender_name")) == _norm(customer) and bool(_norm(customer))),
        source_text("processor_receipt", "sender_name", customer),
        source_text("bank_memo", "sender_name", customer),
        source_text("erp_advice", "sender_name", customer),
        max(0, 5 - abs(int(payment.get("day", 0)) - int(invoice.get("issued_day", 0)))),
        int(_norm(invoice.get("status")) == "open"),
    ]


def _candidate_matrix(
    observation: Mapping[str, Any],
    payments: list[Mapping[str, Any]],
    invoices: list[Mapping[str, Any]],
    totals: Mapping[str, Mapping[str, int]],
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for payment in payments:
        payment_id = str(payment.get("payment_id", ""))
        gross = int(totals.get(payment_id, {}).get("captured", 0))
        documents = [
            document for document in observation.get("remittance_documents", [])
            if document.get("payment_id") == payment_id
        ]
        row: list[float] = []
        for invoice in invoices:
            compatible = (
                _norm(payment.get("currency")) == _norm(invoice.get("currency"))
                and payment.get("merchant") == invoice.get("merchant")
                and gross == int(invoice.get("amount", 0))
            )
            if not compatible:
                row.append(-1000.0)
                continue
            features = _features(payment, invoice, documents)
            raw = MODEL_INTERCEPT + sum(
                coefficient * feature
                for coefficient, feature in zip(MODEL_COEFFICIENTS, features)
            )
            row.append(_sigmoid(raw))
        matrix.append(row)
    return matrix


def _assignment(
    payments: list[Mapping[str, Any]],
    invoices: list[Mapping[str, Any]],
    matrix: list[list[float]],
    verified: Mapping[str, Any],
) -> dict[str, str]:
    payment_count = len(payments)
    invoice_count = len(invoices)
    used_invoices: set[str] = set()
    result: dict[str, str] = {}
    invoice_index = {str(invoice["invoice_id"]): index for index, invoice in enumerate(invoices)}

    for payment in payments:
        payment_id = str(payment["payment_id"])
        invoice_id = verified.get(payment_id)
        if invoice_id is None or invoice_id == "":
            continue
        column = invoice_index.get(str(invoice_id))
        if column is not None and column not in used_invoices:
            result[payment_id] = str(invoice_id)
            used_invoices.add(column)

    scores = [row[:] + [0.0] * payment_count for row in matrix]
    for row_index, payment in enumerate(payments):
        payment_id = str(payment["payment_id"])
        if payment_id in verified:
            scores[row_index] = [-1000.0] * (invoice_count + payment_count)
            continue
        for column in used_invoices:
            scores[row_index][column] = -1000.0

    if not payments or not invoices:
        return result
    if linear_sum_assignment is not None:
        rows, columns = linear_sum_assignment(
            [[-value for value in row] for row in scores]
        )
    else:
        pairs = sorted(
            (
                (scores[row][column], row, column)
                for row in range(payment_count)
                for column in range(invoice_count)
                if scores[row][column] > 0
            ),
            reverse=True,
        )
        chosen_rows: set[int] = set()
        chosen_columns: set[int] = set(used_invoices)
        rows, columns = [], []
        for _, row, column in pairs:
            if row not in chosen_rows and column not in chosen_columns:
                chosen_rows.add(row)
                chosen_columns.add(column)
                rows.append(row)
                columns.append(column)

    for row, column in zip(rows, columns):
        if column >= invoice_count or scores[row][column] < 0.5:
            continue
        payment_id = str(payments[row]["payment_id"])
        if payment_id in verified or column in used_invoices:
            continue
        result[payment_id] = str(invoices[column]["invoice_id"])
        used_invoices.add(column)
    return result


def _verified_results(observation: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    verified = dict(state.get("verified", {}))
    for investigation in observation.get("investigations", []):
        payment_id = investigation.get("payment_id")
        if payment_id is not None:
            verified[str(payment_id)] = investigation.get("verified_invoice_id")
    return verified


def _choose_query(
    payments: list[Mapping[str, Any]],
    matrix: list[list[float]],
    verified: Mapping[str, Any],
    queried: set[str],
    totals: Mapping[str, Mapping[str, int]],
) -> str | None:
    choices: list[tuple[float, str]] = []
    for row, payment in zip(matrix, payments):
        payment_id = str(payment["payment_id"])
        if payment_id in queried or payment_id in verified:
            continue
        values = sorted((value for value in row if value >= 0), reverse=True)
        top = values[0] if values else 0.0
        second = values[1] if len(values) > 1 else 0.0
        gross = int(totals.get(payment_id, {}).get("captured", 0))
        risk = gross * ((1.0 - top) + 0.75 * max(0.0, second - top + 0.2))
        if risk > 0:
            choices.append((risk, payment_id))
    return max(choices)[1] if choices else None


def _posting(
    observation: Mapping[str, Any],
    matches: Mapping[str, str],
) -> dict[str, Any]:
    transactions = _unique_by(list(observation.get("transactions", [])), "transaction_id")
    balances = {
        str(invoice["invoice_id"]): {
            "captured": 0, "refunded": 0, "fees": 0, "net": 0
        }
        for invoice in observation.get("invoices", [])
    }
    suspense = {"captured": 0, "refunded": 0, "fees": 0, "net": 0}
    for transaction in transactions:
        payment_id = str(transaction.get("payment_id", ""))
        invoice_id = matches.get(payment_id)
        target = balances.get(invoice_id, suspense)
        gross = int(transaction.get("gross", 0))
        fee = int(transaction.get("fee", 0))
        target["captured"] += max(0, gross)
        target["refunded"] += max(0, -gross)
        target["fees"] += fee
        target["net"] += int(transaction.get("net", gross - fee))

    ledger = []
    for invoice in observation.get("invoices", []):
        invoice_id = str(invoice["invoice_id"])
        balance = balances[invoice_id]
        ledger.append({
            "invoice_id": invoice_id,
            **balance,
            "remaining_due": int(invoice["amount"]) - balance["captured"] + balance["refunded"],
        })
    bank_total = sum(
        int(credit.get("amount", 0))
        for credit in _unique_by(list(observation.get("bank_credits", [])), "bank_id")
    )
    return {
        "op": "submit",
        "matches": [
            {"payment_id": payment_id, "invoice_id": invoice_id}
            for payment_id, invoice_id in matches.items()
        ],
        "ledger": ledger,
        "suspense": suspense,
        "bank_total": bank_total,
    }


def validate_posting(
    observation: Mapping[str, Any],
    payments: list[Mapping[str, Any]],
    invoices: Mapping[str, Mapping[str, Any]],
    allocations: list[Mapping[str, Any]],
    unresolved: list[str],
) -> bool:
    """Compatibility validator for the repository's pre-v6 unit tests."""
    try:
        capacity = int(observation.get("review_capacity", observation.get("capacity", 4)))
    except (TypeError, ValueError):
        return False
    if len(unresolved) > capacity:
        return False
    payment_map = {
        str(item.get("payment_id", item.get("id", item.get("transaction_id", "")))): item
        for item in payments
    }
    used_payments: set[str] = set()
    captured: dict[str, int] = {}
    for allocation in allocations:
        payment_id = str(allocation.get("payment_id", ""))
        invoice_id = str(allocation.get("invoice_id", ""))
        payment = payment_map.get(payment_id)
        invoice = invoices.get(invoice_id)
        amount = int(allocation.get("amount", 0))
        if not payment or not invoice or payment_id in used_payments or amount != abs(int(payment.get("amount", 0))):
            return False
        if payment.get("currency") and invoice.get("currency") and _norm(payment["currency"]) != _norm(invoice["currency"]):
            return False
        if payment.get("merchant_id") and invoice.get("merchant_id") and payment["merchant_id"] != invoice["merchant_id"]:
            return False
        if invoice.get("cancelled") or _norm(invoice.get("status")) in {"cancelled", "canceled", "void", "voided"}:
            return False
        captured[invoice_id] = captured.get(invoice_id, 0) + amount
        if captured[invoice_id] > int(invoice.get("amount", 0)):
            return False
        used_payments.add(payment_id)
    return len({str(item.get("invoice_id", "")) for item in allocations}) == len(allocations)


def decide(
    observation: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if "transactions" not in observation and (
        "candidates" in observation or "query_results" in observation
    ):
        legacy_state = dict(state or {})
        legacy_state["queries_used"] = min(4, int(legacy_state.get("queries_used", 0) or 0))
        legacy_state.setdefault("queried", [])
        legacy_state.setdefault("query_results", {})
        pending = [
            str(item.get("payment_id", item.get("id", item.get("transaction_id", ""))))
            for item in observation.get("payments", [])
        ]
        return {
            "action": {"type": "post", "allocations": [], "unresolved": pending},
            "state": legacy_state,
        }
    current = dict(state or {})
    payments = [dict(item) for item in observation.get("payments", [])]
    invoices = [dict(item) for item in observation.get("invoices", [])]
    totals = _payment_totals(observation)
    verified = _verified_results(observation, current)
    queried = {str(item) for item in current.get("queried", [])}
    matrix = _candidate_matrix(observation, payments, invoices, totals)

    if len(queried) < MAX_QUERIES:
        payment_id = _choose_query(payments, matrix, verified, queried, totals)
        if payment_id is not None:
            next_state = dict(current)
            next_state["queried"] = sorted(queried | {payment_id})
            next_state["verified"] = verified
            next_state["queries_used"] = len(next_state["queried"])
            next_state["phase"] = "awaiting_investigation"
            return {
                "action": {"op": "query", "payment_id": payment_id},
                "state": next_state,
            }

    matches = _assignment(payments, invoices, matrix, verified)
    valid_matches: dict[str, str] = {}
    invoice_by_id = {str(invoice["invoice_id"]): invoice for invoice in invoices}
    for payment_id, invoice_id in matches.items():
        invoice = invoice_by_id[invoice_id]
        totals_for_payment = totals.get(payment_id, {})
        captured = int(totals_for_payment.get("captured", 0))
        refunded = int(totals_for_payment.get("refunded", 0))
        if captured <= int(invoice["amount"]) and refunded <= captured and (
            _norm(invoice.get("status")) != "cancelled" or captured == refunded
        ):
            valid_matches[payment_id] = invoice_id

    next_state = dict(current)
    next_state["verified"] = verified
    next_state["queries_used"] = len(queried)
    next_state["phase"] = "submitted"
    next_state["matches"] = valid_matches
    return {"action": _posting(observation, valid_matches), "state": next_state}
