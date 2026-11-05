"""Conservative deterministic payment-to-invoice reconciliation.

The implementation deliberately treats conflicting references as uncertainty.
It deduplicates transactions and bank credits by their stable IDs, uses a
one-to-one payment-group/invoice assignment, and spends the limited query
budget on high-value assignments with conflicting evidence.
"""
import re

FIELDS = ("captured", "refunded", "fees", "net")


def _norm_text(value):
    return str(value or "").strip().lower()


def _norm_currency(value):
    return str(value or "").strip().upper()


def _norm_tax(value):
    return str(value or "").strip()


def _norm_ref(value):
    # Invoice references in the supplied data contain formatting noise
    # (spaces, slashes, hyphens and case differences).
    return "".join(re.findall(r"[0-9]+", str(value or "")))


def _tokens(value):
    return set(re.findall(r"[a-z0-9]+", _norm_text(value)))


def _unique_rows(rows, key):
    result = {}
    for row in rows:
        # Match the public runner's last-observation-wins deduplication.
        result[row[key]] = row
    return list(result.values())


def _prepare(observation):
    # Work on new lists; do not mutate the observation supplied by the runner.
    obs = dict(observation)
    obs["payments"] = [dict(p, currency=_norm_currency(p.get("currency")))
                       for p in observation.get("payments", [])]
    obs["invoices"] = [dict(i, currency=_norm_currency(i.get("currency")))
                       for i in observation.get("invoices", [])]
    obs["transactions"] = _unique_rows(observation.get("transactions", []),
                                       "transaction_id")
    obs["bank_credits"] = _unique_rows(observation.get("bank_credits", []),
                                       "bank_id")
    obs["remittance_documents"] = list(observation.get("remittance_documents", []))
    obs["investigations"] = list(observation.get("investigations", []))
    return obs


def _payment_transactions(obs):
    by_payment = {}
    for tx in obs["transactions"]:
        by_payment.setdefault(tx["payment_id"], []).append(tx)
    return by_payment


def _payment_value(payment_id, by_payment):
    return sum(max(0, tx["gross"]) for tx in by_payment.get(payment_id, []))


def _payment_refunded(payment_id, by_payment):
    return sum(max(0, -tx["gross"]) for tx in by_payment.get(payment_id, []))


def _docs_for_payment(obs, payment_id):
    return [d for d in obs["remittance_documents"]
            if d.get("payment_id") == payment_id and d.get("status") == "active"]


def _independent_documents(docs):
    # Processor receipts repeat the raw payment evidence.  Other documents
    # are collapsed by lineage so a copied document is not counted twice.
    groups = {}
    for doc in docs:
        if doc.get("source") == "processor_receipt":
            continue
        lineage = doc.get("lineage") or doc.get("document_id")
        groups.setdefault(lineage, doc)
    return list(groups.values())


def _pair_evidence(payment, invoice, docs):
    invoice_ref = _norm_ref(invoice.get("reference"))
    payment_ref = _norm_ref(payment.get("source_ref"))
    payment_tax = _norm_tax(payment.get("tax_suffix"))
    invoice_tax = _norm_tax(invoice.get("tax_suffix"))

    independent = _independent_documents(docs)
    independent_ref = any(
        invoice_ref and _norm_ref(d.get("reference")) == invoice_ref
        for d in independent
    )
    independent_tax = any(
        invoice_tax and _norm_tax(d.get("tax_suffix")) == invoice_tax
        for d in independent
    )
    independent_joint = any(
        invoice_ref
        and _norm_ref(d.get("reference")) == invoice_ref
        and invoice_tax
        and _norm_tax(d.get("tax_suffix")) == invoice_tax
        for d in independent
    )

    # A processor receipt is correlated with the raw payment record, so it
    # is intentionally not treated as an independent reference source.
    payment_reference_match = bool(
        invoice_ref and payment_ref and invoice_ref == payment_ref
    )
    tax_match = bool(payment_tax and invoice_tax and payment_tax == invoice_tax)
    name_overlap = len(_tokens(payment.get("sender_name")) &
                       _tokens(invoice.get("customer_name")))

    return {
        "payment_reference": payment_reference_match,
        "independent_reference": independent_ref,
        "tax_match": tax_match,
        "independent_tax": independent_tax,
        "independent_joint": independent_joint,
        "name_overlap": name_overlap,
    }


def _candidate_options(obs, payment, by_payment):
    payment_id = payment["payment_id"]
    captured = _payment_value(payment_id, by_payment)
    refunded = _payment_refunded(payment_id, by_payment)
    if captured <= 0:
        return []

    docs = _docs_for_payment(obs, payment_id)
    options = []

    for invoice in obs["invoices"]:
        if invoice["merchant"] != payment.get("merchant"):
            continue
        if invoice["currency"] != payment.get("currency"):
            continue
        if invoice["amount"] < captured:
            continue
        if invoice.get("status") == "cancelled" and refunded != captured:
            continue

        ev = _pair_evidence(payment, invoice, docs)
        if not ev["name_overlap"]:
            continue

        # High-confidence deterministic rule: a matching tax suffix plus
        # either the payment reference or an independent remittance reference.
        has_reference = (ev["payment_reference"] or
                         ev["independent_reference"])
        if not (ev["tax_match"] and has_reference):
            continue

        # Prefer corroboration from both independent document and payment
        # reference, but do not count duplicate lineage as multiple votes.
        score = (
            100 * int(ev["tax_match"])
            + 20 * int(ev["payment_reference"])
            + 15 * int(ev["independent_reference"])
            + 5 * int(ev["independent_joint"])
            + 2 * min(ev["name_overlap"], 2)
            + int(invoice["amount"] == captured)
        )
        options.append((score, invoice["invoice_id"], ev))

    options.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return options


def _reference_conflicts(obs, payment_id, assigned_invoice_id):
    invoice_by_ref = {}
    invoice_by_id = {}
    for invoice in obs["invoices"]:
        invoice_by_id[invoice["invoice_id"]] = invoice
        ref = _norm_ref(invoice.get("reference"))
        if ref:
            invoice_by_ref[ref] = invoice["invoice_id"]

    payment = next((p for p in obs["payments"]
                    if p["payment_id"] == payment_id), None)
    if payment is None:
        return False, False

    # Include void/superseded documents when detecting a conflict. They are
    # not positive matching evidence, but they can reveal a contradictory
    # reference that makes an otherwise plausible assignment unsafe.
    docs = [d for d in obs["remittance_documents"]
            if d.get("payment_id") == payment_id]
    observed_refs = set()
    for doc in docs:
        ref = _norm_ref(doc.get("reference"))
        if ref in invoice_by_ref:
            observed_refs.add(ref)

    payment_ref = _norm_ref(payment.get("source_ref"))
    payment_ref_invoice = invoice_by_ref.get(payment_ref)
    multiple_document_targets = len(observed_refs) > 1
    source_conflict = bool(
        payment_ref_invoice and payment_ref_invoice != assigned_invoice_id
    )
    return multiple_document_targets, source_conflict


def _build_mapping(obs):
    by_payment = _payment_transactions(obs)
    verified = {}
    verified_payments = set()
    for result in obs.get("investigations", []):
        pid = result.get("payment_id")
        if pid is None:
            continue
        verified_payments.add(pid)
        iid = result.get("verified_invoice_id")
        if iid is not None:
            verified[pid] = iid

    mapping = dict(verified)
    used_invoices = set(mapping.values())
    choices = {}

    for payment in obs["payments"]:
        pid = payment["payment_id"]
        if pid in verified_payments:
            continue
        options = _candidate_options(obs, payment, by_payment)
        choices[pid] = options
        if not options:
            continue

        # If the best candidate is not clearly better than another invoice,
        # abstain instead of making an unsafe assignment.
        if len(options) > 1 and options[0][0] - options[1][0] < 10:
            continue
        choices[pid] = options

    # Resolve invoice collisions by strongest evidence.  Equal/near-equal
    # competing payments are both left unresolved.
    proposed = []
    for pid, options in choices.items():
        if options and (len(options) == 1 or
                        options[0][0] - options[1][0] >= 10):
            proposed.append((options[0][0], pid, options[0][1]))
    by_invoice = {}
    for score, pid, iid in proposed:
        if iid in used_invoices:
            continue
        by_invoice.setdefault(iid, []).append((score, pid))

    for iid, contenders in by_invoice.items():
        contenders.sort(reverse=True)
        if len(contenders) > 1 and contenders[0][0] - contenders[1][0] < 15:
            continue
        mapping[contenders[0][1]] = iid
        used_invoices.add(iid)

    return mapping, choices, verified_payments, by_payment


def _query_candidate(obs, mapping, choices, verified_payments, by_payment):
    remaining = int(obs.get("queries_remaining", 0) or 0)
    if remaining <= 0:
        return None

    invoice_by_id = {
        invoice["invoice_id"]: invoice for invoice in obs["invoices"]
    }
    payment_by_id = {
        payment["payment_id"]: payment for payment in obs["payments"]
    }
    risk = []
    for pid, payment in payment_by_id.items():
        if pid in verified_payments:
            continue
        value = _payment_value(pid, by_payment)
        if value <= 0:
            continue

        if pid not in mapping:
            # An unresolved payment with a strong candidate that lost a
            # conflict/uniqueness check is highly likely to be matched.
            # Otherwise, larger unresolved payments are still worth checking.
            priority = value * (200 if choices.get(pid) else 20)
            risk.append((priority, value, pid))
            continue

        assigned_invoice_id = mapping[pid]
        invoice = invoice_by_id.get(assigned_invoice_id)
        if invoice is None:
            continue

        multi_docs, source_conflict = _reference_conflicts(
            obs, pid, assigned_invoice_id
        )
        evidence = _pair_evidence(
            payment, invoice, _docs_for_payment(obs, pid)
        )
        weak_independent_support = not evidence["independent_reference"]
        severity = (
            1 * int(multi_docs)
            + 4 * int(source_conflict)
            + 4 * int(weak_independent_support)
        )
        if severity > 0:
            # A review of a risky existing match can avoid a 5x wrong-match
            # penalty; it is lower priority than an unresolved payment.
            priority = value * 0.5 * severity
            risk.append((priority, value, pid))

    if risk:
        risk.sort(reverse=True)
        return risk[0][2]
    return None

def _accounting(obs, mapping, by_payment):
    ledger = {
        invoice["invoice_id"]: dict.fromkeys(FIELDS, 0)
        for invoice in obs["invoices"]
    }
    suspense = dict.fromkeys(FIELDS, 0)

    for tx in obs["transactions"]:
        bucket = ledger.get(mapping.get(tx["payment_id"]))
        if bucket is None:
            bucket = suspense
        bucket["captured"] += max(0, tx["gross"])
        bucket["refunded"] += max(0, -tx["gross"])
        bucket["fees"] += tx["fee"]
        bucket["net"] += tx["net"]

    rows = []
    for invoice in obs["invoices"]:
        balance = ledger[invoice["invoice_id"]]
        rows.append({
            "invoice_id": invoice["invoice_id"],
            "captured": balance["captured"],
            "refunded": balance["refunded"],
            "fees": balance["fees"],
            "net": balance["net"],
            "remaining_due": (
                invoice["amount"] - balance["captured"] + balance["refunded"]
            ),
        })

    return {
        "op": "submit",
        "matches": [
            {"payment_id": pid, "invoice_id": iid}
            for pid, iid in mapping.items()
        ],
        "ledger": rows,
        "suspense": suspense,
        "bank_total": sum(row["amount"] for row in obs["bank_credits"]),
    }


def decide(observation, state):
    obs = _prepare(observation)
    mapping, choices, verified_payments, by_payment = _build_mapping(obs)

    query_payment = _query_candidate(
        obs, mapping, choices, verified_payments, by_payment
    )
    if query_payment is not None:
        return {
            "action": {"op": "query", "payment_id": query_payment},
            "state": {},
        }

    return {
        "action": _accounting(obs, mapping, by_payment),
        "state": {},
    }
