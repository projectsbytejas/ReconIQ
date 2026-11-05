from copy import deepcopy


def _clean_currency(value):
    if value is None:
        return None
    return str(value).strip().upper()


def _clean_string(value):
    if value is None:
        return None
    return " ".join(str(value).strip().split())


def _dedupe(items, key):
    seen = set()
    out = []

    for item in items:
        k = item.get(key)
        if k is None:
            out.append(item)
            continue

        if k in seen:
            continue

        seen.add(k)
        out.append(item)

    return out


def prepare(observation):
    obs = deepcopy(observation)

    # -----------------------
    # Payments
    # -----------------------
    for payment in obs.get("payments", []):

        payment["currency"] = _clean_currency(
            payment.get("currency")
        )

        for field in [
            "merchant_name",
            "customer_name",
            "sender_name",
            "processor_reference",
            "reference",
            "transaction_id",
        ]:
            if field in payment:
                payment[field] = _clean_string(payment[field])

    # -----------------------
    # Invoices
    # -----------------------
    for invoice in obs.get("invoices", []):

        invoice["currency"] = _clean_currency(
            invoice.get("currency")
        )

        for field in [
            "customer_name",
            "merchant_name",
            "invoice_reference",
            "invoice_id",
        ]:
            if field in invoice:
                invoice[field] = _clean_string(invoice[field])

    # -----------------------
    # Transactions
    # Retry deliveries exist.
    # event_id is unique.
    # -----------------------
    obs["transactions"] = _dedupe(
        obs.get("transactions", []),
        "event_id",
    )

    # -----------------------
    # Bank credits
    # Retry deliveries exist.
    # bank_id is unique.
    # -----------------------
    obs["bank_credits"] = _dedupe(
        obs.get("bank_credits", []),
        "bank_id",
    )

    # -----------------------
    # Remittance docs
    # Copies exist.
    # lineage is canonical.
    # -----------------------
    obs["remittance_documents"] = _dedupe(
        obs.get("remittance_documents", []),
        "lineage",
    )

    return obs