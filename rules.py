from difflib import SequenceMatcher


def _similar(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(
        None,
        str(a).lower().strip(),
        str(b).lower().strip(),
    ).ratio()


def score_candidates(payment, invoices):
    """
    Return invoices sorted from best candidate to worst.
    """

    ranked = []

    payment_amount = payment.get("amount", 0)
    payment_currency = payment.get("currency", "").strip().upper()
    payment_day = payment.get("day", 0)

    payment_customer = payment.get("customer_name") or payment.get("sender_name") or ""

    for invoice in invoices:

        score = 0

        if payment_amount == invoice.get("amount"):
            score += 60
        elif abs(payment_amount - invoice.get("amount", 0)) <= 2:
            score += 45

        if payment_currency == invoice.get("currency", "").strip().upper():
            score += 20

        score += int(
            20
            * _similar(
                payment_customer,
                invoice.get("customer_name", ""),
            )
        )

        invoice_day = invoice.get("issue_day", invoice.get("day", 0))

        diff = abs(payment_day - invoice_day)

        if diff <= 1:
            score += 15
        elif diff <= 3:
            score += 10
        elif diff <= 7:
            score += 5

        ranked.append((invoice, score))

    ranked.sort(key=lambda x: x[1], reverse=True)

    return ranked


def merchant_matches(invoice, payment):
    return (
        invoice.get("merchant_name", "").strip().lower()
        == payment.get("merchant_name", "").strip().lower()
    )


def currency_matches(invoice, payment):
    return (
        invoice.get("currency", "").strip().upper()
        == payment.get("currency", "").strip().upper()
    )


def remaining_balance(invoice):
    return invoice.get("amount", 0) - invoice.get("captured_amount", 0)


def can_capture(invoice, payment):
    return payment.get("amount", 0) <= remaining_balance(invoice)