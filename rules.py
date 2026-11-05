from difflib import SequenceMatcher


def _get(record, name, default=""):
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


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

    payment_amount = _get(payment, "amount", 0)
    payment_currency = str(_get(payment, "currency", "") or "").strip().upper()
    payment_day = _get(payment, "day", _get(payment, "capture_day", 0))

    payment_customer = _get(payment, "customer_name") or _get(payment, "sender_name") or ""

    for invoice in invoices:

        score = 0

        invoice_amount = _get(invoice, "amount", 0)
        if payment_amount == invoice_amount:
            score += 60
        elif abs(payment_amount - invoice_amount) <= 2:
            score += 45

        if payment_currency == str(_get(invoice, "currency", "") or "").strip().upper():
            score += 20

        score += int(
            20
            * _similar(
                payment_customer,
                _get(invoice, "customer_name", ""),
            )
        )

        invoice_day = _get(invoice, "issue_day", _get(invoice, "day", 0))

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
    if not isinstance(invoice, dict):
        invoice = vars(invoice)
    if not isinstance(payment, dict):
        payment = vars(payment)
    return (
        str(invoice.get("merchant_name", invoice.get("merchant_id", ""))).strip().lower()
        == str(payment.get("merchant_name", payment.get("merchant_id", ""))).strip().lower()
    )


def currency_matches(invoice, payment):
    if not isinstance(invoice, dict):
        invoice = vars(invoice)
    if not isinstance(payment, dict):
        payment = vars(payment)
    return (
        invoice.get("currency", "").strip().upper()
        == payment.get("currency", "").strip().upper()
    )


def remaining_balance(invoice):
    if not isinstance(invoice, dict):
        invoice = vars(invoice)
    return invoice.get("amount", 0) - invoice.get("captured_amount", 0)


def can_capture(invoice, payment):
    if not isinstance(payment, dict):
        payment = vars(payment)
    return payment.get("amount", 0) <= remaining_balance(invoice)