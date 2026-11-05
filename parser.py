from models import BankCredit, Batch


def load_batch(observation):
    """
    Converts the raw observation into Python objects.
    """

    bank_credits = []

    for credit in observation.get("bank_credits", []):

        bank = BankCredit(
            bank_id=credit["bank_id"],
            payment_id=credit["payment_id"],
            amount=credit["amount"]
        )

        bank_credits.append(bank)

    return Batch(
        invoices=list(observation.get("invoices", [])),
        payments=list(observation.get("payments", observation.get("transactions", []))),
        bank_credits=bank_credits,
        refunds=list(observation.get("refunds", [])),
        raw=dict(observation),
    )