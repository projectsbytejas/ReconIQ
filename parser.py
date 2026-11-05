from models import BankCredit, Batch


def load_batch(observation):
    """
    Converts the raw observation into Python objects.
    """

    bank_credits = []

    for credit in observation["bank_credits"]:

        bank = BankCredit(
            bank_id=credit["bank_id"],
            payment_id=credit["payment_id"],
            amount=credit["amount"]
        )

        bank_credits.append(bank)

    return Batch(bank_credits=bank_credits)