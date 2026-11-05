from parser import load_batch

sample = {
    "bank_credits": [
        {
            "bank_id": "bank1",
            "payment_id": "pay1",
            "amount": 1000
        }
    ]
}

batch = load_batch(sample)

print(batch)