from models import Invoice, Payment
from rules import *

invoice = Invoice(
    invoice_id="INV001",
    merchant_id="M001",
    customer_id="C001",
    amount=10000,
    currency="INR",
    issue_day=1,
    due_day=30,
    status="OPEN"
)

payment = Payment(
    payment_id="PAY001",
    amount=4000,
    merchant_id="M001",
    currency="INR",
    capture_day=5,
    processor_reference=None,
    sender_name="Alice",
    transaction_id="TXN001"
)

print("Merchant:", merchant_matches(invoice, payment))
print("Currency:", currency_matches(invoice, payment))
print("Remaining:", remaining_balance(invoice))
print("Can Capture:", can_capture(invoice, payment))