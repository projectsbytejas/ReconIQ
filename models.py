from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class Invoice:
    invoice_id: str
    merchant_id: str
    customer_id: str
    amount: int
    currency: str
    issue_day: int
    due_day: int
    status: str

    captured_amount: int = 0

@dataclass
class Payment:
    payment_id: str
    amount: int
    merchant_id: str
    currency: str
    capture_day: int
    processor_reference: Optional[str]
    sender_name: Optional[str]
    transaction_id: str

@dataclass
class BankCredit:
    bank_id: str
    payment_id: str
    amount: int

@dataclass
class Refund:
    refund_id: str
    payment_id: str
    amount: int

@dataclass
class Batch:
    invoices: list[Invoice] = field(default_factory=list)
    payments: list[Payment] = field(default_factory=list)
    bank_credits: list[BankCredit] = field(default_factory=list)
    refunds: list[Refund] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)