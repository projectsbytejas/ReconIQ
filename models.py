from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Invoice:
    invoice_id: str
    merchant: str
    currency: str
    gross_amount: int

    captured_amount: int = 0

    status: str = "OPEN"

    due_date: Optional[str] = None


@dataclass
class Payment:
    payment_id: str

    merchant: str

    currency: str

    gross_amount: int

    reference: Optional[str]

    payer: Optional[str]

    timestamp: Optional[str]

    duplicate: bool = False

    refunded: bool = False


@dataclass
class Batch:

    invoices: List[Invoice]

    payments: List[Payment]