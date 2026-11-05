"""
scorer.py - Confidence Scoring Engine for EcoRecon

Calculates a normalized confidence score between a payment and an invoice.
Designed to strongly discourage incorrect matches (false positives cost ~5x more).
"""

from datetime import datetime
from typing import Dict, Any, Optional
from rapidfuzz import fuzz

# ==============================================================================
# CONFIGURATION & WEIGHTS
# ==============================================================================
WEIGHTS = {
    "amount": 0.40,          # Exact or close amount match is critical
    "currency": 0.15,        # Currency must align
    "merchant": 0.15,        # Merchant / counterparty name similarity
    "reference": 0.15,       # Invoice reference / memo similarity
    "date": 0.10,            # Proximity between payment date and invoice date
    "status": 0.05           # Invoice status (e.g., OPEN, PARTIAL)
}

HIGH_CONFIDENCE_THRESHOLD = 0.85
MAX_DATE_DIFFERENCE_DAYS = 45


def score_amount(payment_amount: float, invoice_remaining_balance: float) -> float:
    """
    Score the amount proximity. Exact matches get 1.0. 
    Overpayments or partial matches scale down smoothly.
    """
    if payment_amount <= 0 or invoice_remaining_balance <= 0:
        return 0.0
    
    if payment_amount == invoice_remaining_balance:
        return 1.0
    
    diff = abs(payment_amount - invoice_remaining_balance)
    max_val = max(payment_amount, invoice_remaining_balance)
    ratio = diff / max_val
    
    score = max(0.0, 1.0 - ratio)
    return score


def score_currency(payment_currency: str, invoice_currency: str) -> float:
    """Currency must be an exact match. Hard fail (0.0) on mismatch."""
    if not payment_currency or not invoice_currency:
        return 0.5
    return 1.0 if payment_currency.upper() == invoice_currency.upper() else 0.0


def score_text_similarity(val1: Optional[str], val2: Optional[str]) -> float:
    """Compute token-set string similarity using rapidfuzz."""
    if not val1 or not val2:
        return 0.0
    similarity = fuzz.token_sort_ratio(str(val1), str(val2))
    return similarity / 100.0


def score_date_proximity(payment_date: Any, invoice_date: Any) -> float:
    """Score based on how close payment date is to invoice date."""
    if payment_date is None or invoice_date is None:
        return 0.5

    if isinstance(payment_date, str) and not payment_date.strip():
        return 0.5
    if isinstance(invoice_date, str) and not invoice_date.strip():
        return 0.5

    try:
        if isinstance(payment_date, str):
            p_date = datetime.fromisoformat(payment_date.strip().replace("Z", "+00:00")).date()
        else:
            p_date = payment_date.date() if hasattr(payment_date, "date") else payment_date

        if isinstance(invoice_date, str):
            i_date = datetime.fromisoformat(invoice_date.strip().replace("Z", "+00:00")).date()
        else:
            i_date = invoice_date.date() if hasattr(invoice_date, "date") else invoice_date

        delta_days = (p_date - i_date).days

        if abs(delta_days) > MAX_DATE_DIFFERENCE_DAYS:
            return 0.0

        if delta_days < -2:
            return 0.1

        if delta_days > MAX_DATE_DIFFERENCE_DAYS:
            return 0.0

        score = 1.0 - (delta_days / MAX_DATE_DIFFERENCE_DAYS)
        return min(1.0, max(0.0, score))
    except (AttributeError, OverflowError, TypeError, ValueError):
        return 0.0


def score_status(invoice_status: Optional[str]) -> float:
    """Prioritize open or partially paid invoices over closed/voided ones."""
    if not invoice_status:
        return 0.5
    status = invoice_status.upper()
    if status in ["OPEN", "UNPAID", "PARTIAL"]:
        return 1.0
    elif status in ["DRAFT"]:
        return 0.4
    return 0.0


def score(payment: Dict[str, Any], invoice: Dict[str, Any]) -> float:
    """Computes a weighted confidence score between 0.0 and 1.0 for a payment-invoice pair."""
    s_amount = score_amount(
        payment.get("amount", 0.0), 
        invoice.get("remaining_balance", invoice.get("amount", 0.0))
    )
    
    s_curr = score_currency(payment.get("currency"), invoice.get("currency"))
    if s_curr == 0.0:
        return 0.0

    s_merchant = score_text_similarity(
        payment.get("merchant", payment.get("payer")), 
        invoice.get("merchant", invoice.get("customer"))
    )
    
    s_reference = score_text_similarity(
        payment.get("reference", payment.get("memo")), 
        invoice.get("reference", invoice.get("invoice_number"))
    )
    
    s_date = score_date_proximity(
        payment.get("date", payment.get("payment_date")), 
        invoice.get("date", invoice.get("invoice_date"))
    )
    
    s_status = score_status(invoice.get("status"))

    total_score = (
        (s_amount * WEIGHTS["amount"]) +
        (s_curr * WEIGHTS["currency"]) +
        (s_merchant * WEIGHTS["merchant"]) +
        (s_reference * WEIGHTS["reference"]) +
        (s_date * WEIGHTS["date"]) +
        (s_status * WEIGHTS["status"])
    )

    return round(float(total_score), 4)