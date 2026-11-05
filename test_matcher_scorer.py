"""
test_matcher_scorer.py - Pytest suite for EcoRecon matching & scoring engine.
"""

import pytest
from datetime import datetime, timedelta
import matcher
from scorer import (
    score_amount,
    score_currency,
    score_text_similarity,
    score_date_proximity,
    score_status,
    score,
    HIGH_CONFIDENCE_THRESHOLD
)
from matcher import generate_candidates, match

# ==============================================================================
# SCORER TESTS
# ==============================================================================

def test_score_amount_exact():
    """Exact amounts should return a score of 1.0."""
    assert score_amount(100.0, 100.0) == 1.0

def test_score_amount_partial():
    """Partial payments should scale down smoothly based on ratio difference."""
    s = score_amount(50.0, 100.0)
    assert 0.0 < s < 1.0

def test_score_amount_zero_or_negative():
    """Zero or negative amounts should return 0.0."""
    assert score_amount(0.0, 100.0) == 0.0
    assert score_amount(-50.0, 100.0) == 0.0
    assert score_amount(100.0, 0.0) == 0.0

def test_score_currency_match():
    """Matching currencies must return 1.0."""
    assert score_currency("USD", "usd") == 1.0

def test_score_currency_mismatch():
    """Mismatched currencies must return 0.0 (hard fail)."""
    assert score_currency("USD", "EUR") == 0.0

def test_score_text_similarity_exact():
    """Identical strings should score 1.0."""
    assert score_text_similarity("Acme Corp", "Acme Corp") == 1.0

def test_score_text_similarity_partial():
    """Similar strings should score moderately high."""
    s = score_text_similarity("Acme Corporation", "Acme Corp")
    assert s > 0.7

def test_score_text_similarity_none():
    """Missing text inputs should safely return 0.0."""
    assert score_text_similarity(None, "Acme Corp") == 0.0
    assert score_text_similarity("Acme", None) == 0.0

def test_score_date_proximity_valid():
    """Dates close to each other should receive high scores."""
    d1 = datetime.now()
    d2 = d1 + timedelta(days=2)
    assert score_date_proximity(d1, d2) > 0.9

def test_score_date_proximity_too_far():
    """Dates exceeding the max difference window should score 0.0."""
    d1 = datetime.now()
    d2 = d1 + timedelta(days=60)
    assert score_date_proximity(d1, d2) == 0.0

def test_score_date_proximity_missing_is_neutral():
    assert score_date_proximity(None, "2026-06-01") == 0.5

def test_score_date_proximity_invalid_is_zero():
    assert score_date_proximity("not-a-date", "2026-06-01") == 0.0

def test_score_status():
    """Verify status scoring priorities."""
    assert score_status("OPEN") == 1.0
    assert score_status("PARTIAL") == 1.0
    assert score_status("PAID") == 0.0
    assert score_status(None) == 0.5


# ==============================================================================
# MATCHER TESTS
# ==============================================================================

def test_generate_candidates_filters():
    """Ensure candidate generation filters out currency mismatches and extreme amounts."""
    payments = [
        {"id": "p1", "currency": "USD", "amount": 100.0},
        {"id": "p2", "currency": "EUR", "amount": 500.0},
    ]
    invoices = [
        {"id": "inv1", "currency": "USD", "remaining_balance": 100.0},
        {"id": "inv2", "currency": "USD", "remaining_balance": 10000.0}, # Too large
    ]
    
    candidates = generate_candidates(payments, invoices)
    
    # p1 matches inv1 (USD and amount compatible)
    assert ("p1", "inv1") in candidates
    # p2 should be filtered out due to currency mismatch (EUR vs USD)
    assert ("p2", "inv1") not in candidates
    # inv2 should be filtered out due to extreme amount discrepancy
    assert ("p1", "inv2") not in candidates

def test_match_high_confidence():
    """High confidence matches should be accepted cleanly."""
    payments = [{
        "id": "p1",
        "currency": "USD",
        "amount": 250.0,
        "merchant": "GreenEnergy Inc",
        "reference": "INV-1001",
        "date": "2026-06-01"
    }]
    invoices = [{
        "id": "inv1",
        "currency": "USD",
        "remaining_balance": 250.0,
        "merchant": "GreenEnergy Inc",
        "reference": "INV-1001",
        "date": "2026-05-30",
        "status": "OPEN"
    }]
    
    result = match(payments, invoices)
    
    assert len(result["accepted_matches"]) == 1
    assert result["accepted_matches"][0]["payment_id"] == "p1"
    assert result["accepted_matches"][0]["invoice_id"] == "inv1"
    assert result["accepted_matches"][0]["confidence_score"] >= HIGH_CONFIDENCE_THRESHOLD
    assert len(result["rejected_payments"]) == 0

def test_match_ignores_non_candidates(monkeypatch):
    """Pairs rejected by candidate generation must not be scored or accepted."""
    monkeypatch.setattr(matcher, "score", lambda payment, invoice: 1.0)
    payments = [{"id": "p1", "currency": "EUR", "amount": 100.0}]
    invoices = [{"id": "inv1", "currency": "USD", "remaining_balance": 100.0}]

    result = match(payments, invoices)

    assert result["accepted_matches"] == []
    assert result["rejected_payments"] == ["p1"]
    assert result["rejected_invoices"] == ["inv1"]

@pytest.mark.parametrize(
    "payments,invoices,error_match",
    [
        ([{"amount": 100.0}], [{"id": "inv1", "amount": 100.0}], "payment ID"),
        ([{"id": "p1", "amount": 100.0}], [{"amount": 100.0}], "invoice ID"),
    ],
)
def test_match_rejects_missing_ids(payments, invoices, error_match):
    with pytest.raises(ValueError, match=error_match):
        match(payments, invoices)

def test_match_rejects_duplicate_ids():
    payments = [{"id": "p1", "amount": 100.0}, {"id": "p1", "amount": 100.0}]
    invoices = [{"id": "inv1", "amount": 100.0}]

    with pytest.raises(ValueError, match="Duplicate payment ID"):
        match(payments, invoices)

def test_match_preserves_zero_identifier():
    payments = [{
        "id": 0,
        "currency": "USD",
        "amount": 100.0,
        "merchant": "Acme Corp",
        "reference": "INV-1",
        "date": "2026-06-01",
    }]
    invoices = [{
        "id": "inv1",
        "currency": "USD",
        "remaining_balance": 100.0,
        "merchant": "Acme Corp",
        "reference": "INV-1",
        "date": "2026-06-01",
        "status": "OPEN",
    }]

    result = match(payments, invoices)

    assert result["accepted_matches"][0]["payment_id"] == 0

def test_match_keeps_valid_pair_when_weaker_pairs_conflict(monkeypatch):
    """Below-threshold scores must not crowd out a high-confidence assignment."""
    scores = {
        ("p1", "inv1"): 0.90,
        ("p1", "inv2"): 0.84,
        ("p2", "inv1"): 0.84,
        ("p2", "inv2"): 0.0,
    }
    monkeypatch.setattr(
        matcher,
        "score",
        lambda payment, invoice: scores[(payment["id"], invoice["id"])],
    )
    payments = [{"id": "p1", "amount": 100.0}, {"id": "p2", "amount": 100.0}]
    invoices = [
        {"id": "inv1", "remaining_balance": 100.0},
        {"id": "inv2", "remaining_balance": 100.0},
    ]

    result = match(payments, invoices)

    assert result["accepted_matches"] == [
        {"payment_id": "p1", "invoice_id": "inv1", "confidence_score": 0.90}
    ]
    assert result["rejected_payments"] == ["p2"]
    assert result["rejected_invoices"] == ["inv2"]

def test_match_low_confidence_rejection():
    """Low confidence matches must be rejected rather than guessed (Never Guess Rule)."""
    payments = [{
        "id": "p_bad",
        "currency": "USD",
        "amount": 99.99,
        "merchant": "Random Shop",
        "reference": "REF-999",
        "date": "2026-06-01"
    }]
    invoices = [{
        "id": "inv_other",
        "currency": "USD",
        "remaining_balance": 5000.0,
        "merchant": "Enterprise Corp",
        "reference": "INV-555",
        "date": "2026-01-01",
        "status": "OPEN"
    }]
    
    result = match(payments, invoices)
    
    assert len(result["accepted_matches"]) == 0
    assert "p_bad" in result["rejected_payments"]
    assert "inv_other" in result["rejected_invoices"]