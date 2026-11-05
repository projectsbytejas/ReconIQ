import copy, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import solution as S

def obs():
    return {
        "invoices": [
            {"invoice_id": "inv_a", "merchant": "m1", "currency": "INR", "customer_name": "A", "tax_suffix": "1111",
             "reference": "INV-1000001", "issued_day": 1, "amount": 1000, "status": "open"},
            {"invoice_id": "inv_b", "merchant": "m1", "currency": "INR", "customer_name": "B", "tax_suffix": "2222",
             "reference": "INV-1000002", "issued_day": 1, "amount": 500, "status": "cancelled"}],
        "payments": [
            {"payment_id": "pay_1", "merchant": "m1", "currency": " inr ", "channel": "wallet", "sender_name": "A",
             "tax_suffix": "1111", "source_ref": "INV-1000001", "day": 2}],
        "transactions": [
            {"event_id": "e1", "transaction_id": "t1", "payment_id": "pay_1", "kind": "capture", "gross": 1000, "fee": 20, "net": 980, "parent_transaction_id": None},
            {"event_id": "e2", "transaction_id": "t1", "payment_id": "pay_1", "kind": "capture", "gross": 1000, "fee": 20, "net": 980, "parent_transaction_id": None}],
        "bank_credits": [{"bank_id": "b1", "payment_id": "pay_1", "amount": 980}, {"bank_id": "b1", "payment_id": "pay_1", "amount": 980}],
        "remittance_documents": [], "investigations": [], "queries_remaining": 4}

def test_contract_and_dedup():
    r = S.decide(obs(), {})
    assert set(r) == {"action", "state"}
    a = r["action"]
    assert a["op"] == "submit" and set(a) == {"op", "matches", "ledger", "suspense", "bank_total"}
    assert a["bank_total"] == 980 and len(a["ledger"]) == 2
    assert a["matches"] == [{"payment_id": "pay_1", "invoice_id": "inv_a"}]
    assert a["ledger"][0]["remaining_due"] == 0 and a["ledger"][1]["remaining_due"] == 500

def test_cancelled_needs_full_reversal():
    o = obs()
    assert not S.eligible(o["payments"][0], {"captured": 500, "refunded": 0}, o["invoices"][1])
    assert S.eligible(o["payments"][0], {"captured": 500, "refunded": 500}, o["invoices"][1])

def test_reference_and_currency_normalisation():
    assert S.norm_ref(" inv / 6608957 ") == "6608957" and S.norm_ref("INV-6608957") == "6608957"
    assert S.norm_currency(" inr ") == "INR"

def test_verified_null_is_suspense():
    o = obs(); o["investigations"] = [{"payment_id": "pay_1", "verified_invoice_id": None}]; o["queries_remaining"] = 3
    a = S.decide(o, {})["action"]
    assert a["matches"] == [] and a["suspense"]["net"] == 980

def test_verified_invoice_locked():
    o = obs(); o["investigations"] = [{"payment_id": "pay_1", "verified_invoice_id": "inv_a"}]
    assert S.decide(o, {})["action"]["matches"][0]["invoice_id"] == "inv_a"

def test_failsafe_shape():
    o = obs(); del o["payments"][0]["merchant"]
    a = S.decide(o, {})["action"]
    assert a["op"] == "submit" and a["matches"] == []

def test_integer_money_only():
    import pytest
    for bad in (True, 1.0):
        with pytest.raises(ValueError):
            S._int(bad)
