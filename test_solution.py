import json
import unittest

from solution import decide, validate_posting


class SolutionTests(unittest.TestCase):
    def setUp(self):
        self.payment = {"id": "p", "amount": 100, "currency": "USD", "merchant_id": "m"}
        self.invoice = {"id": "i", "amount": 100, "currency": "USD", "merchant_id": "m"}
        self.observation = {"payments": [self.payment], "invoices": [self.invoice]}

    def test_validator_rejects_duplicate_invoice(self):
        allocations = [{"payment_id": "p", "invoice_id": "i", "amount": 100},
                       {"payment_id": "p2", "invoice_id": "i", "amount": 100}]
        payments = [self.payment, {"id": "p2", "amount": 100}]
        self.assertFalse(validate_posting(self.observation, payments, {"i": self.invoice}, allocations, []))

    def test_validator_rejects_overcapture(self):
        self.assertFalse(validate_posting(self.observation, [self.payment], {"i": {"id": "i", "amount": 50}}, [{"payment_id": "p", "invoice_id": "i", "amount": 100}], []))

    def test_validator_rejects_cross_merchant_and_cancelled(self):
        for invoice in ({"id": "i", "amount": 100, "merchant_id": "other"}, {"id": "i", "amount": 100, "cancelled": True}):
            self.assertFalse(validate_posting(self.observation, [self.payment], {"i": invoice}, [{"payment_id": "p", "invoice_id": "i", "amount": 100}], []))

    def test_query_cap_and_null_answer(self):
        observation = {"payments": [{"id": "p", "amount": 1000}], "invoices": [{"id": "i1", "amount": 1000}, {"id": "i2", "amount": 1000}], "candidates": {"p": [{"invoice_id": "i1", "confidence": 0.5}, {"invoice_id": "i2", "confidence": 0.4}]}, "query_results": {"p": None}}
        result = decide(observation, {"queries_used": 0, "queried": ["p"], "query_results": {}, "duplicate_map": {}, "pending": [], "phase": "awaiting_query"})
        self.assertEqual(result["action"]["allocations"], [])
        self.assertIn("p", result["action"]["unresolved"])
        self.assertLessEqual(result["state"]["queries_used"], 4)

    def test_state_round_trips(self):
        observation = {"payments": [], "invoices": []}
        result = decide(observation, {})
        json.loads(json.dumps(result))


if __name__ == "__main__":
    unittest.main()
