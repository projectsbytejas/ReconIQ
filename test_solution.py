import json
import unittest

from solution import decide


class SolutionTests(unittest.TestCase):
    def _observation(self):
        return {
            "payments": [{"payment_id": "p", "merchant": "m", "currency": "INR"}],
            "invoices": [{"invoice_id": "i", "merchant": "m", "currency": "INR", "amount": 100, "status": "open"}],
            "transactions": [{"transaction_id": "t", "payment_id": "p", "gross": 100, "fee": 2, "net": 98}],
            "bank_credits": [{"bank_id": "b", "payment_id": "p", "amount": 98}],
            "investigations": [],
            "queries_remaining": 4,
        }

    def test_submit_shape_and_state_round_trip(self):
        result = decide(self._observation(), {})
        self.assertEqual(set(result["action"]), {"op", "matches", "ledger", "suspense", "bank_total"})
        json.loads(json.dumps(result))

    def test_null_result_leaves_payment_unresolved(self):
        observation = self._observation()
        observation["investigations"] = [{"payment_id": "p", "verified_invoice_id": None}]
        result = decide(observation, {"queried": ["p"], "queries_used": 1})
        self.assertEqual(result["action"]["matches"], [])
        self.assertEqual(result["action"]["suspense"]["captured"], 100)

    def test_duplicate_invoice_is_not_assigned(self):
        observation = self._observation()
        observation["payments"].append({"payment_id": "p2", "merchant": "m", "currency": "INR"})
        observation["transactions"].append({"transaction_id": "t2", "payment_id": "p2", "gross": 50, "fee": 1, "net": 49})
        observation["bank_credits"].append({"bank_id": "b2", "payment_id": "p2", "amount": 49})
        observation["investigations"] = [
            {"payment_id": "p", "verified_invoice_id": "i"},
            {"payment_id": "p2", "verified_invoice_id": "i"},
        ]
        result = decide(observation, {"queried": ["p", "p2"], "queries_used": 2})
        self.assertEqual(len(result["action"]["matches"]), 1)

    def test_query_cap_and_no_repeat(self):
        state = {"queried": ["p"], "queries_used": 1}
        result = decide(self._observation(), state)
        self.assertLessEqual(result["state"]["queries_used"], 4)
        self.assertEqual(len(result["state"]["queried"]), len(set(result["state"]["queried"])))


if __name__ == "__main__":
    unittest.main()
