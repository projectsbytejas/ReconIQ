"""
matcher.py - Matching Engine for EcoRecon

Orchestrates candidate generation, confidence scoring, and optimal assignment 
while rejecting low-confidence links to avoid costly false matches.
"""

from typing import List, Dict, Any, Tuple, Hashable
from scipy.optimize import linear_sum_assignment
import numpy as np
from scorer import score, HIGH_CONFIDENCE_THRESHOLD


def _index_records(
    records: List[Dict[str, Any]], fallback_key: str, entity_name: str
) -> Dict[Hashable, Dict[str, Any]]:
    indexed = {}
    for record in records:
        identifier = record.get("id")
        if identifier is None or (isinstance(identifier, str) and not identifier.strip()):
            identifier = record.get(fallback_key)

        if identifier is None or (isinstance(identifier, str) and not identifier.strip()):
            raise ValueError(f"{entity_name.lower()} ID must be present and non-empty")

        try:
            hash(identifier)
        except TypeError as exc:
            raise ValueError(f"{entity_name.lower()} ID must be hashable") from exc

        if identifier in indexed:
            raise ValueError(f"Duplicate {entity_name.lower()} ID: {identifier!r}")
        indexed[identifier] = record

    return indexed


def _generate_candidates(
    payment_map: Dict[Hashable, Dict[str, Any]],
    invoice_map: Dict[Hashable, Dict[str, Any]],
) -> List[Tuple[Hashable, Hashable]]:
    candidates = []

    for payment_id, payment in payment_map.items():
        payment_currency = str(payment.get("currency", "")).upper()
        payment_amount = float(payment.get("amount", 0.0))

        for invoice_id, invoice in invoice_map.items():
            invoice_currency = str(invoice.get("currency", "")).upper()
            if payment_currency and invoice_currency and payment_currency != invoice_currency:
                continue

            invoice_balance = float(invoice.get("remaining_balance", invoice.get("amount", 0.0)))
            if (
                payment_amount <= 0
                or invoice_balance <= 0
                or payment_amount > invoice_balance * 3.0
                or invoice_balance > payment_amount * 3.0
            ):
                continue

            candidates.append((payment_id, invoice_id))

    return candidates


def generate_candidates(
    payments: List[Dict[str, Any]], invoices: List[Dict[str, Any]]
) -> List[Tuple[Hashable, Hashable]]:
    """
    Generate plausible payment-invoice candidate pairs based on fast heuristic filters
    (e.g., currency matching, loose date proximity, and non-zero amount overlaps).
    
    Do NOT parse files or handle queries here.
    """
    payment_map = _index_records(payments, "payment_id", "Payment")
    invoice_map = _index_records(invoices, "invoice_id", "Invoice")
    return _generate_candidates(payment_map, invoice_map)


def match(payments: List[Dict[str, Any]], invoices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Executes the full matching pipeline:
    1. Generate candidates.
    2. Score candidate pairs.
    3. Perform optimal assignment (global optimization using scipy).
    4. Filter out low confidence matches (leaving unresolved is preferred over wrong matches).
    
    Returns:
        Dict containing:
            - 'accepted_matches': List of high-confidence matches with scores.
            - 'rejected_payments': List of payment IDs left unresolved.
            - 'rejected_invoices': List of invoice IDs left unresolved.
    """
    # Map IDs to objects for fast lookup
    payment_map = _index_records(payments, "payment_id", "Payment")
    invoice_map = _index_records(invoices, "invoice_id", "Invoice")
    
    p_ids = list(payment_map.keys())
    inv_ids = list(invoice_map.keys())
    
    if not p_ids or not inv_ids:
        return {
            "accepted_matches": [],
            "rejected_payments": p_ids,
            "rejected_invoices": inv_ids
        }

    # Build cost/score matrix (rows: payments, cols: invoices)
    score_matrix = np.zeros((len(p_ids), len(inv_ids)))
    candidate_pairs = set(_generate_candidates(payment_map, invoice_map))
    
    for i, p_id in enumerate(p_ids):
        for j, inv_id in enumerate(inv_ids):
            if (p_id, inv_id) not in candidate_pairs:
                continue

            confidence = score(payment_map[p_id], invoice_map[inv_id])
            if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                score_matrix[i, j] = confidence

    # Convert scores to costs for linear_sum_assignment (Maximizing score == Minimizing negative score)
    cost_matrix = -score_matrix
    
    # Solve optimal bipartite matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    accepted_matches = []
    matched_payments = set()
    matched_invoices = set()
    
    for r, c in zip(row_ind, col_ind):
        p_id = p_ids[r]
        inv_id = inv_ids[c]
        confidence = float(score_matrix[r, c])
        
        # Enforce high confidence threshold (Never guess rule)
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            accepted_matches.append({
                "payment_id": p_id,
                "invoice_id": inv_id,
                "confidence_score": confidence
            })
            matched_payments.add(p_id)
            matched_invoices.add(inv_id)
            
    # Compile unassigned / rejected items
    rejected_payments = [p_id for p_id in p_ids if p_id not in matched_payments]
    rejected_invoices = [inv_id for inv_id in inv_ids if inv_id not in matched_invoices]
    
    return {
        "accepted_matches": accepted_matches,
        "rejected_payments": rejected_payments,
        "rejected_invoices": rejected_invoices
    }