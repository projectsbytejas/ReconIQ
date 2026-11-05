"""ReconIQ - payment-to-invoice reconciliation for the hackathon contract.

Entry point: decide(observation, state) -> {"action": ..., "state": ...}

Standard library only, deterministic, no machine learning.  Pipeline:
  1. normalise + deduplicate (transactions by transaction_id, bank credits by bank_id)
  2. build payment groups from unique transactions (integer minor units)
  3. generate candidates under hard accounting constraints
  4. score candidates with explicit, independent-origin evidence
  5. resolve one-to-one assignment globally (exact enumeration per component)
  6. spend the query budget where a verified answer is worth more than its cost
  7. emit the exact submit schema (full ledger + suspense + bank_total)
"""
import itertools
import math
import re

QUERY_COST = 180
WRONG_MULT = 4
MAX_QUERIES = 4

# ---------------------------------------------------------------- weights --
# Log-odds contributions.  They were set by hand (rounded) after tabulating how
# each evidence pattern behaves on the supplied training scenarios.
GAP_OK = range(0, 4)           # payment day - invoice issued day seen for true links
W_GAP_OUTSIDE = -4.0           # soft only: schema says date is never a hard cutoff
W_UNDERPAID = -3.0             # captured < invoice amount (training truth is exact)
# Origin A = raw payment + processor receipt + copied lineage (ONE vote):
W_A = {                        # (reference state, tax suffix matches)
    ("match", True): 4.3, ("match", False): -0.3,
    ("none", True): 3.0, ("none", False): -2.8,
    ("other", True): 1.8, ("other", False): -3.3,
}
W_A_NAME = (0.2, -0.5)         # payment sender name (match, mismatch): weak
# Independent remittance lineages (active, not the processor receipt):
W_DOC_MATCH = 1.5
W_DOC_OTHER = -0.5
W_DOC_NAME_BAD = -4.0          # document names the payer; invoice customer differs
W_DOC_NAME_OK = 0.3
W_DOC_NAME_CAP = -6.0
NULL_LOGIT = 0.0               # score of "this payment matches nothing"
# A payment whose tax suffix matches no eligible invoice is (in all supplied
# data) never a real match; one whose suffix matches some candidate usually is.
W_NULL_TAX_UNMATCHED = 3.5
W_NULL_TAX_MATCHED = -2.0
TEMPERATURE = 1.35              # >1 flattens posteriors (evidence is correlated)
PRUNE_MARGIN = 9.0             # drop candidates this far below the payment's best


# ------------------------------------------------------------ normalisation --
def _int(v):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError("money must be int")
    return v


def norm_currency(c):
    return str(c).strip().upper() if c is not None else ""


def norm_ref(r):
    """INV followed by seven digits -> canonical digits; otherwise None."""
    if r is None:
        return None
    s = re.sub(r"[^A-Za-z0-9]", "", str(r)).upper()
    m = re.fullmatch(r"INV(\d{7})", s)
    return m.group(1) if m else None


def norm_tax(t):
    return None if t is None else str(t).strip()


def norm_name(n):
    return None if n is None else " ".join(str(n).split()).casefold()


def unique_transactions(obs):
    seen = {}
    for t in obs.get("transactions", []):
        tid = t["transaction_id"]
        if tid not in seen:
            seen[tid] = t
    return list(seen.values())


def unique_bank_credits(obs):
    seen = {}
    for b in obs.get("bank_credits", []):
        if b["bank_id"] not in seen:
            seen[b["bank_id"]] = b
    return list(seen.values())


def payment_groups(obs):
    groups = {}
    for p in obs["payments"]:
        groups[p["payment_id"]] = dict(captured=0, refunded=0, fees=0, net=0)
    for t in unique_transactions(obs):
        g = groups.setdefault(t["payment_id"], dict(captured=0, refunded=0, fees=0, net=0))
        gross = _int(t["gross"])
        g["captured"] += max(gross, 0)
        g["refunded"] += max(-gross, 0)
        g["fees"] += _int(t["fee"])
        g["net"] += _int(t["net"])
    return groups


# ----------------------------------------------------------------- evidence --
def independent_documents(docs):
    """Active, non-receipt documents, one per lineage (copies are not votes)."""
    by_lineage = {}
    for d in sorted(docs, key=lambda d: (d.get("received_day", 0), str(d.get("document_id")))):
        if d.get("status") != "active" or d.get("source") == "processor_receipt":
            continue
        by_lineage.setdefault(d.get("lineage") or d.get("document_id"), d)
    return list(by_lineage.values())


def origin_reference(payment, docs):
    """Reference of the raw-payment origin (source_ref, else its receipt copy)."""
    ref = norm_ref(payment.get("source_ref"))
    if ref is not None:
        return ref
    for d in docs:
        if d.get("source") == "processor_receipt" and d.get("status") == "active":
            ref = norm_ref(d.get("reference"))
            if ref is not None:
                return ref
    return None


def eligible(pay, grp, inv):
    """Hard constraints (merchant, currency, accounting, cancelled rule)."""
    if pay["merchant"] != inv["merchant"]:
        return False
    if norm_currency(pay["currency"]) != norm_currency(inv["currency"]):
        return False
    if grp["captured"] <= 0 or grp["captured"] > inv["amount"]:
        return False
    if grp["refunded"] > grp["captured"]:
        return False
    if inv["status"] == "cancelled" and grp["captured"] != grp["refunded"]:
        return False
    return True


def score(pay, grp, inv, docs):
    s = 0.0
    if grp["captured"] < inv["amount"]:
        s += W_UNDERPAID
    gap = pay["day"] - inv["issued_day"]
    if gap not in GAP_OK:
        s += W_GAP_OUTSIDE
    ref = origin_reference(pay, docs)
    inv_ref = norm_ref(inv["reference"])
    state = "none" if ref is None else ("match" if ref == inv_ref else "other")
    tax_ok = norm_tax(pay.get("tax_suffix")) == norm_tax(inv["tax_suffix"])
    s += W_A[(state, tax_ok)]
    s += W_A_NAME[0] if norm_name(pay.get("sender_name")) == norm_name(inv["customer_name"]) else W_A_NAME[1]
    bad_names = 0.0
    for d in independent_documents(docs):
        dref = norm_ref(d.get("reference"))
        if dref is not None:
            s += W_DOC_MATCH if dref == inv_ref else W_DOC_OTHER
        dn = norm_name(d.get("sender_name"))
        if dn is not None:
            if dn == norm_name(inv["customer_name"]):
                s += W_DOC_NAME_OK
            else:
                bad_names += W_DOC_NAME_BAD
    s += max(bad_names, W_DOC_NAME_CAP)
    return s


# ---------------------------------------------------------------- inference --
def components(cands):
    """Connected components of payments that share candidate invoices."""
    pids = sorted(cands)
    parent = {p: p for p in pids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for p in pids:
        for inv_id in cands[p]:
            if inv_id in owner:
                a, b = find(p), find(owner[inv_id])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owner[inv_id] = p
    comps = {}
    for p in pids:
        comps.setdefault(find(p), []).append(p)
    return [sorted(v) for _, v in sorted(comps.items())]


def marginals(cands, scores, null_scores=None):
    """Exact marginals of the one-to-one assignment distribution.

    P(assignment) ~ prod exp(score of assigned link) * prod exp(NULL) for
    unmatched payments.  Enumerated per connected component, with a budgeted
    fallback (greedy-conditioned) for pathologically large components.
    """
    null_scores = null_scores or {}
    result = {p: {} for p in cands}
    for comp in components(cands):
        # smallest-branching first keeps the enumeration tight
        order = sorted(comp, key=lambda p: len(cands[p]))
        budget = [200000]
        total = [0.0]
        acc = {p: {} for p in comp}
        null_acc = {p: 0.0 for p in comp}

        def rec(k, used, weight, chosen):
            if budget[0] <= 0:
                return
            if k == len(order):
                budget[0] -= 1
                total[0] += weight
                for p, i in chosen:
                    if i is None:
                        null_acc[p] += weight
                    else:
                        acc[p][i] = acc[p].get(i, 0.0) + weight
                return
            p = order[k]
            rec(k + 1, used, weight * math.exp(null_scores.get(p, NULL_LOGIT) / TEMPERATURE), chosen + [(p, None)])
            for inv_id in cands[p]:
                if inv_id not in used:
                    rec(k + 1, used | {inv_id}, weight * math.exp(scores[(p, inv_id)] / TEMPERATURE), chosen + [(p, inv_id)])

        rec(0, frozenset(), 1.0, [])
        if budget[0] <= 0 or total[0] <= 0:
            # fallback: independent softmax per payment (never forces a match)
            for p in comp:
                z = math.exp(null_scores.get(p, NULL_LOGIT) / TEMPERATURE) + sum(math.exp(scores[(p, i)] / TEMPERATURE) for i in cands[p])
                result[p] = {i: math.exp(scores[(p, i)] / TEMPERATURE) / z for i in cands[p]}
        else:
            for p in comp:
                result[p] = {i: w / total[0] for i, w in acc[p].items()}
    return result


def choose_matches(cands, marg, value, threshold=None):
    """Maximise expected score: value * (5p - 4) per accepted link, one-to-one."""
    links = []
    for p, row in marg.items():
        for i, pr in row.items():
            gain = value[p] * ((1 + WRONG_MULT) * pr - WRONG_MULT)
            if gain > 0:
                links.append((p, i, gain))
    best = {}
    by_pay = {}
    for p, i, g in links:
        by_pay.setdefault(p, []).append((i, g))
    for comp in components({p: [i for i, _ in v] for p, v in by_pay.items()}):
        opts = {p: by_pay[p] for p in comp}
        order = sorted(comp)
        top = [0.0, {}]

        def rec(k, used, total, chosen):
            if k == len(order):
                if total > top[0] + 1e-12:
                    top[0], top[1] = total, dict(chosen)
                return
            p = order[k]
            rec(k + 1, used, total, chosen)
            for i, g in sorted(opts[p]):
                if i not in used:
                    chosen[p] = i
                    rec(k + 1, used | {i}, total + g, chosen)
                    del chosen[p]

        rec(0, frozenset(), 0.0, {})
        best.update(top[1])
    return best


# ------------------------------------------------------------------- build --
def build_model(obs):
    groups = payment_groups(obs)
    invoices = {i["invoice_id"]: i for i in obs["invoices"]}
    pays = {p["payment_id"]: p for p in obs["payments"]}
    docs = {}
    for d in obs.get("remittance_documents", []):
        docs.setdefault(d["payment_id"], []).append(d)
    verified = {}
    for inv in obs.get("investigations", []):
        verified[inv["payment_id"]] = inv.get("verified_invoice_id")
    return groups, invoices, pays, docs, verified


def infer(obs, extra_verified=None):
    groups, invoices, pays, docs, verified = build_model(obs)
    if extra_verified:
        for k, v in extra_verified.items():
            verified.setdefault(k, v)
    taken = {v for v in verified.values() if v is not None}
    cands, scores, null_scores = {}, {}, {}
    value = {p: groups[p]["captured"] for p in pays}
    for pid, pay in pays.items():
        if pid in verified:
            continue
        grp = groups[pid]
        row, tax_hit = [], False
        for iid, inv in invoices.items():
            if not eligible(pay, grp, inv):
                continue
            if norm_tax(pay.get("tax_suffix")) == norm_tax(inv["tax_suffix"]):
                tax_hit = True
            if iid in taken:
                continue
            row.append(iid)
            scores[(pid, iid)] = score(pay, grp, inv, docs.get(pid, []))
        if row:
            top = max(scores[(pid, i)] for i in row)
            row = [i for i in row if scores[(pid, i)] >= top - PRUNE_MARGIN]
            cands[pid] = sorted(row)
            null_scores[pid] = NULL_LOGIT + (W_NULL_TAX_MATCHED if tax_hit else W_NULL_TAX_UNMATCHED)
    marg = marginals(cands, scores, null_scores) if cands else {}
    return dict(groups=groups, invoices=invoices, pays=pays, docs=docs, verified=verified,
                cands=cands, scores=scores, null_scores=null_scores, marg=marg, value=value)


def final_mapping(model):
    mapping = {}
    for pid, iid in model["verified"].items():
        if iid is None:
            continue
        pay, inv = model["pays"].get(pid), model["invoices"].get(iid)
        if pay is None or inv is None or not eligible(pay, model["groups"][pid], inv):
            continue  # verified but impossible under accounting rules: keep integrity
        if iid in mapping.values():
            continue
        mapping[pid] = iid
    picks = choose_matches(model["cands"], model["marg"], model["value"])
    for pid, iid in picks.items():
        if iid not in mapping.values():
            mapping[pid] = iid
    return mapping


def pick_query(model, queried, remaining):
    if remaining <= 0:
        return None
    mapping = final_mapping(model)
    best, best_gain = None, 0.0
    for pid in sorted(model["cands"]):
        if pid in queried or pid in model["verified"]:
            continue
        v = model["value"][pid]
        row = model["marg"].get(pid, {})
        p_real = sum(row.values())
        p_pick = row.get(mapping.get(pid), 0.0) if pid in mapping else 0.0
        now = v * ((1 + WRONG_MULT) * p_pick - WRONG_MULT) if pid in mapping else 0.0
        gain = v * p_real - QUERY_COST - now
        if gain > best_gain:
            best, best_gain = pid, gain
    return best


def ledger_and_suspense(obs, mapping):
    fields = ("captured", "refunded", "fees", "net")
    rows = {i["invoice_id"]: dict.fromkeys(fields, 0) for i in obs["invoices"]}
    suspense = dict.fromkeys(fields, 0)
    for t in unique_transactions(obs):
        row = rows[mapping[t["payment_id"]]] if t["payment_id"] in mapping else suspense
        gross = _int(t["gross"])
        row["captured"] += max(gross, 0)
        row["refunded"] += max(-gross, 0)
        row["fees"] += _int(t["fee"])
        row["net"] += _int(t["net"])
    ledger = []
    for i in obs["invoices"]:
        r = rows[i["invoice_id"]]
        ledger.append(dict(invoice_id=i["invoice_id"], captured=r["captured"], refunded=r["refunded"],
                           fees=r["fees"], net=r["net"],
                           remaining_due=_int(i["amount"]) - r["captured"] + r["refunded"]))
    return ledger, suspense


def submission(obs, mapping):
    ledger, suspense = ledger_and_suspense(obs, mapping)
    return dict(op="submit",
                matches=[dict(payment_id=p, invoice_id=i) for p, i in sorted(mapping.items())],
                ledger=ledger, suspense=suspense,
                bank_total=sum(_int(b["amount"]) for b in unique_bank_credits(obs)))


def safe_submission(obs):
    """Last-resort valid submission: nothing matched, everything in suspense."""
    return submission(obs, {})


def decide(observation, state):
    state = dict(state) if isinstance(state, dict) else {}
    queried = list(state.get("queried", []))
    try:
        for inv in observation.get("investigations", []):
            if inv["payment_id"] not in queried:
                queried.append(inv["payment_id"])
        remaining = observation.get("queries_remaining", 0)
        model = infer(observation)
        used = len(observation.get("investigations", []))
        remaining = min(remaining, MAX_QUERIES - used)
        pid = pick_query(model, set(queried), remaining)
        if pid is not None:
            queried.append(pid)
            return {"action": {"op": "query", "payment_id": pid},
                    "state": {"queried": queried, "v": 1}}
        mapping = final_mapping(model)
        return {"action": submission(observation, mapping), "state": {"queried": queried, "v": 1}}
    except Exception as exc:  # fail safe, but with a valid submission shape
        return {"action": safe_submission(observation),
                "state": {"queried": queried, "v": 1, "error": repr(exc)[:200]}}
