"""Run the public v6 lifecycle and print per-batch scores."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_solution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _expected(obs, mapping):
    fields = ("captured", "refunded", "fees", "net")
    balances = {i["invoice_id"]: dict.fromkeys(fields, 0) for i in obs["invoices"]}
    suspense = dict.fromkeys(fields, 0)
    unique = {t["transaction_id"]: t for t in obs["transactions"]}.values()
    for transaction in unique:
        row = balances[mapping[transaction["payment_id"]]] if transaction["payment_id"] in mapping else suspense
        row["captured"] += max(0, transaction["gross"])
        row["refunded"] += max(0, -transaction["gross"])
        row["fees"] += transaction["fee"]
        row["net"] += transaction["net"]
    ledger = [
        {**{"invoice_id": i["invoice_id"]}, **balances[i["invoice_id"]],
         "remaining_due": i["amount"] - balances[i["invoice_id"]]["captured"] + balances[i["invoice_id"]]["refunded"]}
        for i in obs["invoices"]
    ]
    return {
        "op": "submit",
        "matches": [{"payment_id": p, "invoice_id": i} for p, i in mapping.items()],
        "ledger": ledger,
        "suspense": suspense,
        "bank_total": sum(b["amount"] for b in {b["bank_id"]: b for b in obs["bank_credits"]}.values()),
    }


def evaluate(solution_path: Path, dataset_path: Path) -> tuple[float, list[dict]]:
    module = _load(solution_path)
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    reports = []
    for index, item in enumerate(rows):
        obs = item["observation"]
        truth = item["matches"]
        state = {}
        queries = []
        violation = ""
        for _ in range(5):
            response = module.decide(obs, state)
            action, state = response["action"], response["state"]
            if action["op"] == "query":
                pid = action["payment_id"]
                if pid in queries or len(queries) >= 4:
                    violation = "query cap/repeat"
                    break
                queries.append(pid)
                obs["queries_remaining"] -= 1
                obs["investigations"].append({"payment_id": pid, "verified_invoice_id": truth[pid]})
                continue
            mapping = {m["payment_id"]: m["invoice_id"] for m in action["matches"]}
            expected = _expected(obs, mapping)
            if action != expected:
                violation = "posting differs from unique source totals"
            break
        else:
            violation = "no final submission"
        values = {p: sum(max(0, t["gross"]) for t in {x["transaction_id"]: x for x in obs["transactions"]}.values() if t["payment_id"] == p) for p in truth}
        eligible = sum(values[p] for p, i in truth.items() if i is not None)
        correct = sum(values[p] for p, i in truth.items() if i is not None and dict((m["payment_id"], m["invoice_id"]) for m in action.get("matches", [])).get(p) == i)
        wrong = 4 * sum(values[p] for p, i in dict((m["payment_id"], m["invoice_id"]) for m in action.get("matches", [])).items() if truth.get(p) != i)
        unresolved = eligible - correct
        score = max(0.0, 100 * (1 - (unresolved + wrong + 180 * len(queries)) / eligible)) if eligible else 100.0
        reports.append({"batch": index, "score": round(score, 4), "queries": len(queries), "violation": violation})
        print(json.dumps(reports[-1]))
    return (sum(r["score"] for r in reports) / len(reports) if reports else 0.0), reports


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python evaluate.py solution.py dataset.json")
    average, reports = evaluate(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps({"average": round(average, 4), "queries": sum(r["queries"] for r in reports), "violations": sum(bool(r["violation"]) for r in reports)}))
