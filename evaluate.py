"""Evaluate solution.py against reconciliation JSON batches.

The script accepts either a list of observations or an object containing a
``batches`` list.  Expected truth may be supplied as ``expected_allocations``
or ``truth`` on each batch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_solution(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("submitted_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _batches(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("batches", data.get("examples", [data]))
    return [dict(item) for item in data]


def _truth(batch: dict[str, Any]) -> dict[str, str]:
    raw = batch.get("expected_allocations", batch.get("truth", {}))
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    result: dict[str, str] = {}
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            pid = item.get("payment_id", item.get("payment"))
            iid = item.get("invoice_id", item.get("invoice"))
            if pid is not None and iid is not None:
                result[str(pid)] = str(iid)
    return result


def evaluate(solution_path: Path, dataset_path: Path) -> tuple[float, int, list[str]]:
    solution = _load_solution(solution_path)
    scores: list[float] = []
    queries = 0
    violations: list[str] = []
    for index, batch in enumerate(_batches(dataset_path), 1):
        observation = batch.get("observation", batch)
        result = solution.decide(observation, {})
        action = result.get("action", {})
        state = result.get("state", {})
        queries += int(state.get("queries_used", 0))
        if len(state.get("queried", [])) != len(set(state.get("queried", []))) or queries > 4 * index:
            violations.append(f"batch {index}: query cap or repeat")
        truth = _truth(batch)
        payments = {
            str(item.get("payment_id", item.get("id", item.get("transaction_id")))): item
            for item in observation.get("payments", observation.get("transactions", []))
        }
        value = sum(abs(int(item.get("amount", item.get("gross", 0)) or 0)) for item in payments.values())
        wrong = 0
        unresolved = 0
        assigned: dict[str, str] = {}
        for item in action.get("allocations", []):
            pid, iid = str(item.get("payment_id")), str(item.get("invoice_id"))
            assigned[pid] = iid
            if truth and truth.get(pid) != iid:
                wrong += abs(int(payments.get(pid, {}).get("amount", 0) or 0))
        if truth:
            unresolved = sum(
                abs(int(payments.get(pid, {}).get("amount", 0) or 0))
                for pid in truth
                if pid not in assigned
            )
        score = max(0.0, 100 * (1 - (unresolved + 4 * wrong + 180 * int(state.get("queries_used", 0))) / value)) if value else 100.0
        scores.append(score)
    average = sum(scores) / len(scores) if scores else 0.0
    return average, queries, violations


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python evaluate.py solution.py dataset.json")
        return 2
    average, queries, violations = evaluate(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"average score: {average:.2f}")
    print(f"queries used: {queries}")
    print(f"integrity violations: {len(violations)}")
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
