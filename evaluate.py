"""Check the AI's priorities against the synthetic fraud labels.

The labels live in the rule engine's transactions.csv. They are joined here,
locally and after generation -- nothing about is_fraud is ever sent to the LLM.

This measures agreement with a hand-injected label, not correctness: the
labelled data is synthetic and the model only sees four fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_RANKED = Path("output/alerts_ranked.csv")
DEFAULT_TRANSACTIONS = Path(
    r"C:\Users\adixi\Downloads\Transaction Monitoring Rule Engine with UAT Framework"
    r"\data\transactions.csv"
)
ORDER = ["High", "Medium", "Low", "Unrated"]


def evaluate(ranked: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    labels = transactions[["txn_id", "is_fraud", "fraud_type"]].drop_duplicates("txn_id")
    merged = ranked.merge(labels, on="txn_id", how="left")
    unmatched = int(merged["is_fraud"].isna().sum())
    merged = merged[merged["is_fraud"].notna()]

    by_priority = {}
    for priority in ORDER:
        band = merged[merged["priority"] == priority]
        if band.empty:
            continue
        fraud = int(band["is_fraud"].sum())
        by_priority[priority] = {
            "alerts": int(len(band)),
            "fraud_labelled": fraud,
            "fraud_rate": round(100 * fraud / len(band), 1),
        }

    total_fraud = int(merged["is_fraud"].sum())
    return {
        "alerts_evaluated": int(len(merged)),
        "unmatched_alerts": unmatched,
        "fraud_labelled_total": total_fraud,
        "fraud_rate_overall": round(100 * total_fraud / len(merged), 1) if len(merged) else 0.0,
        "by_priority": by_priority,
        "top_fraud_types": (
            merged[merged["priority"] == "High"]["fraud_type"]
            .value_counts().head(5).to_dict()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ranked", default=DEFAULT_RANKED, type=Path)
    parser.add_argument("--transactions", default=DEFAULT_TRANSACTIONS, type=Path)
    parser.add_argument("--out", default=Path("output/priority_vs_labels.json"), type=Path)
    args = parser.parse_args(argv)

    if not args.ranked.exists():
        raise SystemExit(f"error: {args.ranked} not found; run summarise_alerts.py first")
    if not args.transactions.exists():
        raise SystemExit(f"error: {args.transactions} not found")

    result = evaluate(pd.read_csv(args.ranked), pd.read_csv(args.transactions))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"alerts evaluated : {result['alerts_evaluated']}")
    print(f"overall fraud rate on these alerts: {result['fraud_rate_overall']}%")
    print()
    print(f"{'AI priority':<12}{'alerts':>8}{'fraud':>8}{'fraud rate':>12}")
    for priority, row in result["by_priority"].items():
        print(f"{priority:<12}{row['alerts']:>8}{row['fraud_labelled']:>8}"
              f"{str(row['fraud_rate']) + '%':>12}")
    print()
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
