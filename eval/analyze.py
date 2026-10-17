"""
读取 eval/runs/*.json，输出汇总表。

用法:
    python eval/analyze.py
    python eval/analyze.py --csv eval/results_summary.csv
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "eval" / "runs"
MODES = ["ortho_single", "dual", "routed"]


def load_records(mode):
    path = RUNS_DIR / f"{mode}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_stats(records):
    n_med_req = sum(1 for r in records if r.get("medical_required") == 1)
    n_leg_req = sum(1 for r in records if r.get("legal_required") == 1)

    med_cov = sum(
        1 for r in records
        if r.get("medical_required") == 1
        and any(x["domain"] == "medical" for x in r.get("retrieval", []))
    )
    leg_cov = sum(
        1 for r in records
        if r.get("legal_required") == 1
        and any(x["domain"] == "legal" for x in r.get("retrieval", []))
    )
    unnec_med = sum(
        sum(1 for x in r.get("retrieval", []) if x["domain"] == "medical")
        for r in records if r.get("medical_required") == 0
    )
    unnec_leg = sum(
        sum(1 for x in r.get("retrieval", []) if x["domain"] == "legal")
        for r in records if r.get("legal_required") == 0
    )

    cross = [r for r in records if r.get("question_type") == "cross"]
    if cross:
        cross_med = sum(
            sum(1 for x in r["retrieval"] if x["domain"] == "medical")
            for r in cross
        ) / len(cross)
        cross_leg = sum(
            sum(1 for x in r["retrieval"] if x["domain"] == "legal")
            for r in cross
        ) / len(cross)
    else:
        cross_med = cross_leg = 0.0

    routing_acc = "n/a"
    if any("routing_prediction" in r for r in records):
        exact = 0
        n = 0
        for r in records:
            pred = r.get("routing_prediction")
            if not pred:
                continue
            n += 1
            if (pred["medical_required"] == bool(r["medical_required"])
                    and pred["legal_required"] == bool(r["legal_required"])):
                exact += 1
        routing_acc = f"{exact}/{n}" if n else "n/a"

    return {
        "Required Medical Coverage": f"{med_cov}/{n_med_req}",
        "Required Legal Coverage": f"{leg_cov}/{n_leg_req}",
        "Unnecessary Medical Retrieval": unnec_med,
        "Unnecessary Legal Retrieval": unnec_leg,
        "Cross-domain Medical Avg": f"{cross_med:.2f}",
        "Cross-domain Legal Avg": f"{cross_leg:.2f}",
        "Routing Accuracy": routing_acc,
    }


def to_markdown(rows, metrics):
    headers = ["Metric"] + [r["mode"] for r in rows]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for m in metrics:
        line = [m]
        for r in rows:
            line.append(str(r["stats"].get(m, "n/a")))
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "eval" / "results_summary.csv"))
    args = parser.parse_args()

    rows = []
    for mode in MODES:
        records = load_records(mode)
        if records is None:
            print(f"[skip] {mode}.json not found")
            continue
        stats = compute_stats(records)
        rows.append({"mode": mode, "stats": stats, "n": len(records)})

    if not rows:
        print("no eval/runs/*.json found")
        sys.exit(1)

    metrics = [
        "Required Medical Coverage",
        "Required Legal Coverage",
        "Unnecessary Medical Retrieval",
        "Unnecessary Legal Retrieval",
        "Cross-domain Medical Avg",
        "Cross-domain Legal Avg",
        "Routing Accuracy",
    ]

    print("=" * 80)
    print("Evaluation summary")
    print("=" * 80)
    for r in rows:
        print(f"  mode={r['mode']}  n={r['n']}")
    print()
    print(to_markdown(rows, metrics))

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Mode"] + metrics
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            line = [r["mode"]] + [str(r["stats"].get(m, "n/a")) for m in metrics]
            f.write(",".join(line) + "\n")
    print(f"\nwrote: {csv_path}")


if __name__ == "__main__":
    main()