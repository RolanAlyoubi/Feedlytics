"""Run the end-to-end analysis and write every table to outputs/ (git-ignored).

Outputs contain aggregate tables only (no review text), plus a summary that
states the labelling method, sentiment sources, sampling and weighting.

Usage:
  python -m scripts.run_analysis data/raw/reviews.csv                 # non-AI baseline
  python -m scripts.run_analysis data/sample_reviews_SYNTHETIC.csv
  python -m scripts.run_analysis data/raw/reviews.csv --labelling ai  # needs API access
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from app.data_processing import DataValidationError
from app.pipeline import LABELLING_MODES, PipelineError, run_analysis


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Feedlytics analysis pipeline.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--labelling", choices=LABELLING_MODES, default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    try:
        result = run_analysis(args.csv, profile=args.profile, labelling=args.labelling, seed=args.seed)
    except (DataValidationError, PipelineError) as exc:
        print(f"Cannot run the analysis: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, PipelineError) else 1

    out = args.out_dir / result.profile.name
    (out / "tables").mkdir(parents=True, exist_ok=True)
    index = []
    for name, table in result.tables.items():
        index.append({"table": name, "available": table.available, "basis": table.basis,
                      "weighted": table.weighted, "sentiment_col": table.sentiment_col,
                      "n_reviews": table.n_reviews, "reason": table.reason, "note": table.note})
        if table.available:
            table.df.to_csv(out / "tables" / f"{name.replace(':', '__')}.csv", index=False)
    method = result.method
    summary = {
        "profile": result.profile.name,
        "method": {"kind": method.kind, "is_ai": method.is_ai, "model": method.model,
                   "calibrated_baseline": method.calibrated_baseline, "fallback_rows": method.fallback_rows},
        "sampling": result.sampling,
        "kpis": result.kpis,
        "tables": index,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    label = "AI (Claude)" if method.is_ai else ("none" if method.kind == "none" else "NON-AI baseline")
    print(f"Profile {result.profile.name}; labels: {label}; sentiment sources: {result.kpis['sentiment_sources']}")
    print(f"Wrote {sum(t.available for t in result.tables.values())} tables to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
