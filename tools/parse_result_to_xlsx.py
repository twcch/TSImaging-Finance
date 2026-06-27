"""Parse result_long_term_forecast.txt into an xlsx report.

Each experiment in the txt file is two lines:

    long_term_forecast_<data>_<sl>_<ll>_<pl>_<model>_..._sl##_ll##_pl##_..._seed####_0
    mse:..., mae:..., rmse:..., mape:..., mspe:..., r2:..., dtw:..., time:...

The xlsx is written next to the input file with two sheets:
  - "raw":     one row per experiment
               (seed, seql_lens, pred_lens, model, data, MSE, MAE, RMSE, MAPE, R2)
  - "summary": aggregated over seeds, with mean/std per metric
               (data, model, seql_lens, pred_lens, n_seeds, <metric>_mean/_std)

Usage:
    python3 tools/parse_result_to_xlsx.py [path/to/result_long_term_forecast.txt]
"""

import re
import sys
from pathlib import Path

import pandas as pd

# Header looks like:
# long_term_forecast_TSMC-2013-2023_30_14_1_RevTransLSTM-AR_custom_ftMS_sl30_ll14_pl1_..._seed2020_0
HEADER_RE = re.compile(
    r"^long_term_forecast_(?P<data>.+?)"
    r"_\d+_\d+_\d+_"
    r"(?P<model>[^_]+)_"
    r".*?_sl(?P<seql_lens>\d+)_ll\d+_pl(?P<pred_lens>\d+)_"
    r".*?_seed(?P<seed>\d+)_\d+\s*$"
)

METRIC_RE = re.compile(r"(?P<key>[a-z0-9_]+):(?P<val>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

COLUMNS = ["seed", "seql_lens", "pred_lens", "model", "data",
           "MSE", "MAE", "RMSE", "MAPE", "R2",
           "params", "trainable_params", "gpu_mem_peak_mb"]

METRICS = ["MSE", "MAE", "RMSE", "MAPE", "R2"]
GROUP_KEYS = ["data", "model", "seql_lens", "pred_lens"]


def parse(txt_path: Path) -> pd.DataFrame:
    lines = [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]

    rows = []
    i = 0
    while i < len(lines):
        m = HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        if i + 1 >= len(lines):
            print(f"Warning: header without metrics line: {lines[i]}", file=sys.stderr)
            break

        metrics = {mm.group("key"): float(mm.group("val"))
                   for mm in METRIC_RE.finditer(lines[i + 1])}

        rows.append({
            "seed": int(m.group("seed")),
            "seql_lens": int(m.group("seql_lens")),
            "pred_lens": int(m.group("pred_lens")),
            "model": m.group("model"),
            "data": m.group("data"),
            "MSE": metrics.get("mse"),
            "MAE": metrics.get("mae"),
            "RMSE": metrics.get("rmse"),
            "MAPE": metrics.get("mape"),
            "R2": metrics.get("r2"),
            "params": metrics.get("params"),
            "trainable_params": metrics.get("trainable_params"),
            "gpu_mem_peak_mb": metrics.get("gpu_mem_peak_mb"),
        })
        i += 2

    return pd.DataFrame(rows, columns=COLUMNS)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(GROUP_KEYS, sort=False)

    summary = grouped[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = grouped.size()
    summary["params"] = grouped["params"].first()
    summary["trainable_params"] = grouped["trainable_params"].first()
    summary["gpu_mem_peak_mb_mean"] = grouped["gpu_mem_peak_mb"].mean()
    summary["gpu_mem_peak_mb_std"] = grouped["gpu_mem_peak_mb"].std()

    ordered = []
    for metric in METRICS:
        ordered += [f"{metric}_mean", f"{metric}_std"]
    ordered += ["params", "trainable_params",
                "gpu_mem_peak_mb_mean", "gpu_mem_peak_mb_std"]
    summary = summary[["n_seeds"] + ordered].reset_index()

    return summary.sort_values(GROUP_KEYS).reset_index(drop=True)


def main() -> None:
    default = Path(__file__).resolve().parent.parent / "result_long_term_forecast.txt"
    txt_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else default

    if not txt_path.is_file():
        sys.exit(f"Input file not found: {txt_path}")

    df = parse(txt_path)
    if df.empty:
        sys.exit("No experiment records parsed; check the input format.")

    df = df.sort_values(
        ["data", "model", "seql_lens", "pred_lens", "seed"]
    ).reset_index(drop=True)

    summary = summarize(df)

    out_path = txt_path.with_suffix(".xlsx")
    with pd.ExcelWriter(out_path) as writer:
        df.to_excel(writer, sheet_name="raw", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    print(f"Parsed {len(df)} records -> {out_path} "
          f"(raw: {len(df)} rows, summary: {len(summary)} groups)")


if __name__ == "__main__":
    main()
