#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 2 敘述統計工具 — 單表版 (Descriptive Statistics, single table)
================================================================================

只產生「日收盤價 (Close, level) 敘述統計表」這一張表，對應期刊投稿的 Table 2。
原多表工具中的 Log Return 表、績效摘要、相關性矩陣、Ljung-Box 等內容皆已移除。

表格欄位
--------
Series, N, Mean, Median, Std, Min, Max, Skewness, Ex.Kurt,
JB, ADF, KPSS, ARCH(10), Hurst
依 PANELS 設定分組 (Panel A / Panel B) 呈現。

相較前一版的修正 (correctness fixes)
----------------------------------
1. KPSS 顯著性星號：改用「統計量 vs 漸近臨界值」判定，不再使用 statsmodels 回傳的
   p-value。statsmodels 的 KPSS p-value 被截斷在 [0.01, 0.10]；當統計量很大時
   p-value 固定回傳 0.01，配合嚴格不等式 `p < 0.01` 會把本該 *** 的結果誤判為 **。
   ADF 一併改用臨界值判定，與 KPSS 一致。
2. Hurst：原註解標示「R/S analysis」並不正確 — 程式實作其實是 structure-function
   (variance-scaling) 法。已更正函式說明與 LaTeX 註腳。此估計法套用於價格 level，
   隨機漫步式價格本就會得到 H≈0.5，與本資料 0.4x 的結果一致，也與 ADF/KPSS 判定的
   「非穩態、近隨機漫步」相互呼應，並無矛盾。
3. LaTeX 註腳：補上各檢定的 H0、設定 (ADF/KPSS 含截距、lag 選擇)、樣本期間
   (由資料自動帶入)，並移除已刪除指標 (Q / Q^2) 的說明。

使用範例
--------
    python tools/descriptive_stats.py
    python tools/descriptive_stats.py -i ./dataset/2016_2025 -o ./results/desc_2016_2025
"""

from __future__ import annotations

import argparse
import glob
import os
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl-cache"))

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 0. 設定區：要換資料集 / 調整分組，改這裡就好
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "dataset" / "2016_2025"
OUTPUT_DIR = PROJECT_ROOT / "results" / "descriptive_stats_2016_2025"
FILE_PATTERN = "*.csv"

# 已確認：敘述統計以 yfinance 的「Close」欄計算。Close 優先，缺漏才退 Adj Close。
PRICE_COL = "Close"
FALLBACK_COL = "Adj Close"

ARCH_LAGS = 10
FLOAT_FMT = "%.3f"

TABLE_CAPTION = "Descriptive statistics of the daily closing price series."
TABLE_LABEL = "tab:desc_close"

# Panel 分組：key = panel 標題，value = 該 panel 的序列名稱。
# 序列名稱須與「顯示名」一致 (預設為 CSV 檔名 stem；若有 SERIES_RENAME 則為改名後的值)。
PANELS: dict[str, list[str]] = {
    "Panel A: Main dataset (Taiwan/U.S. semiconductors and U.S. benchmarks)":
        ["TSMC", "AAPL", "SOX", "GSPC", "NDX"],
    "Panel B: Cross-market validation (other industries and regions)":
        ["JPM", "N225", "FTSE"],
}

# 若 CSV 檔名與表中要顯示的名稱不同，在這裡 remap (檔名 stem -> 顯示名)；不需要就留空。
SERIES_RENAME: dict[str, str] = {}

ARCH_COL = f"ARCH({ARCH_LAGS})"
DISPLAY_COLUMNS = [
    "Series", "N", "Mean", "Median", "Std", "Min", "Max",
    "Skewness", "Ex.Kurt", "JB", "ADF", "KPSS", ARCH_COL, "Hurst",
]
# 統計量欄位 -> 對應星號欄位
STAR_MAP = {"JB": "JB_s", "ADF": "ADF_s", "KPSS": "KPSS_s", ARCH_COL: "ARCH_s"}


# ---------------------------------------------------------------------------
# 1. 載入資料
# ---------------------------------------------------------------------------
def load_files(input_dir: str, pattern: str = "*.csv") -> dict[str, pd.DataFrame]:
    """讀取資料夾下所有符合 pattern 的 yfinance 檔案 (自動偵測日期欄)。"""
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 找不到符合 {pattern} 的檔案")

    data: dict[str, pd.DataFrame] = {}
    for f in files:
        name = Path(f).stem
        df = pd.read_csv(f)
        date_col = next(
            (c for c in df.columns if c.lower() in ("date", "datetime", "time", "timestamp")),
            None,
        )
        if date_col is not None:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        data[name] = df
        print(f"  ✓ {name}: {len(df)} 筆")
    return data


# ---------------------------------------------------------------------------
# 2. 統計核心
# ---------------------------------------------------------------------------
def hurst_exponent(series, max_lag: int = 100) -> float:
    """Hurst 指數 — structure-function (variance-scaling) 估計法。

    對序列 s，計算各 lag τ 的增量 (s[t+τ] - s[t]) 之標準差；對碎形布朗運動
    std ∝ τ^H，故以 log-log 迴歸斜率估計 H。
        H < 0.5 : 反持續 / 均值回歸
        H ≈ 0.5 : 隨機漫步
        H > 0.5 : 持續 / 趨勢

    註：此法 *非* 經典 R/S analysis。套用於價格 level 時，近隨機漫步的股價會得到
    H≈0.5，屬正常結果，且與 ADF/KPSS 判定的非穩態並不衝突。
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 50:
        return np.nan
    lags = range(2, min(max_lag, len(s) // 2))
    tau, valid = [], []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        std_diff = np.std(diff)
        if std_diff > 0:
            tau.append(np.sqrt(std_diff))
            valid.append(lag)
    if len(tau) < 5:
        return np.nan
    poly = np.polyfit(np.log(valid), np.log(tau), 1)
    return float(poly[0] * 2.0)


def _safe(fn, *args, **kwargs):
    """包裹統計函數，失敗時回傳 None，避免單筆失敗中斷整批。"""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return None


def _pval_stars(p) -> str:
    """以 p-value 判定星號 (用於 JB、ARCH，其 p-value 為真實值、不被截斷)。"""
    if p is None or pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _adf_stars(stat: float, crit: dict) -> str:
    """ADF：以統計量 vs 臨界值判定 (越負越顯著，拒絕 H0=單根)。"""
    if stat < crit["1%"]:
        return "***"
    if stat < crit["5%"]:
        return "**"
    if stat < crit["10%"]:
        return "*"
    return ""


def _kpss_stars(stat: float, crit: dict) -> str:
    """KPSS：以統計量 vs 臨界值判定 (越大越顯著，拒絕 H0=平穩)。

    刻意不用 statsmodels 回傳的 p-value：該 p-value 被截斷在 [0.01, 0.10]，
    統計量很大時固定回傳 0.01，會導致顯著性被低估 (*** 誤判為 **)。
    """
    if stat > crit["1%"]:
        return "***"
    if stat > crit["5%"]:
        return "**"
    if stat > crit["10%"]:
        return "*"
    return ""


def compute_row(series: pd.Series, label: str) -> dict | None:
    """產生單一序列的敘述統計列 (對應 Table 2 的一列)。"""
    s = pd.Series(series).astype(float).dropna()
    if len(s) < 30:
        print(f"  ! {label}: 樣本數 {len(s)} < 30，跳過")
        return None

    row: dict = {
        "Series": label,
        "N": int(len(s)),
        "Mean": s.mean(),
        "Median": s.median(),
        "Std": s.std(ddof=1),
        "Min": s.min(),
        "Max": s.max(),
        "Skewness": float(sps.skew(s)),
        "Ex.Kurt": float(sps.kurtosis(s)),  # Fisher = excess kurtosis
    }

    # Jarque-Bera 常態性 (H0: 常態)
    jb = _safe(sps.jarque_bera, s)
    row["JB"] = float(jb[0]) if jb is not None else np.nan
    row["JB_s"] = _pval_stars(jb[1]) if jb is not None else ""

    # ADF (H0: 有單根 → 拒絕 ⇒ 平穩)；含截距、lag 由 AIC 選
    adf = _safe(adfuller, s, regression="c", autolag="AIC")
    if adf is not None:
        row["ADF"] = float(adf[0])
        row["ADF_s"] = _adf_stars(adf[0], adf[4])  # adf[4] = 臨界值 dict
        row["ADF_lag"] = int(adf[2])
    else:
        row["ADF"], row["ADF_s"], row["ADF_lag"] = np.nan, "", None

    # KPSS (H0: 平穩 → 拒絕 ⇒ 不平穩)；level stationarity、自動 bandwidth
    kp = _safe(kpss, s, regression="c", nlags="auto")
    if kp is not None:
        row["KPSS"] = float(kp[0])
        row["KPSS_s"] = _kpss_stars(kp[0], kp[3])  # kp[3] = 臨界值 dict
    else:
        row["KPSS"], row["KPSS_s"] = np.nan, ""

    # ARCH-LM (H0: 無 ARCH 效應)
    arch = _safe(het_arch, s, nlags=ARCH_LAGS)
    row[ARCH_COL] = float(arch[0]) if arch is not None else np.nan
    row["ARCH_s"] = _pval_stars(arch[1]) if arch is not None else ""

    # Hurst (structure-function 法)
    row["Hurst"] = hurst_exponent(s)

    # 內部用：樣本起訖 (供 Note 自動帶入樣本期間，不進表格欄位)
    row["_start"] = s.index[0] if len(s.index) else None
    row["_end"] = s.index[-1] if len(s.index) else None
    return row


def build_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """對每個資產建立一列敘述統計。"""
    rows = []
    for name, df in data.items():
        col = (
            PRICE_COL if PRICE_COL in df.columns
            else (FALLBACK_COL if FALLBACK_COL in df.columns else None)
        )
        if col is None:
            print(f"  ! {name}: 找不到 {PRICE_COL!r} 或 {FALLBACK_COL!r}，跳過")
            continue
        if col != PRICE_COL:
            print(f"  · {name}: 主欄位缺失，改用備援欄位 {col!r}")
        label = SERIES_RENAME.get(name, name)
        row = compute_row(df[col].dropna(), label)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def resolve_panels(df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """依 PANELS 設定排序、分組；缺漏與未指派的序列都會明確提示。"""
    present = list(df["Series"])
    ordered: list[tuple[str, list[str]]] = []
    assigned: set[str] = set()
    for title, series_list in PANELS.items():
        found = [s for s in series_list if s in present]
        for missing in [s for s in series_list if s not in present]:
            print(f"  ! Panel 設定中的 {missing!r} 在資料裡找不到")
        if found:
            ordered.append((title, found))
            assigned.update(found)
    leftover = [s for s in present if s not in assigned]
    if leftover:
        print(f"  ! 未指派到任何 Panel 的序列 {leftover} → 暫歸入 'Other'")
        ordered.append(("Other (unassigned)", leftover))
    return ordered


# ---------------------------------------------------------------------------
# 3. 格式化與輸出
# ---------------------------------------------------------------------------
def format_table(df: pd.DataFrame, float_fmt: str = FLOAT_FMT) -> pd.DataFrame:
    """把數值轉成顯示字串，並將顯著性星號併入對應統計量。"""
    out = pd.DataFrame()
    out["Series"] = df["Series"]
    out["N"] = df["N"].map(lambda x: "%d" % x if pd.notna(x) else "--")
    for col in ["Mean", "Median", "Std", "Min", "Max", "Skewness", "Ex.Kurt", "Hurst"]:
        out[col] = df[col].map(lambda x: float_fmt % x if pd.notna(x) else "--")
    for col, star_col in STAR_MAP.items():
        base = df[col].map(lambda x: float_fmt % x if pd.notna(x) else "--")
        stars = df[star_col].fillna("") if star_col in df.columns else ""
        out[col] = base + stars
    return out[DISPLAY_COLUMNS]


def _fmt_date(d) -> str | None:
    try:
        return pd.Timestamp(d).date().isoformat()
    except Exception:  # noqa: BLE001
        return None


def build_note(df: pd.DataFrame) -> str:
    """組出 LaTeX 註腳，樣本期間由資料自動帶入。"""
    start = _fmt_date(df["_start"].min()) if "_start" in df.columns else None
    end = _fmt_date(df["_end"].max()) if "_end" in df.columns else None
    period = f"{start} to {end}" if (start and end) else "[請填入樣本期間]"
    note = (
        r"Note: $N$ is the number of trading-day observations; Ex.Kurt denotes "
        r"excess kurtosis (Fisher). Sample period: __PERIOD__. "
        r"$^{*}$, $^{**}$, and $^{***}$ denote significance at the 10\%, 5\%, "
        r"and 1\% levels, respectively. "
        r"ADF: Augmented Dickey--Fuller test ($H_0$: unit root; rejection "
        r"$\Rightarrow$ stationary), estimated with an intercept and lag length "
        r"selected by AIC. "
        r"KPSS: Kwiatkowski--Phillips--Schmidt--Shin test ($H_0$: level "
        r"stationarity; rejection $\Rightarrow$ non-stationary), with automatic "
        r"bandwidth selection; significance is assessed by comparing the "
        r"statistic with asymptotic critical values. "
        r"JB: Jarque--Bera test ($H_0$: normality). "
        r"ARCH(10): Engle's ARCH-LM test for conditional heteroskedasticity up "
        r"to lag 10 ($H_0$: no ARCH effects). "
        r"Hurst: Hurst exponent estimated by the structure-function "
        r"(variance-scaling) method ($H<0.5$ anti-persistent, $H=0.5$ random "
        r"walk, $H>0.5$ persistent). "
        r"All statistics are computed on the daily closing price."
    )
    return note.replace("__PERIOD__", period)


def to_latex(fmt_df: pd.DataFrame, panels: list[tuple[str, list[str]]], note: str) -> str:
    """產生 booktabs 風格、含 Panel 分組的 LaTeX 表格。"""
    ncol = len(DISPLAY_COLUMNS)
    colspec = "l" + "r" * (ncol - 1)
    L = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{%s}" % TABLE_CAPTION,
        r"\label{%s}" % TABLE_LABEL,
        r"% 欄位較多，必要時改用 sidewaystable (需 \usepackage{rotating})"
        r" 或以 \resizebox 縮放。",
        r"\begin{tabular}{%s}" % colspec,
        r"\toprule",
        " & ".join(DISPLAY_COLUMNS) + r" \\",
        r"\midrule",
    ]
    for i, (title, series_list) in enumerate(panels):
        if i > 0:
            L.append(r"\addlinespace")
        L.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol, title))
        for name in series_list:
            sub = fmt_df[fmt_df["Series"] == name]
            if sub.empty:
                continue
            r = sub.iloc[0]
            L.append(" & ".join(str(r[c]) for c in DISPLAY_COLUMNS) + r" \\")
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}\footnotesize",
        note,
        r"\end{flushleft}",
        r"\end{table}",
    ]
    return "\n".join(L)


def save_outputs(df: pd.DataFrame, panels: list[tuple[str, list[str]]], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 依 Panel 順序重排
    order = [s for _, lst in panels for s in lst]
    df = df.set_index("Series").loc[order].reset_index()

    # 原始數值 CSV (全精度，含星號欄與樣本起訖；供核對用)
    raw = df.copy()
    raw["Start"] = raw["_start"].map(_fmt_date)
    raw["End"] = raw["_end"].map(_fmt_date)
    raw = raw.drop(columns=[c for c in ("_start", "_end") if c in raw.columns])
    raw.to_csv(out / "table2_descriptive_close.csv", index=False, encoding="utf-8-sig")

    # LaTeX (投稿用)
    fmt_df = format_table(df)
    note = build_note(df)
    (out / "table2_descriptive_close.tex").write_text(to_latex(fmt_df, panels, note),
                                                      encoding="utf-8")

    print(f"\n✓ 輸出完成 → {out.resolve()}")
    for f in sorted(out.iterdir()):
        print(f"   - {f.name}")
    print("\n--- 表格預覽 ---")
    print(fmt_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 4. 主程式
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table 2 敘述統計工具 — 單表版 (yfinance 收盤價)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, default=INPUT_DIR, help="輸入資料夾路徑")
    parser.add_argument("--output", "-o", type=Path, default=OUTPUT_DIR, help="輸出資料夾路徑")
    parser.add_argument("--pattern", default=FILE_PATTERN, help="檔案匹配模式")
    args = parser.parse_args()

    print(f"\n[1/3] 讀取資料: {args.input}")
    data = load_files(args.input, pattern=args.pattern)

    print(f"\n[2/3] 計算統計量 ({len(data)} 個資產)")
    df = build_table(data)
    if df.empty:
        raise SystemExit("沒有任何序列成功計算，請檢查資料與欄位設定。")
    panels = resolve_panels(df)

    print(f"\n[3/3] 輸出至: {args.output}")
    save_outputs(df, panels, args.output)


if __name__ == "__main__":
    main()