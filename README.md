# TSImaging-Finance

Official code for the CVGIP 2026 paper:

> *A Preliminary Study on Financial Time-Series Forecasting Based on Image and Temporal-Trajectory Representations*
>
> Chih-Chien Hsieh, Mu-Yen Chen
> Department of Engineering Science, National Cheng Kung University, Tainan, Taiwan

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Built on Time-Series-Library](https://img.shields.io/badge/Built%20on-Time--Series--Library-8A2BE2.svg)](https://github.com/thuml/Time-Series-Library)

---

## Overview

Financial price series are noisy and non-stationary, and a one-dimensional view does not necessarily capture the similarity and local patterns between time points. This study takes daily **Open–High–Low–Close (OHLC)** windows of the **S&P 500 index (2016–2025)** and turns each channel into a **2-D image**, then frames a **single-step regression** of the next trading day's closing price.

It compares two imaging methods — **Gramian Angular Field (GAF)** and **Recurrence Plot (RP)** — against one-dimensional (**LSTM**, **1D-CNN**) and multi-periodic 2-D (**TimesNet**) baselines. To handle non-stationarity and to compare every representation on equal footing, **Reversible Instance Normalization (RevIN)** is applied uniformly to all models: each model learns relative changes in the normalized space and is then mapped back to the (train-standardized) input scale. Imaging additionally strips the instance-level scale inside each window, so RevIN's scale compensation is especially critical for the imaging models.

### Key findings

- A simple **random-walk (persistence) baseline** beats every learned model — single-step daily-close regression is dominated by price persistence. All comparisons here should therefore be read as **relative representation comparisons**, not absolute predictive power.
- Among learned models, **TimesNet** is best; **LSTM** reaches a comparable level with **far fewer parameters**.
- Among imaging representations under an identical network and parameter budget, the angle-based **GAF clearly beats** the state-similarity-based **RP**, and also beats the 1D-CNN.
- **Ablation:** removing RevIN collapses the imaging models from a usable level to a large negative R² — their usability rests on **scale compensation**, not on 2-D imaging itself.

---

## Method at a glance

For each time point $t$, the OHLC vector is $z_t = [O_t, H_t, L_t, C_t]^\top$. A length-$L$ sliding window
$X_t = [z_{t-L+1}, \dots, z_t] \in \mathbb{R}^{L\times 4}$ is mapped to the next-day close $\hat C_{t+1} = f(X_t)$.
All models take the same OHLC window and are evaluated only on the Close channel.

| Representation | How the window becomes input |
|---|---|
| **GAF (GASF)** | Per channel: min–max scale to $[-1,1]$, $\phi_i = \arccos(\tilde x_i)$, then $G_{i,j} = \cos(\phi_i + \phi_j)$. Four channels stacked into an $L\times L\times 4$ image. Computed on-the-fly inside `forward`. |
| **RP** | Per channel: window z-score, distance $D_{i,j} = \lvert \bar x_i - \bar x_j \rvert$, continuous similarity $R_{i,j} = \exp(-\gamma D_{i,j})$ with $\gamma = 1$. Four channels stacked into an $L\times L\times 4$ image. |
| **1D-CNN / LSTM** | The raw OHLC window (sequence) directly. |
| **TimesNet** | Multi-periodic 2-D tensor reshaping of the sequence. |

The imaging models share **one lightweight CNN** (two `Conv3×3–BN–ReLU–MaxPool` blocks → global average pooling → linear head, MSE loss, Adam) so the comparison isolates the *representation*, not model capacity. **RevIN** normalizes each input window per channel before imaging and de-normalizes at the output, re-injecting the instance-level scale that imaging discards.

---

## Models

The active model is selected with `--model`. Models are auto-discovered from `models/` (drop in a `.py` with a `Model` class and it is registered; see `exp/exp_basic.py`).

| Paper name | Input form | Main run (with RevIN) | Ablation (without RevIN) |
|---|---|---|---|
| LSTM | OHLC sequence | `revin-LSTM` | `LSTM` |
| 1D-CNN | OHLC sequence | `revin-1D-CNN` | `1D-CNN` |
| TimesNet | OHLC multi-periodic 2-D | `revin-TimesNet` | `TimesNet` |
| GAF-CNN | OHLC GAF image | `revin-GAF-CNN` | `GAF-CNN` |
| RP-CNN | OHLC RP image | `revin-RP-CNN` | `RP-CNN` |

> In the paper, **all five models are reported with RevIN** (the `revin-*` files). The plain files (no `revin-` prefix) are used for the RevIN ablation in Table 4.

---

## Dataset

`dataset/GSPC-2016-2025.csv` — daily S&P 500 (`^GSPC`) OHLC, **2016-01-04 → 2025-12-31, 2,514 trading days**.

- Header: `date,Open,High,Low,Close` (YYYY-MM-DD; **OHLC only, no Volume**).
- **Split:** chronological 70% / 10% / 20% (train / val / test); no random split, to avoid look-ahead leakage. Validation and test window starts are extended back $L$ steps so each window keeps its required history. Standardization parameters are estimated on the train set only, then RevIN is applied per input window.
- **Source / regeneration:** the raw data come from Yahoo Finance via the `yfinance` package. `tools/fetch_yfinance_data.py` downloads `^GSPC` (start `2016-01-01`, end `2026-01-01`), flattens the multi-index, renames `Date → date`, slices `["date","Open","High","Low","Close"]`, and writes the CSV.

---

## Installation

Recommended Python: **3.11**.

```bash
# Core dependencies for reproducing the paper
pip install torch==2.5.1 numpy pandas scikit-learn matplotlib einops sktime
```

The repository is built on [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library); the full upstream environment (foundation-model and Mamba backends not needed for this paper) is captured in the ordered `requirements/reqs_1..4.txt` files and a CUDA `Dockerfile` / `docker-compose.yml`.

> **Hardware:** runs on NVIDIA CUDA or Apple Silicon. On Apple Silicon add `--gpu_type mps`; otherwise the default is CUDA.

---

## Reproducing the experiments

### One command — the full paper sweep

`run_batch_long_term_forecast.py` runs all five RevIN models on `GSPC-2016-2025` with the paper's settings (`seq_len=96`, `label_len=48`, `pred_len=1`, seed `2020`, Adam `lr=1e-4` cosine schedule, batch 32, 30 epochs, early-stopping patience 5, dropout 0.3):

```bash
python run_batch_long_term_forecast.py
```

It shells out to `run.py`, keys each completed run by an MD5 of its command (resumes on restart via `run_batch_progress.log`), records failures in `run_batch_failed.log`, and writes per-experiment logs to `run_batch_logs/`. Edit the `model_configs` / sweep grid at the top of the script to change models or windows.

### A single model by hand

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id GSPC-2016-2025_96_48_1 \
  --model revin-GAF-CNN \
  --data custom \
  --root_path ./dataset/ \
  --data_path GSPC-2016-2025.csv \
  --features MS \
  --target Close \
  --freq b \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 1 \
  --enc_in 4 --dec_in 4 --c_out 1 \
  --train_epochs 30 --batch_size 32 \
  --learning_rate 0.0001 --lradj cosine \
  --patience 5 --dropout 0.3 \
  --rand_seed 2020 --use_dtw
```

Swap `--model` for `revin-LSTM`, `revin-1D-CNN`, `revin-TimesNet`, or `revin-RP-CNN`. For the **RevIN ablation** (Table 4), drop the `revin-` prefix (`GAF-CNN`, `RP-CNN`). `--features MS` = multivariate OHLC input, single target output; `--target Close`; channel dims are `4` for the four OHLC columns; `--freq b` = business-day calendar.

---

## Outputs & evaluation

**Metrics** (`utils/metrics.py`): MAE, MSE, RMSE, MAPE, MSPE, **R²**, and optional **DTW** (`--use_dtw`). MAPE is computed on the standardized Close scale, not on raw index points.

**Where results land:**

- `result_long_term_forecast.txt` — appended human-readable summary (metrics, params, train/inference timing, GPU peak memory).
- `results/<setting>/` — `pred.npy`, `true.npy`, `metrics.npy`.
- `test_results/<setting>/` — auto-generated publication figures (`utils/visualization.py`).
- `checkpoints/<setting>/` — trained checkpoints.

The text log can be parsed into a spreadsheet with `tools/parse_result_to_xlsx.py` (a `raw` sheet plus a seed-aggregated `summary` sheet); `tools/descriptive_stats.py` produces dataset descriptive statistics.

### Headline results (single-step next-day Close, with RevIN)

| Model | MSE | RMSE | MAE | MAPE | R² | Params |
|---|---|---|---|---|---|---|
| **Naive (persistence)** | **0.0052** | **0.0718** | **0.0486** | **0.0148** | **0.9904** | — |
| LSTM | 0.0136 | 0.1166 | 0.0839 | 0.0258 | 0.9746 | 13,353 |
| 1D-CNN | 0.0962 | 0.3102 | 0.2702 | 0.0820 | 0.8203 | 3,689 |
| TimesNet | 0.0085 | 0.0923 | 0.0656 | 0.0201 | 0.9841 | 2,354,058 |
| GAF-CNN | 0.0414 | 0.2036 | 0.1597 | 0.0501 | 0.9226 | 5,369 |
| RP-CNN | 0.1481 | 0.3849 | 0.3465 | 0.1047 | 0.7234 | 5,369 |

### RevIN ablation (imaging models)

| Model | RevIN | R² |
|---|---|---|
| GAF-CNN | ✗ | −20.90 |
| GAF-CNN | ✓ | 0.9226 |
| RP-CNN | ✗ | −19.52 |
| RP-CNN | ✓ | 0.7234 |

Without RevIN both imaging models fall below the trivial "predict the test-set mean" baseline: GAF/RP images are scale-invariant inside each window, so they cannot recover the absolute next-day close on their own — only RevIN re-injects the window's level and amplitude at the output.

---

## Repository structure

```text
TSImaging-Finance/
├── run.py                            # Main entry — CLI, seeding, task dispatch
├── run_batch_long_term_forecast.py   # Batch runner for the paper sweep (5 RevIN models)
├── models/
│   ├── LSTM.py / 1D-CNN.py / TimesNet.py / GAF-CNN.py / RP-CNN.py   # baselines (no RevIN)
│   └── revin-*.py                    # RevIN-wrapped variants = paper's main results
├── layers/RevIN.py                   # Reversible Instance Normalization
├── exp/                              # Task pipelines (exp_long_term_forecasting.py, exp_basic.py registry)
├── data_provider/                    # data_factory.py, data_loader.py (Dataset_Custom)
├── dataset/GSPC-2016-2025.csv        # S&P 500 daily OHLC (2016–2025)
├── utils/                            # metrics.py, visualization.py, tools.py
├── tools/                            # fetch_yfinance_data.py, descriptive_stats.py, parse_result_to_xlsx.py
├── requirements/                     # full upstream environment (reqs_1..4.txt)
├── Dockerfile / docker-compose.yml   # CUDA 12.1 / PyTorch 2.5.1
└── LICENSE
```

---

## Acknowledgements

This codebase is built on [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) by THUML, reusing its experiment framework, the TimesNet implementation, and reproduction scaffolding. The RevIN layer follows Kim et al. (ICLR 2022). Market data are retrieved from Yahoo Finance via the open-source `yfinance` package; due to third-party usage terms the raw download is not redistributed beyond the bundled CSV.

## License

Released under the **MIT License** — see [`LICENSE`](./LICENSE).

- Copyright © 2026 Chih-Chien Hsieh
- Copyright © 2021 THUML @ Tsinghua University (Time-Series-Library)
