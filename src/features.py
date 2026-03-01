# features.py
# Build ML-ready features from a solar log CSV.
#
# Expected columns (minimum):
#   - V, I, duty
# Optional but supported:
#   - t_sec (seconds since start) OR timestamp (monotonic/epoch); used to infer dt
#   - cloud_coverage, mean_gray, std_gray (if you merge vision features later)
#
# Output:
#   - DataFrame with engineered features (P, derivatives, rolling stats, etc.)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    # If you have t_sec or timestamp, dt is inferred. Otherwise fallback_dt_sec is used.
    fallback_dt_sec: float = 0.02

    # Rolling windows in seconds (you can add more)
    windows_sec: Tuple[float, ...] = (0.2, 1.0, 3.0)

    # Columns expected for electrical features
    v_col: str = "V"
    i_col: str = "I"
    duty_col: str = "duty"

    # Time columns (either can exist)
    t_col: str = "t_sec"
    ts_col: str = "timestamp"  # optional alternative

    # If True, drop rows that don't have full rolling windows
    drop_na: bool = True


def _infer_dt(df: pd.DataFrame, cfg: FeatureConfig) -> float:
    # Prefer t_sec if present, else timestamp; else fallback
    if cfg.t_col in df.columns:
        t = pd.to_numeric(df[cfg.t_col], errors="coerce")
        dt = t.diff().median()
        if pd.notna(dt) and dt > 0:
            return float(dt)
    if cfg.ts_col in df.columns:
        t = pd.to_numeric(df[cfg.ts_col], errors="coerce")
        dt = t.diff().median()
        if pd.notna(dt) and dt > 0:
            return float(dt)
    return float(cfg.fallback_dt_sec)


def _rolling(df: pd.DataFrame, col: str, w: int, prefix: str) -> pd.DataFrame:
    r = df[col].rolling(w, min_periods=w)
    df[f"{prefix}_mean"] = r.mean()
    df[f"{prefix}_std"] = r.std()
    df[f"{prefix}_min"] = r.min()
    df[f"{prefix}_max"] = r.max()
    # Robust spread (less sensitive than std)
    df[f"{prefix}_iqr"] = r.quantile(0.75) - r.quantile(0.25)
    return df


def build_features(df_raw: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    df = df_raw.copy()

    # Basic column checks
    for c in (cfg.v_col, cfg.i_col, cfg.duty_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column: '{c}'")

    # Ensure numeric
    df[cfg.v_col] = pd.to_numeric(df[cfg.v_col], errors="coerce")
    df[cfg.i_col] = pd.to_numeric(df[cfg.i_col], errors="coerce")
    df[cfg.duty_col] = pd.to_numeric(df[cfg.duty_col], errors="coerce")

    # Drop rows with NaNs in core signals
    df = df.dropna(subset=[cfg.v_col, cfg.i_col, cfg.duty_col]).reset_index(drop=True)

    # Time step
    dt = _infer_dt(df, cfg)

    V = df[cfg.v_col].astype(np.float32)
    I = df[cfg.i_col].astype(np.float32)
    duty = df[cfg.duty_col].astype(np.float32)

    # Power
    df["P"] = V * I

    # First differences (per-sample deltas)
    df["dV"] = V.diff().fillna(0.0)
    df["dI"] = I.diff().fillna(0.0)
    df["dP"] = df["P"].diff().fillna(0.0)
    df["dD"] = duty.diff().fillna(0.0)

    # Rates (per second)
    df["dV_dt"] = df["dV"] / dt
    df["dI_dt"] = df["dI"] / dt
    df["dP_dt"] = df["dP"] / dt
    df["dD_dt"] = df["dD"] / dt

    # Nonlinear / interaction features (cheap but useful)
    df["VI"] = V * I
    df["V2"] = V * V
    df["I2"] = I * I
    df["logP"] = np.log(np.clip(df["P"].astype(np.float32), 1e-6, None))
    df["eff_proxy"] = df["P"] / np.clip(duty, 1e-3, None)  # proxy only, not true efficiency

    # Rolling features at multiple time scales
    for ws in cfg.windows_sec:
        w = max(2, int(round(ws / dt)))
        df = _rolling(df, "P", w, f"P_{ws:.1f}s")
        df = _rolling(df, cfg.i_col, w, f"I_{ws:.1f}s")
        df = _rolling(df, cfg.v_col, w, f"V_{ws:.1f}s")
        df = _rolling(df, "dP_dt", w, f"dPdt_{ws:.1f}s")

    # Optional: if you already merged vision metrics, add rolling features for them too
    for opt in ("cloud_coverage", "mean_gray", "std_gray"):
        if opt in df.columns:
            df[opt] = pd.to_numeric(df[opt], errors="coerce")
            for ws in cfg.windows_sec:
                w = max(2, int(round(ws / dt)))
                df = _rolling(df, opt, w, f"{opt}_{ws:.1f}s")

    if cfg.drop_na:
        df = df.dropna().reset_index(drop=True)

    # Track inferred dt (helpful for debugging)
    df.attrs["dt_sec_inferred"] = dt
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    # Convenience: return all numeric columns except labels if present
    exclude = {"V_mpp_label", "label", "y"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


if __name__ == "__main__":
    # Quick CLI: python features.py --in solar_log_labeled.csv --out dataset_features.csv
    import argparse
    import os

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="input CSV path")
    p.add_argument("--out", dest="out", required=True, help="output CSV path")
    args = p.parse_args()

    if not os.path.isfile(args.inp):
        raise FileNotFoundError(args.inp)

    raw = pd.read_csv(args.inp)
    feats = build_features(raw)
    print("Rows in:", len(raw), "Rows out:", len(feats), "dt:", feats.attrs.get("dt_sec_inferred"))
    feats.to_csv(args.out, index=False)
    print("Wrote:", args.out)