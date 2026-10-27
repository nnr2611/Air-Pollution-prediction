"""Train and evaluate the PM2.5 forecaster.

    python train.py --horizon 24
    python train.py --horizon 168      # 7 days ahead

Writes model, metrics and test-set predictions to ``artifacts/`` (the dashboard reads these).
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from feature_engineering import build_dataset, feature_columns  # noqa: E402
from model_training import (  # noqa: E402
    apply_window_scaler, blend_weight, evaluate, fit_window_scaler, make_windows,
    predict_lstm, top_features, train_lstm, train_xgboost,
)
from preprocessing import clean_data, load_raw, split_by_time  # noqa: E402
from utils import ensure_directory, save_results  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--horizon", type=int, default=24, help="forecast horizon in hours")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--models", default="xgboost,lstm,hybrid",
                    help="comma list of xgboost,lstm,hybrid (hybrid = val-tuned blend of both)")
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out_dir)
    ensure_directory(out)

    print("Loading and cleaning...")
    clean = clean_data(load_raw(args.data_dir))
    data = build_dataset(clean, horizon=args.horizon)
    use_lstm = any(m in args.models for m in ("lstm", "hybrid"))
    if use_lstm:  # windows must be built before rows are dropped so history is contiguous
        Xw_all, win_ok = make_windows(data, args.seq_len)
    keep = data["y"].notna().to_numpy() & data["PM2.5"].notna().to_numpy()
    if use_lstm:
        keep &= win_ok
        Xw_all = Xw_all[keep]
    data = data[keep]

    train_all, test = split_by_time(data, args.test_fraction)
    train, val = split_by_time(train_all, 0.15)  # later slice of train used for early stopping
    cols = feature_columns(data)
    print(f"train={len(train):,} val={len(val):,} test={len(test):,} features={len(cols)}")

    model = train_xgboost(train[cols], train["y"], val[cols], val["y"])
    pred = model.predict(test[cols])
    preds_by_model = {"xgboost": pred}

    if use_lstm:
        # same chronological masks as split_by_time, applied to the window array
        t_cut = test.index.min()
        v_cut = val.index.min()
        idx = data.index.to_numpy()
        tr_m, va_m, te_m = idx < v_cut, (idx >= v_cut) & (idx < t_cut), idx >= t_cut
        mean, std = fit_window_scaler(Xw_all[tr_m])
        Xs = apply_window_scaler(Xw_all, mean, std)
        sid = data["station"].cat.codes.to_numpy().astype("int32")
        n_st = int(data["station"].nunique())
        lstm, ym, ys = train_lstm(Xs[tr_m], sid[tr_m], train["y"].to_numpy(), Xs[va_m], sid[va_m],
                                  val["y"].to_numpy(), n_st, epochs=args.epochs)
        lstm_test = predict_lstm(lstm, Xs[te_m], sid[te_m], ym, ys)
        preds_by_model["lstm"] = lstm_test
        w = blend_weight(val["y"].to_numpy(), predict_lstm(lstm, Xs[va_m], sid[va_m], ym, ys),
                         model.predict(val[cols]))
        preds_by_model["hybrid"] = w * lstm_test + (1 - w) * pred
        print(f"hybrid blend: {w:.2f} LSTM + {1 - w:.2f} XGBoost (tuned on validation)")
        lstm.save(out / "lstm.keras")

    metrics = {
        "horizon_hours": args.horizon,
        "n_test": int(len(test)),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        **{name: evaluate(test["y"], p) for name, p in preds_by_model.items()},
        "blend_weight_lstm": w if use_lstm else None,
        "persistence": evaluate(test["y"], test["PM2.5"]),
        "same_hour_yesterday": evaluate(test["y"], test["PM2.5_lag24"].fillna(test["PM2.5"])),
        "per_station_MAE": {
            s: evaluate(g["y"], p)["MAE"]
            for s, (g, p) in {
                s: (g, pred[test["station"].to_numpy() == s]) for s, g in test.groupby("station", observed=True)
            }.items()
        },
        "top_features": top_features(model, cols).round(4).to_dict(),
    }
    save_results(metrics, out / "metrics.json")

    best = min(preds_by_model, key=lambda k: metrics[k]["RMSE"])
    metrics["best_model"] = best
    save_results(metrics, out / "metrics.json")
    preds = pd.DataFrame(
        {"station": test["station"].astype(str), "actual": test["y"], "predicted": preds_by_model[best],
         **{f"pred_{k}": v for k, v in preds_by_model.items()}, "current_pm25": test["PM2.5"]}
    )
    preds.index.name = "forecast_made_at"
    preds.to_csv(out / "test_predictions.csv")
    joblib.dump({"model": model, "columns": cols, "horizon": args.horizon}, out / "model.joblib")

    for name in (*preds_by_model, "persistence", "same_hour_yesterday"):
        m = metrics[name]
        print(f"{name:22s} MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} R2={m['R2']:.3f}")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
