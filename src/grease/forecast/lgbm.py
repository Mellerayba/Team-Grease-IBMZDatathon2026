"""LightGBM Poisson forecaster: one model per (acuity, horizon) predicting expected incidents in the next window (decisions D8, M1, M8).

Memory: the feature matrix is built ONCE as a float32 array straight from the table (categories as integer codes), binned ONCE into a shared
LightGBM Dataset, and only the label is swapped between the 8 models. Models are saved as LightGBM text files plus a small JSON file, which is
portable across machines and CPU architectures (the VM is s390x).
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from grease.forecast.features import ACUITY, HORIZONS, target_name

# Parameter names follow LightGBM's scikit-learn interface (so notebook overrides read naturally); _core_params translates them.
DEFAULT_PARAMS = dict(objective="poisson", n_estimators=1500, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
EARLY_STOPPING_ROUNDS = 50


class ForecastModel:
    def __init__(self, features: list[str], params: dict | None = None, horizons: dict = HORIZONS, acuity: list[str] = ACUITY):
        self.features, self.params = list(features), {**DEFAULT_PARAMS, **(params or {})}
        self.horizons, self.acuity = dict(horizons), list(acuity)
        self.categories: dict[str, list[str]] = {}          # categorical column -> training categories (coded as 0..n-1)
        self.boosters: dict[tuple[str, str], lgb.Booster] = {}
        self.best_iteration: dict[tuple[str, str], int] = {}

    def _core_params(self) -> dict:
        p = self.params
        return dict(objective=p["objective"], learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
                    min_data_in_leaf=p["min_child_samples"], bagging_fraction=p["subsample"], bagging_freq=p["subsample_freq"],
                    feature_fraction=p["colsample_bytree"], verbosity=-1, feature_pre_filter=False)

    def _matrix(self, df: pd.DataFrame, rows=None) -> np.ndarray:
        """Float32 matrix of the model features for `rows` (positions; all rows if None). Categorical columns become integer codes (NaN if unseen)."""
        n = len(df) if rows is None else len(rows)
        X = np.empty((n, len(self.features)), dtype=np.float32)
        for j, c in enumerate(self.features):
            s = df[c]
            if c in self.categories:
                cats = self.categories[c]
                if str(s.dtype) == "category" and [str(v) for v in s.cat.categories] == cats:
                    codes = s.cat.codes.to_numpy()
                else:
                    codes = pd.Categorical(s.astype(str), categories=cats).codes
                col = codes if rows is None else codes[rows]
                X[:, j] = np.where(col >= 0, col, np.nan)
            else:
                v = s.to_numpy()
                X[:, j] = v if rows is None else v[rows]
        return X

    def fit(self, df: pd.DataFrame, train_rows, valid_rows, targets=None, log=print) -> "ForecastModel":
        """`train_rows` / `valid_rows`: row positions in `df`. `targets(acuity, horizon)` returns a float32 array aligned to df
        (default: read the `t_<acuity>_<horizon>` columns). Rows used for training must have no missing target."""
        self.categories = {c: [str(v) for v in df[c].cat.categories] for c in self.features if str(df[c].dtype) == "category"}
        cat_idx = [j for j, c in enumerate(self.features) if c in self.categories]
        get = targets or (lambda a, h: df[target_name(a, h)].to_numpy())
        params = self._core_params()
        ds_args = dict(feature_name=self.features, categorical_feature=cat_idx, params=params, free_raw_data=True)
        dtrain = lgb.Dataset(self._matrix(df, train_rows), label=np.zeros(len(train_rows), np.float32), **ds_args).construct()
        dvalid = lgb.Dataset(self._matrix(df, valid_rows), label=np.zeros(len(valid_rows), np.float32), reference=dtrain, **ds_args).construct()
        for a in self.acuity:
            for h in self.horizons:
                y = get(a, h)
                ytr, yva = y[train_rows], y[valid_rows]
                if np.isnan(ytr).any() or np.isnan(yva).any():
                    raise ValueError(f"{target_name(a, h)} has missing values in the training / validation rows: drop incomplete windows first "
                                     f"(features.complete_window_mask)")
                dtrain.set_label(ytr)
                dvalid.set_label(yva)
                booster = lgb.train(params, dtrain, num_boost_round=self.params["n_estimators"], valid_sets=[dvalid],
                                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
                self.boosters[(a, h)] = booster
                self.best_iteration[(a, h)] = int(booster.best_iteration or self.params["n_estimators"])
                log(f"  {target_name(a, h)}: {len(train_rows):,} training rows, best iteration {self.best_iteration[(a, h)]}")
        return self

    def predict(self, df: pd.DataFrame, rows=None) -> pd.DataFrame:
        """Expected incidents in the next window for every row: pred_<acuity>_<horizon> and pred_all_<horizon> (= critical + other)."""
        X = self._matrix(df, rows)
        out = pd.DataFrame(index=df.index if rows is None else df.index[rows])
        for (a, h), booster in self.boosters.items():
            out[f"pred_{a}_{h}"] = booster.predict(X).astype("float32")
        for h in self.horizons:
            if all(f"pred_{a}_{h}" in out for a in ("critical", "other")):
                out[f"pred_all_{h}"] = out[f"pred_critical_{h}"] + out[f"pred_other_{h}"]
        return out

    def save(self, directory) -> None:
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        for (a, h), booster in self.boosters.items():
            booster.save_model(str(d / f"{a}_{h}.txt"))
        (d / "meta.json").write_text(json.dumps({"features": self.features, "params": self.params, "horizons": self.horizons,
                                                  "acuity": self.acuity, "categories": self.categories,
                                                  "best_iteration": {f"{a}_{h}": v for (a, h), v in self.best_iteration.items()}}, indent=2))

    @classmethod
    def load(cls, directory) -> "ForecastModel":
        d = Path(directory); meta = json.loads((d / "meta.json").read_text())
        m = cls(meta["features"], meta["params"], meta["horizons"], meta["acuity"])
        m.categories = meta.get("categories", {})
        for a in m.acuity:
            for h in m.horizons:
                m.boosters[(a, h)] = lgb.Booster(model_file=str(d / f"{a}_{h}.txt"))
        return m

    def feature_importance(self) -> pd.DataFrame:
        """Gain-based importance per model, as a share of that model's total (columns: one per model)."""
        cols = {f"{a}_{h}": pd.Series(b.feature_importance("gain"), index=b.feature_name()) for (a, h), b in self.boosters.items()}
        imp = pd.DataFrame(cols).reindex(self.features)
        return imp / imp.sum()
