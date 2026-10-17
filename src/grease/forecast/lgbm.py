"""LightGBM Poisson forecaster: one model per (acuity, horizon) predicting expected incidents in the next window (decisions D8, M1, M8).

Models are saved as LightGBM text files plus a small JSON file, which is portable across machines and CPU architectures (the VM is s390x).
"""
import inspect
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from grease.forecast.features import ACUITY, HORIZONS, target_name

DEFAULT_PARAMS = dict(objective="poisson", n_estimators=1500, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
EARLY_STOPPING_ROUNDS = 50
# LightGBM >= 4.7 deprecates `eval_set` in favour of `eval_X` / `eval_y`; older versions (possibly the VM's) only have `eval_set`.
_NEW_EVAL_API = "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters


def _eval_kwargs(X, y) -> dict:
    return dict(eval_X=X, eval_y=y) if _NEW_EVAL_API else dict(eval_set=[(X, y)])


class ForecastModel:
    def __init__(self, features: list[str], params: dict | None = None, horizons: dict = HORIZONS, acuity: list[str] = ACUITY):
        self.features, self.params = list(features), {**DEFAULT_PARAMS, **(params or {})}
        self.horizons, self.acuity = dict(horizons), list(acuity)
        self.categories: dict[str, list] = {}          # categorical columns and their training categories, so live rows can be re-coded
        self.boosters: dict[tuple[str, str], lgb.Booster] = {}
        self.best_iteration: dict[tuple[str, str], int] = {}

    def _as_trained(self, X: pd.DataFrame) -> pd.DataFrame:
        """Feature frame with categorical columns coded exactly as in training (LightGBM requires matching categories)."""
        feats = X[self.features].copy()
        for col, cats in self.categories.items():
            feats[col] = pd.Categorical(feats[col].astype(str) if str(feats[col].dtype) != "category" else feats[col], categories=cats)
        return feats

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, log=print) -> "ForecastModel":
        self.categories = {c: [str(v) for v in train[c].cat.categories] for c in self.features if str(train[c].dtype) == "category"}
        train, valid = train.copy(deep=False), valid.copy(deep=False)
        for col, cats in self.categories.items():          # make train and validation use the same coding (a no-op when they already do)
            for d in (train, valid):
                if [str(v) for v in d[col].cat.categories] != cats:
                    d[col] = pd.Categorical(d[col].astype(str), categories=cats)
        for a in self.acuity:
            for h in self.horizons:
                t = target_name(a, h)
                tr, va = train[train[t].notna()], valid[valid[t].notna()]
                model = lgb.LGBMRegressor(**self.params)
                model.fit(tr[self.features], tr[t], **_eval_kwargs(va[self.features], va[t]),
                          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
                self.boosters[(a, h)] = model.booster_
                self.best_iteration[(a, h)] = int(model.best_iteration_ or self.params["n_estimators"])
                log(f"  {t}: {len(tr):,} training rows, best iteration {self.best_iteration[(a, h)]}")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Expected incidents in the next window for every row: pred_<acuity>_<horizon> and pred_all_<horizon> (= critical + other)."""
        out = pd.DataFrame(index=X.index)
        feats = self._as_trained(X)
        for (a, h), booster in self.boosters.items():
            out[f"pred_{a}_{h}"] = booster.predict(feats).astype("float32")
        for h in self.horizons:
            if all(f"pred_{a}_{h}" in out for a in ("critical", "other")):
                out[f"pred_all_{h}"] = out[f"pred_critical_{h}"] + out[f"pred_other_{h}"]
        return out

    def save(self, directory) -> None:
        d = Path(directory); d.mkdir(parents=True, exist_ok=True)
        for (a, h), booster in self.boosters.items():
            booster.save_model(str(d / f"{a}_{h}.txt"))
        (d / "meta.json").write_text(json.dumps({"features": self.features, "params": self.params, "horizons": self.horizons,
                                                  "acuity": self.acuity, "categories": self.categories, "best_iteration": {f"{a}_{h}": v for (a, h), v in self.best_iteration.items()}}, indent=2))

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
