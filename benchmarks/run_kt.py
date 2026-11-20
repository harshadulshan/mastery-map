"""
Benchmark on public tutoring data: predict whether each answer is right from the learner's earlier answers.

Scored as in Gervet et al. (2020): 5-fold cross-validation by learner (each learner's answers are in exactly one
test fold; the model never sees a test learner while learning), AUC per fold, mean ± SD. Three models:

    marks per skill   the learner's share of right answers on the question's skills so far (a mark book)
    Best-LR           the published model, re-implemented here (masterymap.model.best_lr_columns)
    Mastery Map       Best-LR plus recency and pooled practice (the app's model)

The extra features were chosen on ASSISTments 2009; the other five datasets were not used to choose anything.

    python benchmarks/run_kt.py                    # all six datasets (about an hour on 2 cores)
    python benchmarks/run_kt.py assistments09      # one or more by name
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from masterymap import ktdata  # noqa: E402
from masterymap import model as M  # noqa: E402

OUT = os.path.join(ROOT, "benchmarks", "results_kt.json")
ORDER = ["assistments09", "statics", "spanish", "assistments15", "algebra05", "assistments17"]
FOLDS, SEED = 5, 0


def folds_of(users, k=FOLDS, seed=SEED):
    u = np.unique(users)
    f = dict(zip(u, np.random.default_rng(seed).permutation(len(u)) % k))
    return np.array([f[x] for x in users])


def calibration(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [{"from": float(edges[i]), "to": float(edges[i + 1]), "n": int((b == i).sum()),
             "predicted": float(p[b == i].mean()) if (b == i).any() else None,
             "actual": float(y[b == i].mean()) if (b == i).any() else None} for i in range(bins)]


def run(name, log=print):
    t0 = time.time()
    df, Q = ktdata.load(name)
    u, q, y = df["user_id"].to_numpy(), df["item_id"].to_numpy(), df["correct"].to_numpy().astype(float)
    X = M.encode(u, q, y, Q)
    cols = M.best_lr_columns(M.Layout(*Q.shape))
    Xb = X[:, cols]
    marks = M.marks_per_skill(u, q, y, Q)
    fold = folds_of(u)
    pred = {"Mastery Map": np.zeros(len(y)), "Best-LR": np.zeros(len(y)), "marks per skill": marks}
    per = {k: [] for k in pred}
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        for nm, XX in (("Best-LR", Xb), ("Mastery Map", X)):
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(XX[tr], y[tr])
            pred[nm][te] = lr.predict_proba(XX[te])[:, 1]
        for nm in pred:
            per[nm].append(roc_auc_score(y[te], pred[nm][te]))
        log(f"  {name} fold {f}: " + ", ".join(f"{k} {v[-1]:.4f}" for k, v in per.items())
            + f"  ({time.time() - t0:.0f} s)")
    p = pred["Mastery Map"]
    band = (p >= 0.6) & (p < 0.8)
    res = {"answers": int(len(y)), "learners": int(len(np.unique(u))), "questions": int(Q.shape[0]),
           "skills": int(Q.shape[1]), "right": round(float(y.mean()), 4), "seconds": round(time.time() - t0),
           "models": {k: {"folds": [round(a, 5) for a in v], "mean": round(float(np.mean(v)), 5),
                          "sd": round(float(np.std(v)), 5), "pooled": round(float(roc_auc_score(y, pred[k])), 5),
                          "rmse": round(float(np.sqrt(np.mean((pred[k] - y) ** 2))), 5)} for k, v in per.items()},
           "calibration": calibration(y, p),
           "band_60_80": {"answers": int(band.sum()), "predicted": round(float(p[band].mean()), 4),
                          "actual": round(float(y[band].mean()), 4)}}
    return res


def main():
    names = [a for a in sys.argv[1:] if a in ktdata.SHA256] or ORDER
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out.setdefault("datasets", {})
    out["protocol"] = (f"{FOLDS}-fold cross-validation by learner (seed {SEED}); AUC per fold, mean ± SD; scikit-learn "
                       "LogisticRegression (L-BFGS, C = 1) for Best-LR and Mastery Map")
    for nm in names:
        print(nm, flush=True)
        out["datasets"][nm] = run(nm, log=lambda s: print(s, flush=True))
        out["order"] = [d for d in ORDER if d in out["datasets"]]
        json.dump(out, open(OUT, "w"), indent=1)
        m = out["datasets"][nm]["models"]
        print(f"{nm}: " + ", ".join(f"{k} {v['mean']:.4f} ± {v['sd']:.4f}" for k, v in m.items()), flush=True)


if __name__ == "__main__":
    main()
