"""
The mastery model: how likely each student is to answer a question of each skill correctly now.

A logistic regression on each student's own history. Its core is "Best-LR" (Gervet, Koedinger, Schneider & Mitchell,
2020, Journal of Educational Data Mining): how easy each question and each skill is, and the student's earlier
attempts and correct answers on the question's skills, on the question itself and in total, each counted as
log(1 + n). Added here, for recency: earlier correct and wrong answers per skill and in total with older answers
counting less (half-life of 4 answers on the skill, 8 answers overall), and the same counts pooled over all skills,
so that a small class shares what is learnt about practice in general. All features of an answer come from the
answers before it, never from the answer itself or later ones.

The skills of each question come from a question-by-skill matrix Q (a question can train several skills).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

HALF_SKILL = 4.0          # answers on the same skill
HALF_ALL = 8.0            # answers on anything
DENSE = ["question_tries", "question_right", "all_tries", "all_right", "recent_right", "recent_wrong",
         "skill_tries_any", "skill_right_any", "skill_recent_right_any", "skill_recent_wrong_any"]


def phi(x):
    return np.log1p(x)


def _prior(keys, y):
    """For rows in answer order: earlier rows with the same key, and how many of them were correct."""
    g = pd.Series(y).groupby(np.asarray(keys), sort=False)
    att = g.cumcount().to_numpy().astype(float)
    win = (g.cumsum() - y).to_numpy().astype(float)
    return att, win


def _decayed_prior(keys, y, half):
    """Earlier correct and wrong answers with the same key, the latest counting 1 and each one before it less
    (halving every `half` answers): sum over earlier answers j of d^(answers since j - 1)."""
    d = 0.5 ** (1.0 / half)
    keys = np.asarray(keys)
    y = np.asarray(y, float)
    order = np.argsort(keys, kind="stable")
    ks, ys = keys[order], y[order]
    new = np.ones(len(ks), bool)
    new[1:] = ks[1:] != ks[:-1]
    w = np.zeros(len(ks))
    f = np.zeros(len(ks))
    cw = cf = 0.0
    for i in range(len(ks)):
        if new[i]:
            cw = cf = 0.0
        w[i], f[i] = cw, cf
        cw = cw * d + ys[i]
        cf = cf * d + 1.0 - ys[i]
    out_w, out_f = np.empty(len(ks)), np.empty(len(ks))
    out_w[order], out_f[order] = w, f
    return out_w, out_f


def _decayed_final(keys, y, half):
    """The same sums as _decayed_prior, for the next answer after all of them: per key (wins, fails)."""
    d = 0.5 ** (1.0 / half)
    s = pd.DataFrame({"k": np.asarray(keys), "y": np.asarray(y, float)})
    g = s.groupby("k", sort=False)
    back = (g["y"].transform("size") - 1 - g.cumcount()).to_numpy()          # answers after this one
    wt = d ** back
    s["w"], s["f"] = wt * s["y"], wt * (1 - s["y"])
    out = s.groupby("k", sort=True)[["w", "f"]].sum()
    return out.index.to_numpy(), out["w"].to_numpy(), out["f"].to_numpy()


@dataclass
class Layout:
    n_items: int
    n_skills: int

    @property
    def blocks(self):
        I, K = self.n_items, self.n_skills
        names = ["question", "skill", "skill_tries", "skill_right", "skill_recent_right", "skill_recent_wrong"]
        sizes = [I, K, K, K, K, K]
        out, s = {}, 0
        for nm, sz in zip(names, sizes):
            out[nm] = (s, s + sz)
            s += sz
        out["dense"] = (s, s + len(DENSE))
        return out

    @property
    def width(self):
        return self.blocks["dense"][1]


def encode(student, item, correct, Q):
    """Feature rows for answers in time order (within each student). student, item: integer ids; correct: 0/1;
    Q: (questions, skills) 0/1 sparse matrix. Returns a sparse (answers, features) matrix."""
    u = np.asarray(student, np.int64)
    q = np.asarray(item, np.int64)
    y = np.asarray(correct, float)
    Q = sparse.csr_matrix(Q)
    n, (I, K) = len(y), Q.shape
    lay = Layout(I, K)
    Qr = Q[q]
    rows = np.repeat(np.arange(n), np.diff(Qr.indptr))
    ks = Qr.indices.astype(np.int64)
    key = u[rows] * K + ks
    att_k, win_k = _prior(key, y[rows])
    dw_k, df_k = _decayed_prior(key, y[rows], HALF_SKILL)
    att_q, win_q = _prior(u * I + q, y)
    att_t, win_t = _prior(u, y)
    fw, ff = _decayed_prior(u, y, HALF_ALL)
    shared = [np.bincount(rows, weights=v, minlength=n) for v in (phi(att_k), phi(win_k), dw_k, df_k)]
    dense = np.column_stack([phi(att_q), phi(win_q), phi(att_t), phi(win_t), fw, ff] + shared)
    B = lay.blocks
    data = [np.ones(n), np.ones(len(rows)), phi(att_k), phi(win_k), dw_k, df_k]
    r_ = [np.arange(n)] + [rows] * 5
    c_ = [q + B["question"][0], ks + B["skill"][0], ks + B["skill_tries"][0], ks + B["skill_right"][0],
          ks + B["skill_recent_right"][0], ks + B["skill_recent_wrong"][0]]
    d0 = B["dense"][0]
    data.append(dense.ravel())
    r_.append(np.repeat(np.arange(n), dense.shape[1]))
    c_.append(np.tile(np.arange(dense.shape[1]) + d0, n))
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(r_), np.concatenate(c_))),
                             shape=(n, lay.width))


def best_lr_columns(layout: Layout):
    """The columns of the published Best-LR model (no recency, no pooled practice): for reproducing it."""
    B = layout.blocks
    keep = [np.arange(*B[k]) for k in ("question", "skill", "skill_tries", "skill_right")]
    return np.concatenate(keep + [np.arange(B["dense"][0], B["dense"][0] + 4)])


def marks_per_skill(student, item, correct, Q):
    """The mark-book view, for comparison: the student's share of right answers on the question's skills so far
    ((right + 1) / (tries + 2), averaged over the skills)."""
    u = np.asarray(student, np.int64)
    q = np.asarray(item, np.int64)
    y = np.asarray(correct, float)
    Q = sparse.csr_matrix(Q)
    Qr = Q[q]
    rows = np.repeat(np.arange(len(y)), np.diff(Qr.indptr))
    att, win = _prior(u[rows] * Q.shape[1] + Qr.indices, y[rows])
    rate = (win + 1) / (att + 2)
    return np.bincount(rows, weights=rate, minlength=len(y)) / np.maximum(np.diff(Qr.indptr), 1)


@dataclass
class State:
    """Everything the model needs about each student's history to predict their next answer."""
    students: np.ndarray            # student ids in row order of the arrays below
    skill: dict                     # name -> (students, skills) array: tries, right, recent_right, recent_wrong
    total: dict                     # name -> (students,) array: tries, right, recent_right, recent_wrong
    question_tries: sparse.csr_matrix   # (students, questions)
    question_right: sparse.csr_matrix


def state(student, item, correct, Q, students=None):
    """The history summed up after all the answers given (answers in time order within each student)."""
    u = np.asarray(student, np.int64)
    q = np.asarray(item, np.int64)
    y = np.asarray(correct, float)
    Q = sparse.csr_matrix(Q)
    I, K = Q.shape
    S = np.unique(u) if students is None else np.asarray(students, np.int64)
    pos = {s: i for i, s in enumerate(S)}
    keep = np.array([s in pos for s in u], bool) if len(u) else np.zeros(0, bool)
    u, q, y = u[keep], q[keep], y[keep]
    r = np.array([pos[s] for s in u], np.int64)
    n = len(S)
    Qr = Q[q]
    rows = np.repeat(np.arange(len(y)), np.diff(Qr.indptr))
    ks = Qr.indices.astype(np.int64)
    tries = np.zeros((n, K))
    right = np.zeros((n, K))
    np.add.at(tries, (r[rows], ks), 1.0)
    np.add.at(right, (r[rows], ks), y[rows])
    rw = np.zeros((n, K))
    rf = np.zeros((n, K))
    if len(rows):
        keys, w, f = _decayed_final(r[rows] * K + ks, y[rows], HALF_SKILL)
        rw[keys // K, keys % K] = w
        rf[keys // K, keys % K] = f
    tw = np.zeros(n)
    tf = np.zeros(n)
    if len(y):
        keys, w, f = _decayed_final(r, y, HALF_ALL)
        tw[keys], tf[keys] = w, f
    qt = sparse.csr_matrix((np.ones(len(y)), (r, q)), shape=(n, I))
    qr = sparse.csr_matrix((y, (r, q)), shape=(n, I))
    return State(S, {"tries": tries, "right": right, "recent_right": rw, "recent_wrong": rf},
                 {"tries": np.bincount(r, minlength=n).astype(float),
                  "right": np.bincount(r, weights=y, minlength=n), "recent_right": tw, "recent_wrong": tf},
                 qt, qr)


class MasteryModel:
    """Fit on answers in time order; predict the chance of a correct answer for any question or skill."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, student, item, correct, Q):
        self.Q = sparse.csr_matrix(Q)
        self.layout = Layout(*self.Q.shape)
        X = encode(student, item, correct, self.Q)
        y = np.asarray(correct, int)
        self.lr = LogisticRegression(C=self.C, max_iter=2000)
        if len(np.unique(y)) < 2:                     # all right or all wrong: nothing to learn but the rate
            self.lr = None
            self.rate = float(y.mean()) if len(y) else 0.5
        else:
            self.lr.fit(X, y)
        self._unpack()
        return self

    def _unpack(self):
        B = self.layout.blocks
        w = self.lr.coef_.ravel() if self.lr is not None else np.zeros(self.layout.width)
        self.b0 = float(self.lr.intercept_[0]) if self.lr is not None else float(np.log(self.rate / (1 - self.rate)
                                                                                      + 1e-12))
        self.w = {k: w[a:b] for k, (a, b) in B.items()}
        # a typical question of each skill: the mean easiness of the questions that train it
        Qc = self.Q.tocsc()
        cnt = np.diff(Qc.indptr)
        self.typical = np.where(cnt > 0, (Qc.T @ self.w["question"]) / np.maximum(cnt, 1), 0.0)

    def predict_proba(self, X):
        if self.lr is None:
            return np.full(X.shape[0], self.rate)
        return self.lr.predict_proba(X)[:, 1]

    def _person(self, st: State):
        d = self.w["dense"]
        t = st.total
        return (d[2] * phi(t["tries"]) + d[3] * phi(t["right"]) + d[4] * t["recent_right"]
                + d[5] * t["recent_wrong"])

    def _skill_terms(self, st: State):
        """(students, skills) log-odds of the skill part of a question, for each student's next answer."""
        s, w, d = st.skill, self.w, self.w["dense"]
        return (w["skill"][None, :] + w["skill_tries"] * phi(s["tries"]) + w["skill_right"] * phi(s["right"])
                + w["skill_recent_right"] * s["recent_right"] + w["skill_recent_wrong"] * s["recent_wrong"]
                + d[6] * phi(s["tries"]) + d[7] * phi(s["right"]) + d[8] * s["recent_right"]
                + d[9] * s["recent_wrong"])

    def mastery(self, st: State):
        """(students, skills): the chance of answering a typical question of each skill correctly now."""
        z = self.b0 + self._person(st)[:, None] + self._skill_terms(st) + self.typical[None, :]
        return 1 / (1 + np.exp(-z))

    def question_chance(self, st: State, questions=None):
        """(students, questions): the chance of answering each question correctly now."""
        Qm = self.Q if questions is None else self.Q[np.asarray(questions)]
        qi = np.arange(self.Q.shape[0]) if questions is None else np.asarray(questions)
        d = self.w["dense"]
        z = (self.b0 + self._person(st)[:, None] + (self._skill_terms(st) @ Qm.T.toarray())
             + self.w["question"][qi][None, :]
             + d[0] * phi(st.question_tries[:, qi].toarray()) + d[1] * phi(st.question_right[:, qi].toarray()))
        return 1 / (1 + np.exp(-z))
