"""
The term, looked at two ways.

replay: the class week by week, as the app would have shown it each Monday: the model is fitted again on the answers
given before that Monday only, so no week is judged with answers from later weeks.

what_if: the next weeks, played forward with two ways of practising: one revision paper for everybody (n questions a
week spread over the skills taught so far) or a personal practice set each (plan.practice_set, refreshed every week
with the new answers). On the demo class the students are the simulated learners whose true state is known; on a
real class they are simulated by the fitted model itself (what the model expects, not a measurement).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import model as M
from . import plan
from .classroom import Classroom


def replay(c: Classroom, progress=None):
    """Mastery each Monday: dict with weeks (dates), mastery (weeks, students, skills), tries (same shape), taught
    (weeks, skills), answers (weeks,) number of answers known by then."""
    say = progress or (lambda *_: None)
    weeks = [w for w in c.weeks() if (pd.to_datetime(c.answers["Date"]) < w).any()] + [c.today]
    weeks = sorted(set(weeks))
    out_m, out_t, taught, cnt = [], [], [], []
    for j, w in enumerate(weeks):
        say(j / len(weeks), f"Week of {w:%d %b}")
        s = plan.snapshot(c, until=w)
        out_m.append(s.mastery)
        out_t.append(s.tries)
        taught.append(s.taught)
        cnt.append(int(s.tries.sum() and len(c.coded(w)[0])))
    say(1.0, "Done")
    return {"weeks": weeks, "mastery": np.array(out_m), "tries": np.array(out_t), "taught": np.array(taught),
            "answers": np.array(cnt)}


class ModelLearners:
    """Learners simulated by the fitted model: a question is answered right with the model's predicted chance, and
    the answers change the model's estimates through what it learnt about practice."""

    def __init__(self, c: Classroom):
        self.c = c
        s, q, y, _ = c.coded()
        self.s, self.q, self.y = list(s), list(q), list(y)
        self.Q = c.Q()
        self.model = M.MasteryModel().fit(s, q, y, self.Q)
        self.n = len(c.students)

    def _state(self):
        return M.state(np.array(self.s), np.array(self.q), np.array(self.y), self.Q, students=np.arange(self.n))

    def chance_skill(self):
        return self.model.mastery(self._state())

    def answer_sets(self, sets, rng):
        """sets: {student index: [question indices]}; everyone answers their set (chances from the start of the
        week); returns {student: answers}."""
        ch = self.model.question_chance(self._state())
        out = {}
        for i, qs_ in sets.items():
            res = (rng.random(len(qs_)) < ch[i, qs_]).astype(float)
            out[i] = res
            self.s += [i] * len(qs_)
            self.q += list(qs_)
            self.y += list(res)
        order = np.argsort(np.array(self.s), kind="stable")
        self.s, self.q, self.y = (list(np.array(v)[order]) for v in (self.s, self.q, self.y))
        return out


class TruthLearners:
    """The demo's simulated learners (demo.Truth): their true chance is known."""

    def __init__(self, c: Classroom):
        self.truth = c.extra["truth"].copy()

    def chance_skill(self):
        return self.truth.chance_skill()

    def answer_sets(self, sets, rng):
        return {i: self.truth.answer([i] * len(qs_), list(qs_), rng) for i, qs_ in sets.items()}


def _same_paper(c: Classroom, snap, week, n):
    """One revision paper for everybody: n medium questions spread over the taught skills, a different spread each
    week."""
    taught = np.flatnonzero(snap.taught)
    Qc = snap.model.Q.tocsc()
    lv = pd.to_numeric(c.questions["Level"], errors="coerce").fillna(2).to_numpy()
    picks = []
    ks = taught[(np.arange(n) * len(taught) // n + week) % len(taught)]
    for j, k in enumerate(ks):
        qs_ = Qc.indices[Qc.indptr[k]:Qc.indptr[k + 1]]
        med = qs_[lv[qs_] == 2] if (lv[qs_] == 2).any() else qs_
        picks.append(int(med[(week + j) % len(med)]))
    return picks


def what_if(c: Classroom, weeks=4, n=10, seed=0, progress=None):
    """Play the next weeks with both ways of practising. Returns dict: kind ('truth' or 'model'), weeks, and for each
    policy ('same', 'personal') the (weeks + 1, students, skills) chances, starting from today."""
    say = progress or (lambda *_: None)
    kind = "truth" if "truth" in c.extra else "model"
    taught0 = plan.snapshot(c).taught
    res = {"kind": kind, "weeks": weeks, "taught": taught0}
    for p_i, policy in enumerate(("same", "personal")):
        rng = np.random.default_rng(seed)
        L = TruthLearners(c) if kind == "truth" else ModelLearners(c)
        cur = Classroom(c.name, c.today, c.students, c.skills, c.questions, c.answers.copy(), {})
        path = [L.chance_skill()]
        for w in range(weeks):
            say((p_i * weeks + w) / (2 * weeks), f"{'One paper for all' if policy == 'same' else 'Personal sets'}: "
                                                 f"week {w + 1}")
            snap = plan.snapshot(cur)
            qpos = {q: j for j, q in enumerate(c.question_ids)}
            if policy == "same":
                paper = _same_paper(c, snap, w, n)
                sets = {i: paper for i in range(len(c.students))}
            else:
                sets = {i: [qpos[q] for q in plan.practice_set(snap, cur, i, n)["Question ID"]]
                        for i in range(len(c.students))}
            got = L.answer_sets(sets, rng)
            day = c.today + pd.Timedelta(days=7 * w + 5)
            new = [(day, c.student_ids[i], c.question_ids[q], int(r)) for i, qs_ in sets.items()
                   for q, r in zip(qs_, got[i])]
            cur.answers = pd.concat([cur.answers, pd.DataFrame(new, columns=cur.answers.columns)], ignore_index=True)
            cur.today = day + pd.Timedelta(days=2)
            path.append(L.chance_skill())
        res[policy] = np.array(path)
    say(1.0, "Done")
    return res


def summary(res, weak=0.5):
    """Per policy: the expected score on a paper with one medium question per taught skill (class mean), and weak
    spots per student (skills below `weak`), at the start and after the weeks played."""
    t = res["taught"]
    out = {}
    for p in ("same", "personal"):
        ch = res[p][:, :, t]
        out[p] = {"score": ch.mean(axis=(1, 2)), "weak": (ch < weak).sum(axis=2).mean(axis=1),
                  "students_score": ch.mean(axis=2)}
    return out
