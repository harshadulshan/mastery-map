"""
From answers to decisions: each student's mastery of each skill, their weak spots, and their next practice set.

A skill is secure from an 80% chance on a typical question, developing from 60%, weak below; with fewer than 3
answers on it the estimate leans on the rest of the student's work, so it is marked "few answers". A practice set
(10 questions by default) spends most questions on the weakest skills whose prerequisites are in place (or on the
prerequisite first, when it is the prerequisite that is weak), keeps one or two for review of secure skills not
practised lately, and checks a skill with too few answers; within a skill it picks questions the student has not
done, with the predicted chance closest to 70% ("desirable difficulty": hard enough to learn from, easy enough to
get right most of the time).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import model as M
from .classroom import Classroom

SECURE, WEAK, EVIDENCE = 0.8, 0.6, 3
TARGET = {"focus": 0.70, "review": 0.80, "check": 0.65}


@dataclass
class Snapshot:
    """Everything the app shows about a class at one moment."""
    model: M.MasteryModel
    state: M.State
    mastery: np.ndarray        # (students, skills)
    chance: np.ndarray         # (students, questions)
    tries: np.ndarray          # (students, skills)
    right: np.ndarray
    since: np.ndarray          # (students, skills) days since the last answer on the skill (inf: never)
    taught: np.ndarray         # (skills,) skills somebody has answered questions on
    done: np.ndarray           # (students, questions) answers given to each question
    until: pd.Timestamp


def snapshot(c: Classroom, until=None) -> Snapshot:
    s, q, y, day = c.coded(until)
    Q = c.Q()
    n = len(c.students)
    mod = M.MasteryModel().fit(s, q, y, Q)
    st = M.state(s, q, y, Q, students=np.arange(n))
    ref = pd.Timestamp(until) if until is not None else c.today
    d0 = pd.to_datetime(c.answers["Date"]).min()
    age = (ref - d0).days - day if len(day) else day
    since = np.full((n, Q.shape[1]), np.inf)
    Qr = Q[q]
    rows = np.repeat(np.arange(len(q)), np.diff(Qr.indptr))
    for r_, k in zip(rows, Qr.indices):
        since[s[r_], k] = min(since[s[r_], k], age[r_])
    done = np.zeros((n, Q.shape[0]))
    np.add.at(done, (s, q), 1)
    return Snapshot(mod, st, mod.mastery(st), mod.question_chance(st), st.skill["tries"], st.skill["right"], since,
                    st.skill["tries"].sum(axis=0) > 0, done, ref)


def status(m, tries):
    """'secure' / 'developing' / 'weak' / 'few answers' for arrays of mastery and answer counts."""
    m, tries = np.asarray(m), np.asarray(tries)
    out = np.where(m >= SECURE, "secure", np.where(m >= WEAK, "developing", "weak")).astype(object)
    out[tries < EVIDENCE] = "few answers"
    return out


def _blocking(m_i, taught, prereq):
    """For each skill, how many taught, not yet secure skills need it as a prerequisite."""
    K = len(m_i)
    out = np.zeros(K)
    for k, ps in prereq.items():
        if taught[k] and m_i[k] < SECURE:
            for p in ps:
                out[p] += 1
    return out


def practice_set(snap: Snapshot, c: Classroom, i: int, n=10):
    """The next practice set of student i: a table with the question, skill, kind (focus / review / check), the
    predicted chance and the reason."""
    K = snap.mastery.shape[1]
    sk = c.skill_ids
    names = dict(zip(c.skills["Skill ID"].astype(str), c.skills["Skill"]))
    prereq = c.prerequisites()
    m, t, taught = snap.mastery[i], snap.tries[i], snap.taught
    block = _blocking(m, taught, prereq)
    ready = np.array([all(m[p] >= WEAK or not taught[p] for p in prereq.get(k, [])) for k in range(K)])
    pri = np.where(taught & (m < SECURE) & (t >= 1), (SECURE - m) * (1 + 0.5 * block) * np.where(ready, 1, 0.4), 0)
    focus = [int(k) for k in np.argsort(-pri) if pri[k] > 0][:3]
    unknown = [int(k) for k in np.argsort(t) if taught[k] and t[k] < 2 and k not in focus]
    review = [int(k) for k in np.argsort(-np.where(np.isfinite(snap.since[i]), snap.since[i], 1e6))
              if taught[k] and m[k] >= SECURE and k not in focus]
    slots = []
    n_check = 1 if unknown and n >= 6 else 0
    n_focus = int(round(0.7 * n)) if focus else 0
    if focus:
        w = pri[focus] / pri[focus].sum()
        alloc = np.maximum(2, np.floor(w * n_focus)).astype(int)
        while alloc.sum() > n_focus and alloc.max() > 2:
            alloc[np.argmax(alloc)] -= 1
        while alloc.sum() < n_focus:
            alloc[np.argmin(alloc / np.maximum(w, 1e-9))] += 1
        slots += [("focus", k, int(a)) for k, a in zip(focus, alloc)]
    if n_check:
        slots.append(("check", unknown[0], 1))
    rest = n - sum(x[2] for x in slots)
    rv = (review or [k for k in np.argsort(-m) if taught[k] and k not in focus])[:max(1, (rest + 1) // 2)]
    for j, k in enumerate(rv):
        if rest <= 0:
            break
        a = int(np.ceil(rest / (len(rv) - j)))
        slots.append(("review", int(k), a))
        rest -= a
    Qc = snap.model.Q.tocsc()
    rows, used = [], set()
    for kind, k, a in slots:
        qs_ = Qc.indices[Qc.indptr[k]:Qc.indptr[k + 1]]
        if not len(qs_):
            continue
        ch = snap.chance[i, qs_]
        fresh = snap.done[i, qs_] == 0
        lvl = pd.to_numeric(c.questions["Level"], errors="coerce").fillna(0).to_numpy()[qs_]
        picked = []
        for _ in range(a):                     # closest to the target chance, unseen first, levels mixed
            score = (np.abs(ch - TARGET[kind]) + np.where(fresh, 0, 0.5) + np.isin(qs_, list(used)) * 9
                     + 0.06 * np.array([sum(lvl[qs_ == p_][0] == lv for p_ in picked) for lv in lvl]))
            q = int(qs_[np.argmin(score)])
            if q in used:
                break
            picked.append(q)
            used.add(q)
        for q in picked:
            if kind == "focus":
                why = (f"weak spot: {m[k]:.0%} on a typical question ({int(t[k])} answers)" if m[k] < WEAK else
                       f"developing: {m[k]:.0%} ({int(t[k])} answers)")
                if block[k]:
                    why += f"; needed for {int(block[k])} skill{'s' if block[k] > 1 else ''} still to secure"
                if not ready[k]:
                    why += "; its own prerequisites are shaky too"
            elif kind == "check":
                why = f"check: only {int(t[k])} answer{'s' if t[k] != 1 else ''} so far"
            else:
                d = snap.since[i, k]
                why = f"review: secure ({m[k]:.0%})" + (f", last practised {int(d)} days ago" if np.isfinite(d) else "")
            rows.append({"Question ID": c.question_ids[q], "Skill ID": sk[k], "Skill": names[sk[k]], "Kind": kind,
                         "Level": c.questions.at[q, "Level"], "Question": c.questions.at[q, "Question"],
                         "Answer": c.questions.at[q, "Answer"], "Chance": float(snap.chance[i, q]), "Why": why})
    order = {"focus": 0, "check": 1, "review": 2}
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["Kind", "Skill ID", "Chance"], key=lambda col: col.map(order) if col.name == "Kind"
                              else (-col if col.name == "Chance" else col), kind="stable").reset_index(drop=True)
    return out


def class_plan(snap: Snapshot, c: Classroom, n=10):
    """Practice sets for everybody, the skills to re-teach to the whole class, and small groups by first focus."""
    sets = {sid: practice_set(snap, c, i, n) for i, sid in enumerate(c.student_ids)}
    st = status(snap.mastery, snap.tries)
    sk = c.skill_ids
    names = dict(zip(c.skills["Skill ID"].astype(str), c.skills["Skill"]))
    weak_share = ((st == "weak") & snap.taught[None, :]).mean(axis=0)
    whole = [(sk[k], names[sk[k]], float(weak_share[k])) for k in np.argsort(-weak_share)
             if snap.taught[k] and weak_share[k] >= 0.4]
    groups = {}
    for sid, df in sets.items():
        f = df[df["Kind"] == "focus"]
        if len(f):
            groups.setdefault(f["Skill ID"].iloc[0], []).append(sid)
    groups = sorted(((k, names[k], v) for k, v in groups.items()), key=lambda g: -len(g[2]))
    return {"sets": sets, "whole_class": whole, "groups": groups}
