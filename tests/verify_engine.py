"""
Engine checks: the mastery model (no look-ahead, the published Best-LR reproduced), the class workbook, the question
generators, practice sets, the term replay, the what-if, the demo class, the printable sheets and the animations.

    python tests/verify_engine.py      # about 2 minutes (the Statics reproduction needs the benchmark data, which
                                       # is downloaded on first use; without it that check is skipped)
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.environ.get("MM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from masterymap import anim, demo, ktdata, plan, questions, report, term  # noqa: E402
from masterymap import classroom as cl  # noqa: E402
from masterymap import model as M  # noqa: E402

passed, failed = 0, []


def ok(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}  {detail}")


rng = np.random.default_rng(3)
I, K = 40, 7
Qd = (rng.random((I, K)) < 0.2).astype(float)
Qd[np.arange(I), rng.integers(0, K, I)] = 1
Q = sparse.csr_matrix(Qd)
n = 4000
u = np.sort(rng.integers(0, 60, n))
q = rng.integers(0, I, n)
y = (rng.random(n) < 0.62).astype(float)

print("Mastery model")
X = M.encode(u, q, y, Q).toarray()
y2 = y.copy()
cut = 2000
y2[cut:] = 1 - y2[cut:]
X2 = M.encode(u, q, y2, Q).toarray()
same = np.flatnonzero(np.ones(n, bool))
first_changed = cut
ok("an answer's features come only from earlier answers (flipping later ones changes nothing before them)",
   np.allclose(X[:cut + 1], X2[:cut + 1]) and not np.allclose(X[cut + 1:], X2[cut + 1:]))
s0 = np.array([0, 0, 0, 0])
kk = M._decayed_prior(s0, np.array([1.0, 0.0, 1.0, 1.0]), 1.0)          # half-life 1: d = 0.5
ok("recency: earlier right/wrong answers halve with each newer one", np.allclose(kk[0], [0, 1, 0.5, 1.25]) and
   np.allclose(kk[1], [0, 0, 1, 0.5]), str(kk))
mod = M.MasteryModel().fit(u, q, y, Q)
p_tr = mod.predict_proba(M.encode(u, q, y, Q))
ends = np.r_[np.flatnonzero(np.diff(u)), n - 1][:30]
gap = 0.0
for e in ends:
    b = (u == u[e]) & (np.arange(n) < e)
    st = M.state(u[b], q[b], y[b], Q, students=[u[e]])
    gap = max(gap, abs(mod.question_chance(st, [q[e]])[0, 0] - p_tr[e]))
ok("the app's prediction for the next answer equals the trained model's (same features)", gap < 1e-9, f"{gap:.2e}")
Qs = sparse.csr_matrix(np.eye(K)[rng.integers(0, K, I)])                # one skill per question
ms = M.MasteryModel().fit(u, q, y, Qs)
st = M.state(u, q, y, Qs, students=np.arange(5))
qc = ms.question_chance(st)
want = np.zeros((5, K))
for k in range(K):
    qs_ = np.flatnonzero(Qs[:, k].toarray().ravel())
    z = np.log(qc[:, qs_] / (1 - qc[:, qs_])) - ms.w["dense"][0] * np.log1p(st.question_tries[:, qs_].toarray()) \
        - ms.w["dense"][1] * np.log1p(st.question_right[:, qs_].toarray())
    want[:, k] = 1 / (1 + np.exp(-z.mean(axis=1)))
ok("mastery is the chance on a typical question of the skill (its questions' mean log-odds)",
   np.allclose(ms.mastery(st), want, atol=1e-9))
mk = M.marks_per_skill([0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 1, 1], sparse.csr_matrix([[1.0]]))
ok("marks per skill: (right + 1) / (tries + 2) from earlier answers only", np.allclose(mk, [0.5, 2 / 3, 0.5, 0.5]),
   str(mk))
try:
    sdf, sQ = ktdata.load("statics")
except Exception as exc:                                                 # no network and no cache
    sdf = None
    print(f"  (Statics data not available, skipping the reproduction: {exc})")
if sdf is not None:
    su, sq, sy = sdf["user_id"].to_numpy(), sdf["item_id"].to_numpy(), sdf["correct"].to_numpy().astype(float)
    SX = M.encode(su, sq, sy, sQ)
    SXb = SX[:, M.best_lr_columns(M.Layout(*sQ.shape))]
    users = np.unique(su)
    fmap = dict(zip(users, np.random.default_rng(0).permutation(len(users)) % 5))
    fold = np.array([fmap[x] for x in su])
    aucs, mine = [], []
    for f in range(5):
        tr, te = fold != f, fold == f
        aucs.append(roc_auc_score(sy[te], LogisticRegression(max_iter=2000).fit(SXb[tr], sy[tr])
                                  .predict_proba(SXb[te])[:, 1]))
        if f == 0:
            mine.append(roc_auc_score(sy[te], LogisticRegression(max_iter=2000).fit(SX[tr], sy[tr])
                                      .predict_proba(SX[te])[:, 1]))
    ok("Best-LR reproduced on Statics: 5-fold AUC within 0.005 of the published 0.819",
       abs(np.mean(aucs) - 0.819) < 0.005, f"{np.mean(aucs):.4f}")
    ok("... and Mastery Map's features do better on the first fold", mine[0] > aucs[0], f"{mine[0]:.4f} vs {aucs[0]:.4f}")

print("Class workbook")
c = demo.demo()
back = cl.from_excel(cl.to_excel(c))
ok("Excel round trip keeps students, skills, questions, answers and today",
   back.student_ids == c.student_ids and back.skill_ids == c.skill_ids and back.question_ids == c.question_ids
   and len(back.answers) == len(c.answers) and back.today == c.today
   and np.array_equal(c.coded()[2], back.coded()[2]))
ok("right/wrong read from 1/0, yes/no, right/wrong, true/false",
   cl.parse_correct(pd.Series([1, 0, "yes", "No", "right", "WRONG", True, "x"])).tolist() ==
   [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
bad = cl.Classroom("b", c.today, c.students, c.skills, c.questions.copy(), c.answers.copy())
bad.questions.loc[0, "Skills"] = "ZZ9"
bad.answers["Correct"] = bad.answers["Correct"].astype(object)
bad.answers.loc[3, "Correct"] = "maybe"
stop, _ = cl.check(bad)
ok("problems are reported: a skill that does not exist, an unreadable right/wrong",
   any("ZZ9" in p for p in stop) and any("Correct" in p for p in stop), str(stop))
tiny = cl.Classroom("t", pd.Timestamp("2026-03-10"),
                    pd.DataFrame({"Student ID": ["A", "B"], "Name": ["a", "b"], "Group": ["", ""]}),
                    pd.DataFrame({"Skill ID": ["K"], "Skill": ["k"], "Topic": ["T"], "Prerequisites": [""],
                                  "Order": [1]}),
                    pd.DataFrame({"Question ID": ["q1", "q2"], "Skills": ["K", "K"], "Level": [1, 1],
                                  "Question": ["", ""], "Answer": ["", ""]}),
                    pd.DataFrame({"Date": ["2026-03-05", "2026-03-01", "2026-03-05", "2026-03-02"],
                                  "Student ID": ["A", "A", "A", "B"], "Question ID": ["q2", "q1", "q1", "q2"],
                                  "Correct": [1, 0, 0, 1]}))
s_, q_, y_, d_ = tiny.coded()
ok("answers in time order per student; same-day answers keep their sheet order",
   s_.tolist() == [0, 0, 0, 1] and q_.tolist() == [0, 1, 0, 1] and y_.tolist() == [0, 1, 0, 1])
ok("'until' keeps only answers before that date", len(tiny.coded("2026-03-05")[0]) == 2)

print("Questions")
bank = questions.bank(per_level=4)
cnt = pd.Series([r[1] for r in bank]).value_counts()
ok("a bank of distinct questions with answers for every skill and level",
   set(cnt.index) == set(questions.SKILL_IDS) and cnt.min() >= 9 and all(r[3] and r[4] for r in bank)
   and len({r[3] for r in bank}) == len(bank))
okk = True
for sd in range(40):
    t, a = questions.make("N2", 1, sd)
    p_, x_ = map(int, re.findall(r"\d+", t)[:2])
    okk &= float(a.replace(",", "")) == p_ * x_ / 100
    t, a = questions.make("A6", 3, sd)
    nums = list(map(int, re.findall(r"-?\d+", t)))
    xs, ys = map(int, re.findall(r"-?\d+", a))
    okk &= nums[0] * xs + nums[1] * ys == nums[2] and nums[3] * xs - nums[4] * ys == nums[5]
    t, a = questions.make("G2", 2, sd)
    h, s1 = map(int, re.findall(r"\d+", t)[:2])
    okk &= int(a.split()[0]) ** 2 + s1 ** 2 == h ** 2
ok("answers are right (percentages, simultaneous equations substituted back, Pythagoras)", okk)

print("Practice sets")
snap = plan.snapshot(c)
ps = [plan.practice_set(snap, c, i, 10) for i in range(len(c.students))]
taught = set(np.array(c.skill_ids)[snap.taught])
ok("every set: 10 different questions, taught skills only",
   all(len(p) == 10 and p["Question ID"].is_unique and set(p["Skill ID"]) <= taught for p in ps))
kpos = {k: j for j, k in enumerate(c.skill_ids)}
foc = pd.concat([p.assign(i=i) for i, p in enumerate(ps)])
foc = foc[foc["Kind"] == "focus"]
ok("focus questions are on skills not yet secure, about 70% of each set",
   all(snap.mastery[r.i, kpos[r._2]] < plan.SECURE for r in foc.itertuples()) and
   abs(len(foc) / (10 * len(ps)) - 0.7) < 0.08, f"{len(foc) / (10 * len(ps)):.2f}")
near = True
qpos = {qq: j for j, qq in enumerate(c.question_ids)}
for i, p in enumerate(ps[:12]):
    for k, grp in p[p["Kind"] == "focus"].groupby("Skill ID"):
        cand = np.flatnonzero(c.Q()[:, kpos[k]].toarray().ravel())
        fresh = cand[snap.done[i, cand] == 0]
        if len(fresh) >= 3:
            d = np.abs(snap.chance[i, fresh] - plan.TARGET["focus"])
            picked = np.abs(snap.chance[i, [qpos[x] for x in grp["Question ID"]]] - plan.TARGET["focus"])
            near &= picked.min() <= np.sort(d)[0] + 1e-9
ok("within a focus skill, the first question is the unseen one closest to a 70% chance", near)
pre = c.prerequisites()
skip = with_weak = pre_pick = dep_pick = 0
for i in range(len(c.students)):
    m = snap.mastery[i]
    f = set(ps[i].loc[ps[i]["Kind"] == "focus", "Skill ID"])
    for k, pl in pre.items():
        for p_ in pl:
            if snap.taught[k] and snap.taught[p_] and m[p_] < plan.WEAK:
                if c.skill_ids[k] in f:
                    with_weak += 1
                    skip += c.skill_ids[p_] not in f
                if m[k] < plan.WEAK:
                    pre_pick += c.skill_ids[p_] in f
                    dep_pick += c.skill_ids[k] in f
ok("prerequisites first: no set practises a skill but skips its weak prerequisite; when both are weak the "
   "prerequisite is picked far more often", skip == 0 and pre_pick > 5 * max(dep_pick, 1),
   f"skipped {skip}/{with_weak}, picked {pre_pick} vs {dep_pick}")
cp = plan.class_plan(snap, c, 10)
st_ = plan.status(snap.mastery, snap.tries)
ok("whole-class topics are weak for at least 40% of the students; everybody gets a set",
   len(cp["sets"]) == len(c.students) and all((st_[:, kpos[k]] == "weak").mean() >= 0.4 for k, _, _ in
                                              cp["whole_class"]))

print("Term replay and what if")
rp = term.replay(c)
w5 = rp["weeks"][5]
c2 = cl.Classroom(c.name, c.today, c.students, c.skills, c.questions, c.answers.copy())
late = pd.to_datetime(c2.answers["Date"]) >= w5
c2.answers.loc[late, "Correct"] = 1 - c2.answers.loc[late, "Correct"]
rp2 = term.replay(c2)
ok("the replay never looks ahead: other answers from a week on leave every earlier week unchanged",
   np.allclose(rp["mastery"][:6], rp2["mastery"][:6]) and not np.allclose(rp["mastery"][6:], rp2["mastery"][6:]))
before = c.extra["truth"].done.copy()
w1 = term.what_if(c, weeks=2, n=10, seed=4)
w2 = term.what_if(c, weeks=2, n=10, seed=4)
ok("what if: the demo's simulated students are copied, not changed; the same seed gives the same result",
   np.array_equal(before, c.extra["truth"].done) and np.allclose(w1["personal"], w2["personal"])
   and np.allclose(w1["same"], w2["same"]))
sm = term.summary(w1)
ok("what if: both ways start from the same state; practice raises the expected score",
   np.isclose(sm["same"]["score"][0], sm["personal"]["score"][0]) and sm["same"]["score"][-1] > sm["same"]["score"][0]
   and sm["personal"]["score"][-1] > sm["personal"]["score"][0])

print("Demo class")
d1, d2 = demo.demo(), demo.demo()
ok("the demo class is the same on every run", d1.students.equals(d2.students) and d1.answers.equals(d2.answers))
tr = c.extra["truth"]
tk = snap.taught
true = tr.chance_skill()[:, tk]
est = snap.mastery[:, tk]
marks = ((snap.right + 1) / (snap.tries + 2))[:, tk]
r_e, r_m = np.corrcoef(est.ravel(), true.ravel())[0, 1], np.corrcoef(marks.ravel(), true.ravel())[0, 1]
ok("on the demo, the estimates follow the hidden truth closer than marks per skill", r_e > r_m + 0.03 and r_e > 0.8,
   f"{r_e:.3f} vs {r_m:.3f}")
we, wm, wt = est < 0.5, marks < 0.5, true < 0.5
ok("... and the weak spots it flags are more often real", (we & wt).sum() / we.sum() > (wm & wt).sum() / wm.sum())

print("Sheets and animations")
names = dict(zip(c.student_ids, c.students["Name"]))
pdf = report.worksheets({k: cp["sets"][k].head(3) for k in c.student_ids[:5]}, names, c.name, c.today)
pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf))
ok("practice sheets: each student on a page of their own (even short sets), then the answer key",
   pdf[:4] == b"%PDF" and pages == 6, f"{pages} pages")
xl = pd.read_excel(__import__("io").BytesIO(report.sets_excel(cp["sets"], names)))
ok("the Excel copy lists every question of every set", len(xl) == sum(len(v) for v in cp["sets"].values()))
evil = "</script><img src=x onerror=alert(1)>"
doc = anim.skill_map(["K1"], [evil], ["T"], {0: []}, [("w", [0.5], [3])])
doc2 = anim.heat_replay([evil], ["K1"], ["k"], np.full((1, 1, 1), 0.5), ["w"], np.ones((1, 1), bool))
ok("names inside the animations cannot break out of them", "</script><img" not in doc and "</script><img" not in doc2)

print(f"\n{passed} passed, {len(failed)} failed")
if failed:
    print("FAILED: " + "; ".join(failed))
sys.exit(1 if failed else 0)
