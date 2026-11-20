"""
A fictional class for trying the app: 36 students (generated names), the 23 skills of masterymap.questions, a bank of
practice questions, and a 16-week term of weekly quizzes ending the Saturday before "today".

The answers come from a simple learner model whose truth is kept (Classroom.extra["truth"]), so the app can show how
close its estimates are to the real (simulated) state of each student, and what-if questions can be run on it:

    chance of a right answer = sigmoid(ability + skill gap + skill easiness + question easiness
                                       + 1.1 once the topic is taught + 0.45 * log(1 + questions done on the skill)
                                       - 0.7 for each prerequisite not yet secure)

Students who miss the week a topic is taught start that topic with an extra gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import questions as qs
from .classroom import Classroom

FIRST = ["Kavindi", "Tharindu", "Sachini", "Malith", "Dilini", "Ravindu", "Sanduni", "Pasindu", "Chamodi", "Yasiru",
         "Nethmi", "Thilina", "Senuri", "Akila", "Imesha", "Shehan", "Kaveesha", "Oshadi", "Lahiru", "Hasini",
         "Fathima", "Arjun", "Priya", "Nirosha", "Ishara", "Dinuka", "Amaya", "Sahan", "Ruwani", "Kasun", "Methmi",
         "Ayesha", "Vinod", "Gayani", "Tharushi", "Nadun", "Pramodi", "Hiruni", "Janith", "Sewmini"]
LAST = ["Perera", "Fernando", "Silva", "Bandara", "Herath", "Dissanayake", "Weerasinghe", "Gunasekara",
        "Karunaratne", "Liyanage", "Mendis", "Pathirana", "Samarasinghe", "Abeysekara", "Rodrigo", "Nadarajah",
        "Sivakumar", "Ismail", "Kumarasinghe", "Ekanayake", "Wanniarachchi", "Senaratne"]
EASE = {"N1": .6, "N2": .5, "N3": .4, "N4": .3, "N5": -.1, "N6": .4, "N7": -.4, "A1": .7, "A2": .4, "A3": -.1,
        "A4": -.6, "A5": .5, "A6": 0, "A7": -.5, "A8": .2, "G1": .7, "G2": .3, "G3": .5, "G4": 0, "G5": -.4,
        "S1": .7, "S2": .2, "S3": .4}
LEVEL_EASE = {1: 0.9, 2: 0.0, 3: -0.9}
TAUGHT, LEARN, PREREQ, MISSED = 1.1, 0.45, 0.7, 1.0
SECURE = 0.6            # a prerequisite counts as secure from this chance on a medium question


class Truth:
    """The simulated learners: their real chance of answering any question right, given what they have done."""

    def __init__(self, ability, gap, skill_ids, prereq, item_skill, item_ease):
        self.ability, self.gap = ability, gap                 # (students,), (students, skills)
        self.skill_ids, self.prereq = skill_ids, prereq
        self.item_skill, self.item_ease = item_skill, item_ease
        self.ease = np.array([EASE[k] for k in skill_ids])
        n, K = gap.shape
        self.done = np.zeros((n, K))                          # questions done per skill
        self.taught = np.zeros(K, bool)

    def logit_skill(self):
        """(students, skills) log-odds on a medium question of each skill, now."""
        base = self.ability[:, None] + self.gap + self.ease[None, :] + TAUGHT * self.taught[None, :] - TAUGHT \
            + LEARN * np.log1p(self.done)
        out = base.copy()
        for k in range(base.shape[1]):                        # prerequisites come earlier in the syllabus
            for p in self.prereq[k]:
                out[:, k] -= PREREQ * (1 / (1 + np.exp(-out[:, p])) < SECURE)
        return out

    def chance_skill(self):
        return 1 / (1 + np.exp(-self.logit_skill()))

    def chance(self, students, items):
        z = self.logit_skill()[students, self.item_skill[items]] + self.item_ease[items]
        return 1 / (1 + np.exp(-z))

    def answer(self, students, items, rng):
        """Students answer items (in order); doing a question teaches its skill."""
        out = np.zeros(len(students))
        for j, (s, i) in enumerate(zip(students, items)):
            out[j] = float(rng.random() < self.chance([s], [i])[0])
            self.done[s, self.item_skill[i]] += 1
        return out

    def copy(self):
        t = Truth(self.ability, self.gap.copy(), self.skill_ids, self.prereq, self.item_skill, self.item_ease)
        t.done, t.taught = self.done.copy(), self.taught.copy()
        return t


def demo(today="2026-11-02", seed=7, n_students=36, weeks=16):
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(today)
    names = []
    while len(names) < n_students:                       # a list, not a set: the same order on every run
        nm = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if nm not in names and nm.split()[0] not in {x.split()[0] for x in names}:
            names.append(nm)
    students = pd.DataFrame({"Student ID": [f"S{i + 1:02d}" for i in range(n_students)], "Name": names,
                             "Group": ["Saturday" if i % 2 == 0 else "Sunday" for i in range(n_students)]})
    skills = pd.DataFrame({"Skill ID": [s[0] for s in qs.SKILLS], "Skill": [s[1] for s in qs.SKILLS],
                           "Topic": [s[2] for s in qs.SKILLS], "Prerequisites": [", ".join(s[3]) for s in qs.SKILLS],
                           "Order": range(1, len(qs.SKILLS) + 1)})
    bank = qs.bank(per_level=4, seed=0)
    questions = pd.DataFrame(bank, columns=["Question ID", "Skills", "Level", "Question", "Answer"])
    K = len(skills)
    kpos = {k: j for j, k in enumerate(skills["Skill ID"])}
    prereq = {kpos[s[0]]: [kpos[p] for p in s[3]] for s in qs.SKILLS}
    item_skill = questions["Skills"].map(kpos).to_numpy()
    item_ease = questions["Level"].map(LEVEL_EASE).to_numpy() + rng.normal(0, 0.3, len(questions))
    ability = rng.normal(0, 0.9, n_students)
    gap = rng.normal(0, 0.7, (n_students, K))
    truth = Truth(ability, gap, list(skills["Skill ID"]), prereq, item_skill, item_ease)
    # the term: one quiz every Saturday; the syllabus taught in order, about 1.5 skills a week
    last_sat = today - pd.Timedelta(days=(today.weekday() - 5) % 7 or 7)
    quiz_days = [last_sat - pd.Timedelta(weeks=w) for w in range(weeks - 1, -1, -1)]
    plan = np.array_split(np.arange(K), weeks)
    absent_rate = np.where(rng.random(n_students) < 0.2, 0.25, 0.07)
    rows = []
    by_skill = {k: np.flatnonzero(item_skill == k) for k in range(K)}
    for w, day in enumerate(quiz_days):
        here = rng.random(n_students) > absent_rate
        for k in plan[w]:
            truth.taught[k] = True
            truth.gap[~here, k] -= MISSED                      # missed the lesson
        new = list(plan[w])
        seen = [k for k in range(K) if truth.taught[k] and k not in new]
        for s in np.flatnonzero(here):
            ks = [int(rng.choice(new)) for _ in range(7)] + \
                ([int(rng.choice(seen)) for _ in range(5)] if seen else [int(rng.choice(new)) for _ in range(5)])
            items = [int(rng.choice(by_skill[k])) for k in ks]
            res = truth.answer([s] * len(items), items, rng)
            rows += [(day, students.at[s, "Student ID"], questions.at[i, "Question ID"], int(c))
                     for i, c in zip(items, res)]
    answers = pd.DataFrame(rows, columns=["Date", "Student ID", "Question ID", "Correct"])
    c = Classroom("Grade 11 Mathematics, weekend class (fictional)", today, students, skills, questions, answers)
    c.extra["truth"] = truth
    return c
