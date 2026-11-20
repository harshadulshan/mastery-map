"""
A class as the app sees it: students, skills (with prerequisites and syllabus order), questions (each tagged with the
skills it trains) and answers (date, student, question, right or wrong). One Excel workbook with four sheets.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

STUDENT_COLS = ["Student ID", "Name", "Group"]
SKILL_COLS = ["Skill ID", "Skill", "Topic", "Prerequisites", "Order"]
QUESTION_COLS = ["Question ID", "Skills", "Level", "Question", "Answer"]
ANSWER_COLS = ["Date", "Student ID", "Question ID", "Correct"]
TRUE = {"1", "1.0", "y", "yes", "true", "correct", "right", "✓", "t"}
FALSE = {"0", "0.0", "n", "no", "false", "wrong", "incorrect", "x", "✗", "f"}


def split_ids(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return [p for p in re.split(r"[,;/\s]+", str(v).strip()) if p]


@dataclass
class Classroom:
    name: str
    today: pd.Timestamp
    students: pd.DataFrame
    skills: pd.DataFrame
    questions: pd.DataFrame
    answers: pd.DataFrame
    extra: dict = field(default_factory=dict)          # the demo keeps its hidden truth here

    # -------------------------------------------------------------- indexing
    @property
    def student_ids(self):
        return self.students["Student ID"].astype(str).tolist()

    @property
    def skill_ids(self):
        return self.skills.sort_values("Order", kind="stable")["Skill ID"].astype(str).tolist()

    @property
    def question_ids(self):
        return self.questions["Question ID"].astype(str).tolist()

    def skill_name(self, k):
        return dict(zip(self.skills["Skill ID"].astype(str), self.skills["Skill"]))[k]

    def Q(self):
        """(questions, skills) 0/1 matrix in the order of question_ids and skill_ids."""
        kpos = {k: j for j, k in enumerate(self.skill_ids)}
        r, c = [], []
        for i, v in enumerate(self.questions["Skills"]):
            for k in split_ids(v):
                if k in kpos:
                    r.append(i)
                    c.append(kpos[k])
        return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(self.questions), len(kpos)))

    def prerequisites(self):
        """{skill index: [prerequisite skill indices]}"""
        kpos = {k: j for j, k in enumerate(self.skill_ids)}
        sk = self.skills.set_index(self.skills["Skill ID"].astype(str))
        return {kpos[k]: [kpos[p] for p in split_ids(sk.at[k, "Prerequisites"]) if p in kpos] for k in kpos}

    def coded(self, until=None):
        """Answers as integer arrays (student index, question index, correct, day), in time order within each
        student (rows on the same day keep their order in the sheet). until: only answers before that date."""
        a = self.answers.copy()
        a["Date"] = pd.to_datetime(a["Date"])
        if until is not None:
            a = a[a["Date"] < pd.Timestamp(until)]
        spos = {s: i for i, s in enumerate(self.student_ids)}
        qpos = {q: i for i, q in enumerate(self.question_ids)}
        a = a[a["Student ID"].astype(str).isin(spos) & a["Question ID"].astype(str).isin(qpos)]
        a = a.assign(_row=np.arange(len(a))).sort_values(["Student ID", "Date", "_row"], kind="stable")
        s = a["Student ID"].astype(str).map(spos).to_numpy(np.int64)
        q = a["Question ID"].astype(str).map(qpos).to_numpy(np.int64)
        order = np.lexsort((np.arange(len(s)), s))            # students in sheet order, time order inside
        day = (a["Date"] - a["Date"].min()).dt.days.to_numpy() if len(a) else np.zeros(0, int)
        return s[order], q[order], parse_correct(a["Correct"]).to_numpy(float)[order], day[order]

    def weeks(self):
        """Mondays from the week of the first answer to the week of today."""
        d = pd.to_datetime(self.answers["Date"])
        start = (d.min() - pd.Timedelta(days=d.min().weekday())).normalize()
        return pd.date_range(start, self.today, freq="7D")


def parse_correct(col):
    def one(v):
        if isinstance(v, (bool, np.bool_)):
            return float(v)
        if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
            return 1.0 if float(v) >= 0.5 else 0.0
        t = str(v).strip().lower()
        return 1.0 if t in TRUE else (0.0 if t in FALSE else np.nan)
    return pd.Series([one(v) for v in col], index=col.index, dtype=float)


def check(c: Classroom):
    """Problems that stop the app (first) and warnings, as lists of sentences."""
    stop, warn = [], []
    for nm, df, cols in (("Students", c.students, STUDENT_COLS[:2]), ("Skills", c.skills, SKILL_COLS[:2]),
                         ("Questions", c.questions, QUESTION_COLS[:2]), ("Answers", c.answers, ANSWER_COLS)):
        miss = [x for x in cols if x not in df.columns]
        if miss:
            stop.append(f"Sheet {nm}: missing column(s) {', '.join(miss)}.")
    if stop:
        return stop, warn
    for nm, df, col in (("Students", c.students, "Student ID"), ("Skills", c.skills, "Skill ID"),
                        ("Questions", c.questions, "Question ID")):
        dup = df[col].astype(str)[df[col].astype(str).duplicated()].unique()
        if len(dup):
            stop.append(f"Sheet {nm}: the same {col} twice: {', '.join(dup[:5])}.")
    skills = set(c.skills["Skill ID"].astype(str))
    bad = [f"{q} ({k})" for q, v in zip(c.questions["Question ID"].astype(str), c.questions["Skills"])
           for k in split_ids(v) if k not in skills]
    if bad:
        stop.append(f"Questions with a skill that is not on the Skills sheet: {', '.join(bad[:5])}.")
    none = c.questions.loc[[len(split_ids(v)) == 0 for v in c.questions["Skills"]], "Question ID"].astype(str)
    if len(none):
        stop.append(f"Questions without a skill: {', '.join(none[:5])}.")
    pre = [f"{k} ({p})" for k, v in zip(c.skills["Skill ID"].astype(str), c.skills.get("Prerequisites", []))
           for p in split_ids(v) if p not in skills]
    if pre:
        warn.append(f"Prerequisites that are not skills (ignored): {', '.join(pre[:5])}.")
    a = c.answers
    if not len(a):
        stop.append("The Answers sheet is empty.")
        return stop, warn
    dates = pd.to_datetime(a["Date"], errors="coerce")
    if dates.isna().any():
        stop.append(f"{int(dates.isna().sum())} answers without a readable date (e.g. 2026-08-15).")
    corr = parse_correct(a["Correct"])
    if corr.isna().any():
        stop.append(f"{int(corr.isna().sum())} answers whose Correct is not 1/0, yes/no or right/wrong.")
    us = set(a["Student ID"].astype(str)) - set(c.students["Student ID"].astype(str))
    if us:
        warn.append(f"Answers from students not on the Students sheet (ignored): {', '.join(sorted(us)[:5])}.")
    uq = set(a["Question ID"].astype(str)) - set(c.questions["Question ID"].astype(str))
    if uq:
        warn.append(f"Answers to questions not on the Questions sheet (ignored): {', '.join(sorted(uq)[:5])}.")
    if not stop and len(a) < 200:
        warn.append("Fewer than 200 answers: the estimates will be rough; they sharpen as answers come in.")
    return stop, warn


def to_excel(c: Classroom) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"Setting": ["Class", "Today"], "Value": [c.name, c.today.date().isoformat()]}) \
            .to_excel(xw, sheet_name="Class", index=False)
        c.students[STUDENT_COLS].to_excel(xw, sheet_name="Students", index=False)
        c.skills[SKILL_COLS].to_excel(xw, sheet_name="Skills", index=False)
        c.questions[QUESTION_COLS].to_excel(xw, sheet_name="Questions", index=False)
        a = c.answers[ANSWER_COLS].copy()
        a["Date"] = pd.to_datetime(a["Date"]).dt.date
        a.to_excel(xw, sheet_name="Answers", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                w = max(len(str(x.value or "")) for x in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 70)
            ws.freeze_panes = "A2"
    return buf.getvalue()


def from_excel(data) -> Classroom:
    x = pd.read_excel(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data, sheet_name=None)
    get = {k.strip().lower(): v for k, v in x.items()}
    meta = get.get("class", pd.DataFrame(columns=["Setting", "Value"]))
    m = dict(zip(meta.get("Setting", []), meta.get("Value", [])))
    students = get.get("students", pd.DataFrame(columns=STUDENT_COLS))
    if "Group" not in students:
        students["Group"] = ""
    skills = get.get("skills", pd.DataFrame(columns=SKILL_COLS))
    for col, v in (("Topic", ""), ("Prerequisites", ""), ("Order", None)):
        if col not in skills:
            skills[col] = v
    skills["Order"] = pd.to_numeric(skills["Order"], errors="coerce").fillna(pd.Series(range(len(skills)),
                                                                                         index=skills.index))
    questions = get.get("questions", pd.DataFrame(columns=QUESTION_COLS))
    for col in ("Level", "Question", "Answer"):
        if col not in questions:
            questions[col] = ""
    answers = get.get("answers", pd.DataFrame(columns=ANSWER_COLS))
    today = pd.Timestamp(m.get("Today")) if m.get("Today") is not None and not pd.isna(m.get("Today")) else \
        pd.to_datetime(answers["Date"], errors="coerce").max().normalize() + pd.Timedelta(days=1)
    for df, cols in ((students, ["Student ID"]), (skills, ["Skill ID"]), (questions, ["Question ID"]),
                     (answers, ["Student ID", "Question ID"])):
        for col in cols:
            if col in df:
                df[col] = df[col].astype(str).str.strip()
    return Classroom(str(m.get("Class", "My class")), pd.Timestamp(today).normalize(), students.reset_index(drop=True),
                     skills.reset_index(drop=True), questions.reset_index(drop=True), answers.reset_index(drop=True))
