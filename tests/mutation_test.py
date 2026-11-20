"""
Mutation test: do the engine checks notice when the code is wrong?

Each mutant copies the package, plants one bug (features that peek at the answer being predicted, a replay that
looks ahead, practice sets that ignore prerequisites, a wrong answer key, ...) and runs verify_engine.py against the
copy. Every mutant must be caught.

    python tests/mutation_test.py            # about 20 minutes
    python tests/mutation_test.py replay     # only the mutants whose name contains "replay"
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MUTANTS = [
    ("features count the answer being predicted", "model.py",
     "    win = (g.cumsum() - y).to_numpy().astype(float)", "    win = g.cumsum().to_numpy().astype(float)"),
    ("recency counts the answer being predicted", "model.py",
     "        w[i], f[i] = cw, cf", "        w[i], f[i] = cw * d + ys[i], cf * d + 1.0 - ys[i]"),
    ("app's recency weights off by one answer", "model.py",
     '    back = (g["y"].transform("size") - 1 - g.cumcount()).to_numpy()',
     '    back = (g["y"].transform("size") - g.cumcount()).to_numpy()'),
    ("mastery forgets the typical question's easiness", "model.py",
     "        z = self.b0 + self._person(st)[:, None] + self._skill_terms(st) + self.typical[None, :]",
     "        z = self.b0 + self._person(st)[:, None] + self._skill_terms(st)"),
    ("Best-LR without question easiness", "model.py",
     '    keep = [np.arange(*B[k]) for k in ("question", "skill", "skill_tries", "skill_right")]',
     '    keep = [np.arange(*B[k]) for k in ("skill", "skill_tries", "skill_right")]'),
    ("marks per skill smoothed wrongly", "model.py",
     "    rate = (win + 1) / (att + 2)", "    rate = (win + 1) / (att + 1)"),
    ("answers not put in date order", "classroom.py",
     '.sort_values(["Student ID", "Date", "_row"], kind="stable")', '.sort_values(["Student ID", "_row"], kind="stable")'),
    ("'until' lets that day's answers in", "classroom.py",
     '        a = a[a["Date"] < pd.Timestamp(until)]', '        a = a[a["Date"] <= pd.Timestamp(until)]'),
    ("unknown skills not reported", "classroom.py",
     "    if bad:\n        stop.append(f\"Questions with a skill", "    if False:\n        stop.append(f\"Questions with a skill"),
    ("practice ignores prerequisites", "plan.py",
     "(SECURE - m) * (1 + 0.5 * block) * np.where(ready, 1, 0.4), 0)", "(SECURE - m), 0)"),
    ("sets barely focus on weak skills", "plan.py",
     "    n_focus = int(round(0.7 * n)) if focus else 0", "    n_focus = int(round(0.3 * n)) if focus else 0"),
    ("questions not chosen near the target chance", "plan.py",
     "            score = (np.abs(ch - TARGET[kind])", "            score = (0 * np.abs(ch - TARGET[kind])"),
    ("replay looks ahead", "term.py",
     "        s = plan.snapshot(c, until=w)", "        s = plan.snapshot(c)"),
    ("what if changes the demo's students", "term.py",
     '        self.truth = c.extra["truth"].copy()', '        self.truth = c.extra["truth"]'),
    ("names can break out of the animations", "anim.py",
     '    return json.dumps(obj).replace("</", "<\\\\/")', "    return json.dumps(obj)"),
    ("wrong answer to simultaneous equations", "questions.py",
     'f"{b * x - a * y}.", f"x = {x}, y = {y}")', 'f"{b * x + a * y}.", f"x = {x}, y = {y}")'),
    ("practice sheets run into each other", "report.py",
     "        story += part + [PageBreak()]", "        story += part"),
    ("demo answers do not depend on the student", "demo.py",
     "        z = self.logit_skill()[students, self.item_skill[items]] + self.item_ease[items]",
     "        z = self.logit_skill()[0, self.item_skill[items]] + self.item_ease[items]"),
]


def main():
    only = sys.argv[1:]
    todo = [m for m in MUTANTS if not only or any(w in m[0] for w in only)]
    caught = 0
    for name, fname, old, new in todo:
        tmp = tempfile.mkdtemp(prefix="mutant_")
        try:
            shutil.copytree(os.path.join(ROOT, "masterymap"), os.path.join(tmp, "masterymap"),
                            ignore=shutil.ignore_patterns("__pycache__"))
            if os.path.isdir(os.path.join(ROOT, "data_cache")):
                os.symlink(os.path.join(ROOT, "data_cache"), os.path.join(tmp, "data_cache"))
            path = os.path.join(tmp, "masterymap", fname)
            src = open(path, encoding="utf-8").read()
            assert src.count(old) == 1, f"{name}: the text to mutate is not found exactly once"
            open(path, "w", encoding="utf-8").write(src.replace(old, new))
            try:
                res = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tests", "verify_engine.py")],
                                     env=dict(os.environ, MM_ROOT=tmp), capture_output=True, text=True, timeout=1500)
            except subprocess.TimeoutExpired:
                print(f"  MISSED  {name:50s}  (checks did not finish)", flush=True)
                continue
            fails = [ln.strip()[6:] for ln in res.stdout.splitlines() if ln.strip().startswith("FAIL")]
            hit = res.returncode != 0
            caught += hit
            why = fails[0] if fails else (res.stderr.strip().splitlines() or ["crash"])[-1]
            print(f"  {'CAUGHT' if hit else 'MISSED'}  {name:50s}  {why[:90] if hit else ''}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{caught} of {len(todo)} mutants caught")
    sys.exit(0 if caught == len(todo) else 1)


if __name__ == "__main__":
    main()
