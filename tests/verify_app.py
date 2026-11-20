"""
App checks: clicks through every tab of the Streamlit app (Streamlit's AppTest, no browser) with the demo class.

    python tests/verify_app.py     # about 1 minute
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

passed, failed = 0, []


def ok(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}  {detail}")


def button(at, label):
    return [b for b in at.button if b.label == label][0]


def metric(at, label):
    m = [m for m in at.metric if m.label == label]
    return m[0] if m else None


def errors(at):
    return [str(e.value)[:300] for e in at.exception]


at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900)
at.run()
ok("app starts without errors", not errors(at), str(errors(at)))
ok("eight tabs", [t.label for t in at.tabs][:8] == ["📥 Data", "🗺️ Class", "🧑‍🎓 Student", "📝 Practice sets",
                                                    "🎬 Term replay", "🔮 What if", "🧪 Benchmark", "ℹ️ How it works"])
ok("before a class is loaded: ask to load one", any("Load the demo" in i.value for i in at.info))
ok("the template can be downloaded", any("template" in d.label for d in at.get("download_button")))

button(at, "Load the demo class").click().run()
ok("demo loads; every tab renders without errors", not errors(at), str(errors(at)))
ok("36 students, 23 skills, 273 questions", metric(at, "Students").value == "36" and
   metric(at, "Skills").value == "23" and metric(at, "Questions").value == "273")
ok("data check passed", any("workbook is complete" in s.value for s in at.success))
ok("practice sets: one per student, mostly on weak or developing skills",
   metric(at, "Practice sets").value == "36" and 60 <= int(metric(at, "On weak or developing skills").value[:-1]) <= 80)
labels = [d.label for d in at.get("download_button")]
ok("all practice sheets (PDF) and the Excel copy can be downloaded",
   any("All practice sheets" in x for x in labels) and any("All practice sets (Excel)" in x for x in labels), str(labels))
ok("the student tab offers that student's sheet", any(x.startswith("Practice sheet for") for x in labels))
ok("re-teach list and small groups shown", any("Re-teach" in m.value for m in at.markdown) and
   any("Small groups" in m.value for m in at.markdown))

at.slider(key="set_size").set_value(15).run()
ok("a larger set size gives larger sets", metric(at, "Questions in all").value == f"{36 * 15:,}" and not errors(at),
   metric(at, "Questions in all").value)
at.slider(key="set_size").set_value(10).run()
at.selectbox(key="pick").set_value(4).run()
ok("another student", not errors(at) and any("Next practice set" in m.value for m in at.markdown))

ok("what if waits for Run", any("Press ▶ Run" in i.value for i in at.info))
button(at, "▶ Run").click().run()
ok("what if runs without errors", not errors(at), str(errors(at)))
a, b = metric(at, "After 4 weeks: one paper for all"), metric(at, "After 4 weeks: personal sets")
ok("what if: both ways shown with their change", a is not None and b is not None and str(a.delta).startswith("+")
   and str(b.delta).startswith("+"))
at.slider(key="wi_weeks").set_value(6).run()
ok("changing the weeks asks for a new run", any("settings changed" in w.value for w in at.warning))
ok("benchmark tab renders with the results", not errors(at) and any("Predicting the next answer" in m.value
                                                                    for m in at.markdown))
ok("how it works explains the method", any("Best-LR" in m.value for m in at.markdown))

print(f"\n{passed} passed, {len(failed)} failed")
if failed:
    print("FAILED: " + "; ".join(failed))
sys.exit(1 if failed else 0)
