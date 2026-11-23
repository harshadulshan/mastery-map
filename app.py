"""
Mastery Map
-----------
A tutor with forty students cannot see, from a mark book, who is weak at what: a 60% on a mixed test hides which
skills are missing, and the next worksheet goes to everybody. Mastery Map reads the answers to quizzes and homework,
estimates for every student how likely they are to get a typical question of each skill right now, finds the weak
spots (and the weak prerequisites behind them), and makes a personal practice set for each student, ready to print.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from masterymap import anim, plan, report, term
from masterymap import classroom as cl
from masterymap import demo as demomod

st.set_page_config(page_title="Mastery Map", page_icon="🗺️", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
DARK = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
C_BLUE, C_GREY, C_RED, C_GREEN, C_AMBER, C_VIOLET = "#2a78d6", "#9b9a95", "#e34948", "#1baf7a", "#eda100", "#6250d6"
STATUS_COL = {"weak": C_RED, "developing": C_AMBER, "secure": C_GREEN, "few answers": C_GREY}
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
SCALE = [[0, C_RED], [0.35, C_RED], [0.6, C_AMBER], [0.85, C_GREEN], [1, C_GREEN]]   # 0.3 .. 1.0

ss = st.session_state
for key, default in {"cls": None, "key": None, "set_size": 10, "pick": None, "wi_weeks": 4, "wi": None,
                     "wi_sig": None}.items():
    ss.setdefault(key, default)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def layout(fig, height=300, **kw):
    top = 56 if kw.get("title") else 16
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=top, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig


def html(doc, height):
    """An animated piece (our own HTML and JavaScript; class data inside it is escaped)."""
    if hasattr(st, "iframe"):
        st.iframe(doc, height=height)
    else:
        components.html(doc, height=height, scrolling=False)


def load(c):
    ss.cls, ss.key, ss.pick, ss.wi, ss.wi_sig = c, uuid.uuid4().hex, None, None, None


def problems(c):
    return cl.check(c)


# cached by a key unique to each loaded workbook (never shared between users or files)
@st.cache_data(show_spinner="Estimating every student's mastery…", max_entries=6)
def _snap(_c, key):
    return plan.snapshot(_c)


@st.cache_data(show_spinner="Making the practice sets…", max_entries=12)
def _plan(_c, key, n):
    return plan.class_plan(_snap(_c, key), _c, n)


@st.cache_data(show_spinner=False, max_entries=6)
def _replay(_c, key):
    bar = st.progress(0.0, "Replaying the term week by week")
    out = term.replay(_c, progress=lambda f, m: bar.progress(min(f, 1.0), m))
    bar.empty()
    return out


def names_of(c):
    return dict(zip(c.student_ids, c.students["Name"].astype(str)))


def student_label(c, i):
    return f"{c.student_ids[i]} · {c.students['Name'].iloc[i]}"


def skill_frames(c, rp, rows=None):
    """Frames for the skill map from the replay: class average (rows None) or one student (rows=[i])."""
    out = []
    for j, w in enumerate(rp["weeks"]):
        m = rp["mastery"][j] if rows is None else rp["mastery"][j][rows]
        n = rp["tries"][j].sum(axis=0) if rows is None else rp["tries"][j][rows].sum(axis=0)
        tk = rp["taught"][j]
        lab = "today" if j == len(rp["weeks"]) - 1 else f"week of {w:%d %b}"
        out.append((lab, [float(m[:, k].mean()) if tk[k] else None for k in range(m.shape[1])], list(n)))
    return out


def map_height(c):
    from masterymap.anim import map_layout
    sk = c.skill_ids
    return int(map_layout(sk, c.skills.set_index("Skill ID").loc[sk, "Topic"].astype(str).tolist())[3] + 96)


def ready(c):
    if c is None:
        return False
    stop, _ = problems(c)
    return not stop


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗺️ Mastery Map")
    st.caption("Who is weak at what, why, and what each student should practise next.")
    st.divider()
    side_steps = st.container()
    st.divider()
    st.markdown("**Practice sets**")
    st.slider("Questions per student", 5, 20, key="set_size",
              help="About 70% go to the student's weakest skills whose prerequisites are in place, one checks a "
                   "skill with too few answers, the rest review secure skills not practised lately.")
    st.divider()
    st.caption("Mastery: a logistic model of each student's answers (Best-LR, Gervet et al. 2020, plus recency). "
               "See **ℹ️ How it works**.")

st.title("🗺️ Mastery Map")
st.caption("From quiz and homework answers: **how likely each student is to get each skill right now**, the weak "
           "spots and the weak prerequisites behind them, and **a personal practice set for every student**, ready "
           "to print.")
tabs = st.tabs(["📥 Data", "🗺️ Class", "🧑‍🎓 Student", "📝 Practice sets", "🎬 Term replay", "🔮 What if",
                "🧪 Benchmark", "ℹ️ How it works"])

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Demo: a Grade 11 maths class**")
        st.caption("36 students, 23 O/L-style skills with prerequisites, 273 practice questions, 16 weeks of "
                   "Saturday quizzes. Fictional class, simulated answers.")
        if st.button("Load the demo class", type="primary", width="stretch"):
            load(demomod.demo())
            st.rerun()
    with c2:
        st.markdown("**Your class**")
        st.caption("One Excel workbook: Students, Skills (with prerequisites), Questions (each tagged with its "
                   "skills) and Answers (date, student, question, right or wrong).")
        up = st.file_uploader("Workbook", type=["xlsx"], label_visibility="collapsed")
        if up is not None and st.button(f"Use {up.name}", type="primary", width="stretch"):
            try:
                load(cl.from_excel(up.getvalue()))
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read {up.name}: {exc}")
    with c3:
        st.markdown("**Template**")
        st.caption("The demo class as a workbook: keep the sheet and column names, replace the rows with yours. "
                   "Correct can be 1/0, yes/no or right/wrong.")
        st.download_button("Download the template (Excel)", lambda: cl.to_excel(demomod.demo()),
                           "mastery_map_template.xlsx", XLSX, width="stretch")
    c = ss.cls
    if c is None:
        st.info("Load the demo class or your own workbook to begin.")
    else:
        st.divider()
        st.subheader(c.name)
        stop, warn = problems(c)
        a = c.answers
        m = st.columns(5)
        m[0].metric("Students", len(c.students))
        m[1].metric("Skills", len(c.skills))
        m[2].metric("Questions", len(c.questions))
        m[3].metric("Answers", f"{len(a):,}")
        m[4].metric("Right", f"{cl.parse_correct(a['Correct']).mean():.0%}" if len(a) and not stop else "–")
        for p in stop:
            st.error(p)
        for p in warn:
            st.warning(p)
        if not stop:
            st.success("✅ The workbook is complete: every answer has a student, a question with skills, a date and "
                       "a right/wrong mark.")
            d = a.assign(Date=pd.to_datetime(a["Date"]), ok=cl.parse_correct(a["Correct"]))
            wk = d.groupby(d["Date"].dt.to_period("W-SUN").dt.start_time).agg(n=("ok", "size"), right=("ok", "mean"))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=wk.index, y=wk["n"], name="answers", marker=dict(color=C_BLUE, cornerradius=4),
                                 hovertemplate="week of %{x|%d %b}: %{y} answers<extra></extra>"))
            fig.add_trace(go.Scatter(x=wk.index, y=wk["right"] * 100, name="% right", yaxis="y2",
                                     line=dict(color=C_AMBER, width=2.5), mode="lines+markers",
                                     hovertemplate="%{y:.0f}% right<extra></extra>"))
            layout(fig, 260, title="Answers each week",
                   yaxis2=dict(overlaying="y", side="right", range=[0, 100], dtick=25, showgrid=False, ticksuffix="%"))
            st.plotly_chart(fig, width="stretch")
            with st.expander("The sheets"):
                for nm, df in (("Students", c.students), ("Skills", c.skills), ("Questions", c.questions),
                               ("Answers (latest first)", c.answers.iloc[::-1])):
                    st.markdown(f"**{nm}**")
                    st.dataframe(df.head(300), hide_index=True, width="stretch", height=220)

# ---------------------------------------------------------------------------
# 2. Class
# ---------------------------------------------------------------------------
with tabs[1]:
    c = ss.cls
    if not ready(c):
        st.info("Load a class first (📥 Data).")
    else:
        sn = _snap(c, ss.key)
        st_ = plan.status(sn.mastery, sn.tries)
        tk = sn.taught
        weak_n = ((st_ == "weak") & tk[None, :]).sum(axis=1)
        sec_n = ((st_ == "secure") & tk[None, :]).sum(axis=1)
        cp = _plan(c, ss.key, ss.set_size)
        html(anim.kpi_cards([
            ("Skills taught so far", int(tk.sum()), "", f" of {len(tk)}", 0, "with answers from the class", "accent"),
            ("Secure skills per student", float(sec_n.mean()), "", "", 1, "80% chance or more on a typical question",
             "ok"),
            ("Weak spots per student", float(weak_n.mean()), "", "", 1, "skills below 60%, with 3+ answers", "bad"),
            ("Students with 5+ weak spots", int((weak_n >= 5).sum()), "", f" of {len(weak_n)}", 0,
             "candidates for extra help", "mid"),
            ("Skills to re-teach", len(cp["whole_class"]), "", "", 0, "weak for 40% of the class or more",
             "violet")], DARK), 150)
        st.markdown("#### The class on the map · how the skills grew over the term")
        rp = _replay(c, ss.key)
        sk = c.skill_ids
        sk_names = [c.skill_name(k) for k in sk]
        topics = c.skills.set_index("Skill ID").loc[sk, "Topic"].astype(str).tolist()
        html(anim.skill_map(sk, sk_names, topics, c.prerequisites(), skill_frames(c, rp), dark=DARK),
             map_height(c))
        st.caption("Class average chance of answering a typical question of each skill right, week by week. Arrows "
                   "run from each prerequisite to the skill that needs it.")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Re-teach to the whole class**")
            if cp["whole_class"]:
                for k, nm, share in cp["whole_class"]:
                    st.markdown(f"- **{nm}** ({k}): weak for {share:.0%} of the students")
            else:
                st.caption("No skill is weak for 40% of the class or more.")
        with g2:
            st.markdown("**Small groups** · students whose first focus is the same skill")
            for k, nm, who in cp["groups"][:6]:
                st.markdown(f"- **{nm}** · {len(who)} student{'s' if len(who) > 1 else ''}: "
                            + ", ".join(names_of(c)[s].split()[0] for s in who[:9]) + ("…" if len(who) > 9 else ""))
        st.markdown("#### Every student, every skill")
        order = np.argsort(-sn.mastery[:, tk].mean(axis=1)) if tk.any() else np.arange(len(c.students))
        z = np.where(tk[None, :], sn.mastery, np.nan)[order]
        lab = [f"{c.students['Name'].iloc[i]}" for i in order]
        txt = np.where(np.isnan(z), "", np.char.mod("%.0f", np.nan_to_num(z) * 100))
        few = (sn.tries[order] < plan.EVIDENCE) & tk[None, :]
        txt = np.where(few & ~np.isnan(z), np.char.add(txt, "·"), txt)
        fig = go.Figure(go.Heatmap(
            z=z, x=sk, y=lab, zmin=0.3, zmax=1.0, colorscale=SCALE, text=txt, texttemplate="%{text}",
            textfont=dict(size=10), xgap=2, ygap=2, colorbar=dict(title="chance", tickformat=".0%", len=0.6),
            customdata=np.dstack([np.tile(sk_names, (len(order), 1)), sn.tries[order].astype(int)]),
            hovertemplate="%{y} · %{customdata[0]}<br>%{z:.0%} chance · %{customdata[1]} answers<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(side="top")
        layout(fig, 140 + 19 * len(order))
        st.plotly_chart(fig, width="stretch")
        st.caption("Students sorted from strongest to weakest. A dot after the number: fewer than 3 answers on that "
                   "skill, so the estimate leans on the student's other work. Grey: not taught yet.")

# ---------------------------------------------------------------------------
# 3. Student
# ---------------------------------------------------------------------------
with tabs[2]:
    c = ss.cls
    if not ready(c):
        st.info("Load a class first (📥 Data).")
    else:
        sn = _snap(c, ss.key)
        n = len(c.students)
        if ss.pick is None or ss.pick >= n:                 # start with the student with most weak spots
            ss.pick = int(np.argmax((plan.status(sn.mastery, sn.tries)[:, sn.taught] == "weak").sum(axis=1)))
        i = st.selectbox("Student", list(range(n)), format_func=lambda k: student_label(c, k), key="pick")
        tk = sn.taught
        stt = plan.status(sn.mastery[i], sn.tries[i])
        s_, q_, y_, _ = c.coded()
        mine = s_ == i
        html(anim.kpi_cards([
            ("Average chance, taught skills", float(sn.mastery[i, tk].mean() * 100) if tk.any() else 0, "", "%", 0,
             f"class average {sn.mastery[:, tk].mean():.0%}" if tk.any() else "", "accent"),
            ("Secure skills", int(((stt == "secure") & tk).sum()), "", f" of {int(tk.sum())}", 0, "80% and up", "ok"),
            ("Weak spots", int(((stt == "weak") & tk).sum()), "", "", 0, "below 60%, 3+ answers", "bad"),
            ("Answers so far", int(mine.sum()), "", "", 0, f"{y_[mine].mean():.0%} right" if mine.any() else "",
             "violet")], DARK), 150)
        rp = _replay(c, ss.key)
        sk = c.skill_ids
        sk_names = [c.skill_name(k) for k in sk]
        topics = c.skills.set_index("Skill ID").loc[sk, "Topic"].astype(str).tolist()
        g1, g2 = st.columns([3, 2])
        with g1:
            st.markdown(f"**{c.students['Name'].iloc[i]} on the map** · week by week")
            html(anim.skill_map(sk, sk_names, topics, c.prerequisites(), skill_frames(c, rp, [i]), dark=DARK),
                 map_height(c))
        with g2:
            order = np.argsort(np.where(tk, sn.mastery[i], 9))
            order = [k for k in order if tk[k]]
            fig = go.Figure(go.Bar(
                y=[sk_names[k] for k in order], x=[sn.mastery[i, k] * 100 for k in order], orientation="h",
                marker=dict(color=[STATUS_COL[stt[k]] for k in order], cornerradius=4),
                text=[f"{sn.mastery[i, k]:.0%} · {int(sn.tries[i, k])} ans." for k in order], textposition="outside",
                hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
            fig.add_vline(x=60, line=dict(color=C_RED, dash="dot", width=1))
            fig.add_vline(x=80, line=dict(color=C_GREEN, dash="dot", width=1))
            fig.update_xaxes(range=[0, 118], ticksuffix="%")
            fig.update_yaxes(autorange="reversed")
            layout(fig, 90 + 22 * len(order), title="Chance on a typical question, weakest first")
            st.plotly_chart(fig, width="stretch")
            st.caption("Red: weak · amber: developing · green: secure · grey: fewer than 3 answers.")
        st.markdown("**Every answer** · one row per skill, right ● and wrong ✕")
        if mine.any():
            d0 = pd.to_datetime(c.answers["Date"]).min()
            _, _, _, day = c.coded()
            Qm = c.Q()
            rows = []
            for q, yv, dd in zip(q_[mine], y_[mine], day[mine]):
                for k in Qm[q].indices:
                    rows.append((d0 + pd.Timedelta(days=int(dd)), sk_names[k], yv, c.question_ids[q]))
            ev = pd.DataFrame(rows, columns=["date", "skill", "right", "q"])
            fig = go.Figure()
            for val, nm, sym, colr in ((1.0, "right", "circle", C_GREEN), (0.0, "wrong", "x", C_RED)):
                e = ev[ev["right"] == val]
                fig.add_trace(go.Scatter(x=e["date"], y=e["skill"], mode="markers", name=nm,
                                         marker=dict(symbol=sym, size=9, color=colr, line=dict(width=0)),
                                         customdata=e["q"], hovertemplate="%{x|%d %b}: %{customdata} (" + nm
                                                                          + ")<extra></extra>"))
            fig.update_yaxes(categoryorder="array", categoryarray=[sk_names[k] for k in range(len(sk))][::-1])
            layout(fig, 120 + 19 * ev["skill"].nunique())
            st.plotly_chart(fig, width="stretch")
        ps = _plan(c, ss.key, ss.set_size)["sets"][c.student_ids[i]]
        st.markdown(f"**Next practice set** · {len(ps)} questions")
        st.dataframe(ps[["Question", "Skill", "Kind", "Chance", "Why"]], hide_index=True, width="stretch",
                     column_config={"Chance": st.column_config.ProgressColumn("Chance", format="percent", min_value=0,
                                                                              max_value=1)})
        sid = c.student_ids[i]
        st.download_button(f"Practice sheet for {c.students['Name'].iloc[i]} (PDF)",
                           lambda: report.worksheets({sid: ps}, names_of(c), c.name, c.today),
                           f"practice_{sid}.pdf", "application/pdf")

# ---------------------------------------------------------------------------
# 4. Practice sets
# ---------------------------------------------------------------------------
with tabs[3]:
    c = ss.cls
    if not ready(c):
        st.info("Load a class first (📥 Data).")
    else:
        cp = _plan(c, ss.key, ss.set_size)
        sets = cp["sets"]
        nm = names_of(c)
        allq = pd.concat([df.assign(Student=sid) for sid, df in sets.items()], ignore_index=True)
        m = st.columns(4)
        m[0].metric("Practice sets", len(sets))
        m[1].metric("Questions in all", f"{len(allq):,}")
        m[2].metric("On weak or developing skills", f"{(allq['Kind'] == 'focus').mean():.0%}")
        m[3].metric("Average chance of a right answer", f"{allq['Chance'].mean():.0%}",
                    help="Questions are chosen near 70%: hard enough to learn from, easy enough to get right most "
                         "of the time.")
        d1, d2 = st.columns(2)
        d1.download_button("📄 All practice sheets + answer key (PDF)",
                           lambda: report.worksheets(sets, nm, c.name, c.today), "practice_sets.pdf",
                           "application/pdf", type="primary", width="stretch")
        d2.download_button("📊 All practice sets (Excel)", lambda: report.sets_excel(sets, nm), "practice_sets.xlsx",
                           XLSX, width="stretch")
        g1, g2 = st.columns([3, 2])
        with g1:
            f = allq[allq["Kind"] == "focus"].groupby("Skill")["Student"].nunique().sort_values()
            fig = go.Figure(go.Bar(y=f.index, x=f.values, orientation="h", marker=dict(color=C_BLUE, cornerradius=4),
                                   text=f.values, textposition="outside",
                                   hovertemplate="%{y}: %{x} students<extra></extra>"))
            layout(fig, 90 + 22 * len(f), title="Students practising each skill as a focus")
            st.plotly_chart(fig, width="stretch")
        with g2:
            st.markdown("**Re-teach to the whole class**")
            for k, nm_, share in cp["whole_class"] or []:
                st.markdown(f"- **{nm_}**: weak for {share:.0%}")
            if not cp["whole_class"]:
                st.caption("None.")
            st.markdown("**Small groups** (first focus)")
            for k, nm_, who in cp["groups"]:
                st.markdown(f"- **{nm_}**: " + ", ".join(nm[s] for s in who))
        st.markdown("**Each student's set**")
        summ = pd.DataFrame([{"Student": f"{sid} · {nm[sid]}",
                              "Focus": ", ".join(dict.fromkeys(df.loc[df["Kind"] == "focus", "Skill"])),
                              "Check": ", ".join(df.loc[df["Kind"] == "check", "Skill"]),
                              "Review": ", ".join(dict.fromkeys(df.loc[df["Kind"] == "review", "Skill"])),
                              "Average chance": df["Chance"].mean()} for sid, df in sets.items()])
        st.dataframe(summ, hide_index=True, width="stretch", height=420,
                     column_config={"Average chance": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                                       max_value=1)})

# ---------------------------------------------------------------------------
# 5. Term replay
# ---------------------------------------------------------------------------
with tabs[4]:
    c = ss.cls
    if not ready(c):
        st.info("Load a class first (📥 Data).")
    else:
        rp = _replay(c, ss.key)
        sk = c.skill_ids
        sk_names = [c.skill_name(k) for k in sk]
        st.markdown("**The term again, as the app would have shown it each Monday** · the model is refitted on the "
                    "answers given before that day only")
        labs = ["today" if j == len(rp["weeks"]) - 1 else f"{w:%d %b}" for j, w in enumerate(rp["weeks"])]
        html(anim.heat_replay(c.students["Name"].astype(str).tolist(), sk, sk_names, rp["mastery"], labs,
                              rp["taught"], DARK), int(120 + 18 * len(c.students) + 120))
        topics = c.skills.set_index("Skill ID").loc[sk, "Topic"].astype(str).to_numpy()
        fig = go.Figure()
        pal = [C_BLUE, C_VIOLET, C_GREEN, C_AMBER, C_RED, C_GREY]
        for j, t in enumerate(dict.fromkeys(topics)):
            cols = np.flatnonzero(topics == t)
            ys = [float(rp["mastery"][w][:, cols[rp["taught"][w][cols]]].mean()) * 100
                  if rp["taught"][w][cols].any() else None for w in range(len(rp["weeks"]))]
            fig.add_trace(go.Scatter(x=rp["weeks"], y=ys, name=t, mode="lines+markers",
                                     line=dict(color=pal[j % len(pal)], width=2.5),
                                     hovertemplate=t + ": %{y:.0f}%<extra></extra>"))
        weak = [float(((rp["mastery"][w] < plan.WEAK) & (rp["tries"][w] >= plan.EVIDENCE)
                       & rp["taught"][w][None, :]).sum(axis=1).mean()) for w in range(len(rp["weeks"]))]
        g1, g2 = st.columns(2)
        with g1:
            fig.update_yaxes(ticksuffix="%")
            layout(fig, 300, title="Class average by topic (skills taught so far)")
            st.plotly_chart(fig, width="stretch")
        with g2:
            f2 = go.Figure(go.Scatter(x=rp["weeks"], y=weak, mode="lines+markers", fill="tozeroy",
                                      line=dict(color=C_RED, width=2.5), fillcolor="rgba(227,73,72,.15)",
                                      hovertemplate="%{x|%d %b}: %{y:.1f} weak spots per student<extra></extra>"))
            layout(f2, 300, title="Weak spots per student (below 60% with 3+ answers)")
            st.plotly_chart(f2, width="stretch")
        st.caption("Weak spots rise as new topics are taught and fall as they are practised. Averages over the "
                   "skills taught by each week.")

# ---------------------------------------------------------------------------
# 6. What if
# ---------------------------------------------------------------------------
with tabs[5]:
    c = ss.cls
    if not ready(c):
        st.info("Load a class first (📥 Data).")
    else:
        truth = "truth" in c.extra
        st.markdown("**The next weeks, two ways of practising** · one revision paper for everybody, or a personal "
                    f"practice set each, {ss.set_size} questions a week")
        if truth:
            st.caption("Demo class: the students are simulated learners whose true state is known (rules in "
                       "masterymap/demo.py: practice helps a little each time, weak prerequisites hold skills back).")
        else:
            st.caption("Your class: the students are simulated by the fitted model itself, so this is what the "
                       "model expects, not a measurement.")
        w1, w2 = st.columns([3, 1])
        w1.slider("Weeks to play forward", 2, 8, key="wi_weeks")
        sig = (ss.key, ss.wi_weeks, ss.set_size)
        if w2.button("▶ Run", type="primary", width="stretch"):
            bar = st.progress(0.0, "Simulating")
            ss.wi = term.what_if(c, weeks=ss.wi_weeks, n=ss.set_size,
                                 progress=lambda f, m: bar.progress(min(f, 1.0), m))
            ss.wi_sig = sig
            bar.empty()
        if ss.wi is not None and ss.wi_sig != sig:
            st.warning("The settings changed since the last run: press ▶ Run again.")
        if ss.wi is None:
            st.info("Press ▶ Run to play the next weeks both ways (a few seconds).")
        else:
            w = ss.wi
            s = term.summary(w)
            t = w["taught"]
            m = st.columns(3)
            a0, a1, b1 = s["same"]["score"][0], s["same"]["score"][-1], s["personal"]["score"][-1]
            m[0].metric("Expected score today", f"{a0:.1%}", help="Average chance on a paper with one medium "
                                                                   "question per skill taught so far.")
            m[1].metric(f"After {w['weeks']} weeks: one paper for all", f"{a1:.1%}",
                        delta=f"{(a1 - a0) * 100:+.1f} points")
            m[2].metric(f"After {w['weeks']} weeks: personal sets", f"{b1:.1%}", delta=f"{(b1 - a0) * 100:+.1f} points")
            m = st.columns(3)
            ws0, wsa, wsb = s["same"]["weak"][0], s["same"]["weak"][-1], s["personal"]["weak"][-1]
            m[0].metric("Weak spots per student today", f"{ws0:.1f}", help="Taught skills below 50% chance.")
            m[1].metric("… after: one paper for all", f"{wsa:.1f}", delta=f"{wsa - ws0:+.1f}", delta_color="inverse")
            m[2].metric("… after: personal sets", f"{wsb:.1f}", delta=f"{wsb - ws0:+.1f}", delta_color="inverse")
            html(anim.race(w["same"][:, :, t].mean(axis=2), w["personal"][:, :, t].mean(axis=2),
                           (w["same"][:, :, t] < 0.5).sum(axis=2), (w["personal"][:, :, t] < 0.5).sum(axis=2),
                           c.students["Name"].astype(str).tolist(), dark=DARK), 340)
            fixed_a, fixed_b = ws0 - wsa, ws0 - wsb
            st.caption(f"Personal sets cleared {fixed_b:.1f} weak spots per student against {fixed_a:.1f} for one "
                       f"paper for everybody; the average score moved {(b1 - a0) * 100:+.1f} against "
                       f"{(a1 - a0) * 100:+.1f} points. Same number of questions both ways.")

# ---------------------------------------------------------------------------
# 7. Benchmark
# ---------------------------------------------------------------------------
with tabs[6]:
    path = os.path.join(HERE, "benchmarks", "results_kt.json")
    pub = json.load(open(os.path.join(HERE, "benchmarks", "kt_published.json")))
    if not os.path.exists(path):
        st.info("No benchmark results yet: run `python benchmarks/run_kt.py`.")
    else:
        R = json.load(open(path))
        st.markdown("#### Predicting the next answer on public tutoring data")
        st.caption("Six datasets published with Gervet et al. (2020), the same preprocessed files and the same "
                   "protocol: 5-fold cross-validation by learner (a learner's answers are never seen while "
                   "learning), AUC per fold (1.0: always ranks a right answer above a wrong one; 0.5: chance). "
                   "Mastery Map's extra features were chosen on ASSISTments 2009 only.")
        rows = []
        for d in R.get("order", []):
            r = R["datasets"][d]
            p = pub["datasets"].get(d, {})
            mm, bl, mk = (r["models"][k] for k in ("Mastery Map", "Best-LR", "marks per skill"))
            rows.append({"Dataset": pub["names"].get(d, d), "Answers": f"{r['answers']:,}",
                         "Marks per skill": mk["mean"], "Best-LR (published)": p.get("Best-LR", [None])[0],
                         "Best-LR (here)": bl["mean"], "DKT (published)": p.get("DKT", [None])[0],
                         "Mastery Map": mm["mean"], "± (folds)": mm["sd"]})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            k: st.column_config.NumberColumn(format="%.3f") for k in df.columns if k not in ("Dataset", "Answers")})
        fig = go.Figure()
        for k, colr in (("Marks per skill", C_GREY), ("DKT (published)", C_VIOLET), ("Best-LR (published)", C_AMBER),
                        ("Mastery Map", C_BLUE)):
            fig.add_trace(go.Bar(x=df["Dataset"], y=df[k], name=k, marker=dict(color=colr, cornerradius=3),
                                 hovertemplate=k + ": %{y:.3f}<extra></extra>"))
        layout(fig, 360, title="AUC by dataset (higher is better)", barmode="group")
        fig.update_yaxes(range=[0.5, 0.9])
        st.plotly_chart(fig, width="stretch")
        a09 = R["datasets"].get("assistments09")
        g1, g2 = st.columns(2)
        if a09:
            with g1:
                cal = pd.DataFrame(a09["calibration"]).dropna()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C_GREY, dash="dot"),
                                         name="perfect", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers", name="Mastery Map",
                                         marker=dict(size=np.clip(np.sqrt(cal["n"]) / 6, 5, 22), color=C_BLUE),
                                         line=dict(color=C_BLUE, width=2.5), customdata=cal["n"],
                                         hovertemplate="predicted %{x:.0%} · right %{y:.0%} · %{customdata:,} answers"
                                                       "<extra></extra>"))
                fig.update_xaxes(tickformat=".0%", range=[0, 1], title="predicted chance")
                fig.update_yaxes(tickformat=".0%", range=[0, 1], title="share answered right")
                layout(fig, 340, title="Calibration on ASSISTments 2009 (out-of-fold)")
                st.plotly_chart(fig, width="stretch")
                b = a09["band_60_80"]
                st.caption(f"Practice questions are picked near a 70% chance. Of the {b['answers']:,} answers predicted "
                           f"between 60% and 80%, {b['actual']:.1%} were right (predicted {b['predicted']:.1%}).")
        with g2:
            c = ss.cls
            if c is not None and "truth" in c.extra and ready(c):
                sn = _snap(c, ss.key)
                tr = c.extra["truth"]
                tk = sn.taught
                true = tr.chance_skill()[:, tk]
                est = sn.mastery[:, tk]
                marks = ((sn.right + 1) / (sn.tries + 2))[:, tk]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=true.ravel(), y=marks.ravel(), mode="markers", name="marks per skill",
                                         marker=dict(color=C_GREY, size=5, opacity=0.45)))
                fig.add_trace(go.Scatter(x=true.ravel(), y=est.ravel(), mode="markers", name="Mastery Map",
                                         marker=dict(color=C_BLUE, size=5, opacity=0.6)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=C_GREY, dash="dot"),
                                         showlegend=False, hoverinfo="skip"))
                fig.update_xaxes(tickformat=".0%", range=[0, 1], title="true chance (simulated learner)")
                fig.update_yaxes(tickformat=".0%", range=[0, 1], title="estimate")
                layout(fig, 340, title="Demo class: estimates against the hidden truth")
                st.plotly_chart(fig, width="stretch")
                wt = true < 0.5

                def pr(e):
                    we = e < 0.5
                    return (we & wt).sum() / max(we.sum(), 1), (we & wt).sum() / max(wt.sum(), 1)
                p1, r1 = pr(est)
                p0, r0 = pr(marks)
                st.caption(f"Correlation with the truth: Mastery Map {np.corrcoef(est.ravel(), true.ravel())[0, 1]:.2f}, "
                           f"marks per skill {np.corrcoef(marks.ravel(), true.ravel())[0, 1]:.2f}. Weak spots "
                           f"(true chance below 50%): of those flagged, {p1:.0%} are real ({p0:.0%} with marks); "
                           f"{r1:.0%} of the real ones are found ({r0:.0%}).")
            else:
                st.caption("Load the demo class to compare the estimates with the simulated students' true state.")

# ---------------------------------------------------------------------------
# 8. How it works
# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("""
#### The problem
A mark book shows that a student got 60% on a mixed test, not which skills are missing. The next worksheet then goes
to everybody: too easy for some, too hard for others, and the gap that holds a student back (often a prerequisite
from an earlier topic) stays.

#### 1. Mastery from answers
Every answer is a data point: who answered, which question, which skills the question trains, right or wrong, when.
A logistic model predicts each answer from the answers **before** it: how easy the question and its skills are, how
many questions of each skill the student has tried and got right (counted as log(1 + n)), the same for the question
itself and in total — the "Best-LR" model of Gervet et al. (2020) — plus recency: right and wrong answers with older
ones counting less (half-life 4 answers on the skill, 8 overall), so a student who has just started getting a skill
right moves up quickly. A student's **mastery** of a skill is the model's chance that they answer a typical question
of that skill right now.

Secure: 80% and up · developing: 60–80% · weak: below 60% · fewer than 3 answers: marked, because the estimate then
leans on the student's other work.

#### 2. Practice sets
About 70% of the questions go to the student's weakest skills whose prerequisites are in place; if a prerequisite is
itself weak, it comes first. One question checks a skill with too few answers; the rest review secure skills not
practised lately. Within a skill, the questions are ones the student has not done, with a predicted chance near 70%:
hard enough to learn from, easy enough to get right most of the time. Every question comes with its reason.

#### 3. Term replay and what if
The replay refits the model each Monday on the answers given before it, so it shows what the app would have said
then. What if plays the next weeks forward two ways (one paper for all, a personal set each) with the same number of
questions.

#### Tested
See 🧪 Benchmark: six public tutoring datasets, the published protocol and published results.
""")

with side_steps:
    c = ss.cls
    done = [c is not None, c is not None and ready(c)]
    for ok, lab in zip(done, ["Class loaded", "Data checked"]):
        st.markdown(f"{'✅' if ok else '⬜'} {lab}")
    if c is not None:
        st.caption(c.name)
