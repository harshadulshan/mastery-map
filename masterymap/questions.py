"""
Practice questions for a G.C.E. O/L style Mathematics syllabus, made from templates: every skill has a generator that
writes a question and its answer at three levels (1 easy, 2 medium, 3 hard), with numbers chosen so the answers come
out clean. Used for the demo class and for printable practice sheets.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

# id, name, topic, prerequisites
SKILLS = [
    ("N1", "Fractions", "Number", []),
    ("N2", "Decimals and percentages", "Number", ["N1"]),
    ("N3", "Ratio and proportion", "Number", ["N1"]),
    ("N4", "Profit, loss and discount", "Number", ["N2"]),
    ("N5", "Simple and compound interest", "Number", ["N2"]),
    ("N6", "Indices", "Number", []),
    ("N7", "Logarithms", "Number", ["N6"]),
    ("A1", "Algebraic expressions", "Algebra", []),
    ("A2", "Expanding brackets", "Algebra", ["A1"]),
    ("A3", "Factorising", "Algebra", ["A2"]),
    ("A4", "Algebraic fractions", "Algebra", ["A3", "N1"]),
    ("A5", "Linear equations", "Algebra", ["A1"]),
    ("A6", "Simultaneous equations", "Algebra", ["A5"]),
    ("A7", "Quadratic equations", "Algebra", ["A3", "A5"]),
    ("A8", "Inequalities", "Algebra", ["A5"]),
    ("G1", "Angles and parallel lines", "Geometry", []),
    ("G2", "Pythagoras' theorem", "Geometry", ["N6"]),
    ("G3", "Area and perimeter", "Geometry", ["N2"]),
    ("G4", "Volume and surface area", "Geometry", ["G3"]),
    ("G5", "Trigonometry", "Geometry", ["G2"]),
    ("S1", "Mean, median and mode", "Statistics", ["N2"]),
    ("S2", "Probability", "Statistics", ["N1"]),
    ("S3", "Sets and Venn diagrams", "Statistics", []),
]
SKILL_IDS = [s[0] for s in SKILLS]


def _frac(f: Fraction) -> str:
    if f.denominator == 1:
        return str(f.numerator)
    whole, rest = divmod(abs(f.numerator), f.denominator)
    sign = "-" if f < 0 else ""
    return f"{sign}{whole} {rest}/{f.denominator}" if whole else f"{sign}{rest}/{f.denominator}"


def _num(x) -> str:
    x = round(float(x), 2)
    return f"{x:,.0f}" if abs(x - round(x)) < 1e-9 else f"{x:,.2f}".rstrip("0").rstrip(".")


def _signed(c, var="", first=False):
    if c == 0:
        return ""
    s = "-" if c < 0 else ("" if first else "+")
    a = abs(c)
    body = (var if a == 1 and var else f"{a}{var}")
    return f"{s} {body}".strip() if not first else f"{s}{body}"


def _poly(*terms):
    out = []
    for c, v in terms:
        if c:
            out.append(_signed(c, v, first=not out))
    return " ".join(out) or "0"


def g_N1(r, lv):
    d1, d2 = r.choice([2, 3, 4, 5, 6, 8]), r.choice([3, 4, 5, 6, 10, 12])
    a, b = Fraction(int(r.integers(1, d1)), d1), Fraction(int(r.integers(1, d2)), d2)
    if lv == 1:
        return f"Find {_frac(a)} + {_frac(b)}.", _frac(a + b)
    if lv == 2:
        w = int(r.integers(1, 4))
        return f"Find {_frac(w + a)} - {_frac(b)}.", _frac(w + a - b)
    w = int(r.integers(1, 3))
    return f"Find {_frac(w + a)} × {_frac(b)} ÷ {_frac(a)}.", _frac((w + a) * b / a)


def g_N2(r, lv):
    if lv == 1:
        p, x = r.choice([10, 20, 25, 50]), int(r.integers(2, 20)) * 20
        return f"Find {p}% of {x}.", _num(p * x / 100)
    if lv == 2:
        x, y = int(r.integers(2, 10)) * 10, int(r.integers(1, 8)) * 10
        y = min(y, x - 10)
        return f"Write {y} as a percentage of {x}.", f"{_num(100 * y / x)}%"
    x, p = int(r.integers(4, 30)) * 50, r.choice([5, 10, 12, 15, 20])
    up = bool(r.integers(0, 2))
    return (f"A price of Rs. {x:,} is {'increased' if up else 'decreased'} by {p}%. Find the new price.",
            f"Rs. {_num(x * (1 + (p if up else -p) / 100))}")


def g_N3(r, lv):
    a, b = int(r.integers(1, 6)), int(r.integers(2, 7))
    if a == b:
        b += 1
    k = int(r.integers(2, 12)) * (a + b)
    if lv == 1:
        return f"Share Rs. {k * 10:,} in the ratio {a} : {b}. Find the larger share.", f"Rs. {max(a, b) * k * 10 // (a + b):,}"
    if lv == 2:
        n, t = int(r.integers(2, 9)), int(r.integers(3, 13))
        return (f"{n} workers finish a job in {t * n} days. How many days would {n * 2} workers take?",
                f"{_num(t * n / 2)} days")
    c = int(r.integers(1, 6))
    return (f"A : B = {a} : {b} and B : C = {b} : {c}. If C = {c * 6}, find A.", _num(a * 6))


def g_N4(r, lv):
    cost = int(r.integers(4, 40)) * 100
    p = r.choice([10, 15, 20, 25, 30])
    if lv == 1:
        return f"An item bought for Rs. {cost:,} is sold for Rs. {cost + cost * p // 100:,}. Find the profit.", \
            f"Rs. {cost * p // 100:,}"
    if lv == 2:
        return (f"An item bought for Rs. {cost:,} is sold at a {p}% loss. Find the selling price.",
                f"Rs. {_num(cost * (100 - p) / 100)}")
    d = r.choice([5, 10, 20])
    marked = cost * 2
    return (f"An item marked Rs. {marked:,} is sold at a {d}% discount. It cost Rs. {cost:,}. Find the profit "
            f"percentage.", f"{_num((marked * (100 - d) / 100 - cost) / cost * 100)}%")


def g_N5(r, lv):
    P = int(r.integers(2, 20)) * 5000
    rate = r.choice([4, 5, 6, 8, 10, 12])
    t = int(r.integers(2, 5))
    if lv == 1:
        return f"Find the simple interest on Rs. {P:,} at {rate}% a year for {t} years.", f"Rs. {_num(P * rate * t / 100)}"
    if lv == 2:
        I_ = P * rate * t / 100
        return (f"Simple interest of Rs. {_num(I_)} is earned on Rs. {P:,} in {t} years. Find the annual rate.",
                f"{rate}%")
    return (f"Find the compound interest on Rs. {P:,} at {rate}% a year for 2 years.",
            f"Rs. {_num(P * ((1 + rate / 100) ** 2 - 1))}")


def g_N6(r, lv):
    a, m, n = r.choice([2, 3, 5]), int(r.integers(2, 6)), int(r.integers(2, 6))
    if lv == 1:
        return f"Simplify {a}^{m} × {a}^{n}, giving your answer as a power of {a}.", f"{a}^{m + n}"
    if lv == 2:
        return f"Simplify ({a}^{m})^{n} ÷ {a}^{m}, giving your answer as a power of {a}.", f"{a}^{m * n - m}"
    k = r.choice([4, 8, 9, 16, 25, 27])
    root = {4: 2, 8: 2, 9: 3, 16: 2, 25: 5, 27: 3}[k]
    p = round(math.log(k, root))
    return f"Find the value of {k}^(-{1}/{p}).", _frac(Fraction(1, root))


def g_N7(r, lv):
    b = r.choice([2, 3, 5, 10])
    n = int(r.integers(2, 5))
    if lv == 1:
        return f"Find log base {b} of {b ** n}.", str(n)
    if lv == 2:
        k = int(r.integers(2, 5))
        return f"Find log {2 ** k} + log {5 ** k} (logarithms to base 10).", str(k)
    return (f"Solve log base {b} of (x + 1) = {n}.", str(b ** n - 1))


def g_A1(r, lv):
    a, b, c = (int(v) for v in r.integers(2, 10, 3))
    if lv == 1:
        return f"Simplify {a}x + {b}x - {c}x.", _poly((a + b - c, "x"))
    if lv == 2:
        x = int(r.integers(-4, 6))
        return f"Find the value of {_poly((a, 'x²'), (-b, 'x'), (c, ''))} when x = {x}.", str(a * x * x - b * x + c)
    return (f"Simplify {a}x + {b}y - {c}x + {a + 1}y - {b}.", _poly((a - c, "x"), (b + a + 1, "y"), (-b, "")))


def g_A2(r, lv):
    a, b = int(r.integers(2, 8)), int(r.integers(1, 7))
    if lv == 1:
        return f"Expand {a}(x + {b}).", _poly((a, "x"), (a * b, ""))
    if lv == 2:
        return f"Expand (x + {a})(x - {b}).", _poly((1, "x²"), (a - b, "x"), (-a * b, ""))
    return f"Expand (2x - {a})².", _poly((4, "x²"), (-4 * a, "x"), (a * a, ""))


def g_A3(r, lv):
    a, b = int(r.integers(1, 8)), int(r.integers(1, 8))
    if lv == 1:
        return f"Factorise {a * b}x + {a * (b + 1)}.", f"{a}({b}x + {b + 1})"
    if lv == 2:
        return f"Factorise x² {_signed(a + b, 'x')} {_signed(a * b)}.", f"(x + {a})(x + {b})"
    return f"Factorise x² - {a * a}.", f"(x + {a})(x - {a})"


def g_A4(r, lv):
    a, b = int(r.integers(2, 6)), int(r.integers(2, 6))
    if a == b:
        b += 1
    if lv == 1:
        return f"Simplify x/{a} + x/{b}.", f"{a + b}x/{a * b}"
    if lv == 2:
        return f"Simplify (x² - {a * a}) / (x + {a}).", f"x - {a}"
    return (f"Simplify 1/(x + {a}) + 1/(x - {a}).", f"2x/(x² - {a * a})")


def g_A5(r, lv):
    x = int(r.integers(-5, 10))
    a, b = int(r.integers(2, 8)), int(r.integers(1, 20))
    if lv == 1:
        return f"Solve {a}x + {b} = {a * x + b}.", f"x = {x}"
    if lv == 2:
        c = int(r.integers(1, a))
        return f"Solve {a}x - {b} = {_poly((c, 'x'), ((a - c) * x - b, ''))}.", f"x = {x}"
    return f"Solve {a}(x - {b}) = {a * (x - b)}.", f"x = {x}"


def g_A6(r, lv):
    x, y = int(r.integers(-3, 8)), int(r.integers(-3, 8))
    a, b = int(r.integers(2, 6)), int(r.integers(2, 6))
    if lv == 1:
        return f"Solve x + y = {x + y} and x - y = {x - y}.", f"x = {x}, y = {y}"
    if lv == 2:
        return f"Solve {_poly((a, 'x'), (1, 'y'))} = {a * x + y} and x + y = {x + y}.", f"x = {x}, y = {y}"
    return (f"Solve {_poly((a, 'x'), (b, 'y'))} = {a * x + b * y} and {_poly((b, 'x'), (-a, 'y'))} = "
            f"{b * x - a * y}.", f"x = {x}, y = {y}")


def g_A7(r, lv):
    p, q = (int(v) for v in r.choice(np.arange(-6, 7), 2, replace=False))
    if lv == 1:
        return f"Solve (x - {p})(x - {q}) = 0.".replace("- -", "+ "), f"x = {p} or x = {q}"
    if lv == 2:
        return f"Solve x² {_signed(-(p + q), 'x')} {_signed(p * q)} = 0.".replace("  ", " "), \
            f"x = {min(p, q)} or x = {max(p, q)}"
    k = int(r.integers(2, 9))
    return f"Solve x² = {k * k}.", f"x = {k} or x = -{k}"


def g_A8(r, lv):
    a, b, c = int(r.integers(2, 7)), int(r.integers(1, 15)), int(r.integers(1, 30))
    if lv == 1:
        return f"Solve x + {b} > {b + c}.", f"x > {c}"
    if lv == 2:
        return f"Solve {a}x - {b} ≤ {a * c - b}.", f"x ≤ {c}"
    return f"Solve {b + a * c} - {a}x ≥ {b}.", f"x ≤ {c}"


def g_G1(r, lv):
    a = int(r.integers(20, 160))
    if lv == 1:
        return f"Two angles on a straight line; one is {a}°. Find the other.", f"{180 - a}°"
    if lv == 2:
        b = int(r.integers(20, 160 - a + 20)) if a < 150 else 10
        return f"Two angles of a triangle are {a // 2}° and {b // 2}°. Find the third angle.", f"{180 - a // 2 - b // 2}°"
    n = r.choice([5, 6, 8, 9, 10, 12])
    return f"Find the size of each interior angle of a regular polygon with {n} sides.", f"{_num(180 * (n - 2) / n)}°"


def g_G2(r, lv):
    t = [(3, 4, 5), (5, 12, 13), (6, 8, 10), (8, 15, 17), (9, 12, 15), (7, 24, 25)][int(r.integers(0, 6))]
    if lv == 1:
        return f"A right triangle has shorter sides {t[0]} cm and {t[1]} cm. Find the hypotenuse.", f"{t[2]} cm"
    if lv == 2:
        return f"A right triangle has hypotenuse {t[2]} cm and one side {t[0]} cm. Find the other side.", f"{t[1]} cm"
    return (f"A ladder {t[2]} m long reaches {t[1]} m up a wall. How far is its foot from the wall?", f"{t[0]} m")


def g_G3(r, lv):
    a, b = int(r.integers(3, 15)), int(r.integers(3, 15))
    if lv == 1:
        return f"Find the area of a rectangle {a} cm by {b} cm.", f"{a * b} cm²"
    if lv == 2:
        return f"Find the area of a triangle with base {2 * a} cm and height {b} cm.", f"{a * b} cm²"
    rr = r.choice([7, 14, 21])
    return f"Find the area of a circle of radius {rr} cm (take π = 22/7).", f"{22 * rr * rr // 7} cm²"


def g_G4(r, lv):
    a, b, c = (int(v) for v in r.integers(2, 10, 3))
    if lv == 1:
        return f"Find the volume of a cuboid {a} cm × {b} cm × {c} cm.", f"{a * b * c} cm³"
    if lv == 2:
        return f"Find the total surface area of a cuboid {a} cm × {b} cm × {c} cm.", f"{2 * (a * b + b * c + a * c)} cm²"
    rr, h = r.choice([7, 14]), int(r.integers(2, 10))
    return f"Find the volume of a cylinder of radius {rr} cm and height {h} cm (take π = 22/7).", f"{22 * rr * rr * h // 7} cm³"


def g_G5(r, lv):
    t = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)][int(r.integers(0, 4))]
    if lv == 1:
        return (f"In a right triangle the side opposite angle θ is {t[0]} cm and the hypotenuse {t[2]} cm. "
                f"Find sin θ.", _frac(Fraction(t[0], t[2])))
    if lv == 2:
        return (f"In a right triangle the sides next to the right angle are {t[0]} cm (opposite θ) and {t[1]} cm. "
                f"Find tan θ.", _frac(Fraction(t[0], t[1])))
    h = r.choice([10, 20, 30])
    return (f"From {h} m away, the angle of elevation of a tower's top is 45°. Find the tower's height.", f"{h} m")


def g_S1(r, lv):
    n = 5 if lv < 3 else 6
    xs = sorted(int(v) for v in r.integers(1, 20, n))
    if lv == 1:
        s = sum(xs)
        xs[-1] += (n - s % n) % n
        return f"Find the mean of {', '.join(map(str, xs))}.", _num(sum(xs) / n)
    if lv == 2:
        return f"Find the median of {', '.join(map(str, r.permutation(xs)))}.", str(xs[n // 2])
    xs[1] = xs[2] = xs[3]
    return f"Find the mode of {', '.join(map(str, r.permutation(xs)))}.", str(xs[2])


def g_S2(r, lv):
    red, blue = int(r.integers(1, 8)), int(r.integers(1, 8))
    if lv == 1:
        return (f"A bag has {red} red and {blue} blue balls. Find the probability of drawing a red ball.",
                _frac(Fraction(red, red + blue)))
    if lv == 2:
        return ("A fair die is rolled. Find the probability of getting a number greater than "
                f"{int(r.integers(1, 5))}.", None)
    return (f"A bag has {red} red and {blue} blue balls. Two are drawn with replacement. Find the probability "
            f"that both are red.", _frac(Fraction(red, red + blue) ** 2))


def g_S3(r, lv):
    n, a, b = int(r.integers(30, 50)), int(r.integers(10, 25)), int(r.integers(10, 25))
    both = int(r.integers(2, min(a, b)))
    if lv == 1:
        return f"n(A) = {a}, n(B) = {b}, n(A ∩ B) = {both}. Find n(A ∪ B).", str(a + b - both)
    if lv == 2:
        return (f"In a class of {n}, {a} study Science, {b} study Art and {both} study both. How many study "
                f"neither?", str(n - (a + b - both)))
    return (f"n(A ∪ B) = {a + b - both}, n(A) = {a}, n(B) = {b}. Find n(A ∩ B).", str(both))


GENERATORS = {k: globals()["g_" + k] for k in SKILL_IDS}


def make(skill, level, seed):
    """(question, answer) for a skill at level 1-3; the same seed gives the same question."""
    r = np.random.default_rng([sum(ord(c) * 31 ** i for i, c in enumerate(skill)), int(level), int(seed)])
    text, ans = GENERATORS[skill](r, int(level))
    if ans is None:                                   # die question: the answer depends on the number asked
        k = int(text.split("greater than ")[1].rstrip("."))
        ans = _frac(Fraction(6 - k, 6))
    return text, ans


def bank(per_level=4, seed=0):
    """A question bank: per_level different questions at each of the 3 levels for every skill.
    Rows: (question id, skill id, level, question, answer); ids like N1-2b (skill N1, level 2, second question)."""
    rows = []
    for k in SKILL_IDS:
        for lv in (1, 2, 3):
            seen, j = set(), 0
            while len(seen) < per_level and j < 200:
                t, a = make(k, lv, seed * 1000 + j)
                j += 1
                if t not in seen:
                    seen.add(t)
                    rows.append((f"{k}-{lv}{chr(96 + len(seen))}", k, lv, t, a))
    return rows
