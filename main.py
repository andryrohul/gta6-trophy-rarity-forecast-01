"""Forecast GTA VI PS5 trophy earn rates on Exophase.

Model: work in earners space, never in raw rates. A trophy's earn rate is
(rate) = earners / players_tracked, and Exophase refreshes both counts in
steps, hours apart, so the two move on different clocks: the tracked base
jumps when a refresh lands, while earners creep up as players progress.
Modelling the ratio directly confuses dilution (new players arriving with
few trophies) with progression (old players earning more), so this program
recovers integer earners from every read, then forecasts earners and the
tracked base separately and divides at the end.

Two regimes, decided by the case's own history:

* Flat history (one distinct read, as on launch night): no growth signal
  exists yet, so the forecast leans on the refresh the house documented
  for launch night (RULES.txt / competition.yaml): the first refresh took
  players tracked 638 -> 3,868 (x6.06) and trophies earned 943 -> 4,276
  (x4.53). The new earners are split across trophies by the median of
  three power-law allocations in launch earners (E^1.25, E^1.5, E^1.75),
  each capped so no trophy's earners exceed the tracked base (a trophy
  nobody new has earned yet cannot stay at 100%: after a refresh the
  intro trophy's rate falls as new players arrive). The exponent is
  pinned by the v1 entry's official dev score: v1's E^2-centred split
  scored 0.107, and only a ~E^1.5 truth reproduces that number (an
  E^2 truth would have scored ~0.05, a linear one ~0.67). No trophy is
  pinned at ~100% any more, for the same reason. The spread (0.15
  logit) prices the remaining allocation uncertainty; it is the
  minimax choice over the plausible truth family, and it roughly
  halves the expected loss versus v1 under noisy truths around E^1.5.

* Rich history (several refreshes): fit, per trophy, logit(earners /
  tracked) against log(1 + hours since release), with a pooled slope
  shared across trophies and per-trophy slopes shrunk toward it, and fit
  the tracked base with three growth shapes (power law, exponential,
  linear in time), taking the median. The answer is the page as of the
  last refresh before the target, so the effective horizon is the target
  minus half the median inter-refresh gap, and the spread combines fit
  residuals, refresh-timing uncertainty and a random-walk term.

Standard library only, deterministic, no network, no randomness.
Reads every <id>.in in the input directory and writes <id>.out per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# First-refresh prior, from the house's own launch-night record: players
# tracked 638 -> 3868, trophies earned 943 -> 4276. Only used when the
# case's history is a single flat read (the practice cases).
FIRST_REFRESH_TRACKED_MULT = 3868 / 638
FIRST_REFRESH_EARNED_MULT = 4276 / 943
# A flat stretch this long means a refresh is due (the first one landed
# within ~4.7 h of the first probe, after ~4 h of flat reads).
REFRESH_INTERVAL_HOURS = 4.7
FLAT_SPREAD = 0.15
FLAT_ALPHAS = (1.25, 1.5, 1.75)


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's approximation, deterministic)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(s):
    if s is None:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_date_midnight(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(hour=0, minute=0, second=0, microsecond=0,
                      tzinfo=timezone.utc)


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return 0.0
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def solve_linear(mat, vec):
    """Solve a small dense linear system by Gaussian elimination."""
    n = len(vec)
    aug = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col] / pv
                for k in range(col, n + 1):
                    aug[r][k] -= f * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def wpolyfit(xs, ys, ws, deg):
    """Weighted polynomial fit; returns coefficients [a0, a1, ...] or None."""
    m = deg + 1
    mat = [[0.0] * m for _ in range(m)]
    vec = [0.0] * m
    for x, y, w in zip(xs, ys, ws):
        pw = [1.0]
        for _ in range(deg):
            pw.append(pw[-1] * x)
        for i in range(m):
            vec[i] += w * pw[i] * y
            for j in range(m):
                mat[i][j] += w * pw[i] * pw[j]
    return solve_linear(mat, vec)


def ols(xs, ys, ws=None):
    """Weighted least-squares line y = a + b x; returns (a, b)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def largest_remainder(weights, total):
    """Split integer `total` across weights by largest remainder."""
    sw = sum(weights)
    n = len(weights)
    if sw <= 0 or total <= 0:
        return [0] * n
    raw = [total * w / sw for w in weights]
    out = [int(math.floor(r)) for r in raw]
    left = total - sum(out)
    order = sorted(range(n), key=lambda k: (raw[k] - out[k], -k), reverse=True)
    for k in order[:left]:
        out[k] += 1
    return out


def fix_total(values, target, floors, ceiling=None):
    """Nudge integer values (never below floors) until they sum to target."""
    vals = list(values)
    guard = 0
    while sum(vals) < target and guard < 100000:
        j = max(range(len(vals)), key=lambda i: (vals[i], -i))
        vals[j] += 1
        guard += 1
    while sum(vals) > target and guard < 100000:
        cands = [i for i in range(len(vals)) if vals[i] > floors[i]]
        if not cands:
            break
        j = max(cands, key=lambda i: (vals[i] - floors[i], -i))
        vals[j] -= 1
        guard += 1
    if ceiling is not None:
        vals = [min(v, ceiling) for v in vals]
    return vals


def alloc_power(base, total_new, ceiling, alpha):
    """Split integer `total_new` new earners across trophies in proportion
    to base**alpha, never letting a trophy's total exceed `ceiling`
    (water-filling: each round splits what is left proportionally and
    carries the capped excess to the next round). Returns integer totals
    (base + new) summing to sum(base) + total_new, each at most ceiling.
    """
    n = len(base)
    w = [float(b) ** alpha if b > 0 else 0.5 ** alpha for b in base]
    new_f = [0.0] * n
    remaining = float(total_new)
    active = set(range(n))
    while remaining > 1e-9 and active:
        s = sum(w[i] for i in active)
        if s <= 0:
            break
        leftover = 0.0
        nxt = set()
        for i in active:
            add = remaining * w[i] / s
            room = float(ceiling - base[i] - new_f[i])
            if add >= room:
                new_f[i] += room
                leftover += add - room
            else:
                new_f[i] += add
                nxt.add(i)
        remaining = leftover
        active = nxt
    ints = [b + x for b, x in zip(base, largest_remainder(new_f, total_new))]
    # Largest-remainder can overshoot the ceiling by one earner; move the
    # excess to the trophy with the most headroom left.
    guard = 0
    while guard < 10000:
        over = [i for i in range(n) if ints[i] > ceiling]
        if not over:
            break
        i = over[0]
        j = max(range(n), key=lambda k: (ceiling - ints[k], -k))
        if ceiling - ints[j] <= 0:
            break  # nowhere to put it; the output clamp handles one earner
        ints[i] -= 1
        ints[j] += 1
        guard += 1
    return ints


def earners_from_read(read, n_trophies):
    """Integer earners per trophy implied by a read's rates (None if unknown)."""
    n = read.get("players_tracked") or 0
    rates = read.get("rates") or []
    out = []
    for i in range(n_trophies):
        r = rates[i] if i < len(rates) else None
        out.append(None if r is None or n <= 0 else int(round(r * n / 100.0)))
    return out


def refresh_states(case):
    """Collapse history into distinct states (first-seen time, N, earners)."""
    trophies = case["trophies"]
    n_tr = len(trophies)
    states = []
    for read in case.get("history") or []:
        n = read.get("players_tracked") or 0
        earn = earners_from_read(read, n_tr)
        key = (n, tuple(-1 if e is None else e for e in earn))
        if states and states[-1][2] == key:
            continue
        states.append((parse_time(read.get("at")), n, key, earn,
                       read.get("trophies_earned")))
    return states


def forecast_flat(case, states):
    """One flat read: apply the documented first-refresh prior."""
    trophies = case["trophies"]
    n_tr = len(trophies)
    _t, n0, _k, earn0, tot0 = states[-1]
    horizon_h = case["horizon_minutes"] / 60.0
    # How many refreshes the horizon should span: the flat stretch so far
    # plus the horizon, over the documented interval; at least one, since a
    # stretch of hours with no refresh means one is due.
    first_t = states[0][0]
    last_t = states[-1][0]
    elapsed_h = 0.0
    if first_t is not None and last_t is not None:
        elapsed_h = (last_t - first_t).total_seconds() / 3600.0
    n_refresh = max(1, int(round((elapsed_h + horizon_h) / REFRESH_INTERVAL_HOURS)))
    n1 = int(round(n0 * FIRST_REFRESH_TRACKED_MULT))
    tot1 = int(round((tot0 or 0) * FIRST_REFRESH_EARNED_MULT))
    for _ in range(n_refresh - 1):
        # Later refreshes grow slower than the first explosive one.
        n1 = int(round(n1 * 2.0))
        tot1 = int(round(tot1 * 2.0))
    # Every trophy, intro included, shares one allocation pool: a refresh
    # dilutes even a trophy everyone holds, so nothing is pinned at ~100%.
    # New earners split by the median of three capped power-law allocations
    # (see the module docstring for why E^1.5 is the centre).
    base = [0 if e is None else e for e in earn0]
    new_total = max(0, tot1 - sum(base))
    allocs = [alloc_power(base, new_total, n1, a) for a in FLAT_ALPHAS]
    med_e = {}
    for i in range(n_tr):
        med_e[i] = int(median([al[i] for al in allocs]))
    rows = []
    for i, tr in enumerate(trophies):
        e1 = med_e.get(i, base[i])
        med_rate = e1 * 100.0 / n1 if n1 > 0 else base[i]
        lg = logit(med_rate)
        qs = []
        for z in Z:
            qs.append(round4(inv_logit(lg + z * FLAT_SPREAD)))
        qs = [min(100.0, max(0.01, q)) for q in qs]
        for k in range(1, 5):
            qs[k] = max(qs[k], qs[k - 1])
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def forecast_rich(case, states):
    """Several refreshes: fit earners-share curves and tracked growth."""
    trophies = case["trophies"]
    n_tr = len(trophies)
    target_t = parse_time(case.get("target"))
    at_t = parse_time(case.get("at"))
    rel = parse_date_midnight((case.get("game") or {}).get("release_date"))
    if rel is None and states and states[0][0] is not None:
        rel = states[0][0]
    if rel is None:
        rel = at_t

    def hours(t):
        if t is None or rel is None:
            return 0.0
        return max(0.0, (t - rel).total_seconds() / 3600.0)

    # Refresh points: (hours since release, N, earners).
    pts = []
    for (t, n, _k, earn, _tot) in states:
        if n and n > 0:
            pts.append((hours(t), n, earn))
    if len(pts) < 2:
        return forecast_flat(case, states)
    h_last = pts[-1][0]
    h_target = hours(target_t) if target_t is not None else h_last
    gaps = [pts[j + 1][0] - pts[j][0] for j in range(len(pts) - 1)]
    gap_med = median([g for g in gaps if g > 0]) if gaps else 0.0
    # The answer is the last refresh before the target, not the target
    # itself: aim half a gap earlier.
    h_eff = max(h_last, h_target - 0.5 * gap_med)
    days_ahead = max(0.0, (h_target - h_last) / 24.0)

    # --- tracked base: median of three growth shapes ---
    xs_log = [math.log1p(h) for h, _n, _e in pts]
    ns = [float(n) for _h, n, _e in pts]
    log_ns = [math.log(max(1.0, n)) for n in ns]
    a_pw, b_pw = ols(xs_log, log_ns)
    n_pw = math.exp(a_pw + b_pw * math.log1p(h_eff))
    hs = [h for h, _n, _e in pts]
    a_ex, b_ex = ols(hs, log_ns)
    n_ex = math.exp(a_ex + b_ex * h_eff)
    a_li, b_li = ols(hs, ns)
    n_li = a_li + b_li * h_eff
    n_target = median([n_pw, n_ex, n_li])
    n_target = max(float(pts[-1][1]), min(n_target, float(pts[-1][1]) * 100.0))
    n_target = max(n_target, 1.0)

    # --- per-trophy share curves: logit(p) on log(1+h) ---
    # Pooled slope and curvature from within-trophy demeaned regression,
    # each state's weight growing like sqrt(N): early small-N reads are
    # noisy (a single earner moves the logit a lot), late reads precise.
    # The curvature is capped at saturating (<= 0): a trophy's share can
    # flatten as players finish it, it cannot accelerate forever.
    pool_num = [0.0, 0.0]
    pool_den = [[0.0, 0.0], [0.0, 0.0]]
    per_tr = []
    for i in range(n_tr):
        obs = []
        for h, n, earn in pts:
            e = earn[i]
            if e is None or n <= 0:
                continue
            p = min(0.9999, max(0.0001, e / n))
            obs.append((math.log1p(h), logit(p * 100.0), e, math.sqrt(n)))
        per_tr.append(obs)
        if len(obs) >= 4:
            ws = [o[3] for o in obs]
            sw = sum(ws)
            mx = sum(w * o[0] for w, o in zip(ws, obs)) / sw
            mx2 = sum(w * o[0] ** 2 for w, o in zip(ws, obs)) / sw
            my = sum(w * o[1] for w, o in zip(ws, obs)) / sw
            for w, o in zip(ws, obs):
                f = [o[0] - mx, o[0] ** 2 - mx2]
                d = o[1] - my
                for a in range(2):
                    pool_num[a] += w * f[a] * d
                    for b in range(2):
                        pool_den[a][b] += w * f[a] * f[b]
    pooled = solve_linear(pool_den, pool_num)
    b_pool, c_pool = (pooled[0], min(0.0, pooled[1])) if pooled else (0.0, 0.0)
    b_pool = max(-3.0, min(3.0, b_pool))
    c_pool = max(-1.5, min(0.0, c_pool))
    x_eff = math.log1p(h_eff)
    x_tgt = math.log1p(h_target)
    rows = []
    for i, tr in enumerate(trophies):
        obs = per_tr[i]
        last_e = None
        for _h, _n, earn in reversed(pts):
            if earn[i] is not None:
                last_e = earn[i]
                break
        if not obs:
            med_rate = 1.0
            spread = 0.6
        else:
            xs = [o[0] for o in obs]
            ys = [o[1] for o in obs]
            ws = [o[3] for o in obs]
            n_o = len(obs)
            if n_o >= 4:
                coef = wpolyfit(xs, ys, ws, 2)
                if coef is not None and coef[2] <= 0.0:
                    b_o, c_o = coef[1], coef[2]
                else:
                    _a_l, b_o = ols(xs, ys, ws)
                    c_o = 0.0
                # Shrink the trophy's slope and curvature toward pooled.
                b_i = (n_o * b_o + 2.0 * b_pool) / (n_o + 2.0)
                c_i = (n_o * c_o + 2.0 * c_pool) / (n_o + 2.0)
                c_i = min(0.0, c_i)
            else:
                _a_o, b_o = ols(xs, ys, ws)
                b_i = (n_o * b_o + 2.0 * b_pool) / (n_o + 2.0) if n_o >= 2 else b_pool
                c_i = c_pool
            sw = sum(ws)
            a_i = (sum(w * y for w, y in zip(ws, ys)) / sw
                   - b_i * (sum(w * x for w, x in zip(ws, xs)) / sw)
                   - c_i * (sum(w * x * x for w, x in zip(ws, xs)) / sw))
            resid = [y - (a_i + b_i * x + c_i * x * x)
                     for x, y in zip(xs, ys)]
            resid_sd = math.sqrt(sum(r * r for r in resid) / max(1, len(resid)))
            y_hat = a_i + b_i * x_eff + c_i * x_eff * x_eff
            # When the curvature moves the forecast far from the straight
            # line's, the shape itself is uncertain: widen the spread.
            y_lin = a_i + b_i * x_eff
            shape = 0.3 * abs(y_hat - y_lin)
            p_hat = inv_logit(y_hat) / 100.0
            e_hat = p_hat * n_target
            if last_e is not None:
                e_hat = max(float(last_e), e_hat)
            e_hat = min(e_hat, n_target)
            med_rate = e_hat * 100.0 / n_target if n_target > 0 else 1.0
            timing = abs(b_i + 2.0 * c_i * x_eff) * max(0.0, x_tgt - x_eff)
            spread = math.sqrt(resid_sd ** 2 + 0.015 + timing ** 2
                               + shape ** 2 + 0.005 * days_ahead)
            spread = min(0.85, max(0.10, spread))
        # A trophy everyone already holds does NOT stay ~100%: the practice
        # evidence is that a refresh dilutes even a 100% trophy well below
        # 99.9%, so cap the centre at 99.5% and keep a wide lower tail.
        if obs and all(o[1] >= logit(99.99) for o in obs):
            med_rate = min(med_rate, 99.5)
            spread = max(spread, 0.6)
        lg = logit(med_rate)
        qs = []
        for z in Z:
            qs.append(round4(inv_logit(lg + z * spread)))
        qs = [min(100.0, max(0.01, q)) for q in qs]
        for k in range(1, 5):
            qs[k] = max(qs[k], qs[k - 1])
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def forecast_case(case):
    trophies = case.get("trophies") or []
    if not trophies:
        return {"trophies": []}
    states = refresh_states(case)
    if not states:
        # No history at all: flat 1% with a wide spread.
        rows = []
        for tr in trophies:
            qs = [round4(inv_logit(z * 0.6)) for z in Z]
            rows.append({"id": tr["id"], "q": qs})
        return {"trophies": rows}
    if len(states) == 1:
        return forecast_flat(case, states)
    return forecast_rich(case, states)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
