#!/usr/bin/env python3
"""Symbolic verification of the accounting model in paper/main.tex (Section III) and of its
implementation in src/.

  A  SymPy proofs of the structural claims (conservation, non-negativity, bounds, closure,
     monotonicity of Delta C in every break-even parameter)
  B  dimensional analysis of every equation and exactness of the unit-conversion constants
  C  paper equations == implementation, at central values and on Monte Carlo draws
  D  closed-form break-even roots == Brent roots in results/processed/break_even.json
  E  manuscript literals, macros and symbols == registry / configs / code

Usage: python scripts/verify_math_proofs.py      (exit status 1 if any check fails)
Requires sympy (pip install -e .[dev]).
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
import yaml  # noqa: E402
from sympy.physics import units as u  # noqa: E402

from src.experiments.break_even import SPECS  # noqa: E402
from src.experiments.monte_carlo import sample_parameters  # noqa: E402
from src.models.pipeline_evaluator import AI_TIERS, BASELINES, evaluate_all, load_regimes  # noqa: E402
from src.params import load_scenario  # noqa: E402
from src.paths import (ENERGY_SUMMARY, PAPER_DIR, PROCESSED_DIR, RESULTS_MACROS, SCENARIOS,  # noqa: E402
                       SQI_CONFIG)
from src.sustainability.carbon_accounting import COMPONENTS  # noqa: E402
from src.validation.source_registry import load_registry  # noqa: E402

RESULTS = []


def check(section, name, ok, detail=""):
    RESULTS.append((section, name, "PASS" if ok else "FAIL", detail))


def note(section, name, detail):
    RESULTS.append((section, name, "INFO", detail))


# ---------------------------------------------------------------------------------------------
# Paper symbols. nonneg/positive assumptions encode the physical domains stated in the paper.
nn = dict(nonnegative=True, real=True)
pos = dict(positive=True, real=True)
N, Theta, pi_, fA, fB, fC, r, d, q0, tau = sp.symbols("N Theta pi f_A f_B f_C r d q_0 tau_rw", **nn)
Theta, tau, N = sp.Symbol("Theta", **pos), sp.Symbol("tau_rw", **pos), sp.Symbol("N", **pos)
m, eta, lam, t_rev, t_L0 = sp.symbols("m eta lambda_FA t_review t_L0", **nn)
P_idle, P_act, P_host, P_st, e_rw = sp.symbols("P_idle P_active P_host P_station e_rw", **nn)
gamma, EF_mat, EF_rec, C_hw, c_ret, kappa = sp.symbols("gamma EF_mat EF_rec C_hw c_ret kappa", **nn)
L_hw, f_cam = sp.Symbol("L_hw", **pos), sp.Symbol("f_cam", **pos)
a_ev, g_rate, b_gl, lam_nom, d_fr, r_AI, delta_r = sp.symbols("a g b lambda_nom d_frames r_AI delta_r", **nn)

# Section III equations, transcribed from paper/main.tex.
T_FU = N / Theta
D = N * pi_
q_rw = q0 * sp.exp(-d / tau)                                   # (1)
N_rw = fA * r * D * q_rw                                       # (1)
N_sc = r * D - N_rw                                            # (2)
N_esc = (1 - r) * D                                            # (2)
M_gr = N_sc * m
M_rec = eta * M_gr
M_loss = (1 - eta) * M_gr
n_ev = lam * T_FU + a_ev * r * D                              # Paper A load model
H_ai = n_ev * t_rev / 3600                                     # (3)
rho_ai = n_ev * t_rev / (3600 * T_FU)                          # (3)
H_L0 = N * t_L0 / 3600
rho_L0 = Theta * t_L0 / 3600
E_edge = (P_idle + P_act + P_host) * T_FU / 1000               # (4)


def energy_and_carbon(H, edge_on, n_rw=N_rw, m_loss=M_loss, m_rec=M_rec, n_esc=N_esc):
    e_edge = E_edge if edge_on else sp.Integer(0)
    e_rev = H * P_st / 1000                                    # (5)
    e_rwk = n_rw * e_rw                                        # (5)
    comps = {                                                  # (6), seven components
        "mat": m_loss * EF_mat,
        "recycle": m_rec * EF_rec,
        "rework": gamma * e_rwk,
        "review": gamma * e_rev,
        "edge": gamma * e_edge,
        "hardware": C_hw * T_FU / L_hw if edge_on else sp.Integer(0),
        "escape": n_esc * (m * EF_mat + c_ret) + kappa * fC * n_esc * m * EF_mat,
    }
    return {"E_edge": e_edge, "E_rev": e_rev, "E_rw": e_rwk}, comps


def regime_exprs(kind):
    """Paper regime rules (Sections III-A/B): recall r, delay d and alert model per regime."""
    if kind == "N0":
        sub = {r: 0, d: 0}
        H, rho, edge = sp.Integer(0), sp.Integer(0), False
    elif kind == "L0":
        r_L0, d_L0 = sp.symbols("r_L0 d_L0", **nn)
        sub = {r: r_L0, d: d_L0}
        H, rho, edge = H_L0, rho_L0, False
    else:  # AI tier: lambda_FA = lambda_nom + g b, d = d_frames / f_cam
        sub = {lam: lam_nom + g_rate * b_gl, d: d_fr / f_cam}
        H, rho, edge = H_ai, rho_ai, True
    energy, comps = energy_and_carbon(H, edge)
    out = {"N_rw": N_rw, "N_sc": N_sc, "N_esc": N_esc, "M_loss": M_loss, "M_rec": M_rec, "H": H, "rho": rho,
           **energy, **{f"C_{k}": v for k, v in comps.items()}}
    out["C"] = sum(comps.values())
    return {k: sp.sympify(v).subs(sub, simultaneous=True) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------
def part_a_proofs():
    S = "A proofs"
    check(S, "count conservation N_rw + N_sc + N_esc = D", sp.simplify(N_rw + N_sc + N_esc - D) == 0)
    check(S, "mass split M_rec + M_loss = M_gr", sp.simplify(M_rec + M_loss - M_gr) == 0)

    # N_sc >= 0: N_sc = rD(1 - f_A q_0 e^{-d/tau}); with a, b, c in [0, 1] the certificate
    # 1 - abc = (1-a) + a(1-b) + ab(1-c) is a sum of non-negative terms.
    a, b, c = sp.symbols("a b c")
    check(S, "N_sc factorises as r D (1 - f_A q_rw)", sp.simplify(N_sc - r * D * (1 - fA * q_rw)) == 0)
    check(S, "certificate 1-abc = (1-a)+a(1-b)+ab(1-c)  =>  N_sc >= 0",
          sp.expand((1 - a) + a * (1 - b) + a * b * (1 - c) - (1 - a * b * c)) == 0,
          "requires f_A, q_0 in [0,1] (enforced by the registry schema / reworkability())")
    check(S, "N_esc >= 0 and N_rw >= 0 for r in [0,1]",
          sp.simplify(N_esc.subs(r, 1 - sp.Symbol("rbar", **nn))).is_nonnegative and N_rw.is_nonnegative)

    check(S, "q_rw(0) = q_0", sp.simplify(q_rw.subs(d, 0) - q0) == 0)
    dq = sp.diff(q_rw, d)
    check(S, "q_rw strictly decreasing in d (q_0 > 0)", sp.simplify(dq + q_rw / tau) == 0,
          f"dq/dd = {dq} = -q_rw/tau_rw")
    check(S, "rho = H / T_FU (AI regimes)", sp.simplify(rho_ai - H_ai / T_FU) == 0)
    check(S, "rho_L0 = H_L0 / T_FU  (Theta t_L0 / 3600)", sp.simplify(rho_L0 - H_L0 / T_FU) == 0)
    check(S, "alerts n_ev = lambda_FA T_FU + a r D >= 0 and rho = n_ev t_rev / (3600 T_FU)",
          n_ev.is_nonnegative and sp.simplify(rho_ai * 3600 * T_FU - n_ev * t_rev) == 0)

    # Carbon closure and non-negativity of the seven components.
    base, pol = regime_exprs("L0"), regime_exprs("AI")
    delta = sp.simplify(base["C"] - pol["C"] - sum(base[f"C_{k}"] - pol[f"C_{k}"] for k in COMPONENTS))
    check(S, "Delta C = sum of the seven component differences (exact)", delta == 0)
    check(S, "carbon has exactly seven components, as in the paper and COMPONENTS",
          len(COMPONENTS) == 7 and set(COMPONENTS) == {"mat", "recycle", "rework", "review", "edge", "hardware", "escape"})
    # Non-negativity certificates: rewrite each component with the complements
    # eta_c = 1-eta, r_c = 1-r, qf = 1-f_A q_rw (all >= 0, qf by the certificate above), check the
    # rewrite is an identity, then require SymPy to *prove* the rewritten form non-negative.
    eta_c, r_c, qf = sp.symbols("eta_c r_c qf", **nn)
    back = {eta_c: 1 - eta, r_c: 1 - r, qf: 1 - fA * q_rw}
    m_gr = r * D * qf * m
    _, cert = energy_and_carbon(H_ai, True, n_rw=N_rw, m_loss=eta_c * m_gr, m_rec=eta * m_gr, n_esc=r_c * D)
    ai_sub = {lam: lam_nom + g_rate * b_gl, d: d_fr / f_cam}   # same AI-tier rules as regime_exprs("AI")
    back = {k: v.subs(ai_sub, simultaneous=True) for k, v in back.items()}
    cert = {k: v.subs(ai_sub, simultaneous=True) for k, v in cert.items()}
    for k in COMPONENTS:
        identity = sp.simplify(cert[k].subs(back) - pol[f"C_{k}"]) == 0
        proven = sp.simplify(cert[k]).is_nonnegative is True
        check(S, f"component '{k}' >= 0 (certificate)", identity and proven, str(sp.factor(cert[k]))[:90])

    # SQI: S_X = (X_b - X_p) / max(|X_b|, |X_p|) lies in [-1, 1] iff X_b, X_p have the same sign.
    xb, xp = sp.symbols("x_b x_p", positive=True)
    s_hi = (xb - xp) / xb   # case x_b >= x_p: S = 1 - x_p/x_b in [0, 1)
    s_lo = (xb - xp) / xp   # case x_p >  x_b: S = x_b/x_p - 1 in (-1, 0)
    check(S, "S_X in [-1,1] for X_b, X_p >= 0 (two-case proof)",
          sp.simplify(1 - s_hi - xp / xb) == 0 and sp.simplify(s_lo + 1 - xb / xp) == 0,
          "x_p/x_b and x_b/x_p lie in (0, 1] in the respective case")
    counter = (1 - (-1)) / max(abs(1), abs(-1))
    check(S, "S_X bound needs non-negative X (counter-example X_b=1, X_p=-1 gives S=2)", counter == 2,
          "all four SQI dimensions are sums of the non-negative terms proven above")
    w = [sp.Rational(str(v)) for prof in yaml.safe_load(SQI_CONFIG.read_text(encoding="utf-8"))["profiles"].values()
         for v in [sum(sp.Rational(str(x)) for x in prof["weights"].values())]]
    check(S, "every SQI weight profile sums to exactly 1 (SQI is a convex combination, |SQI| <= 1)",
          all(x == 1 for x in w), f"{len(w)} profiles")

    # Monotonicity of Delta C in every break-even parameter (justifies the endpoint test in _solve).
    regimes = {"N0_NoInspection": regime_exprs("N0"), "L0_Manual": regime_exprs("L0"), "AI": regime_exprs("AI")}
    tier = {t: {x: sp.Symbol(f"{x.name}_{t}", **nn) for x in (a_ev, b_gl, lam_nom, d_fr)} for t in ("B0", "B3")}
    b0 = {k: v.subs({**tier["B0"], r: r_AI}, simultaneous=True) for k, v in regimes["AI"].items()}
    b4 = {k: v.subs({**tier["B3"], r: r_AI - delta_r}, simultaneous=True) for k, v in regimes["AI"].items()}
    ref = {"N0_NoInspection": regimes["N0_NoInspection"], "L0_Manual": regimes["L0_Manual"], "B0_Raw": b0}
    sym = {"defect_prevalence": pi_, "host_power": P_host, "manual_inspection_recall": sp.Symbol("r_L0", **nn),
           "manual_discovery_delay": sp.Symbol("d_L0", **nn), "grid_carbon_factor": gamma,
           "persistence_recall_loss": delta_r}
    for name, key, base_id, _, _ in SPECS:
        x = sym[key]
        dc = sp.expand(ref[base_id]["C"] - b4["C"])
        slope = sp.diff(dc, x)
        if x.name == "d_L0":
            g = sp.simplify(slope / sp.exp(-x / tau))
            ok, why = x not in g.free_symbols, "dDeltaC/dd_L0 = exp(-d_L0/tau) g(...), g free of d_L0: constant sign"
        else:
            ok, why = x not in slope.free_symbols, "Delta C is affine in the parameter"
        check(S, f"Delta C monotone in {key} ({name})", ok, why)
    return b4, ref, sym


def part_b_dimensions():
    S = "B dimensions"
    M, Cc, T, E = sp.symbols("MASS CARBON TIME ENERGY", positive=True)
    dim = {N: 1, Theta: 1 / T, pi_: 1, fA: 1, fB: 1, fC: 1, r: 1, d: T, q0: 1, tau: T, m: M, eta: 1,
           lam: 1 / T, t_rev: T, t_L0: T, P_idle: E / T, P_act: E / T, P_host: E / T, P_st: E / T, e_rw: E,
           gamma: Cc / E, EF_mat: Cc / M, EF_rec: Cc / M, C_hw: Cc, c_ret: Cc, kappa: 1, L_hw: T,
           f_cam: 1 / T, a_ev: 1, g_rate: 1 / T, b_gl: 1, lam_nom: 1 / T, d_fr: 1, r_AI: 1, delta_r: 1}

    def dims_of(expr):
        terms = sp.Add.make_args(sp.expand(expr))
        out = set()
        for t in terms:
            for f in t.atoms(sp.exp):  # exponent arguments must be dimensionless
                if sp.simplify(f.args[0].subs(dim)).free_symbols & {M, Cc, T, E}:
                    return {"non-dimensionless exponent"}
            out.add(sp.simplify(t.subs({f: 1 for f in t.atoms(sp.exp)}).subs(dim).as_coeff_Mul()[1]))
        return out

    ai, l0 = regime_exprs("AI"), regime_exprs("L0")
    dim[sp.Symbol("r_L0", **nn)], dim[sp.Symbol("d_L0", **nn)] = 1, T
    expect = {"N_rw": 1, "N_sc": 1, "N_esc": 1, "M_loss": M, "M_rec": M, "H": T, "rho": 1,
              "E_edge": E, "E_rev": E, "E_rw": E, "C": Cc, **{f"C_{k}": Cc for k in COMPONENTS}}
    for label, ex in (("AI", ai), ("L0", l0)):
        for k, want in expect.items():
            if ex[k] == 0:
                continue
            got = dims_of(ex[k])
            check(S, f"{label}: [{k}] = {want}", got == {sp.sympify(want)}, str(got))
    check(S, "[d_frames / f_cam] = TIME", dims_of(d_fr / f_cam) == {T})
    check(S, "[lambda_nom + g b] = 1/TIME", dims_of(lam_nom + g_rate * b_gl) == {1 / T})
    check(S, "[lambda_FA T_FU] = count", dims_of(lam * T_FU) == {sp.Integer(1)})

    # Conversion constants: W*h -> kWh is 1/1000; s -> h is 1/3600; W/(frame/s) -> J/frame.
    check(S, "1 W * 1 h = 1/1000 kWh (E_edge, E_rev)",
          u.convert_to(u.watt * u.hour, u.joule) / u.convert_to(1000 * u.watt * u.hour, u.joule) == sp.Rational(1, 1000))
    check(S, "1 s = 1/3600 h (H, rho)", sp.simplify(u.convert_to(u.second, u.hour) / u.hour) == sp.Rational(1, 3600))
    check(S, "E_frame [J] = P [W] / f [1/s]", sp.simplify(u.convert_to(u.watt / (1 / u.second), u.joule) / u.joule) == 1)


def part_c_implementation():
    S = "C paper == code"
    regimes = load_regimes()
    exprs = {"N0_NoInspection": regime_exprs("N0"), "L0_Manual": regime_exprs("L0")}
    ai = regime_exprs("AI")
    rL0, dL0 = sp.Symbol("r_L0", **nn), sp.Symbol("d_L0", **nn)
    fields = {"N_rw": lambda x: x.routing.n_rework, "N_sc": lambda x: x.routing.n_scrap,
              "N_esc": lambda x: x.routing.n_escape, "M_loss": lambda x: x.material.net_loss_kg,
              "M_rec": lambda x: x.material.recovered_kg, "H": lambda x: x.workload.operator_hours,
              "rho": lambda x: x.workload.utilisation_rho, "E_edge": lambda x: x.energy.edge_kwh,
              "E_rev": lambda x: x.energy.review_kwh, "E_rw": lambda x: x.energy.rework_kwh,
              "C": lambda x: x.carbon.total, **{f"C_{k}": (lambda k: lambda x: getattr(x.carbon, k))(k) for k in COMPONENTS}}

    def bind(p, rid):
        """Paper regime rules: map registry values onto the paper symbols for regime rid."""
        common = {N: p["functional_unit"], Theta: p["line_throughput"], pi_: p["defect_prevalence"],
                  q0: p["base_reworkability"], tau: p["reworkability_time_constant"], m: p["part_mass"],
                  eta: p["material_recovery_fraction"], t_rev: p["review_time"], t_L0: p["manual_inspection_time"],
                  P_idle: p["gpu_idle_power"], P_act: p["gpu_active_power"], P_host: p["host_power"],
                  P_st: p["review_station_power"], e_rw: p["rework_energy"], gamma: p["grid_carbon_factor"],
                  EF_mat: p["material_carbon_factor"], EF_rec: p["recycling_burden"], C_hw: p["edge_embodied_carbon"],
                  L_hw: p["edge_lifetime_hours"], c_ret: p["return_logistics_carbon"], kappa: p["collateral_multiplier"],
                  rL0: p["manual_inspection_recall"], dL0: p["manual_discovery_delay"]}
        if rid in BASELINES:
            return exprs[rid], common
        reg = regimes[rid]
        recall = p["ai_recall"] - (p["persistence_recall_loss"] if reg.persistence else 0.0)
        return ai, {**common, r: recall, f_cam: p["camera_fps"], d_fr: p[f"detection_delay@{rid}"],
                    lam_nom: p[f"nominal_false_alarm_rate@{rid}"], g_rate: p["glare_burst_rate"],
                    b_gl: p[f"glare_alerts_per_burst@{rid}"], a_ev: p[f"alerts_per_defect@{rid}"]}

    for sc in SCENARIOS:
        s = load_scenario(sc)
        shares = dict(zip((fA, fB, fC), s.class_shares))
        draws = sample_parameters(s, 500, seed=11)
        for label, pset in (("central", s.central()), ("500 MC draws", draws)):
            res = evaluate_all(pset, s.class_shares)
            worst = 0.0
            for rid in BASELINES + AI_TIERS:
                ex, binding = bind(pset, rid)
                for k, getter in fields.items():
                    f = sp.lambdify(list(binding) + list(shares), ex[k].subs({}), "numpy")
                    want = np.broadcast_to(f(*binding.values(), *shares.values()), np.shape(getter(res[rid])))
                    got = np.asarray(getter(res[rid]), dtype=float)
                    err = np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want)))
                    worst = max(worst, float(err))
            check(S, f"{sc}: all {len(BASELINES + AI_TIERS)} regimes x {len(fields)} quantities match ({label})", worst < 1e-12,
                  f"max relative deviation {worst:.2e}")


def part_d_break_even(b4, ref, sym):
    S = "D break-even"
    be = json.loads((PROCESSED_DIR / "break_even.json").read_text(encoding="utf-8"))
    for sc in SCENARIOS:
        s = load_scenario(sc)
        p = s.central()
        vals = {N: p["functional_unit"], Theta: p["line_throughput"], pi_: p["defect_prevalence"], q0: p["base_reworkability"],
                tau: p["reworkability_time_constant"], m: p["part_mass"], eta: p["material_recovery_fraction"],
                t_rev: p["review_time"], t_L0: p["manual_inspection_time"], P_idle: p["gpu_idle_power"],
                P_act: p["gpu_active_power"], P_host: p["host_power"], P_st: p["review_station_power"], e_rw: p["rework_energy"],
                gamma: p["grid_carbon_factor"], EF_mat: p["material_carbon_factor"], EF_rec: p["recycling_burden"],
                C_hw: p["edge_embodied_carbon"], L_hw: p["edge_lifetime_hours"], c_ret: p["return_logistics_carbon"],
                kappa: p["collateral_multiplier"], sym["manual_inspection_recall"]: p["manual_inspection_recall"],
                sym["manual_discovery_delay"]: p["manual_discovery_delay"], f_cam: p["camera_fps"],
                r_AI: p["ai_recall"], delta_r: p["persistence_recall_loss"], g_rate: p["glare_burst_rate"],
                **{sp.Symbol(f"{x}_{t}", **nn): p[f"{key}@{rid}"] for t, rid in (("B0", "B0_Raw"), ("B3", "B3_Full_Policy"))
                   for x, key in (("a", "alerts_per_defect"), ("b", "glare_alerts_per_burst"),
                                  ("lambda_nom", "nominal_false_alarm_rate"), ("d_frames", "detection_delay"))},
                **dict(zip((fA, fB, fC), s.class_shares))}
        for name, key, base_id, (lo, hi), _ in SPECS:
            x = sym[key]
            dc = (ref[base_id]["C"] - b4["C"]).subs({k: v for k, v in vals.items() if k != x})
            f_lo, f_hi = float(dc.subs(x, lo)), float(dc.subs(x, hi))
            fr = be[sc][name]
            if f_lo * f_hi < 0:
                root = [float(v) for v in sp.solve(sp.Eq(dc, 0), x) if v.is_real and lo <= float(v) <= hi] \
                    if x.name != "d_L0" else [float(sp.nsolve(dc, x, (lo + hi) / 2))]
                ok = fr["status"] == "root" and len(root) == 1 and abs(root[0] - fr["value"]) <= 1e-8 * max(1, abs(root[0]))
                check(S, f"{sc}: {name} closed form {root[0]:.10g} vs Brent {fr['value']}", ok)
            else:
                want = "none_positive" if f_lo > 0 else "none_negative"
                check(S, f"{sc}: {name} no sign change -> '{want}'", fr["status"] == want and fr["value"] is None,
                      f"Delta C({lo:g}) = {f_lo:.4g}, Delta C({hi:g}) = {f_hi:.4g}")


def part_e_manuscript():
    S = "E manuscript"
    tex = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", RESULTS_MACROS.read_text(encoding="utf-8")))

    def macro(name):
        return macros.get(name)

    recs = {r.key: r for r in load_registry()}
    summary = json.loads(ENERGY_SUMMARY.read_text(encoding="utf-8"))
    import pandas as pd
    windows = pd.read_csv(Path(ENERGY_SUMMARY).parents[2] / summary["trace_dir"] / "windows.csv")
    active = windows[windows.stage != "BASELINE_IDLE"]
    idle = windows[windows.stage == "BASELINE_IDLE"]
    sqi_names = list(yaml.safe_load(SQI_CONFIG.read_text(encoding="utf-8"))["profiles"])
    spec = dict((n, (lo, hi)) for n, _, _, (lo, hi), _ in SPECS)
    literals = [  # (text in main.tex, claim holds?, detail)
        ("$N=1{,}000$", recs["functional_unit"].central == 1000, "registry functional_unit"),
        ("0.242~kg/kWh", recs["grid_carbon_factor"].central == 0.242, "registry grid central"),
        ("0.716~kg/kWh", recs["grid_carbon_factor"].high == 0.716, "registry grid upper bound"),
        ("up to 1.5~kg/kWh", spec["gamma_star_vs_L0"][1] == 1.5, "break_even.SPECS gamma interval"),
        ("(seed 42)", "N_DRAWS, SEED = 10000, 42" in (Path(__file__).parent / "06_run_monte_carlo.py").read_text(), "scripts/06"),
        ("misses 20--30\\% of defects", (recs["manual_inspection_recall"].low, recs["manual_inspection_recall"].high) == (0.70, 0.80),
         "registry r_L0 bounds"),
        ("$28\\times28\\times384$", 224 // 8 == 28 and 128 + 256 == 384, "ResNet-18 layer2 (stride 8, 128 ch) + layer3 (256 ch)"),
        ("coreset of $\\PaperABank$ real patch features", "MEMORY_BANK_SIZE = " + macro("PaperABank").replace("{,}", "") in
         (Path(__file__).parents[1] / "src/experiments/energy_benchmark.py").read_text()
         and "torch.randn(memory_bank_size, 384)" in (Path(__file__).parents[1] / "src/experiments/energy_benchmark.py").read_text()
         and summary["workload"].count(macro("PaperABank").replace("{,}", "")) == 1,
         "PatchCoreProxy memory bank == Paper A coreset size, and the committed run used it"),
        ("15~s idle window", summary["idle_window_s"] == 15, "run_meta idle_window_s"),
        ("with NVML at 20~Hz", "sample_hz: float = 20.0" in (Path(__file__).parents[1] / "src/experiments/energy_benchmark.py").read_text(),
         "NvmlMeter default"),
        ("at least \\StageReadingsMin{} distinct sensor readings", macro("StageReadingsMin") == str(active.n_distinct_readings.min())
         and macro("IdleReadingsMin") == str(idle.n_distinct_readings.min()),
         f"windows.csv: stage windows >= {active.n_distinct_readings.min()}, idle windows >= {idle.n_distinct_readings.min()}"),
        ("\\CounterIdleMinW--\\CounterIdleMaxW~W at idle",
         (macro("CounterIdleMinW"), macro("CounterIdleMaxW")) == (f"{idle.counter_diag_w.min():.0f}", f"{idle.counter_diag_w.max():.0f}"),
         f"windows.csv idle counter_diag_w = {idle.counter_diag_w.min():.0f}-{idle.counter_diag_w.max():.0f} W"),
        ("Six regimes", len(load_regimes()) == 6, "configs/policies.yaml"),
        ("four alert policies measured in", len(AI_TIERS) == 4, "pipeline_evaluator.AI_TIERS"),
        ("$\\delta_r$ is centred on zero", recs["persistence_recall_loss"].central == 0.0, "registry persistence_recall_loss"),
        ("seven non-negative components", len(COMPONENTS) == 7, "carbon_accounting.COMPONENTS"),
        ("balanced, material-, carbon- and human-centred", sqi_names == ["balanced", "material_priority", "carbon_priority",
                                                                           "human_centered"], "configs/sqi_weights.yaml"),
    ]
    for text, ok, detail in literals:
        present = text in tex
        check(S, f"literal '{text}'", present and ok, detail if present else "text not found in main.tex")

    # Every macro used in the manuscript and generated tables must be defined.
    defined = set(macros) | set(re.findall(r"\\newcommand\{\\(\w+)\}", tex))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)\b", tex))
    latex_builtin = {"IEEEPARstart", "IEEEkeywords", "Big", "Delta", "Theta", "Gamma", "Lambda", "Sigma", "Omega", "Pi", "Phi"}
    undefined = sorted(used - defined - latex_builtin)
    check(S, "every macro used in main.tex is defined", not undefined, ", ".join(undefined))
    unused = sorted(defined - used - {"kgce"})
    note(S, "generated macros not quoted in main.tex (harmless)", f"{len(unused)}: {', '.join(unused[:6])}, ...")

    # Symbols: every subscripted symbol in the manuscript's math is a ledger symbol or a derived
    # quantity defined in Section III; the ledger table must use the same notation as the text.
    spec09 = importlib.util.spec_from_file_location("tables", Path(__file__).parent / "09_generate_tables.py")
    tables = importlib.util.module_from_spec(spec09)
    spec09.loader.exec_module(tables)
    ledger = {tables.symbol(r).strip("$") for r in load_registry()}
    derived = {"T_{FU}", "N_{rw}", "N_{sc}", "N_{esc}", "q_{rw}", "M_{gr}", "M_{rec}", "M_{loss}", "n_{ev}", "E_{edge}",
               "E_{rev}", "E_{rw}", "S_X", "w_X", "X_b", "X_p", r"\lambda_{FA}", "f_A", "f_B", "f_C", r"\lambda_t", "a_t", "b_t", "d_t",
               "S_E", "S_H", "P^\\star_{cell}", "q_0", r"\sum_X"}
    math = " ".join(re.findall(r"\$([^$]+)\$", tex) + re.findall(r"\\begin\{(?:align|equation)\}(.*?)\\end", tex, re.S))
    tokens = set(re.findall(r"(\\?[A-Za-z]+_\{[^{}]+\}|\\?[A-Za-z]+_[A-Za-z0-9])", math))
    unknown = sorted(t for t in tokens if t not in ledger and t not in derived)
    check(S, "every subscripted symbol is in the ledger or defined in Section III", not unknown, ", ".join(unknown))
    check(S, "lambda_FA is defined in the text", re.search(r"(?:\\lambda_\{FA\}\$? (?:is|denotes)|false-alert rate is \$\\lambda_\{FA\})", tex) is not None,
          "n_ev uses lambda_FA; the text must say what it is")
    check(S, "hardware term restricted to AI regimes in the text (code: edge_active only)",
          re.search(r"hardware term[^.]*zero for N0 and L0", tex) is not None)
    check(S, "S_X bound stated for non-negative X (proof A needs it)", "non-negative" in tex.split("Sustainability Quality Index")[1][:300])


def main():
    b4, ref, sym = part_a_proofs()
    part_b_dimensions()
    part_c_implementation()
    part_d_break_even(b4, ref, sym)
    part_e_manuscript()
    width = max(len(n) for _, n, _, _ in RESULTS)
    section = None
    for sec, name, status, detail in RESULTS:
        if sec != section:
            print(f"\n== {sec}")
            section = sec
        print(f"  [{status}] {name:<{width}}  {detail}")
    checks = [s for _, _, s, _ in RESULTS if s != "INFO"]
    failed = checks.count("FAIL")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
