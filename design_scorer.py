"""
Holistic design scorer: generate 100k random tokamak designs,
score each against absolute engineering desirability functions,
return top 100. No ML, no data.
"""

import math
import random
import time
from physics_engine import quick_eval

SEED = 42
N_SAMPLES = 500_000
TOP_K = 100

DESIGN_BOUNDS = {
    "R0": (0.3, 8.0),
    "a": (0.2, 2.5),
    "B0": (0.2, 20.0),
    "Ip": (100.0, 20000.0),
    "elongation": (1.0, 3.0),
    "triangularity_upper": (0.0, 0.8),
    "triangularity_lower": (0.0, 0.8),
    "q95": (2.5, 6.0),
}
PARAM_NAMES = list(DESIGN_BOUNDS.keys())

DEFAULT_S = 1.0

METRIC_SPECS = [
    ("B_coil_max_T",        "LB", 6.0,    20.0,   2.0,     2.0),
    ("beta_N",              "LB", 2.0,    4.0,    4.0,     2.0),
    ("disruption_prob",     "LB", 0.05,   0.6,    1.5,     1.5),
    ("neutron_wall_MWm2",   "LB", 0.5,    5.0,    1.5,     1.5),
    ("TBR",                 "HB", 0.80,   1.05,   2.0,     1.5),
    ("Q",                   "HB", 1.0,    8.0,    0.7,     1.5),
    ("stability_score",     "HB", -3.0,   2.0,    1.0,     1.0),
    ("P_net_electric_MW",   "HB", 0.0,    200.0,  0.7,     1.0),
    ("cost_MS",             "LB", 1500.0, 12000.0, 0.7,     1.0),
    ("TF_ripple_pct",       "LB", 0.3,    3.0,    1.0,     0.7),
    ("q_div_MWm2",          "LB", 2.0,    10.0,   0.7,     0.7),
    ("volume_m3",           "LB", 30.0,   1500.0, 0.5,     0.5),
    ("q95_margin",          "HB", 0.0,    1.0,    0.7,     0.3),
    ("density_margin",      "HB", 0.0,    1.0,    0.7,     0.3),
    ("lh_margin",           "HB", -1.0,   0.5,    0.7,     0.2),
    ("burn_time_s",         "HB", 30.0,   500.0,  0.3,     0.3),
    ("f_bs",                "HB", 0.03,   0.4,    0.5,     0.2),
    ("tau_E_s",             "HB", 0.2,    2.0,    0.5,     0.2),
    ("tritium_burn_frac",   "HB", 0.001,  0.03,   0.5,     0.2),
]


def desirability(value, direction, L, T_U, U_L, exponent):
    if direction == "HB":
        if value <= L:
            return 0.0
        if value >= T_U:
            return 1.0
        return ((value - L) / (T_U - L)) ** exponent

    elif direction == "LB":
        if value <= T_U:
            return 1.0
        if value >= U_L:
            return 0.0
        return ((U_L - value) / (U_L - T_U)) ** exponent

    return 0.0


def random_designs(n, seed=SEED):
    rng = random.Random(seed)
    return [{name: rng.uniform(lo, hi) for name, (lo, hi) in DESIGN_BOUNDS.items()}
            for _ in range(n)]


def evaluate_batch(designs):
    return [quick_eval(d) for d in designs]


def compute_desirabilities(results):
    out = []
    for r in results:
        d_i = []
        for metric, direction, L_or_T, T_or_U, exponent, weight in METRIC_SPECS:
            val = r[metric]
            if direction == "HB":
                d = desirability(val, "HB", L_or_T, T_or_U, None, exponent)
            else:
                d = desirability(val, "LB", None, L_or_T, T_or_U, exponent)
            d = max(d, 1e-9)
            d_i.append((d, weight))
        out.append(d_i)
    return out


def overall_desirability(d_list):
    log_sum = 0.0
    w_sum = 0.0
    for d, w in d_list:
        log_sum += w * math.log(d)
        w_sum += w
    return math.exp(log_sum / w_sum) if w_sum > 0 else 0.0


def print_design(d_params, r, score):
    param_names = ["R0", "a", "B0", "Ip", "elongation",
                   "triangularity_upper", "triangularity_lower", "q95"]
    print(f"  Parameters:")
    for pn in param_names:
        unit = "m" if pn in ("R0", "a") else ("T" if pn == "B0" else ("kA" if pn == "Ip" else ""))
        val = d_params[pn]
        print(f"    {pn:20s} = {val:.4f}  [{unit}]" if unit else f"    {pn:20s} = {val:.4f}")

    print(f"\n  Desirability: {score:.4f}")
    print(f"  Key outputs:")
    for key, direction, l, t, exp, w in METRIC_SPECS:
        val = r[key]
        d = desirability(val, direction, l, t, None if direction == "HB" else t,
                         exp) if direction == "HB" else desirability(val, direction, None, l, t, exp)
        d = max(d, 1e-9)
        print(f"    {key:25s} = {val:.6e}  d={d:.4f}" if isinstance(val, float) and (abs(val) < 0.001 or abs(val) > 1e6)
              else f"    {key:25s} = {val:.6f}  d={d:.4f}")
    print()


def main():
    print(f"Generating {N_SAMPLES:,} random designs...")
    t0 = time.perf_counter()
    designs = random_designs(N_SAMPLES)
    t1 = time.perf_counter()

    print(f"Evaluating...")
    results = evaluate_batch(designs)
    t2 = time.perf_counter()

    print(f"Scoring via Harrington desirability ({len(METRIC_SPECS)} metrics)...")
    d_list = compute_desirabilities(results)
    scores = [overall_desirability(ds) for ds in d_list]
    t3 = time.perf_counter()

    print(f"\n{'='*70}")
    print(f"Timing:")
    print(f"  Generate: {t1-t0:.3f}s | Evaluate: {t2-t1:.3f}s | Score: {t3-t2:.3f}s | Total: {t3-t0:.3f}s")
    print(f"{'='*70}")

    ranked = sorted(range(N_SAMPLES), key=lambda i: scores[i], reverse=True)

    print(f"\n{'='*70}")
    print(f"TOP {TOP_K} DESIGNS — Harrington desirability (absolute engineering targets)")
    print(f"{'='*70}")
    for rank in range(min(TOP_K, 20)):
        idx = ranked[rank]
        print(f"\n{'─'*70}")
        print(f"RANK #{rank+1:3d} | Overall D: {scores[idx]:.4f} | "
              f"Q={results[idx]['Q']:.2f} β_N={results[idx]['beta_N']:.2f} "
              f"Bmax={results[idx]['B_coil_max_T']:.1f}T vol={results[idx]['volume_m3']:.0f}m³ "
              f"nwall={results[idx]['neutron_wall_MWm2']:.2f}MW/m² "
              f"cost={results[idx]['cost_MS']:.0f}M$ "
              f"burn={results[idx]['burn_time_s']:.0f}s")
        print_design(designs[idx], results[idx], scores[idx])

    with open("top100_designs.txt", "w") as f:
        f.write(f"TOP {TOP_K} DESIGNS out of {N_SAMPLES:,} random candidates\n")
        f.write("Scored via Harrington desirability against absolute engineering targets\n\n")
        for rank in range(TOP_K):
            idx = ranked[rank]
            d = designs[idx]
            r = results[idx]
            f.write(f"Rank #{rank+1:3d} | Overall D: {scores[idx]:.6f}\n")
            for pn in ["R0", "a", "B0", "Ip", "elongation",
                        "triangularity_upper", "triangularity_lower", "q95"]:
                f.write(f"  {pn:20s} = {d[pn]:.6f}\n")
            for key, direction, l, t, exp, w in METRIC_SPECS:
                val = r[key]
                dv = desirability(val, direction, l, t, None if direction == "HB" else t,
                                  exp) if direction == "HB" else desirability(val, direction, None, l, t, exp)
                f.write(f"  {key:25s} = {val:.6e}  d={max(dv,1e-9):.4f}\n" if isinstance(val, float) and (abs(val) < 0.001 or abs(val) > 1e6)
                        else f"  {key:25s} = {val:.6f}  d={max(dv,1e-9):.4f}\n")
            f.write("\n")
    print(f"\nSaved to top100_designs.txt")


if __name__ == "__main__":
    main()
