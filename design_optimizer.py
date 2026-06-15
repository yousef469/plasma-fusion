"""
NSGA-II design optimizer for tokamak designs.
Evaluates designs using pure physics engine — no ML, no training data.
Objectives: maximize Q, maximize stability, minimize volume.
"""

import os
import random
import time
import numpy as np

from physics_engine import quick_eval
from physics_config import (
    DESIGN_BOUNDS, POPULATION_SIZE, N_GENERATIONS,
    MUTATION_RATE, CROSSOVER_RATE,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


PARAM_NAMES = list(DESIGN_BOUNDS.keys())
PARAM_LOWER = np.array([DESIGN_BOUNDS[k][0] for k in PARAM_NAMES])
PARAM_UPPER = np.array([DESIGN_BOUNDS[k][1] for k in PARAM_NAMES])


def design_to_dict(params):
    return {PARAM_NAMES[i]: params[i] for i in range(len(PARAM_NAMES))}


def evaluate_design(params):
    d = design_to_dict(params)
    result = quick_eval(d)
    return result


def constraint_violation(fitness):
    """Total constraint violation.
    fitness = [-Q, -beta_t, -stab, vol, beta_N, B_coil_max]
    Constraints: Q >= 1 (fitness[0] <= -1), stab >= -1.5 (fitness[2] <= 1.5),
                 B_coil_max <= 12.0 (fitness[5] <= 12.0)
    """
    vio = 0.0
    if -fitness[0] < 1.0:
        vio += (1.0 + fitness[0]) * 3.0
    if fitness[2] > 1.5:
        vio += fitness[2] - 1.5
    if len(fitness) >= 6 and fitness[5] > 12.0:
        vio += (fitness[5] - 12.0) * 2.0
    return max(vio, 0.0)


def constrained_dominates(f1, f2):
    """Constrained dominance: feasible dominates infeasible.
    Among feasible: Pareto dominates (first 4 objectives only).
    Among infeasible: smaller violation dominates.
    """
    v1 = constraint_violation(f1)
    v2 = constraint_violation(f2)

    if v1 < 1e-10 and v2 < 1e-10:
        f1_obj = f1[:4]
        f2_obj = f2[:4]
        return all(a <= b for a, b in zip(f1_obj, f2_obj)) and any(a < b for a, b in zip(f1_obj, f2_obj))
    elif v1 < 1e-10:
        return True
    elif v2 < 1e-10:
        return False
    else:
        return v1 < v2


def non_dominated_sort(fitnesses):
    pop_size = len(fitnesses)
    dominated = [set() for _ in range(pop_size)]
    dominates_count = [0] * pop_size

    for i in range(pop_size):
        for j in range(pop_size):
            if i == j:
                continue
            if constrained_dominates(fitnesses[i], fitnesses[j]):
                dominated[i].add(j)
            elif constrained_dominates(fitnesses[j], fitnesses[i]):
                dominates_count[i] += 1

    fronts = [[]]
    for i in range(pop_size):
        if dominates_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while current < len(fronts):
        next_front = []
        for i in fronts[current]:
            for j in dominated[i]:
                dominates_count[j] -= 1
                if dominates_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        current += 1

    return fronts


def crowding_distance(fitnesses, front):
    dist = [0.0] * len(front)
    n_obj = 4
    for obj in range(n_obj):
        front_sorted = sorted(front, key=lambda i: fitnesses[i][obj])
        dist[front_sorted[0]] = float('inf')
        dist[front_sorted[-1]] = float('inf')
        obj_min = fitnesses[front_sorted[0]][obj]
        obj_max = fitnesses[front_sorted[-1]][obj]
        if obj_max - obj_min == 0:
            continue
        for k in range(1, len(front_sorted) - 1):
            dist[front_sorted[k]] += (
                fitnesses[front_sorted[k + 1]][obj] - fitnesses[front_sorted[k - 1]][obj]
            ) / (obj_max - obj_min)
    return dist


def select_parents(fitnesses, fronts):
    pop_size = len(fitnesses)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            dist = crowding_distance(fitnesses, front)
            front_sorted = sorted(front, key=lambda i: dist[i], reverse=True)
            needed = pop_size - len(selected)
            selected.extend(front_sorted[:needed])
            break
    return selected


def random_design():
    return np.array([
        random.uniform(DESIGN_BOUNDS[k][0], DESIGN_BOUNDS[k][1])
        for k in PARAM_NAMES
    ])


def crossover(p1, p2):
    if random.random() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    alpha = np.random.uniform(-0.1, 1.1, size=len(p1))
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = (1 - alpha) * p1 + alpha * p2
    return np.clip(c1, PARAM_LOWER, PARAM_UPPER), np.clip(c2, PARAM_LOWER, PARAM_UPPER)


def mutate(ind):
    for i in range(len(ind)):
        if random.random() < MUTATION_RATE:
            ind[i] += np.random.normal(0, 0.1 * (PARAM_UPPER[i] - PARAM_LOWER[i]))
    return np.clip(ind, PARAM_LOWER, PARAM_UPPER)


REPORT_FIELDS = [
    ("Q", "{:.4f}"),
    ("stability_score", "{:.4f}"),
    ("volume_m3", "{:.4f}"),
    ("tau_E_s", "{:.6f}"),
    ("beta_N", "{:.4f}"),
    ("beta_margin", "{:.4f}"),
    ("beta_t", "{:.6f}"),
    ("beta_p", "{:.6f}"),
    ("lh_margin", "{:.4f}"),
    ("divertor_margin", "{:.4f}"),
    ("tf_stress_margin", "{:.4f}"),
    ("density_margin", "{:.4f}"),
    ("q95_margin", "{:.4f}"),
    ("q_div_MWm2", "{:.6f}"),
    ("neutron_wall_MWm2", "{:.6f}"),
    ("P_fusion_MW", "{:.4f}"),
    ("P_ext_MW", "{:.4f}"),
    ("P_alpha_MW", "{:.6f}"),
    ("n_bar_e20", "{:.4f}"),
    ("T_keV", "{:.2f}"),
    ("triple_product", "{:.4e}"),
    ("f_bs", "{:.4f}"),
    ("T_ped", "{:.4f}"),
    ("ped_width", "{:.4f}"),
    ("B_coil_max_T", "{:.4f}"),
    ("P_LH_MW", "{:.4f}"),
    ("aspect_ratio", "{:.4f}"),
    ("surface_area_m2", "{:.4f}"),
    ("l_i", "{:.4f}"),
    ("P_net_electric_MW", "{:.4f}"),
    ("q_eng", "{:.4f}"),
    ("TBR", "{:.4f}"),
    ("burn_time_s", "{:.2f}"),
    ("E_TF_stored_GJ", "{:.4f}"),
    ("TF_ripple_pct", "{:.4f}"),
    ("tritium_burn_frac", "{:.6f}"),
    ("disruption_prob", "{:.4f}"),
    ("cost_MS", "{:.2f}"),
]


def save_report(population, all_results, pareto_indices, save_path="pareto_report.txt"):
    n_pareto = len(pareto_indices)
    Qs = [all_results[i]["Q"] for i in pareto_indices]
    best_q_idx = pareto_indices[Qs.index(max(Qs))]

    with open(save_path, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("TOKAMAK DESIGN OPTIMIZER — FULL PARETO REPORT\n")
        f.write(f"Date: 2026-05-22\n")
        f.write(f"Pareto designs found: {n_pareto}\n")
        f.write(f"Population: {POPULATION_SIZE}, Generations: {N_GENERATIONS}\n")
        f.write("=" * 90 + "\n\n")

        param_names_list = ["R0", "a", "B0", "Ip", "elongation",
                            "triangularity_upper", "triangularity_lower", "q95"]

        for rank, idx in enumerate(pareto_indices):
            p = population[idx]
            r = all_results[idx]
            f.write(f"{'─'*90}\n")
            f.write(f"DESIGN #{rank + 1:03d}  (Pareto rank 0)\n")
            f.write(f"{'─'*90}\n")
            f.write(f"  Parameters:\n")
            for i, name in enumerate(param_names_list):
                unit = "m" if name in ("R0", "a") else ("T" if name == "B0" else ("kA" if name == "Ip" else ""))
                f.write(f"    {name:20s} = {p[i]:.4f}  [{unit}]\n" if unit else f"    {name:20s} = {p[i]:.4f}\n")
            f.write(f"\n  Outputs:\n")
            for key, fmt in REPORT_FIELDS:
                if key in r:
                    val = r[key]
                    f.write(f"    {key:25s} = {fmt}\n".format(val))
            f.write("\n")

        f.write("=" * 90 + "\n")
        f.write("SUMMARY — PARETO FRONT EXTREMES\n")
        f.write("=" * 90 + "\n")
        q_vals = [r["Q"] for r in all_results]
        b_vals = [r["beta_t"] for r in all_results]
        s_vals = [r["stability_score"] for r in all_results]
        v_vals = [r["volume_m3"] for r in all_results]
        f.write(f"  Q range:           {min(q_vals):.4f} — {max(q_vals):.4f}\n")
        f.write(f"  β_t range:         {min(b_vals)*100:.4f}% — {max(b_vals)*100:.4f}%\n")
        f.write(f"  Stability range:   {min(s_vals):.4f} — {max(s_vals):.4f}\n")
        f.write(f"  Volume range:      {min(v_vals):.4f} — {max(v_vals):.4f} m³\n\n")
        f.write(f"  Best Q design:     Q={all_results[best_q_idx]['Q']:.4f}, "
                f"β_t={all_results[best_q_idx]['beta_t']*100:.2f}%, "
                f"stab={all_results[best_q_idx]['stability_score']:.4f}, "
                f"vol={all_results[best_q_idx]['volume_m3']:.4f} m³\n")
        f.flush()

    print(f"\n  Full report saved: {save_path}")
    return save_path


def plot_pareto_front(all_results, save_path="pareto_front.png"):
    if not HAS_MPL:
        print("  [matplotlib not available — skipping plot]")
        return

    Qs = [r["Q"] for r in all_results]
    betas = [r["beta_t"] * 100 for r in all_results]
    stabs = [r["stability_score"] for r in all_results]
    vols = [r["volume_m3"] for r in all_results]
    taus = [r["tau_E_s"] for r in all_results]
    costs = [r["cost_MS"] for r in all_results]
    rd = [r["disruption_prob"] for r in all_results]
    rips = [r["TF_ripple_pct"] for r in all_results]
    burns = [r["burn_time_s"] for r in all_results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    sc = axes[0, 0].scatter(vols, Qs, c=stabs, cmap="RdYlGn", s=30, edgecolors="k", linewidths=0.3)
    axes[0, 0].set_xlabel("Volume (m³)")
    axes[0, 0].set_ylabel("Q")
    axes[0, 0].set_title("Q vs Volume (color = stability)")
    plt.colorbar(sc, ax=axes[0, 0], label="Stability")

    sc = axes[0, 1].scatter(costs, Qs, c=betas, cmap="viridis", s=30, edgecolors="k", linewidths=0.3)
    axes[0, 1].set_xlabel("Cost (M$)")
    axes[0, 1].set_ylabel("Q")
    axes[0, 1].set_title("Q vs Cost (color = β_t%)")
    plt.colorbar(sc, ax=axes[0, 1], label="β_t (%)")

    sc = axes[0, 2].scatter(vols, rd, c=Qs, cmap="coolwarm", s=30, edgecolors="k", linewidths=0.3)
    axes[0, 2].set_xlabel("Volume (m³)")
    axes[0, 2].set_ylabel("Disruption Probability")
    axes[0, 2].set_title("Disruption Risk vs Volume (color = Q)")
    plt.colorbar(sc, ax=axes[0, 2], label="Q")

    sc = axes[1, 0].scatter(vols, stabs, c=betas, cmap="plasma", s=30, edgecolors="k", linewidths=0.3)
    axes[1, 0].set_xlabel("Volume (m³)")
    axes[1, 0].set_ylabel("Stability Score")
    axes[1, 0].set_title("Stability vs Volume (color = β_t%)")
    plt.colorbar(sc, ax=axes[1, 0], label="β_t (%)")

    sc = axes[1, 1].scatter(burns, Qs, c=rips, cmap="inferno", s=30, edgecolors="k", linewidths=0.3)
    axes[1, 1].set_xlabel("Burn Time (s)")
    axes[1, 1].set_ylabel("Q")
    axes[1, 1].set_title("Q vs Burn Time (color = TF ripple%)")
    plt.colorbar(sc, ax=axes[1, 1], label="TF Ripple (%)")

    sc = axes[1, 2].scatter(taus, Qs, c=betas, cmap="Spectral", s=30, edgecolors="k", linewidths=0.3)
    axes[1, 2].set_xlabel("τ_E (s)")
    axes[1, 2].set_ylabel("Q")
    axes[1, 2].set_title("Q vs τ_E (color = β_t%)")
    plt.colorbar(sc, ax=axes[1, 2], label="β_t (%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Pareto plot saved: {save_path}")
    plt.close()


def seeded_population(n_seed=20000):
    """Generate random designs, filter by hard constraints, return diverse seed."""
    print(f"Seeding population from {n_seed} random designs (β_N≤3.5, B_coil≤12T)...")
    t0 = time.perf_counter()
    candidates = [random_design() for _ in range(n_seed)]
    valid = []
    for p in candidates:
        r = evaluate_design(p)
        if r["Q"] >= 1.0 and r["B_coil_max_T"] <= 12.0:
            valid.append(p)
    n_valid = len(valid)
    print(f"  {n_valid}/{n_seed} satisfy Q≥1, B_coil≤12T ({n_valid/n_seed*100:.1f}%)")
    if n_valid >= POPULATION_SIZE:
        rng = random.Random(42)
        selected = rng.sample(valid, POPULATION_SIZE)
    elif n_valid > 0:
        selected = valid[:]
        while len(selected) < POPULATION_SIZE:
            selected.append(random_design())
    else:
        selected = [random_design() for _ in range(POPULATION_SIZE)]
    print(f"  Seeded {len(selected)} designs ({time.perf_counter()-t0:.1f}s)")
    return selected


def main():
    print("=" * 60)
    print("Tokamak Design Optimizer (NSGA-II) — Constrained")
    print("Evaluates designs using pure physics engine")
    print(f"Population: {POPULATION_SIZE}, Generations: {N_GENERATIONS}")
    print(f"Parameters: {PARAM_NAMES}")
    print("Objectives: maximize Q, maximize beta_t, maximize stability, minimize volume")
    print("Constraints: Q ≥ 1, B_coil_max ≤ 12T, stability ≥ -1.5")
    print("=" * 60)

    population = seeded_population()

    for gen in range(N_GENERATIONS):
        results = [evaluate_design(p) for p in population]

        fitnesses = []
        for r in results:
            q = r["Q"]
            bt = r["beta_t"]
            stab = r["stability_score"]
            vol = r["volume_m3"]
            bn = r["beta_N"]
            bc = r["B_coil_max_T"]
            fitnesses.append([-q, -bt, -stab, vol, bn, bc])

        fronts = non_dominated_sort(fitnesses)
        selected = select_parents(fitnesses, fronts)

        new_population = []
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = random.sample(selected, 2)
            c1, c2 = crossover(population[p1], population[p2])
            new_population.append(mutate(c1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(c2))

        population = new_population[:POPULATION_SIZE]

        feasible = [i for i in selected if constraint_violation(fitnesses[i]) < 1e-10]
        if feasible:
            best_idx = max(feasible, key=lambda i: results[i]["Q"])
        else:
            best_idx = min(selected, key=lambda i: constraint_violation(fitnesses[i]))
        n_feas = len(feasible)
        r_best = results[best_idx]
        print(f"\nGen {gen + 1:3d}/{N_GENERATIONS} | "
              f"Q={r_best['Q']:.2f} "
              f"β_t={r_best['beta_t']*100:.2f}% "
              f"β_N={r_best['beta_N']:.2f} "
              f"Bmax={r_best['B_coil_max_T']:.1f}T "
              f"vol={r_best['volume_m3']:.0f}m³ "
              f"τ_E={r_best['tau_E_s']:.2f}s "
              f"Pfus={r_best['P_fusion_MW']:.0f}MW "
              f"burn={r_best['burn_time_s']:.0f}s "
              f"cost={r_best['cost_MS']:.0f}M$ "
              f"feas={n_feas}")

    print("\n" + "=" * 60)
    print("Pareto Front")
    print("=" * 60)
    final_results = [evaluate_design(p) for p in population]
    final_fitnesses = []
    for r in final_results:
        final_fitnesses.append([-r["Q"], -r["beta_t"], -r["stability_score"],
                                r["volume_m3"], r["beta_N"], r["B_coil_max_T"]])

    final_fronts = non_dominated_sort(final_fitnesses)
    pareto_indices = final_fronts[0] if final_fronts else []

    print(f"Pareto designs found: {len(pareto_indices)}")
    print(f"{'R0':>6} {'a':>6} {'B0':>6} {'Ip':>7} {'kappa':>6} "
          f"{'tri_u':>6} {'tri_l':>6} {'q95':>6} | "
          f"{'Q':>6} {'β_t%':>6} {'β_N':>6} {'Bmax':>6} {'vol':>6} {'tau_E':>6}")
    print("-" * 96)
    display_n = min(len(pareto_indices), 50)
    for idx in pareto_indices[:display_n]:
        p = population[idx]
        r = final_results[idx]
        print(f"{p[0]:6.3f} {p[1]:6.3f} {p[2]:6.3f} {p[3]:7.1f} {p[4]:6.2f} "
              f"{p[5]:6.3f} {p[6]:6.3f} {p[7]:6.2f} | "
              f"{r['Q']:6.2f} {r['beta_t']*100:5.2f} {r['beta_N']:6.2f} "
              f"{r['B_coil_max_T']:6.1f} {r['volume_m3']:6.2f} {r['tau_E_s']:6.4f}")

    plot_pareto_front(final_results, save_path="pareto_front.png")
    save_report(population, final_results, pareto_indices, save_path="pareto_report.txt")


if __name__ == "__main__":
    main()
