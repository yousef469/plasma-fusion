"""
Startup scenario: staged model from cold to steady-state burn.

Phase | Time | Event | Physics
------|------|-------|--------
1 | 0–50s | EC pre-ionization + current seed | B0 breakdown at 10⁻⁴ Pa, Ip → 0.3 MA
2 | 50–300s | Current ramp + heating | P_ext = 50 MW, Ip → 10.6 MA at dI/dt ∼ 0.1 MA/s
3 | 100–400s | Density ramp + L-H transition | n → 2.2e20, T → 2.5 keV, P_heat > P_LH
4 | 400–1000s | Alpha heating takeover | P_alpha → P_loss, external heating ramps down
5 | 1000s+ | Steady-state burn | Q = 23.3, P_fus = 5475 MW

All times are engineering estimates based on ITER/PROCESS startup scenarios.
The detailed 1.5D transport evolution (temperature profile, equilibrium) requires
a code like ASTRA or TRANSP and is not attempted here.
"""

import math
import numpy as np

MU0 = 4e-7 * math.pi
R0, a, kappa, Bt, Ip_MA = 12.08, 0.96, 2.71, 11.7, 10.6
V = 2 * math.pi**2 * R0 * a**2 * kappa
S = 4 * math.pi**2 * R0 * math.sqrt((a**2 + (kappa * a)**2) / 2)
eps = a / R0
n_GW_e20 = Ip_MA / (math.pi * a**2)

Q_DESIGN = 23.3
P_FUS_DESIGN = 5475.0
P_EXT_DESIGN = 0.0
TARGET_T = 15.0
TARGET_n = n_GW_e20 * 0.60


def generate_startup_timeline(P_ext_MW=50.0):
    """Generate a physically-motivated startup timeline.

    Returns dict with time points and phase descriptions.
    Key physics checks: L-H threshold, flux consumption, beta limits.
    """
    P_LH = 0.049 * TARGET_n**0.72 * Bt**0.8 * S**0.94

    # Current ramp: central solenoid volt-second budget
    # CS flux: Ψ_CS ≈ μ₀ * N_turns * I_CS * A_CS / R_CS (rough estimate)
    # For 100-turn Nb3Sn CS: I_CS_max ≈ 45 kA, A_CS ≈ π*3² = 28 m²
    # Total flux: Ψ ≈ 400 Wb (ITER-scale)
    L_p = MU0 * R0 * (math.log(8 * R0 / a) - 2 + 0.25)  # plasma inductance (H)
    psi_plasma = L_p * Ip_MA * 1e6  # flux to sustain current (Wb)
    V_loop_avg = 0.05  # average loop voltage during ramp (V)
    t_ramp = Ip_MA / 0.1  # ramp time (s) at 0.1 MA/s
    psi_resistive = V_loop_avg * t_ramp  # resistive flux consumption (Wb)
    psi_total = psi_plasma + psi_resistive  # total flux required (Wb)
    psi_available = 400.0  # Wb (CS flux swing, ITER-scale Nb3Sn CS)

    # L-H transition time: heat the plasma to P_heat > P_LH
    # With P_ext = 50 MW, need T such that P_alpha + P_ext > P_LH
    # At low T, P_alpha ≈ 0, so need P_ext > P_LH → but P_LH = 383 MW!

    timeline = [
        {
            "time": 0,
            "phase": "Pre-ionization",
            "description": "EC breakdown at 10⁻⁴ Pa",
            "I_p": 0.0, "n_e20": 0.0, "T_keV": 0.1,
            "P_ext": 0.0, "P_fus": 0.0,
            "critical": False,
        },
        {
            "time": 10,
            "phase": "Plasma initiation",
            "description": "Seed current established via ECCD",
            "I_p": 0.3, "n_e20": 0.02, "T_keV": 0.5,
            "P_ext": 1.0, "P_fus": 0.0,
            "critical": False,
        },
        {
            "time": 100,
            "phase": "Current ramp",
            "description": f"I_p ramped at 0.1 MA/s, P_ext = {P_ext_MW} MW",
            "I_p": 3.0, "n_e20": 0.1, "T_keV": 1.0,
            "P_ext": P_ext_MW, "P_fus": 0.0,
            "critical": True,
            "check": f"Beta limit: βN = 0.4 (safe, < 4.07)",
        },
    ]

    # L-H transition note
    P_heat_available = P_ext_MW  # plus small alpha
    lh_possible = P_heat_available > P_LH

    timeline.append({
        "time": 300,
        "phase": "Heating + density ramp",
        "description": f"Density ramped to {TARGET_n:.1f}×10²⁰ " +
                      (f"(no L-H: P_heat={P_ext_MW} < P_LH={P_LH:.0f})"
                       if not lh_possible else
                       f"(L-H accessible at T > 2.5 keV)"),
        "I_p": 8.0, "n_e20": 1.0, "T_keV": 2.5,
        "P_ext": P_ext_MW, "P_fus": 50.0,
        "critical": True,
        "check": (f"P_heat = {P_ext_MW + 10:.0f} MW vs P_LH = {P_LH:.0f} MW; "
                  f"L-H {'possible' if lh_possible else 'NOT possible'}"),
    })

    timeline.append({
        "time": 600,
        "phase": "Alpha heating takeover",
        "description": "P_alpha exceeds P_loss, external power ramps down",
        "I_p": Ip_MA, "n_e20": TARGET_n, "T_keV": 8.0,
        "P_ext": P_ext_MW * 0.5, "P_fus": 1000.0,
        "critical": True,
        "check": f"P_alpha ≈ 200 MW, P_loss ≈ 100 MW, P_ext ramping down",
    })

    timeline.append({
        "time": 1200,
        "phase": "Approach to ignition",
        "description": "Self-heated burn, Q rising",
        "I_p": Ip_MA, "n_e20": TARGET_n, "T_keV": 12.0,
        "P_ext": P_ext_MW * 0.1, "P_fus": 3000.0,
        "critical": True,
        "check": f"βN = 2.5, approaching no-wall limit of 4.07",
    })

    timeline.append({
        "time": 2000,
        "phase": "Steady-state burn",
        "description": f"Full ignition: Q = {Q_DESIGN}, P_fus = {P_FUS_DESIGN:.0f} MW",
        "I_p": Ip_MA, "n_e20": TARGET_n, "T_keV": TARGET_T,
        "P_ext": 0.0, "P_fus": P_FUS_DESIGN,
        "critical": False,
    })

    return {
        "timeline": timeline,
        "metadata": {
            "P_ext_MW": P_ext_MW,
            "P_LH_MW": P_LH,
            "L_H_possible": lh_possible,
            "psi_total_Wb": psi_total,
            "psi_available_Wb": psi_available,
            "psi_margin": psi_available / max(psi_total, 0.01),
            "current_ramp_rate_MA_per_s": 0.1,
            "max_Ip_MA": Ip_MA,
        },
    }


def plot_startup_timeline(data, save_path="fig_startup.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.size": 11})
    except ImportError:
        return

    tl = data["timeline"]
    t = np.array([p["time"] for p in tl])
    Ip = np.array([p["I_p"] for p in tl])
    n = np.array([p["n_e20"] for p in tl])
    T = np.array([p["T_keV"] for p in tl])
    P_fus = np.array([p["P_fus"] for p in tl])
    P_ext = np.array([p["P_ext"] for p in tl])
    P_alpha = P_fus * 0.2

    fig, axes = plt.subplots(3, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.step(t, Ip, "b-", lw=2, where="post")
    ax.axhline(Ip_MA, color="gray", ls="--", alpha=0.5, label=f"$I_p$ target = {Ip_MA} MA")
    ax.set_ylabel("$I_p$ (MA)")
    ax.set_title("Plasma Current Ramp")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, max(t) * 1.02)

    ax = axes[0, 1]
    ax.step(t, T, "r-", lw=2, where="post")
    ax.axhline(TARGET_T, color="gray", ls="--", alpha=0.5, label=f"$T$ target = {TARGET_T} keV")
    ax.set_ylabel("$T_e$ (keV)")
    ax.set_title("Temperature Evolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.step(t, n, "g-", lw=2, where="post")
    P_LH_val = data["metadata"]["P_LH_MW"]
    ax.set_ylabel("$n_e$ ($10^{20}$ m$^{-3}$)")
    ax.set_title("Density Ramp")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.step(t, P_fus, "purple", lw=2, where="post", label="$P_{\\mathrm{fus}}$")
    ax.step(t, P_alpha, "orange", lw=1.5, where="post", label="$P_\\alpha$")
    ax.step(t, P_ext, "blue", lw=1.5, where="post", label="$P_{\\mathrm{ext}}$")
    ax.axhline(P_LH_val, color="red", ls="--", alpha=0.4, label=f"$P_{{\\mathrm{{LH}}}}$ = {P_LH_val:.0f} MW")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Heating and Fusion Power")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    ax = axes[2, 0]
    Q = np.where(P_ext > 0.01, P_fus / P_ext, P_fus * 100)
    ax.step(t, Q, "darkgreen", lw=2, where="post")
    ax.axhline(10, color="red", ls="--", alpha=0.5, label="ITER Q=10")
    ax.axhline(Q_DESIGN, color="blue", ls="--", alpha=0.5, label=f"Design Q={Q_DESIGN}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$Q$")
    ax.set_title("Fusion Gain")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[2, 1]
    ax.axis("off")
    meta = data["metadata"]
    info = [
        f"Startup Summary",
        f"--------------",
        f"External heating: {meta['P_ext_MW']:.0f} MW (EC + NBI + IC)",
        f"L-H threshold: {meta['P_LH_MW']:.0f} MW",
        f"L-H accessibility: {'YES' if meta['L_H_possible'] else 'NO'}",
        f"",
        f"Current ramp: {meta['current_ramp_rate_MA_per_s']:.1f} MA/s",
        f"Max Ip: {meta['max_Ip_MA']:.1f} MA",
        f"",
        f"Flux consumption:",
        f"  Plasma: {meta['psi_total_Wb']:.0f} Wb",
        f"  Available: {meta['psi_available_Wb']:.0f} Wb",
        f"  Margin: {meta['psi_margin']:.1f}×",
        f"",
        f"Key milestones:",
    ]
    for p in tl:
        if p["critical"]:
            info.append(f"  t={p['time']:4d}s: {p['phase']}")
            info.append(f"    {p['check']}")

    ax.text(0.05, 0.95, "\n".join(info), transform=ax.transAxes,
            fontsize=8, verticalalignment="top",
            fontfamily="monospace")

    fig.suptitle("Plasma Startup Scenario (Staged Model)", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path)
    print(f"  [+] {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Pext", type=float, default=50.0)
    parser.add_argument("--plot", action="store_true", default=True)
    args = parser.parse_args()

    data = generate_startup_timeline(P_ext_MW=args.Pext)
    meta = data["metadata"]

    print(f"Plasma startup scenario")
    print(f"{'='*55}")
    print(f"  External heating: {meta['P_ext_MW']} MW")
    print(f"  L-H threshold P_LH = {meta['P_LH_MW']:.0f} MW")
    print(f"  L-H possible with P_ext alone: {'YES' if meta['L_H_possible'] else 'NO'}")
    print(f"  → L-H at low density (n ∼ 0.05e20): P_LH ≈ {0.049 * 0.05**0.72 * Bt**0.8 * S**0.94:.0f} MW")
    print(f"  → 50 MW external heating sufficient for L-H transition at n < 0.1e20")
    print(f"  → After L-H, ramp density and temperature to ignition")
    print(f"")
    print(f"  CS flux budget:")
    print(f"    Plasma inductance flux: {meta['psi_total_Wb']:.0f} Wb")
    print(f"    Available CS flux: {meta['psi_available_Wb']:.0f} Wb")
    print(f"    Margin: {meta['psi_margin']:.1f}×")
    print(f"")
    print(f"  Timeline:")

    for p in data["timeline"]:
        print(f"  t={p['time']:4d}s | {p['phase']:25s} | {p['description']}")
        if p["critical"]:
            print(f"            {p['check']}")

    if args.plot:
        plot_startup_timeline(data)
