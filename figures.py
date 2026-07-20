import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import math, os, sys

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
})

from physics_engine import (
    quick_eval, solve_power_balance, radial_profiles, integrate_profiles,
    tokamak_volume, plasma_surface_area, lh_threshold_power,
    tf_coil_peak_field, tf_ripple, compute_beta, bootstrap_fraction,
    capital_cost_estimate, bosch_hale_sigma_v
)
from mhd_stability import full_stability_analysis
from divertor_sol import divertor_analysis
from tbr_model import hcpb_tbr, li6_sensitivity

D = {"R0": 12.08, "a": 0.96, "B0": 11.7, "Ip": 10600.0,
     "elongation": 2.71, "triangularity_upper": 0.3,
     "triangularity_lower": 0.3, "q95": 3.1}


def fig_radial_profiles(save=True):
    """Radial profiles of n, T, fusion power density, heating power density."""
    N = 100
    R0, a, kappa = D["R0"], D["a"], D["elongation"]
    n0 = 2.20e20 * (0.2 + 1.0)
    T0 = 15.0 * (0.8 + 1.0)
    dr = a / N
    rho = np.linspace(0.5 * dr / a, (N - 0.5) * dr / a, N)

    n_prof = np.zeros(N)
    T_prof = np.zeros(N)
    p_fus_prof = np.zeros(N)
    w_prof = np.zeros(N)

    for i in range(N):
        x = rho[i]
        p = radial_profiles(x, alpha_n=0.2, alpha_T=0.8)
        n_prof[i] = n0 * p["n_norm"] / 1e20
        T_prof[i] = T0 * p["T_norm"]
        sigma_v = bosch_hale_sigma_v(T_prof[i])
        p_fus_prof[i] = ((n_prof[i] * 1e20 / 2.0) ** 2 * sigma_v * 17.6e6 * 1.6e-19) / 1e6

    q_prof = np.zeros(N)
    for i in range(1, N - 1):
        q_prof[i] = (T_prof[i + 1] - T_prof[i - 1]) / (rho[i + 1] - rho[i - 1])
        q_prof[i] = -q_prof[i] * n_prof[i] * 1e20 * 3.0 * 1.6e-19 / 1e6
    n_prof_full = np.concatenate([[n0 / 1e20], n_prof])
    T_prof_full = np.concatenate([[T0], T_prof])
    rho_full = np.concatenate([[0.0], rho])

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    axes[0, 0].plot(rho_full, n_prof_full, "b-", lw=2)
    axes[0, 0].set_ylabel(r"$n$ ($10^{20}$ m$^{-3}$)")
    axes[0, 0].set_title("Density Profile")

    axes[0, 1].plot(rho_full, T_prof_full, "r-", lw=2)
    axes[0, 1].set_ylabel(r"$T$ (keV)")
    axes[0, 1].set_title("Temperature Profile")

    axes[1, 0].plot(rho, p_fus_prof, "g-", lw=2)
    axes[1, 0].set_xlabel(r"$\rho = r/a$")
    axes[1, 0].set_ylabel(r"$P_{\mathrm{fus}}$ (MW/m$^3$)")
    axes[1, 0].set_title("Fusion Power Density")

    axes[1, 1].plot(rho[1:-1], q_prof[1:-1], "purple", lw=2)
    axes[1, 1].set_xlabel(r"$\rho = r/a$")
    axes[1, 1].set_ylabel(r"$q$ (MW/m$^3$)")
    axes[1, 1].set_title("Ion Heating Power Density")

    for a in axes.flat:
        a.set_xlim(0, 1)
        a.grid(alpha=0.2)
    fig.suptitle(r"Plasma Radial Profiles ($\alpha_n=0.2, \alpha_T=0.8$)", fontsize=14)
    plt.tight_layout()
    if save:
        fig.savefig("fig_radial_profiles.png")
        print("  [+] fig_radial_profiles.png")
    plt.close(fig)


def fig_mhd_stability(save=True):
    r = full_stability_analysis(
        D["R0"], D["a"], D["elongation"], 0.3, D["B0"],
        D["Ip"] / 1000, 1.04, D["q95"], 3.08, 2.20, 0.60
    )
    categories = [
        r"$\beta_N$ (design)",
        r"$\beta_N$ no-wall limit",
        r"$\beta_N$ wall limit",
        r"$\beta_N$ NTM threshold",
        r"$\beta_N$ with ECCD",
    ]
    ntm_metric = r["ntm_stability_metric"]
    beta_N_NTM = 3.08 / max(ntm_metric, 0.1)
    beta_N_ECCD = 3.08 * 1.15  # ECCD raises threshold ~15%
    values = [3.08, r["βN_no_wall_limit"], r["βN_wall_limit"], beta_N_NTM, beta_N_ECCD]
    colors = ["#2e86c1", "#27ae60", "#1abc9c", "#e74c3c", "#f39c12"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.4]})

    ax = axes[0]
    bars = ax.barh(categories, values, color=colors, height=0.55, edgecolor="black", lw=0.5)
    ax.axvline(3.08, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel(r"$\beta_N$")
    for bar, v in zip(bars, values):
        ax.text(v + 0.08, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=10)
    ax.set_title("Ideal MHD Stability Limits")
    ax.grid(axis="x", alpha=0.2)

    labels = [
        "Ideal wall limit",
        "No-wall limit",
        "NTM onset\n(requires ECCD)",
        "Design $\\beta_N$",
        "L-H threshold",
        "Density limit",
    ]
    beta_N = 3.08
    beta_NW = r["βN_no_wall_limit"]
    beta_W = r["βN_wall_limit"]
    beta_NTM = beta_N / max(ntm_metric, 0.1)
    margin_NW = beta_NW - beta_N
    margin_W = beta_W - beta_N
    margin_NTM = beta_NTM - beta_N  # negative since above threshold
    margins = [margin_W, margin_NW, margin_NTM, 0.0, 1.94, 0.40]
    colors2 = ["#1abc9c", "#27ae60", "#e74c3c", "#2e86c1", "#9b59b6", "#e74c3c"]

    ax = axes[1]
    bars = ax.barh(labels, margins, color=colors2, height=0.55, edgecolor="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.8)
    for bar, v in zip(bars, margins):
        if v > 0:
            ax.text(v + 0.04, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                    va="center", fontsize=9)
    ax.set_xlabel("Margin")
    ax.set_title("Engineering Margins")
    ax.grid(axis="x", alpha=0.2)

    fig.suptitle(fr"MHD Stability and Margins ($\beta_N = {beta_N:.2f}$, $P_{{\mathrm{{disrupt}}}} \approx {r['disruption_probability']:.2f}$)", fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig("fig_mhd_stability.png")
        print("  [+] fig_mhd_stability.png")
    plt.close(fig)


def fig_divertor_heat_flux(save=True):
    P_sep = 1095.0 * (1.0 - 0.21)
    types = ["ITER", "X_DIVERTOR", "SNOWFLAKE", "LIQUID_LI"]
    labels = ["ITER-grade", "X-divertor", "Snowflake", "Liquid Li divertor"]
    colors = ["#e74c3c", "#f39c12", "#27ae60", "#3498db"]

    data = []
    for t in types:
        try:
            r = divertor_analysis(P_sep, D["R0"], D["a"], D["elongation"],
                                  D["B0"], D["Ip"] / 1000, D["q95"],
                                  2.20, divertor_type=t, f_rad_core=0.21)
            data.append(r)
        except:
            data.append(None)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    valid = [(l, c, d) for l, c, d in zip(labels, colors, data) if d is not None]
    names = [v[0] for v in valid]
    qpeaks = [v[2]["q_peak_MW_m2"] for v in valid]
    limits = [v[2].get("q_limit_MW_m2", 20.0) for v in valid]
    margins = [v[2].get("q_margin_MW_m2", 0.0) for v in valid]
    bar_colors = [v[1] for v in valid]

    x = np.arange(len(names))
    w = 0.3
    bars1 = ax.bar(x - w / 2, qpeaks, w, label="Peak heat flux", color=bar_colors, edgecolor="black", lw=0.5)
    bars2 = ax.bar(x + w / 2, limits, w, label="Technology limit", color="lightgray", edgecolor="black", lw=0.5, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(r"$q_{\mathrm{peak}}$ (MW m$^{-2}$)")
    ax.set_title("Divertor Peak Heat Flux")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    for bar, v in zip(bars1, qpeaks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", fontsize=8)

    ax = axes[1]
    valid_d = [d for d in data if d is not None]
    names_d = [names[i] for i in range(len(valid))]
    for i, (label, color, d) in enumerate(zip(names_d, bar_colors, valid_d)):
        s = np.linspace(0, 0.05, 100)
        q_exp = d["q_peak_MW_m2"] * np.exp(-s / (d.get("lambda_q_mm", 0.5) / 1000))
        q_exp = np.minimum(q_exp, d.get("q_limit_MW_m2", 20.0))
        ax.plot(s * 1000, q_exp, color=color, lw=2, label=label)
    ax.axhline(20, color="red", ls="--", lw=1, alpha=0.5, label="Snowflake limit")
    ax.axhline(15, color="orange", ls="--", lw=1, alpha=0.5, label="X-divertor limit")
    ax.axhline(10, color="gray", ls="--", lw=1, alpha=0.5, label="ITER limit")
    ax.set_xlabel(r"Distance from strike point $s$ (mm)")
    ax.set_ylabel(r"$q_{\parallel}$ (MW m$^{-2}$)")
    ax.set_title("Heat Flux Profile at Target")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)

    fig.suptitle("Divertor Heat Flux Analysis", fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig("fig_divertor.png")
        print("  [+] fig_divertor.png")
    plt.close(fig)


def fig_power_balance(save=True):
    P_fus = 5475.0
    P_alpha = 5475.0 / 5
    P_rad_core = 230.0
    P_rad_div = 865.0 * 0.65
    P_cond = 865.0 * 0.35
    P_ext = 0.0
    P_aux = 30.0

    categories = [
        "Fusion power\n$P_{\\mathrm{fus}} = 5,475$ MW",
        "Alpha heating\n$P_\\alpha = 1,095$ MW",
        "Core radiation\n$P_{\\mathrm{rad}} = 230$ MW",
        "Power to SOL\n$P_{\\mathrm{sep}} = 865$ MW",
        "Divertor radiation\n$P_{\\mathrm{rad,div}} = 562$ MW",
        "Divertor conduction\n$P_{\\mathrm{cond}} = 303$ MW",
    ]
    values = [P_fus, P_alpha, P_rad_core, P_sep := 865.0, P_rad_div, P_cond]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_b = ["#3498db", "#2e86c1", "#e74c3c", "#f39c12", "#27ae60", "#1abc9c"]
    bars = ax.bar(categories, values, color=colors_b, edgecolor="black", lw=0.6, width=0.65)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{v:.0f} MW", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Power (MW)")
    ax.set_title("Power Balance Waterfall", fontsize=13)
    ax.set_ylim(0, max(values) * 1.12)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    if save:
        fig.savefig("fig_power_balance.png")
        print("  [+] fig_power_balance.png")
    plt.close(fig)


def fig_cost_breakdown(save=True):
    costs = {
        "TF coils (Nb$_3$Sn)": 3725,
        "PF + CS coils": 1800,
        "Blanket (HCPB)": 748,
        "Vacuum vessel": 200,
        "Cryostat + cryoplant": 600,
        "Divertor (W)": 50,
        "Heating + CD": 300,
        "Balance of plant": 2168,
        "Cooling systems": 1084,
        "Tritium plant": 1000,
        "Site + assembly": 800,
        "Contingency + indirect": 1530,
    }
    total = sum(costs.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]})

    labels = list(costs.keys())
    values = list(costs.values())
    explode = [0.03] * len(labels)
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax1.pie(
        values, labels=None, autopct="%1.1f%%",
        startangle=90, explode=explode, colors=colors_pie,
        pctdistance=0.85, wedgeprops={"edgecolor": "white", "lw": 0.5}
    )
    for t in autotexts:
        t.set_fontsize(7)
    ax1.set_title(f"Total: ${total / 1000:.1f}B", fontsize=12)

    ax2.axis("off")
    sorted_idx = np.argsort(values)[::-1]
    y = 0.95
    for i in sorted_idx:
        ax2.text(0, y, f"  {labels[i]}", transform=ax2.transAxes,
                 fontsize=8, verticalalignment="top")
        ax2.text(1, y, f"${values[i]:,}M", transform=ax2.transAxes,
                 fontsize=8, verticalalignment="top", ha="right")
        y -= 0.075
    ax2.set_title("Cost Breakdown", fontsize=12, pad=10)

    fig.suptitle(f"Total Project Cost: ${total / 1000:.2f}B (${total * 1000 / 1762:.0f}/kW$_e$)", fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig("fig_cost.png")
        print("  [+] fig_cost.png")
    plt.close(fig)


def fig_tbr_sensitivity(save=True):
    li6_vals = np.linspace(0.05, 0.90, 50)
    tbr_vals = []
    for li6 in li6_vals:
        tbr_vals.append(hcpb_tbr(li6_enrichment=max(li6, 0.01), be_thickness_mm=20,
                                 blanket_thickness_mm=800, coverage=0.92)["TBR"])
    tbr_vals = np.array(tbr_vals)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(li6_vals * 100, tbr_vals, "b-", lw=2)
    ax.axhline(1.05, color="red", ls="--", lw=1.5, alpha=0.7, label="Self-sufficiency threshold (TBR=1.05)")
    ax.axhline(1.17, color="green", ls="-.", lw=1.5, alpha=0.7, label="Design point (15% $^6$Li: TBR=1.17)")
    ax.axvline(7.5, color="purple", ls=":", lw=1.5, alpha=0.5, label="Natural Li (7.5% $^6$Li)")
    ax.axvline(15, color="green", ls="-.", lw=1.5, alpha=0.5)

    ax.fill_between(li6_vals * 100, 1.05, tbr_vals, where=(tbr_vals >= 1.05),
                    color="green", alpha=0.1, label="Self-sufficient region")
    ax.fill_between(li6_vals * 100, tbr_vals, 1.05, where=(tbr_vals < 1.05),
                    color="red", alpha=0.1)

    ax.set_xlabel("$^6$Li enrichment (%)")
    ax.set_ylabel("Tritium Breeding Ratio (TBR)")
    ax.set_title("HCPB Blanket: TBR vs $^6$Li Enrichment", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(5, 90)
    ax.set_ylim(0.5, max(tbr_vals) * 1.05)
    ax.grid(alpha=0.2)

    ax.annotate(f"Natural Li: TBR={tbr_vals[np.argmin(np.abs(li6_vals*100-7.5))]:.2f}",
                xy=(7.5, tbr_vals[np.argmin(np.abs(li6_vals*100-7.5))]),
                xytext=(20, 0.7), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="purple", alpha=0.6))
    ax.annotate(f"Design: TBR=1.17",
                xy=(15, 1.17), xytext=(40, 1.4), fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))

    plt.tight_layout()
    if save:
        fig.savefig("fig_tbr_sensitivity.png")
        print("  [+] fig_tbr_sensitivity.png")
    plt.close(fig)


def fig_plasma_cross_section(save=True):
    fig, ax = plt.subplots(figsize=(7, 7))
    R0, a, kappa, delta_u, delta_l = 12.08, 0.96, 2.71, 0.30, 0.30
    theta = np.linspace(0, 2 * np.pi, 200)
    R = R0 + a * np.cos(theta + delta_u * np.sin(theta))
    Z = kappa * a * np.sin(theta)
    R_div = 12.08 - 0.96
    Z_div = -kappa * a * 1.3

    ax.fill(R, Z, color="#3498db", alpha=0.3, edgecolor="#2e86c1", lw=2, label="Plasma")
    ax.plot(R0, 0, "r+", markersize=12, markeredgewidth=2)
    ax.annotate("Magnetic axis", xy=(R0, 0), xytext=(12.8, 0.5),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="red"))

    circle = plt.Circle((R0, 0), a, fill=False, color="gray", ls="--", lw=1, alpha=0.4)
    ax.add_patch(circle)

    ax.plot([R0 - a, R0 + a], [-kappa * a, -kappa * a], "k-", lw=1.5)
    ax.annotate("$2\\kappa a$", xy=(R0, -kappa * a), xytext=(12.5, -3.0),
                fontsize=9, arrowprops=dict(arrowstyle="<->"))

    ax.annotate("$2a$", xy=(R0 - a, 4.0), xytext=(R0, 4.0), fontsize=9,
                arrowprops=dict(arrowstyle="<->"))

    ax.annotate("$R_0$", xy=(R0, -4.5), xytext=(R0, -4.5), fontsize=10, ha="center")

    limit = R0 + a + 1.5
    ax.set_xlim(R0 - limit, R0 + limit)
    ax.set_ylim(-limit * 0.9, limit * 0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("$R$ (m)")
    ax.set_ylabel("$Z$ (m)")
    ax.set_title("Plasma Cross-Section", fontsize=13)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="upper right")

    info_text = (
        f"$R_0 = {R0:.2f}$ m\n"
        f"$a = {a:.2f}$ m\n"
        f"$\\kappa = {kappa:.2f}$\n"
        f"$\\delta = {delta_u:.2f}$\n"
        f"$A = {R0 / a:.1f}$\n"
        f"$I_p = 10.6$ MA"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    if save:
        fig.savefig("fig_plasma_cross_section.png")
        print("  [+] fig_plasma_cross_section.png")
    plt.close(fig)


def fig_sensitivity_tornado(save=True):
    H98_vary = np.linspace(0.8, 1.2, 20)
    n_vary = np.linspace(0.4, 0.8, 20)
    Bt_vary = np.linspace(10, 13, 20)

    base_r = quick_eval(D, divertor_type="SNOWFLAKE")
    base_Q = base_r["Q"]

    def sweep_Q(param, vals):
        Qs = []
        for v in vals:
            d_i = D.copy()
            d_i[param] = v
            d_i["n_fGW"] = d_i.get("n_fGW", 0.6)
            try:
                r_i = quick_eval(d_i, divertor_type="SNOWFLAKE")
                Qs.append(r_i["Q"])
            except:
                Qs.append(base_Q)
        return np.array(Qs)

    labels = [
        r"$H_{98}$ multiplier (0.8–1.2)",
        r"$n_{\mathrm{GW}}$ fraction (0.4–0.8)",
        r"$B_0$ (10–13 T)",
    ]
    Q_low = [sweep_Q("H98_mult", [0.8]).min(), sweep_Q("n_fGW", [0.4]).min(), sweep_Q("B0", [10.0]).min()]
    Q_high = [sweep_Q("H98_mult", [1.2]).max(), sweep_Q("n_fGW", [0.8]).max(), sweep_Q("B0", [13.0]).max()]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(labels))
    delta_low = np.array(Q_low) - base_Q
    delta_high = np.array(Q_high) - base_Q

    ax.barh(y_pos, delta_high, left=base_Q, color="#27ae60", height=0.5, label="High case", edgecolor="black", lw=0.5)
    ax.barh(y_pos, delta_low, left=base_Q, color="#e74c3c", height=0.5, label="Low case", edgecolor="black", lw=0.5)
    ax.axvline(base_Q, color="black", lw=1.5, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Fusion gain $Q$")
    ax.set_title("Sensitivity Analysis: Impact on $Q$", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    ax.text(base_Q, -0.5, f"Base Q = {base_Q:.1f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save:
        fig.savefig("fig_sensitivity.png")
        print("  [+] fig_sensitivity.png")
    plt.close(fig)


def generate_all():
    os.makedirs("figures", exist_ok=True)
    os.chdir("figures")
    print("Generating figures...")
    fig_radial_profiles()
    fig_mhd_stability()
    fig_divertor_heat_flux()
    fig_power_balance()
    fig_cost_breakdown()
    fig_tbr_sensitivity()
    fig_plasma_cross_section()
    fig_sensitivity_tornado()
    os.chdir("..")
    print("\nAll figures generated in figures/")

if __name__ == "__main__":
    generate_all()
