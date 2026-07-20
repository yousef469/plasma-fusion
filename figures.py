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
    capital_cost_estimate, bosch_hale_sigma_v, pf_coil_system
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
    eng = quick_eval(D, divertor_type='SNOWFLAKE')
    r = full_stability_analysis(
        D["R0"], D["a"], D["elongation"], 0.3, D["B0"],
        D["Ip"] / 1000, eng["l_i"], D["q95"],
        eng["n_bar_e20"], eng["beta_N"], eng["T_ped"],
    )
    categories = [
        r"$\beta_N$ (design)",
        r"$\beta_N$ no-wall limit",
        r"$\beta_N$ wall limit",
        r"$\beta_N$ NTM threshold",
        r"$\beta_N$ with ECCD",
    ]
    bn = eng["beta_N"]
    ntm_metric = r["ntm_stability_metric"]
    beta_N_NTM = bn / max(ntm_metric, 0.1)
    beta_N_ECCD = bn * 1.15  # ECCD raises threshold ~15%
    values = [bn, r["βN_no_wall_limit"], r["βN_wall_limit"], beta_N_NTM, beta_N_ECCD]
    colors = ["#2e86c1", "#27ae60", "#1abc9c", "#e74c3c", "#f39c12"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.4]})

    ax = axes[0]
    bars = ax.barh(categories, values, color=colors, height=0.55, edgecolor="black", lw=0.5)
    ax.axvline(bn, color="black", ls="--", lw=1, alpha=0.5)
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
    beta_N = bn
    beta_NW = r["βN_no_wall_limit"]
    beta_W = r["βN_wall_limit"]
    beta_NTM = beta_N / max(ntm_metric, 0.1)
    margin_NW = beta_NW - beta_N
    margin_W = beta_W - beta_N
    margin_NTM = beta_NTM - beta_N  # negative since above threshold
    margins = [margin_W, margin_NW, margin_NTM, 0.0,
               eng["lh_margin"], eng["density_margin"]]
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
    r = quick_eval(D, "SNOWFLAKE")
    # Map engine costs to chart categories
    C_TF = r["cost_TF_coils_MS"]
    C_PF = r["cost_PF_coils_MS"]
    C_PF_CS = C_PF + 500  # combine PF + CS (CS ~$500M)
    C_blanket = r["cost_blanket_MS"]
    C_bop = r["cost_turbine_MS"]
    C_cool = r["cost_cooling_MS"]
    C_trit = r["cost_tritium_plant_MS"]
    C_site = r["cost_site_MS"]
    C_IC = r["cost_IC_MS"]
    C_aux = r["cost_aux_MS"]
    base = C_TF + C_PF_CS + C_blanket + C_bop + C_cool + C_trit + C_site + C_IC + C_aux
    contingency = base * 0.15

    costs = {
        "TF coils (Nb$_3$Sn)": C_TF,
        "PF coils (NbTi)": C_PF,
        "Blanket (HCPB)": C_blanket,
        "Balance of plant": C_bop,
        "Cooling systems": C_cool,
        "Tritium plant": C_trit,
        "Site + assembly": C_site,
        "I&C + controls": C_IC,
        "Heating + CD": max(C_aux, 1),
        "Contingency + indirect": round(contingency),
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

    P_net = r.get("P_net_electric_MW", 1762)
    fig.suptitle(f"Total Project Cost: ${total / 1000:.2f}B (${total * 1000 / max(P_net, 1):.0f}/kW$_e$)", fontsize=13)
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


def fig_eccd_system(save=True):
    """ECCD system design — power, cost, TRL assessment."""
    eng = quick_eval(D, divertor_type='SNOWFLAKE')

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8),
                             gridspec_kw={"width_ratios": [1, 0.9, 1.1]})

    # Panel 1: Power breakdown
    ax = axes[0]
    categories = [
        "At plasma\n(q=3/2 + q=2)",
        "Transmission\nloss (32%)",
        "Wall-plug\n(40% eff.)",
    ]
    values = [
        eng["P_ECCD_MW"],
        eng["P_ECCD_launched_MW"] - eng["P_ECCD_MW"],
        eng["ECCD_recirc_MW"] - eng["P_ECCD_launched_MW"],
    ]
    colors_p = ["#27ae60", "#f39c12", "#e74c3c"]
    bars = ax.barh(categories, values, color=colors_p, height=0.5,
                   edgecolor="black", lw=0.5)
    ax.set_xlabel("Power (MW)")
    ax.set_title("ECCD Power Budget", fontsize=12)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} MW", va="center", fontsize=9)
    total = sum(values)
    ax.text(total * 0.5, -0.6, f"Total wall-plug: {total:.0f} MW  "
             f"({total/1762*100:.1f}% of P$_{{\\text{{net}}}}$)",
             ha="center", fontsize=9, transform=ax.get_xaxis_transform())
    ax.grid(axis="x", alpha=0.2)

    # Panel 2: Gyrotron frequency / TRL
    ax = axes[1]
    ax.axis("off")
    freq = eng["ECCD_freq_GHz"]
    trl = eng["ECCD_gyrotron_TRL"]
    status = eng["ECCD_gyrotron_status"].title()
    n_gyro = eng["ECCD_gyrotrons"]
    n_ports = eng["ECCD_n_ports"]
    info_lines = [
        f"Frequency:  {freq:.0f} GHz",
        f"TRL:  {trl}  ({status})",
        f"Gyrotrons:  {n_gyro} × 0.5 MW",
        f"Ports:  {n_ports}",
        f"Cost:  {eng['ECCD_cost_MS']:.0f} M USD",
        "",
        f"Recirculated:  {eng['ECCD_recirc_MW']:.0f} MW",
        f"Net reduction:  {eng['ECCD_recirc_MW']/1762*100:.1f}% of P$_{{\\text{{net}}}}$",
    ]
    y0 = 0.85
    for i, line in enumerate(info_lines):
        ax.text(0.1, y0 - i * 0.10, line, fontsize=9.5,
                transform=ax.transAxes, va="top")
    ax.set_title("ECCD System Specs", fontsize=12)

    # Panel 3: Comparison to other reactors
    ax = axes[2]
    reactors = ["ITER", "SPARC", "This design"]
    P_ec = [20, 12, eng["P_ECCD_MW"]]
    P_net_ref = [500, 50, 1762]
    rect_recirc = [40, 24, eng["ECCD_recirc_MW"]]
    x = np.arange(len(reactors))
    w = 0.30
    bars1 = ax.bar(x - w/2, P_ec, w, label="EC power (MW)", color="#2e86c1",
                   edgecolor="black", lw=0.5)
    bars2 = ax.bar(x + w/2, rect_recirc, w, label="Recirc (MW)", color="#e74c3c",
                   edgecolor="black", lw=0.5)
    for bar, v in zip(bars1, P_ec):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{v} MW", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(reactors)
    ax.set_ylabel("Power (MW)")
    ax.set_title("ECCD Comparison", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    if save:
        fig.savefig("fig_eccd.png")
        print("  [+] fig_eccd.png")
    plt.close(fig)


def fig_tokamak_layout(save=True):
    """
    Full tokamak poloidal cross-section with all major components,
    coil systems, and structural regions dimensioned.
    """
    R0, a, kappa = 12.08, 0.96, 2.71
    delta_u, delta_l = 0.30, 0.30

    # Derived geometry from physics_engine
    R_TF_min = 9.32       # m, TF inner leg (plasma-facing edge)
    w_TF_inner = 2.5      # m, TF inner leg radial width (winding + case)
    h_TF = 20.0           # m, TF coil total height (±10 m)
    R_TF_outer = 14.8     # m, TF outer leg approx position

    # CS geometry
    R_CS_inner = 0.75
    R_CS_outer = 3.10
    h_CS = 15.0

    # Inboard build
    r_blk = 0.8
    r_shd = 0.5
    r_vv = 0.3
    r_gap = 0.1
    inboard_build = r_blk + r_shd + r_vv + r_gap

    R_plasma_inner = R0 - a  # 11.12 m
    R_plasma_outer = R0 + a  # 13.04 m

    # PF coil positions (quick_eval data)
    pf_coords = {
        "PF1": (4.0, 9.5), "PF2": (9.5, 8.0), "PF3": (15.5, 4.5),
        "PF4": (15.5, -4.5), "PF5": (9.5, -8.0), "PF6": (4.0, -9.5),
        "D1": (11.5, -3.0), "D2": (13.5, -3.2),
    }

    fig, ax = plt.subplots(figsize=(10, 12))

    # ── Plasma boundary D-shape ──────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 300)
    delta_arr = np.where(theta < np.pi, delta_u, delta_l)
    R_plasma = R0 + a * np.cos(theta + delta_arr * np.sin(theta))
    Z_plasma = kappa * a * np.sin(theta)
    ax.fill(R_plasma, Z_plasma, color="#3498db", alpha=0.25,
            edgecolor="#2e86c1", lw=2, label="Plasma", zorder=5)

    # ── Inboard structural column (TF inner leg + case + support) ────────
    z_col = np.linspace(-h_TF / 2, h_TF / 2, 50)
    R_col_left = np.full_like(z_col, R_TF_min)
    R_col_right = np.full_like(z_col, R_TF_min + w_TF_inner)
    ax.fill_betweenx(z_col, R_col_left, R_col_right,
                     color="#d5d8dc", edgecolor="#808b96", lw=1.5,
                     hatch="////", alpha=0.5, label="TF inner leg", zorder=3)

    # ── TF coil D-shape outline (outer arc) ──────────────────────────────
    # Outer D-curve: from top inner to bottom inner via outer midplane
    N_d = 150
    theta_d = np.linspace(np.pi / 2, -np.pi / 2, N_d)
    p_shape = 0.65  # D-shape parameter (<1 = more pointed)
    R_outer_curve = (R_TF_min + w_TF_inner
                     + (R_TF_outer - R_TF_min - w_TF_inner)
                     * np.abs(np.cos(theta_d)) ** p_shape)
    Z_outer_curve = (h_TF / 2) * np.sin(theta_d)

    # Inner curve (parallel to outer, offset by w_TF)
    w_TF_thickness = 1.2  # approximate TF coil thickness in R direction
    R_inner_curve = (R_TF_min + w_TF_inner - w_TF_thickness
                     + (R_TF_outer - w_TF_thickness - R_TF_min - w_TF_inner + w_TF_thickness)
                     * np.abs(np.cos(theta_d)) ** p_shape)

    # Combine to make TF coil D-outline
    R_TF_outline = np.concatenate([R_outer_curve, R_inner_curve[::-1], [R_outer_curve[0]]])
    Z_TF_outline = np.concatenate([Z_outer_curve, Z_outer_curve[::-1], [Z_outer_curve[0]]])
    ax.fill(R_TF_outline, Z_TF_outline, color="#d5d8dc", edgecolor="#808b96",
            lw=2, hatch="//", alpha=0.6, label="TF coil", zorder=2)

    # TF coil structural case outer highlight
    ax.plot(R_outer_curve, Z_outer_curve, color="#5d6d7e", lw=3, zorder=4)
    ax.plot(R_inner_curve, Z_outer_curve, color="#5d6d7e", lw=2, zorder=4)

    # Outer leg cross-section indicator
    z_outer_leg = np.linspace(-h_TF * 0.35, h_TF * 0.35, 10)
    R_outer_leg_L = np.full_like(z_outer_leg, R_TF_outer - 0.6)
    R_outer_leg_R = np.full_like(z_outer_leg, R_TF_outer + 0.6)
    ax.fill_betweenx(z_outer_leg, R_outer_leg_L, R_outer_leg_R,
                     color="#d5d8dc", edgecolor="#808b96", hatch="//",
                     alpha=0.5, zorder=3)

    # ── Central solenoid ─────────────────────────────────────────────────
    z_CS = np.linspace(-h_CS / 2, h_CS / 2, 50)
    ax.fill_betweenx(z_CS, R_CS_inner, R_CS_outer,
                     color="#e67e22", alpha=0.5, edgecolor="#d35400", lw=2,
                     label="CS", zorder=5)
    # CS winding indication
    R_CS_mid = (R_CS_inner + R_CS_outer) / 2
    ax.plot([R_CS_mid], [0], marker="s", color="#d35400", markersize=10)

    # ── Vacuum vessel + blanket + shield (inboard) ──────────────────────
    # Inboard: from R_plasma_inner inward
    R_VV_inner_in = R_plasma_inner - r_blk - r_shd - r_vv - r_gap
    R_shield_in = R_plasma_inner - r_blk - r_shd
    R_blanket_in = R_plasma_inner - r_blk

    # Outboard: from R_plasma_outer outward
    outboard_build = 1.6
    R_blanket_out = R_plasma_outer + 0.6
    R_shield_out = R_blanket_out + 0.4
    R_VV_out = R_shield_out + 0.3

    # VV region (inboard)
    z_vv = np.linspace(-h_TF / 2 + 2, h_TF / 2 - 2, 50)
    ax.fill_betweenx(z_vv, R_VV_inner_in, R_plasma_inner - r_blk - r_shd,
                     color="#aeb6bf", alpha=0.3, edgecolor="#5d6d7e", lw=1,
                     label="Shield", zorder=4)

    # Simple VV outline
    z_fill = np.linspace(-h_TF / 2 + 2.5, h_TF / 2 - 2.5, 30)
    ax.fill_betweenx(z_fill, R_VV_inner_in, R_plasma_inner - r_blk - r_shd - r_vv,
                     color="#85929e", alpha=0.2, edgecolor="#5d6d7e", lw=1.5,
                     label="VV", zorder=4)

    # Blanket (inboard + outboard simplified)
    z_blk = np.linspace(-kappa * a - 0.1, kappa * a + 0.1, 30)
    ax.fill_betweenx(z_blk, R_plasma_inner - r_blk, R_plasma_inner,
                     color="#e74c3c", alpha=0.15, edgecolor="#c0392b", lw=1,
                     hatch="...", label="Blanket", zorder=4)
    ax.fill_betweenx(z_blk, R_plasma_outer, R_plasma_outer + 0.6,
                     color="#e74c3c", alpha=0.15, edgecolor="#c0392b", lw=1,
                     hatch="...", zorder=4)

    # ── Divertor targets (lower) ─────────────────────────────────────────
    # Snowflake divertor with two targets
    z_xpt = -kappa * a * 1.02  # X-point Z position
    # Inner divertor leg
    R_div_in = np.linspace(R0 - a * 0.6, R0 - a * 0.3, 10)
    Z_div_in = np.linspace(z_xpt, z_xpt - 0.5, 10)
    ax.plot(R_div_in, Z_div_in, color="#c0392b", lw=4, label="Divertor target", zorder=6)
    # Outer divertor leg
    R_div_out = np.linspace(R0 + a * 0.5, R0 + a * 0.8, 10)
    Z_div_out = np.linspace(z_xpt, z_xpt - 0.5, 10)
    ax.plot(R_div_out, Z_div_out, color="#c0392b", lw=4, zorder=6)
    # Third leg for snowflake
    R_div_mid = np.linspace(R0, R0 + a * 0.3, 10)
    Z_div_mid = np.linspace(z_xpt, z_xpt - 0.7, 10)
    ax.plot(R_div_mid, Z_div_mid, color="#c0392b", lw=4, zorder=6)
    # X-point marker
    ax.plot(R0 - a * 0.1, z_xpt, "rx", markersize=8, markeredgewidth=2, zorder=7)
    ax.text(R0 - a * 0.1 + 0.1, z_xpt - 0.3, "X-point", fontsize=8,
            color="#c0392b", style="italic")

    # ── PF coils ─────────────────────────────────────────────────────────
    pf_colors = {"PF1": "#8e44ad", "PF2": "#2ecc71", "PF3": "#e67e22",
                 "PF4": "#e67e22", "PF5": "#2ecc71", "PF6": "#8e44ad",
                 "D1": "#e74c3c", "D2": "#e74c3c"}

    for name, (R_pf, Z_pf) in pf_coords.items():
        is_div = name.startswith("D")
        r_coil = 0.35 if not is_div else 0.25
        circle = plt.Circle((R_pf, Z_pf), r_coil, color=pf_colors[name],
                            ec="#2c3e50", lw=1.5, alpha=0.8, zorder=10)
        ax.add_patch(circle)
        label_offset = 0.6
        ax.text(R_pf + label_offset, Z_pf, name, fontsize=8, fontweight="bold",
                ha="center", va="center", color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    # ── Ports ────────────────────────────────────────────────────────────
    # Equatorial port
    port_z = 0
    port_R_start = R_plasma_outer + 0.6 + 0.4 + 0.3 + 0.2
    port_R_end = port_R_start + 1.5
    port_half_h = 0.8
    ax.plot([port_R_start, port_R_end], [port_z - port_half_h, port_z - port_half_h],
            color="#5d6d7e", lw=2, zorder=2)
    ax.plot([port_R_start, port_R_end], [port_z + port_half_h, port_z + port_half_h],
            color="#5d6d7e", lw=2, zorder=2)
    ax.plot([port_R_end, port_R_end], [port_z - port_half_h, port_z + port_half_h],
            color="#5d6d7e", lw=2, zorder=2)
    ax.text(port_R_end / 2 + 5, port_z, "Equatorial port", fontsize=8,
            ha="center", va="center", color="#5d6d7e", rotation=90)

    # Upper port
    ax.plot([port_R_start, port_R_end], [h_TF / 2 - 1.5, h_TF / 2 - 1.5],
            color="#5d6d7e", lw=2, zorder=2)
    ax.plot([port_R_start, port_R_end], [h_TF / 2 - 0.3, h_TF / 2 - 0.3],
            color="#5d6d7e", lw=2, zorder=2)
    ax.text(port_R_start + 3, h_TF / 2 - 0.9, "Upper port", fontsize=8,
            ha="center", va="center", color="#5d6d7e", rotation=90)

    # ── Dimension annotations ────────────────────────────────────────────
    # R0 arrow
    ax.annotate("", xy=(0, -11.5), xytext=(R0, -11.5),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
    ax.text(R0 / 2, -12.0, "$R_0 = 12.08$ m", fontsize=10, ha="center",
            va="center", bbox=dict(fc="white", ec="none", alpha=0.7))

    # a arrow
    ax.annotate("", xy=(R0, -12.5), xytext=(R0 + a, -12.5),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
    ax.text(R0 + a / 2, -13.0, "$a = 0.96$ m", fontsize=9, ha="center")

    # κa arrow
    ax.annotate("", xy=(17.0, 0), xytext=(17.0, kappa * a),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
    ax.text(17.5, kappa * a / 2, "$\\kappa a = 2.60$ m", fontsize=9,
            ha="center", va="center")

    # Inboard build
    ax.annotate("", xy=(R_plasma_inner, -11.0), xytext=(R_TF_min + w_TF_inner, -11.0),
                arrowprops=dict(arrowstyle="<->", color="#5d6d7e", lw=1.5))
    ax.text(R0 - a - 0.8, -10.5, "Inboard\nbuild 1.8 m", fontsize=8,
            ha="center", va="center", color="#5d6d7e")

    # CS dimension
    ax.annotate("", xy=(R_CS_inner, 9.0), xytext=(R_CS_outer, 9.0),
                arrowprops=dict(arrowstyle="<->", color="#d35400", lw=1.5))
    ax.text((R_CS_inner + R_CS_outer) / 2, 9.5, "$R_{CS}=0.75\\!-\\!3.10$ m",
            fontsize=8, ha="center", color="#d35400")

    # ── Legend ───────────────────────────────────────────────────────────
    # Custom legend entries using proxy artists
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", alpha=0.25, edgecolor="#2e86c1", label="Plasma"),
        Patch(facecolor="#e74c3c", alpha=0.15, edgecolor="#c0392b", label="Blanket"),
        Patch(facecolor="#aeb6bf", alpha=0.3, edgecolor="#5d6d7e", label="Shield"),
        Patch(facecolor="#85929e", alpha=0.2, edgecolor="#5d6d7e", label="VV"),
        Patch(facecolor="#e67e22", alpha=0.5, edgecolor="#d35400", label="CS"),
        Patch(facecolor="#d5d8dc", alpha=0.5, edgecolor="#808b96", label="TF coil"),
        Patch(facecolor="#8e44ad", alpha=0.8, edgecolor="#2c3e50", label="PF coil"),
        plt.Line2D([0], [0], color="#c0392b", lw=3, label="Divertor target"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left",
              ncol=2, framealpha=0.9)

    # ── Axis settings ────────────────────────────────────────────────────
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-13.5, 13.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$R$ (m)", fontsize=12)
    ax.set_ylabel("$Z$ (m)", fontsize=12)
    ax.set_title("Tokamak Poloidal Cross-Section", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=9)

    # Symmetry line
    ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.4)
    ax.text(0.2, 13.0, "Axis of symmetry", fontsize=8, color="gray", rotation=90)

    plt.tight_layout()
    if save:
        fig.savefig("fig_tokamak_layout.png")
        print("  [+] fig_tokamak_layout.png")
    plt.close(fig)


def fig_power_flow(save=True):
    """
    Power flow diagram showing fusion power → thermal → gross → net,
    with recirculated power breakdown.
    """
    r = quick_eval(D, "SNOWFLAKE")
    P_fus = r["P_fusion_MW"]
    P_gross = r["P_gross_electric_MW"]
    P_net = r["P_net_electric_MW"]
    P_rej = r["P_rejected_MW"]
    eta = r["eta_thermal"]
    recirc = r["P_recirc_breakdown_MW"]

    P_recirc_total = sum(recirc.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [2, 1.2]})

    # ── Left: Waterfall chart ────────────────────────────────────────────
    categories = [
        "Fusion\npower",
        f"Gross\nelectric\n({eta*100:.0f}% η)",
        "Recirculated\npower",
        "Net electric\n(to grid)",
    ]
    values = [P_fus, P_gross, -P_recirc_total, P_net]
    colors_bar = ["#e74c3c", "#3498db", "#e67e22", "#2ecc71"]

    bars = ax1.bar(categories, values, color=colors_bar, edgecolor="white",
                   width=0.5, zorder=3)

    for bar, v in zip(bars, values):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 60 if y >= 0 else -60
        ax1.text(bar.get_x() + bar.get_width() / 2, y + offset,
                 f"{abs(v):.0f} MW" if abs(y) > 1 else "0 MW",
                 ha="center", va=va, fontsize=10, fontweight="bold",
                 color="#2c3e50")

    # Connecting arrows
    y_max = max(P_fus, P_gross) * 1.15
    for i in range(len(values) - 1):
        ax1.annotate("", xy=(i + 1, values[i] + (values[i + 1] if values[i + 1] < 0 else 0)),
                     xytext=(i + 0.25, values[i]),
                     arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5, alpha=0.5))

    ax1.set_ylabel("Power (MW)", fontsize=11)
    ax1.set_title("Power Flow", fontsize=13, fontweight="bold")
    ax1.axhline(0, color="black", lw=1)
    ax1.grid(axis="y", alpha=0.2)
    ax1.set_ylim(-P_recirc_total * 1.3, y_max)

    # Efficiency annotation
    ax1.text(0.95, 0.95,
             f"$\\eta_{{\\text{{thermal}}}} = {eta*100:.1f}\\%$\n"
             f"$\\eta_{{\\text{{net}}}} = {P_net/P_fus*100:.1f}\\%$",
             transform=ax1.transAxes, fontsize=9, va="top", ha="right",
             bbox=dict(fc="white", ec="gray", alpha=0.8, boxstyle="round"))

    # ── Right: Recirc breakdown ──────────────────────────────────────────
    recirc_labels = {
        "He_circulators": "He circulators",
        "cryoplant": "Cryoplant",
        "cooling_pumps": "Cooling pumps",
        "vacuum": "Vacuum pumps",
        "tritium": "Tritium proc.",
        "BOP_aux": "BOP auxiliaries",
        "plasma_control": "Plasma control",
        "ECCD": "ECCD (NTM)",
    }
    labels = [recirc_labels[k] for k in recirc]
    vals = [recirc[k] for k in recirc]

    colors_recirc = plt.cm.Set2(np.linspace(0, 1, len(vals)))
    wedges, texts, autotexts = ax2.pie(
        vals, labels=labels, autopct="%1.1f%%",
        startangle=90, colors=colors_recirc,
        pctdistance=0.75, wedgeprops={"edgecolor": "white", "lw": 1}
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax2.set_title(f"Recirculated Power: {P_recirc_total:.0f} MW",
                  fontsize=12, fontweight="bold")

    fig.suptitle(
        f"Net electric: {P_net:.0f} MW$_e$  |  "
        f"Engineering Q = {P_net / max(P_recirc_total, 0.01):.1f}",
        fontsize=13, y=1.02
    )

    plt.tight_layout()
    if save:
        fig.savefig("fig_power_flow.png")
        print("  [+] fig_power_flow.png")
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
    fig_eccd_system()
    fig_tokamak_layout()
    fig_power_flow()
    os.chdir("..")
    print("\nAll figures generated in figures/")

if __name__ == "__main__":
    generate_all()
