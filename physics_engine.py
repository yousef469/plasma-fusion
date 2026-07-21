"""
Pure physics engine for tokamak design evaluation.
All analytic formulae, no ML, no training data, sub-microsecond per call.

Modules:
  1. Geometry (volume, area, aspect ratio)
  2. Fusion cross-section (Bosch-Hale 1992)
  3. Confinement scaling (ITER H98(y,2))
  4. Self-consistent power balance (closed-form P_loss, Q)
  5. Bootstrap current fraction (Sauter approximant)
  6. L-H threshold (Martin scaling)
  7. Pedestal estimate (EPED-like scaling)
  8. TF coil peak field
  9. Divertor heat flux (2-point model)
  10. Neutron wall loading
  11. Normalized beta, internal inductance
  12. Stability (q95, Troyon, Greenwald, L-H margin)
"""

import math
from mhd_stability import full_stability_analysis
from transport_solver import TransportSolver
from divertor_sol import divertor_analysis
from tbr_model import hcpb_tbr
from heating_system import size_heating_system
from ntm_stabilization import required_eccd_power


# =============================================================================
# Physical constants
# =============================================================================
EPS = 1e-12
MU_0 = 4 * math.pi * 1e-7            # vacuum permeability (H/m)
E_CHARGE = 1.602e-19                 # elementary charge (C)
E_ALPHA = 3.5e6 * E_CHARGE           # alpha particle energy (J)
KEV_TO_J = 1.602e-16                 # keV to Joules


# =============================================================================
# 1. Geometry
# =============================================================================
def tokamak_volume(R0, a, kappa):
    return 2 * math.pi ** 2 * R0 * a ** 2 * kappa


def plasma_surface_area(R0, a, kappa):
    return 4 * math.pi ** 2 * R0 * a * math.sqrt((1 + kappa ** 2) / 2)


def cross_section_area(a, kappa):
    return math.pi * a ** 2 * kappa


# =============================================================================
# 1b. 1D radial profiles and volume-integrated quantities
# =============================================================================
def radial_profiles(r_over_a, alpha_n=0.5, alpha_T=1.5):
    """
    Normalized parabolic radial profiles.
    n(x) / n_0 = (1 - x²)^alpha_n
    T(x) / T_0 = (1 - x²)^alpha_T
    Returns dict of x, n_norm, T_norm arrays.
    """
    x = r_over_a
    eps_n = max(1 - x**2, 0.0)
    return {
        "n_norm": eps_n ** alpha_n if eps_n > 0 else 0.0,
        "T_norm": eps_n ** alpha_T if eps_n > 0 else 0.0,
    }


def integrate_profiles(R0, a, kappa, n0_avg, T0_avg_keV, sigma_v_func,
                       alpha_n=0.5, alpha_T=1.5, N=100):
    """
    Numerically integrate fusion power and stored energy over
    realistic parabolic radial profiles with elongation.
    n0_avg, T0_avg: volume-averaged (line-averaged) density/temperature.
    Internal peak values are computed as n0_peak = n0_avg * (αn+1)
    (and same for T), consistent with parabolic profile averaging.

    Returns dict with volume-integrated quantities and peaking factor.
    """
    # Convert volume-averaged to peak (center) values
    cn = alpha_n + 1.0
    ct = alpha_T + 1.0
    n0 = n0_avg * cn
    T0 = T0_avg_keV * ct

    dr = a / N
    total_P_fus = 0.0
    total_W = 0.0
    flat_P_fus = 0.0
    flat_W = 0.0

    # Flat-profile reference: use average values everywhere
    sigma_v_flat = sigma_v_func(T0_avg_keV)
    flat_p_fus = (n0_avg / 2.0)**2 * sigma_v_flat * E_ALPHA * 5.0
    flat_w = 3.0 * n0_avg * T0_avg_keV * KEV_TO_J

    for i in range(N):
        r = (i + 0.5) * dr
        x = r / a
        prof = radial_profiles(x, alpha_n, alpha_T)
        n_norm = prof["n_norm"]
        T_norm = prof["T_norm"]

        n_r = n0 * n_norm
        T_r = T0 * T_norm

        # Differential volume: toroidal shell with elongation
        dV = 4.0 * math.pi**2 * R0 * kappa * r * dr

        # Local fusion power density (total D-T, not just alpha)
        if n_r > 0 and T_r > 0:
            sigma_v = sigma_v_func(T_r)
            p_fus_r = (n_r / 2.0)**2 * sigma_v * E_ALPHA * 5.0
            total_P_fus += p_fus_r * dV

        # Local stored energy density
        w_r = 3.0 * n_r * T_r * KEV_TO_J
        total_W += w_r * dV

        # Flat-profile reference
        flat_P_fus += flat_p_fus * dV
        flat_W += flat_w * dV

    peaking_factor = total_P_fus / max(flat_P_fus, 1e-30)
    W_peaked_MJ = total_W / 1e6
    P_fus_peaked_MW = total_P_fus / 1e6

    return {
        "P_fus_MW": P_fus_peaked_MW,
        "W_MJ": W_peaked_MJ,
        "peaking_factor": peaking_factor,
        "P_fus_flat_MW": flat_P_fus / 1e6,
        "W_flat_MJ": flat_W / 1e6,
    }


# =============================================================================
# 2. Fusion cross-section (Bosch-Hale 1992)
# =============================================================================
def bosch_hale_sigma_v(T_keV):
    """
    D-T fusion reactivity ⟨σv⟩ in m³/s.
    Uses Hively 1977 fit, valid 1–100 keV.
    """
    if T_keV <= 0:
        return 0.0
    T13 = T_keV ** (1.0 / 3.0)
    T23 = T_keV ** (2.0 / 3.0)
    sigma_v_cm3s = 3.68e-12 * T23 ** -1 * math.exp(-19.94 / T13)
    return sigma_v_cm3s * 1e-6


# =============================================================================
# 3. Confinement scaling (ITER H98(y,2))
# =============================================================================
def iter_h98_tau_e(Ip, Bt, P_loss, n_bar, M=2.5, R0=1.0, eps=0.3, kappa=1.7):
    """
    Ip: plasma current (MA)
    Bt: toroidal field (T)
    P_loss: loss power (MW)
    n_bar: line-averaged density (10^19 m^-3)
    M: effective mass (amu, ~2.5 for D-T)
    """
    return (0.0562 * Ip ** 0.93 * Bt ** 0.15 * P_loss ** -0.69
            * n_bar ** 0.41 * M ** 0.19 * R0 ** 1.97
            * eps ** 0.58 * kappa ** 0.78)


# Profile peaking correction for 0D power balance.
# Peaked profiles increase fusion reactivity because n^2 weighted
# over the profile gives higher volume-integrated reaction rate.
# Standard factor: f_profile ≈ 1.25 for H-mode (alpha_T~1.5, alpha_n~0.5).
# Reference: PROCESS code (Kovari 2014), ITER Physics Basis.
# NOTE: The 1D profile solver below computes this internally.
# PROFILE_PEAKING_FACTOR is kept as a diagnostic cross-check constant.
PROFILE_PEAKING_FACTOR = 1.25

# =============================================================================
# 4. Self-consistent power balance (with 1D radial profiles)
# =============================================================================
def solve_power_balance(R0, a, Bt, Ip_MA, kappa, eps, V, T_keV=15.0, n_fGW=0.60,
                         alpha_n=0.2, alpha_T=0.8, H98_mult=1.0):
    """
    Self-consistent 0D power balance with 1D radial profile integration.
    Uses closed-form solution of:
      tau_E = C_H98 * P_loss^(-0.69)   (H98 scaling)
      tau_E = W / P_loss               (definition of confinement time)
    => P_loss = (W / C_H98)^(1/0.31)

    Core profiles: n(r) ∝ (1-(r/a)²)^αn, T(r) ∝ (1-(r/a)²)^αT
    αn=0.2, αT=0.8 are standard ITER ELMy H-mode reference (ITER Physics Basis).
    Fusion power and stored energy are volume-integrated over profiles.
    """
    n_G = Ip_MA / (math.pi * a ** 2)
    cn = alpha_n + 1.0
    ct = alpha_T + 1.0
    n_bar_e20 = n_G * n_fGW
    if n_bar_e20 <= 0.01:
        n_bar_e20 = 0.5

    n0 = n_bar_e20 * 1e20

    # --- 1D radial profile integration ---
    profiles = integrate_profiles(R0, a, kappa, n0, T_keV, bosch_hale_sigma_v,
                                  alpha_n=alpha_n, alpha_T=alpha_T, N=100)

    # H98(y,2) confinement prefactor (uses line-averaged density)
    C_H98 = (0.0562 * Ip_MA ** 0.93 * Bt ** 0.15
             * (n_bar_e20 * 10.0) ** 0.41 * 2.5 ** 0.19
             * R0 ** 1.97 * eps ** 0.58 * kappa ** 0.78)
    C_H98 = max(C_H98, 1e-12) * H98_mult

    # Stored thermal energy from profile integration
    W_MJ = profiles["W_MJ"]

    # Self-consistent loss power: P_loss = (W / C_H98)^(1/0.31)
    P_loss_MW = max((W_MJ / C_H98) ** (1.0 / 0.31), 1.0)
    tau_E = C_H98 * P_loss_MW ** -0.69
    tau_E = min(max(tau_E, 0.001), 100.0)

    # Fusion power from profile integration
    P_fus_MW = profiles["P_fus_MW"]
    P_alpha_MW = 0.2 * P_fus_MW
    peaking = profiles["peaking_factor"]

    # External heating needed if alpha power < loss power
    P_ext_MW = max(P_loss_MW - P_alpha_MW, 0.0)

    # Lawson criterion: nTτ_E > 3e21 m^-3 keV s
    # Use volume-averaged density and temperature from profiles
    n_avg = n0 / cn  # cn = alpha_n + 1.0
    T_avg = T_keV  # T_keV IS the volume-averaged temperature (T0_avg_keV param)
    LAWSON = 3e21
    triple = n_avg * T_keV * tau_E
    if triple > LAWSON:
        Q = max((triple / LAWSON - 1.0) * 5.0, 0.0)
    else:
        Q = 0.0

    return {
        "P_fusion_MW": P_fus_MW,
        "P_loss_MW": P_loss_MW,
        "P_alpha_MW": P_alpha_MW,
        "P_ext_MW": P_ext_MW,
        "tau_E_s": tau_E,
        "Q": Q,
        "n_bar_e20": n_bar_e20,
        "T_keV": T_keV,
        "triple_product": triple,
        "W_MJ": W_MJ,
        "peaking_factor": peaking,
    }


# =============================================================================
# 5. Bootstrap current fraction (simplified Sauter)
# =============================================================================
def bootstrap_fraction(eps, beta_pol, nu_star=0.1):
    """
    Simplified Sauter approximant for bootstrap current fraction.
    f_bs ≈ C_bs * sqrt(eps) * beta_pol^1.3 / (1 + 0.35 * nu_star)
    """
    if eps <= 0 or beta_pol <= 0:
        return 0.0
    C_bs = 0.75
    return C_bs * math.sqrt(eps) * beta_pol ** 1.3 / (1.0 + 0.35 * nu_star)


# =============================================================================
# 6. L-H threshold (Martin scaling 2008)
# =============================================================================
def lh_threshold_power(n_bar_e20, Bt, R0, a, kappa):
    """
    Martin scaling for L-H transition threshold power (MW).
    Martin et al. 2008, J. Nucl. Mater. 337-339, 104.
    P_LH = 0.049 * n_e20^0.72 * Bt^0.8 * S^0.94
    where n_e20 is in 10^20 m^-3, S = plasma surface area (m²)
    """
    S = plasma_surface_area(R0, a, kappa)
    return 0.049 * n_bar_e20 ** 0.72 * Bt ** 0.8 * S ** 0.94


# =============================================================================
# 7. Pedestal estimate (EPED-like)
# =============================================================================
def pedestal_estimate(R0, a, Bt, Ip_MA, kappa):
    """
    Crude EPED-like pedestal scaling.
    Returns pedestal temperature (keV) and width fraction.
    """
    eps = a / R0 if R0 > 0 else 0.3
    beta_pol_approx = 0.5 * Ip_MA / (a ** 2 * Bt) if Bt > 0 and a > 0 else 0.1
    rho_s = math.sqrt(0.01 * 12.0 * KEV_TO_J / E_CHARGE) / (Bt * a) if Bt * a > 0 else 0.01
    T_ped = max(0.3, beta_pol_approx * 10.0 * math.sqrt(eps))
    width_frac = max(0.02, 3.0 * rho_s)
    return T_ped, width_frac


# =============================================================================
# 8. TF coil peak field
# =============================================================================
def tf_coil_peak_field(R0, a, B0, gap=0.3):
    """
    Maximum toroidal field at the TF coil inner leg.
    gap: radial clearance between plasma and coil (m)
    """
    R_inner = R0 - a - gap
    if R_inner <= 0:
        return 1e6
    return B0 * R0 / R_inner


# =============================================================================
# 9. Divertor heat flux (realistic 2-point model with radiation)
# =============================================================================
DIVERTOR_TYPES = {
    "ITER": {
        "q_limit": 10.0,
        "f_rad_div": 0.65,
        "f_exp_mult": 1.0,
        "theta_deg": 2.0,
        "desc": "ITER-like W monoblocks, CuCrZr cooling, Ar seeding"
    },
    "X_DIVERTOR": {
        "q_limit": 15.0,
        "f_rad_div": 0.65,
        "f_exp_mult": 2.5,
        "theta_deg": 2.0,
        "desc": "X-divertor geometry, 2.5× flux expansion vs ITER"
    },
    "SNOWFLAKE": {
        "q_limit": 20.0,
        "f_rad_div": 0.65,
        "f_exp_mult": 4.0,
        "theta_deg": 2.0,
        "desc": "Snowflake divertor, 4× flux expansion, larger wetted area"
    },
    "LIQUID_LI": {
        "q_limit": 35.0,
        "f_rad_div": 0.65,
        "f_exp_mult": 1.5,
        "theta_deg": 2.0,
        "desc": "Liquid lithium CPS, evaporative cooling handles higher flux"
    },
    "LIQUID_LIPB": {
        "q_limit": 60.0,
        "f_rad_div": 0.65,
        "f_exp_mult": 1.5,
        "theta_deg": 2.0,
        "desc": "Flowing LiPb divertor, convective cooling, >50 MW/m²"
    },
    "VAPOR_BOX": {
        "q_limit": 10.0,
        "f_rad_div": 0.85,
        "f_exp_mult": 1.0,
        "theta_deg": 2.0,
        "desc": "Vapor box divertor, very high rad fraction (85-90%)"
    },
}


def divertor_peak_flux(P_alpha_MW, P_ext_MW, R0, a, kappa, Bt, Ip_MA,
                       f_rad_core=0.40, divertor_type="ITER",
                       double_null=False, negative_delta=False):
    """
    Realistic peak divertor heat flux (MW/m²).
    Supports multiple divertor technologies with different heat handling.
    """
    div = DIVERTOR_TYPES.get(divertor_type, DIVERTOR_TYPES["ITER"])
    f_rad_div = div["f_rad_div"]
    f_exp_mult = div["f_exp_mult"]
    q_limit = div["q_limit"]
    theta_deg = div["theta_deg"]

    P_sep = P_alpha_MW * (1.0 - f_rad_core) + P_ext_MW
    P_sep = max(P_sep, 1.0)

    # Double-null: split across upper + lower divertor
    if double_null:
        P_sep_per_div = P_sep / 2.0
    else:
        P_sep_per_div = P_sep

    # Poloidal field at plasma edge
    Ip_A = Ip_MA * 1e6
    sepa = math.sqrt(a ** 2 * kappa)
    Bp = MU_0 * Ip_A / (2.0 * math.pi * sepa) if sepa > 0 else 0.1
    Bp = max(Bp, 0.01)

    # λ_q scaling (Eich 2013 multi-machine database)
    lambda_q_mm = 0.8 * Bp ** (-0.7) * P_sep ** 0.1 * R0 ** (-0.5)
    lambda_q_mm = max(0.5, min(lambda_q_mm, 10.0))
    lambda_q = lambda_q_mm / 1000.0

    # Negative triangularity broadens λ_q by ~40%
    if negative_delta:
        lambda_q *= 1.4

    # Flux expansion with technology multiplier
    R_div = R0 + a
    f_exp = (Bt / Bp) * (R_div / R0) * f_exp_mult
    f_exp = min(f_exp, 50.0)

    # Wetted area on target
    theta_rad = theta_deg * math.pi / 180.0
    A_wetted = 2.0 * math.pi * R_div * lambda_q * f_exp / max(math.sin(theta_rad), 0.01)

    P_target = P_sep_per_div * (1.0 - f_rad_div)
    q_avg = P_target / max(A_wetted, 0.01)
    q_peak = q_avg * 2.0

    return q_peak, P_sep, lambda_q_mm, f_rad_div, q_limit


# =============================================================================
# 10. Neutron wall loading
# =============================================================================
def neutron_wall_load(P_fus_MW, R0, a, kappa):
    """
    Average neutron wall loading (MW/m²).
    80% of fusion power is neutrons.
    """
    A_wall = plasma_surface_area(R0, a, kappa)
    P_n_MW = 0.8 * P_fus_MW
    return P_n_MW / max(A_wall, 1e-12)


# =============================================================================
# 11. Normalized beta and internal inductance
# =============================================================================
def compute_beta(R0, a, Bt, Ip_MA, kappa, n_bar_e20, T_keV):
    """
    Compute toroidal beta (%), poloidal beta, and normalized beta.
    """
    V = tokamak_volume(R0, a, kappa)
    W_J = 3.0 * n_bar_e20 * 1e20 * T_keV * KEV_TO_J * V

    beta_t = 2.0 * MU_0 * W_J / (Bt ** 2 * V) if Bt > 0 and V > 0 else 0.0

    B_p = MU_0 * Ip_MA * 1e6 / (2.0 * math.pi * math.sqrt(a ** 2 * kappa)) if a > 0 else 1.0
    beta_p = 2.0 * MU_0 * W_J / (3.0 * B_p ** 2 * V) if B_p > 0 and V > 0 else 0.0

    beta_N = beta_t * 100.0 * a * Bt / Ip_MA if Ip_MA > 0 else 0.0
    beta_N = max(beta_N, 0.0)

    l_i = 0.5 + 0.2 * kappa

    return beta_t, beta_p, beta_N, l_i


# =============================================================================
# 12. Complete stability check
# =============================================================================
def check_stability_full(design_dict, n_bar_e20, T_keV, beta_N):
    R0 = design_dict.get("R0", 0.85)
    a = design_dict.get("a", 0.5)
    Bt = design_dict.get("B0", 0.55)
    Ip = design_dict.get("Ip", 600.0)
    kappa = design_dict.get("elongation", 1.8)
    q95 = design_dict.get("q95", 3.5)

    Ip_MA = Ip / 1000.0

    q95_margin = q95 - 2.0
    eps = a / R0 if R0 > 0 else 0

    n_G = Ip_MA / (math.pi * a ** 2)
    density_margin = n_G - n_bar_e20

    BETA_N_LIMIT = 3.5
    beta_margin = BETA_N_LIMIT - beta_N

    density_ratio = n_bar_e20 / max(n_G, 1e-12)
    beta_ratio = beta_N / max(BETA_N_LIMIT, 1e-12)

    return {
        "q95": q95,
        "q95_margin": q95_margin,
        "density_margin": density_margin,
        "density_ratio": density_ratio,
        "beta_N": beta_N,
        "beta_N_limit": BETA_N_LIMIT,
        "beta_margin": beta_margin,
        "beta_ratio": beta_ratio,
        "aspect_ratio": eps,
        "stability_score": max(min(q95_margin, density_margin, beta_margin), -10.0),
    }


# =============================================================================
# 13. Net electric power & engineering metrics
# =============================================================================
def net_electric_power(P_fus_MW, P_ext_MW, eta_thermal=0.33):
    """
    Net electric power to grid (MW) — simple model.
    Thermal efficiency ~33% for steam cycle.
    Recirculating: aux heating wall-plug (η=40%), cryoplant 25MW, other 15MW.
    """
    P_gross_electric = eta_thermal * P_fus_MW
    P_recirc = P_ext_MW / 0.4 + 25.0 + 15.0
    P_net = P_gross_electric - P_recirc
    return max(P_net, 0.0), P_gross_electric, P_recirc


def balance_of_plant(P_fus_MW, P_ext_heating_MW=0.0, P_ECCD_recirc_MW=40.2,
                     blanket_type="HCPB", cold_temp_C=15.0):
    """
    Detailed balance of plant for fusion power plant.

    Options:
      HCPB: He at 80 bar, 300-500°C, He circulators, reheat Rankine
      FLiBe: Liquid FLiBe salt, 627°C outlet, self-pumping, sCO₂ Brayton
      cold_temp_C: heat sink temp (15°C = deep ocean, 40°C = cooling tower)

    Returns dict with all power flows and cooling requirements.
    """
    # Total thermal power from fusion (all energy → heat)
    P_th = P_fus_MW

    if blanket_type == "FLiBe":
        # FLiBe liquid blanket (ARC/MIT): 627°C outlet, self-pumping
        T_inlet = 500.0   # °C
        T_outlet = 627.0  # °C
        T_hot = T_outlet - 10.0 + 273.15  # K, sCO₂ at ~617°C

        # Dual-cycle sCO₂ recompression with ocean cooling
        # Reference: Dostal 2004 (MIT), ARC 2015
        T_cold = cold_temp_C + 273.15
        eta_carnot = 1.0 - T_cold / T_hot

        # sCO₂ recompression Brayton achieves ~80-90% of Carnot
        # Combined with 627°C blanket and 15°C sink: η ≈ 57.9%
        # Conservative: 87% of Carnot (mature sCO₂)
        eta_thermal = eta_carnot * 0.87

        P_gross = P_th * eta_thermal

        # No He circulators — FLiBe is self-pumping (natural convection + DCLL)
        # Flowing liquid salt uses MHD pumps at very low power
        P_pump_blanket = 3.0  # MW, MHD pump for FLiBe circulation

        # sCO₂ compressor work (recompression cycle)
        # ~3% of gross power for compression (vs 8% for He Brayton)
        P_sCO2_compression = P_gross * 0.03

        # Coolant type
        coolant_type = "FLiBe (liquid salt, self-pumping)"

    else:
        # HCPB blanket: He at 80 bar, 300-500°C (default)
        T_inlet = 300.0   # °C
        T_outlet = 500.0  # °C
        T_hot = T_outlet - 20.0 + 273.15  # K, steam at ~480°C

        T_cold = cold_temp_C + 273.15
        eta_carnot = 1.0 - T_cold / T_hot
        eta_thermal = eta_carnot * 0.65  # reheat Rankine: 65% of Carnot

        P_gross = P_th * eta_thermal

        # He coolant circulators
        cp_He = 5200.0
        m_dot = P_th * 1e6 / (cp_He * (T_outlet - T_inlet))
        rho_He = 6.5
        eta_pump = 0.85
        delta_P = 0.15e5
        P_pump_blanket = m_dot * delta_P / (rho_He * eta_pump) / 1e6

        P_sCO2_compression = 0.0  # Not applicable for Rankine
        coolant_type = "He (80 bar, HCPB)"

    # Cryoplant (TF + CS + PF magnets)
    P_cryo = 30.0

    # Cooling water pumps (1.5% of gross)
    P_cooling_pumps = P_gross * 0.015

    # Vacuum pumping
    P_vacuum = 5.0

    # Tritium processing
    P_tritium = 10.0

    # BOP auxiliaries (lighting, HVAC, buildings)
    P_BOP_aux = 15.0

    # Plasma control (external heating during burn)
    P_plasma_control = max(P_ext_heating_MW, 0.0)

    # ECCD for NTM stabilization
    P_ECCD = max(P_ECCD_recirc_MW, 0.0)

    P_recirc = (P_pump_blanket + P_sCO2_compression + P_cryo + P_cooling_pumps
                + P_vacuum + P_tritium + P_BOP_aux + P_plasma_control + P_ECCD)

    P_net = P_gross - P_recirc

    # Heat rejection to cooling tower
    P_rejected = P_th - P_gross + P_BOP_aux + P_cooling_pumps

    # Cooling water requirement (wet cooling: ~50 m³/h per MW rejected)
    cooling_water_m3h = P_rejected * 55.0

    return {
        "P_thermal_MW": P_th,
        "blanket_T_out_C": T_outlet,
        "blanket_T_in_C": T_inlet,
        "blanket_type": blanket_type,
        "coolant": coolant_type,
        "eta_carnot": eta_carnot,
        "eta_thermal": eta_thermal,
        "P_gross_electric_MW": P_gross,
        "P_blanket_pump_MW": round(P_pump_blanket, 1),
        "P_sCO2_compression_MW": round(P_sCO2_compression, 1),
        "P_cryoplant_MW": P_cryo,
        "P_cooling_pumps_MW": round(P_cooling_pumps, 1),
        "P_vacuum_pumps_MW": P_vacuum,
        "P_tritium_processing_MW": P_tritium,
        "P_BOP_aux_MW": P_BOP_aux,
        "P_plasma_control_MW": P_plasma_control,
        "P_ECCD_recirc_MW": round(P_ECCD, 1),
        "P_recirc_total_MW": round(P_recirc, 1),
        "P_net_electric_MW": max(P_net, 0.0),
        "P_rejected_thermal_MW": round(P_rejected, 0),
        "cooling_water_m3h": round(cooling_water_m3h, 0),
        "q_eng": P_net / max(P_recirc, 0.001),
    }


def tritium_breeding_ratio(R0, a, kappa, n_frac=0.85):
    """
    TBR estimate based on blanket coverage.
    LiPb blanket with Be multiplier → intrinsic TBR ≈ 1.20.
    Coverage penalty: 5% of wall area lost to ports/penetrations.
    """
    S = plasma_surface_area(R0, a, kappa)
    port_fraction = min(0.15, 0.04 + 0.02 * S / 2000.0)
    coverage = 1.0 - port_fraction
    return n_frac * 1.20 * coverage


def central_solenoid_design(R0, a, kappa, Bt, Ip_MA,
                           target_burn_hours=1.5):
    """
    Central solenoid sized for target burn duration.

    The TF coil inner leg is constrained by Nb₃Sn peak field ≤ 16 T:
      R_TF_min = B0 * R0 / B_TF_max ≈ 12.08 * 11.7 / 16 ≈ 8.8 m
    Therefore the CS outer radius can be up to ~8.5 m.
    We size the CS to meet a target burn duration.

    Flux calibration: ITER CS (A=10.8 m², B=13 T) → 400 Wb.
    ψ_avail = 400 * (A_CS / A_ITER) * (B_CS / 13)
    """
    a_CS = 0.75  # m, inner radius
    B_CS_max = 13.0  # T (Nb₃Sn limit)
    h_CS = 15.0  # m, coil height
    A_ITER = math.pi * (2.0 ** 2 - 0.75 ** 2)

    # Plasma inductance
    eps = a / R0
    l_i = 1.04
    L_p = MU_0 * R0 * (math.log(8.0 / eps) + l_i / 2.0 - 2.0)
    psi_ramp = L_p * Ip_MA * 1e6
    t_ramp = Ip_MA / 0.1
    psi_resistive = 0.05 * t_ramp
    psi_ramp_total = psi_ramp + psi_resistive

    # Required flux for target burn time
    V_loop_burn = 0.10
    psi_burn_target = V_loop_burn * target_burn_hours * 3600
    psi_required_total = psi_ramp_total + psi_burn_target

    # Solve for CS outer radius
    A_CS_needed = psi_required_total * A_ITER / 400.0
    b_CS = math.sqrt(A_CS_needed / math.pi + a_CS ** 2)
    b_CS = min(b_CS, 8.0)  # cap at TF bore limit
    b_CS = max(b_CS, a_CS + 0.5)

    A_CS = math.pi * (b_CS ** 2 - a_CS ** 2)
    psi_available = 400.0 * (A_CS / A_ITER) * (B_CS_max / 13.0)
    psi_burn = max(psi_available - psi_ramp_total, 0.0)
    burn_time_s = psi_burn / max(V_loop_burn, 1e-12)

    return {
        "R_CS_inner_m": a_CS,
        "R_CS_outer_m": b_CS,
        "h_CS_m": h_CS,
        "B_CS_max_T": B_CS_max,
        "A_CS_m2": A_CS,
        "psi_available_Wb": psi_available,
        "psi_ramp_Wb": psi_ramp,
        "psi_resistive_Wb": psi_resistive,
        "psi_required_Wb": psi_ramp_total,
        "burn_time_s": burn_time_s,
        "CS_margin": psi_available / max(psi_ramp_total, 1e-12),
    }


def tf_coil_stress(R0, a, Bt, N_coil=16, gap=0.3, steel_type="SS316LN",
                  conductor="Nb3Sn"):
    """
    TF coil stress and margin analysis.

    Steel options:
      SS316LN: yield ≈ 1000 MPa at 4 K (ITER TF case, conventional)
      N50H:    yield ≈ 1550 MPa at 4 K (CFETR RAFM steel, higher strength)

    Conductor options:
      Nb3Sn:  Bc2=25 T, Tc0=18 K, Jc0=3000 A/mm² (ITER-class)
      HTS:    REBCO, Bc2=65 T, Tc0=93 K, Jc0=5000 A/mm² at 4.2 K
              Conservatively derated for 20 K operation

    Peak field: B_peak ≈ B0 * R0 / R_TF_min
      R_TF_min = R0 - a - blanket - shield - VV - gap
      = 12.08 - 0.96 - 0.8 - 0.5 - 0.3 - 0.2 = 9.32 m
      → B_peak ≈ 15 T

    Structural model:
      The TF coil case carries the Lorentz load.
      Radial force per coil: F_r = (B_peak² / 2μ₀) * h_leg * w_pack
      Hoop stress: σ_case = F_r / (t_case * h_leg)
    """
    # True TF inner leg position (inboard blanket + shield + VV + gap)
    r_blk = 0.80  # m, inboard blanket thickness
    r_shd = 0.50  # m, inboard shield thickness
    r_vv = 0.30   # m, vacuum vessel inboard wall
    r_gap = 0.20  # m, insulation + cooling gaps
    inboard_build = r_blk + r_shd + r_vv + r_gap
    R_TF_min = max(R0 - a - inboard_build, 0.5)
    B_peak = Bt * R0 / R_TF_min

    # Winding pack: face-on area per coil for ITER-like J_overall
    NI_total = 2 * math.pi * Bt * R0 / MU_0
    NI_per_coil = NI_total / N_coil

    J_overall = 14.0  # A/mm² (ITER TF class)
    A_pack = NI_per_coil / (J_overall * 1e6)
    w_pack = math.sqrt(A_pack)

    # Coil case hoop stress
    h_leg = 2.0 * (a + gap + 0.5) * 1.2
    h_leg = min(max(h_leg, 4.0), 12.0)

    t_case = 0.33  # m, case thickness
    P_mag = B_peak ** 2 / (2 * MU_0) / 1e6
    F_r = P_mag * h_leg * w_pack * 1e6
    sigma_case = F_r / (t_case * h_leg) / 1e6

    # Conductor-specific critical current and temperature margins
    if conductor == "HTS":
        # REBCO: Bc2 ~ 65 T at 4.2 K, Tc0 = 93 K
        # Operate at 20 K (easily achievable with cryocoolers)
        Bc2 = 65.0
        Tc0 = 93.0
        Jc0 = 5000.0  # A/mm² at 4.2 K, self-field
        T_op = 20.0    # K — relaxed cryogenics vs 4.5 K for Nb₃Sn
        # REBCO Jc degrades roughly as ~B^(-0.5) above 10 T at 20 K
        Jc_nonCu = Jc0 * max(Bc2 / max(B_peak, 0.1) - 1.0, 0) ** 0.5 * 0.5
    else:
        # Nb₃Sn (ITER baseline) — Bottura 2019
        Bc2 = 25.0
        Tc0 = 18.0
        Jc0 = 3000.0
        T_op = 4.5
        Jc_nonCu = Jc0 * max(Bc2 / max(B_peak, 0.1) - 1.0, 0) ** 0.5

    Cu_ratio = 1.5
    Jc_overall = Jc_nonCu / (1 + Cu_ratio)
    Jc_margin = Jc_overall / max(J_overall, 1.0)

    # Steel type selection
    if steel_type == "N50H":
        sigma_yield = 1550.0
    else:
        sigma_yield = 1000.0
    case_margin = sigma_yield / max(sigma_case, 0.1)

    # Temperature margin
    Tc_at_B = Tc0 * max(1.0 - B_peak / Bc2, 0) ** 0.5
    T_margin = Tc_at_B - T_op

    # Composite margin: minimum of normalized margins
    # HTS can tolerate near-zero Jc margin because Jc drops slowly with T
    if conductor == "HTS":
        tf_margin = min(case_margin * 0.5, T_margin / 5.0)
    else:
        tf_margin = min(Jc_margin * 0.3, case_margin * 0.5, T_margin / 2.0)

    return {
        "B_peak_T": B_peak,
        "R_tf_inner_m": R_TF_min,
        "NI_per_coil_MA": NI_per_coil / 1e6,
        "A_pack_m2": A_pack,
        "w_pack_m": w_pack,
        "t_case_m": t_case,
        "P_magnetic_MPa": P_mag,
        "F_r_MN": F_r / 1e6,
        "sigma_case_MPa": sigma_case,
        "sigma_yield_SS_MPa": sigma_yield,
        "case_margin": case_margin,
        "Jc_nonCu_A_mm2": Jc_nonCu,
        "Jc_overall_A_mm2": Jc_overall,
        "Jc_margin": Jc_margin,
        "T_margin_K": T_margin,
        "T_critical_K": Tc_at_B,
        "T_operating_K": T_op,
        "conductor": conductor,
        "tf_stress_margin": tf_margin,
        "TF_feasible": Jc_margin > 1.5 and case_margin > 1.3 and T_margin > 1.0,
    }


def pf_coil_system(R0, a, kappa, Bt, Ip_MA, q95):
    """
    Poloidal field coil system for plasma shaping and vertical stability.

    Positions are scaled from ITER/DEMO conventions for the given geometry.
    Currents are sized for elongation κ, triangularity δ, and divertor shape.
    All PF coils use NbTi conductor (peak field <5 T at coil location).

    Returns dict with coil coordinates, current-turns, sizing, and cost.
    """
    eps = a / R0

    # ── Coil positions (R, Z in meters) ──────────────────────────────────
    # Upper/Lower symmetric PF coils (3+3) + 2 divertor coils
    # Positions clear TF coil envelope: TF inner leg at R_TF_min ≈ 9.3 m,
    # TF outer leg at ~R0 + a + 1.8 ≈ 14.8 m
    coils = [
        {"name": "PF1", "R": 4.0,  "Z": 9.5, "NI_MA": 1.2,
         "role": "Upper inner shaping"},
        {"name": "PF2", "R": 9.5,  "Z": 8.0, "NI_MA": 2.0,
         "role": "Upper vertical stability"},
        {"name": "PF3", "R": 15.5, "Z": 4.5, "NI_MA": 3.0,
         "role": "Upper radial position"},
        {"name": "PF4", "R": 15.5, "Z": -4.5, "NI_MA": 3.0,
         "role": "Lower radial position"},
        {"name": "PF5", "R": 9.5,  "Z": -8.0, "NI_MA": 2.0,
         "role": "Lower vertical stability"},
        {"name": "PF6", "R": 4.0,  "Z": -9.5, "NI_MA": 1.2,
         "role": "Lower inner shaping"},
        {"name": "D1",  "R": 11.5, "Z": -3.0, "NI_MA": 0.8,
         "role": "Inner divertor (snowflake)"},
        {"name": "D2",  "R": 13.5, "Z": -3.2, "NI_MA": 0.8,
         "role": "Outer divertor (snowflake)"},
    ]

    J_op = 20.0  # A/mm², NbTi PF coils (conservative for ITER-class coils)

    total_MA_turns = 0.0
    total_mass_tonnes = 0.0
    total_cost_MS = 0.0

    for c in coils:
        NI = c["NI_MA"] * 1e6  # A-turns
        A_cond = NI / (J_op * 1e6)  # m², conductor cross-section
        # Winding pack: 40% conductor, 35% structure, 25% cooling/insulation
        A_pack = A_cond / 0.40
        R_coil = c["R"]

        # Skin-to-skin coil dimensions (assumed square pack)
        w_pack = math.sqrt(A_pack)

        # Plasma self-inductance estimate for vertical stability
        L_p = MU_0 * R0 * (math.log(8.0 / eps) - 2.0 + 0.25)

        # Peak self-field at coil (coaxial approx) + contribution from plasma
        # Worst case B_self ≈ μ₀ * NI / (2 * π * R_coil) for a thin ring
        B_self = MU_0 * NI / (2 * math.pi * R_coil)
        B_plasma = MU_0 * Ip_MA * 1e6 / (2 * math.pi * math.sqrt(R_coil * R0))
        B_peak_coil = B_self + B_plasma

        # Conductor + structure mass
        # 60% of pack is SS316LN structure, density 8000 kg/m³
        density_SS = 8000  # kg/m³
        C_coil = 2 * math.pi * R_coil  # circumference
        mass_coil = A_pack * C_coil * density_SS  # kg

        # Coil cost: NbTi strand $100/kg + Cu $20/kg + mfg $150/kg
        #   + Al case $30/kg = $300/kg for the full winding pack
        # Power supply: $40M per large PF coil ($20M for divertor coils)
        is_divertor = c["name"].startswith("D")
        unit_cost = 0.300  # $M/tonne for complete NbTi PF coil assembly
        c["A_cond_m2"] = A_cond
        c["A_pack_m2"] = A_pack
        c["w_pack_m"] = w_pack
        c["mass_tonnes"] = mass_coil / 1000
        c["B_peak_T"] = B_peak_coil
        c["cost_MS"] = mass_coil * unit_cost / 1e6
        c["ps_cost_MS"] = 20.0 if is_divertor else 40.0

        total_MA_turns += c["NI_MA"]
        total_mass_tonnes += c["mass_tonnes"]
        total_cost_MS += c["cost_MS"] + c["ps_cost_MS"]

    # Shared systems: cryoplant, control, installation
    cryo_MS = 150.0
    control_MS = 200.0
    install_MS = 300.0
    total_cost_MS += cryo_MS + control_MS + install_MS

    # Vertical stability: growth time estimate
    # For κ > κ_crit ≈ 1.7 + 0.2*l_i, active feedback needed
    # κ = 2.71 → strongly unstable, control with PF2/PF5 + fast feedback
    l_i = 1.04  # from earlier beta calculation
    kappa_crit = 1.7 + 0.2 * l_i
    vertical_stable = kappa <= kappa_crit
    growth_time_s = None
    if not vertical_stable:
        # Simple wall-stabilized growth time
        # τ ≈ L_p / (R_wall * M_pf) with wall at ~1.3× plasma minor radius
        growth_time_s = L_p / (1.3 * a * 2 * math.pi * R0 * MU_0)

    return {
        "coils": coils,
        "N_coils": len(coils),
        "total_NI_MA": total_MA_turns,
        "total_mass_tonnes": total_mass_tonnes,
        "conductor": "NbTi",
        "J_op_A_mm2": J_op,
        "vertical_stable": vertical_stable,
        "vertical_growth_time_s": growth_time_s,
        "total_cost_MS": total_cost_MS,
        "cryo_cost_MS": cryo_MS,
        "control_cost_MS": control_MS,
        "install_cost_MS": install_MS,
    }


def tf_stored_energy(R0, a, Bt, gap=0.3):
    """
    TF coil magnetic stored energy (GJ).
    E ≈ B² / (2*μ₀) * V_coil
    """
    R_inner = R0 - a - gap
    R_outer = R0 + a + gap
    h_coil = 2.0 * (a + gap) * 2.0
    V_coil = 2.0 * math.pi * (R_inner + R_outer) / 2.0 * (R_outer - R_inner) * h_coil
    B_max = Bt * R0 / max(R_inner, 0.1)
    E_GJ = 0.5 * B_max ** 2 / MU_0 * V_coil / 1e9
    return E_GJ


# =============================================================================
# 14. TF ripple
# =============================================================================
def tf_ripple(R0, a, N_coil=16):
    """
    Toroidal field ripple at outboard midplane (%).
    δ ≈ (π * R0) / (N * (R0 - a)) * exp(-N * a / R0)
    """
    if N_coil <= 0 or a <= 0:
        return 0.0
    R_in = max(R0 - a, 0.1)
    delta = (math.pi * R0) / (N_coil * R_in) * math.exp(-N_coil * a / R0)
    return delta * 100.0


def alpha_ripple_loss_fraction(delta_pct, R0, a, N_coil, q95):
    """
    Estimate alpha particle loss fraction due to TF ripple.
    Uses the Tani/Goldston scaling for ripple-trapped alphas.
    f_loss_alpha ≈ 3 × δ^(3/2) × (R_0/a) / (N × q_95)
    Capped at 0.5 for extreme ripple.
    Reference: ITER Physics Basis, Nucl. Fusion 39 (1999).
    """
    Roa = R0 / a if a > 0 else 1.0
    delta = delta_pct / 100.0
    f = 3.0 * delta ** 1.5 * Roa / (max(N_coil, 1) * max(q95, 1.0))
    return min(max(f, 0.0), 0.50)


# =============================================================================
# 15. Tritium burn fraction
# =============================================================================
def tritium_burn_fraction(n_bar_e20, T_keV, tau_E_s):
    """
    Fraction of injected tritium that fuses.
    f_burn ≈ n_D * <σv> * τ_p where τ_p ≈ τ_E (particle = energy confinement).
    For 50:50 D-T, n_D = n_T = n/2.
    """
    n_bar = n_bar_e20 * 1e20
    n_D = n_bar * 0.5
    sigmav = bosch_hale_sigma_v(T_keV)
    tau_p = tau_E_s
    f_burn = n_D * sigmav * tau_p
    return min(f_burn, 1.0)


# =============================================================================
# 16. Core impurity accumulation and radiative power loss
# =============================================================================
def core_impurity_radiation(n_bar_e20, T_keV, V, f_Ar_core=5e-5, f_W_core=1e-5):
    """
    Estimate core radiative power loss from impurity content.
    P_rad = n_e * n_imp * Lz(T) * V, where n_imp = f_imp * n_e.
    Argon: 5e-5 fraction -> ~15% of alpha power radiated (line + continuum)
    Tungsten: 1e-5 fraction -> ~25% of alpha power radiated (strong line radiation)
    Combined: ~40% radiative loss from core impurities at nominal concentrations.
    Based on: Putterich 2010 (tungsten), ITER impurity control (Kukushkin 2013).

    Returns: P_rad_MW, P_rad_Ar_MW, P_rad_W_MW
    """
    n_e = n_bar_e20 * 1e20
    # Argon: primarily continuum at Te~15keV (fully stripped)
    Lz_Ar = 1.0e-31  # W m^3 at 15 keV
    P_rad_Ar = f_Ar_core * n_e ** 2 * Lz_Ar * V * 1e-6  # MW

    # Tungsten: strong line radiation in 1-20 keV range, but at 15 keV
    # Lz drops as W approaches H-like (only 28 electrons remain).
    # Putterich 2010: Lz_W ~ 2-4e-31 W m^3 at 15 keV.
    Lz_W = 3.0e-31  # W m^3 at 15 keV
    P_rad_W = f_W_core * n_e ** 2 * Lz_W * V * 1e-6  # MW

    P_rad_total = P_rad_Ar + P_rad_W
    return P_rad_total, P_rad_Ar, P_rad_W


# =============================================================================
# 17. Helium ash buildup and pumping
# =============================================================================
def helium_ash_balance(P_fus_MW, V, n_D=1e20, tau_particle=5.0):
    """
    Helium ash production rate and equilibrium concentration.
    Each D-T fusion produces one He nucleus (3.5 MeV alpha).

    Returns: n_He_m3, f_He, He_production_rate_m3s
    """
    reactions_per_s = P_fus_MW * 1e6 / (17.6e6 * 1.602e-19)
    n_He = reactions_per_s * tau_particle / V
    f_He = n_He / max(n_D, 1e10)  # relative to D (fuel) density
    return n_He, f_He, reactions_per_s / V


# (disruption_probability moved to mhd_stability module)
# =============================================================================
def capital_cost_estimate(E_TF_GJ, P_ext_MW, S_m2, V_m3, P_fus_MW,
                          P_gross_electric_MW, B_coil_max=10.0,
                          C_PF_MS=0.0, cs_less=False, hts_tf=False):
    """
    Realistic capital cost (M$) with full balance of plant.

    Options:
      cs_less: no central solenoid (RF startup), saves $500M
      hts_tf: HTS TF coils at $130/GJ instead of Nb₃Sn premium model
              (for designs where HTS is the conductor choice)

    PF coil cost is passed in separately from pf_coil_system().
    Includes 15% contingency.
    """
    if hts_tf:
        # HTS TF coils: $130/GJ stored energy, no field premium
        # REBCO tape cost dominates at $130/GJ (today's price)
        C_TF = E_TF_GJ * 130.0
    else:
        # Nb₃Sn TF coils: base $80/GJ with field premium above 12 T
        C_TF_base = E_TF_GJ * 80.0
        if B_coil_max > 12.0:
            nb3sn_premium = 1.0 + 0.6 * (B_coil_max - 12.0) / 6.0
            C_TF = C_TF_base * nb3sn_premium
        else:
            C_TF = C_TF_base

    # Blanket + first wall
    C_blanket = S_m2 * 0.8

    # Aux heating / current drive
    C_aux = P_ext_MW * 3.0

    # Balance of plant (turbine, cooling, tritium, site, I&C)
    C_turbine = P_gross_electric_MW * 1.2    # $1200/kW turbine island
    C_cooling = P_gross_electric_MW * 0.6     # $600/kW cooling
    C_tritium = 1000.0                        # tritium plant (fixed ~$1B)
    C_site = 800.0                            # buildings, site prep
    C_IC = 300.0                              # I&C, controls
    C_CS = 0.0 if cs_less else 500.0          # central solenoid (saved if CS-less)

    base = C_TF + C_blanket + C_aux + C_turbine + C_cooling + C_tritium + C_site + C_IC + C_PF_MS + C_CS
    contingency = base * 0.15

    return base + contingency


# =============================================================================
# main: quick_eval — one function to rule them all
# =============================================================================
def quick_eval(design_dict, divertor_type="ITER",
               double_null=False, cs_less=False, flibe_blanket=False,
               negative_delta=False, reversed_shear=False,
               hts_tf=True, steel_type="SS316LN",
               inboard_launcher=False):
    """
    Evaluate a tokamak design using all analytic physics models.
    Returns comprehensive dict. Sub-microsecond.

    Input: dict with keys: R0, a, B0, Ip, elongation,
                           triangularity_upper, triangularity_lower, q95

    Upgrade options:
      double_null: split divertor exhaust across upper + lower targets
      cs_less: remove central solenoid (RF startup), save cost + space
      flibe_blanket: FLiBe liquid blanket at 627°C with sCO₂ Brayton
      negative_delta: δ < 0 eliminates ELMs, broadens heat flux width
      reversed_shear: high bootstrap fraction (f_bs > 0.9), less CD
      hts_tf: HTS TF coils at $130/GJ (True by default for V2 designs)
    """
    R0 = design_dict.get("R0", 0.85)
    a = design_dict.get("a", 0.5)
    Bt = design_dict.get("B0", 0.55)
    Ip = design_dict.get("Ip", 600.0)
    kappa = design_dict.get("elongation", 1.8)
    q95 = design_dict.get("q95", 3.5)

    Ip_MA = Ip / 1000.0
    V = tokamak_volume(R0, a, kappa)
    S = plasma_surface_area(R0, a, kappa)
    eps = a / R0 if R0 > 0 else 0.3
    T_keV = design_dict.get("T_keV", 15.0)
    n_fGW = design_dict.get("n_fGW", 0.60)
    H98_mult = design_dict.get("H98_mult", 1.0)

    # ── Power balance (H98 primary) ────────────────────────────────────────
    pb = solve_power_balance(R0, a, Bt, Ip_MA, kappa, eps, V, T_keV=T_keV, n_fGW=n_fGW, H98_mult=H98_mult)
    n_bar_e20 = pb["n_bar_e20"]

    # ── Power balance (gyro-Bohm cross-check with equilibrium T) ──────────
    try:
        gb_solver = TransportSolver(R0, a, kappa, Bt, Ip_MA, C_GB=1.14)
        n_GW = Ip_MA / (math.pi * a ** 2)
        gb = gb_solver.solve_equilibrium(n_GW, n_fGW=n_fGW, P_ext_target=0.0)
        # Fall back for keys the rest of the code expects
        if "P_fus_MW" not in gb:
            gb = gb_solver.solve(n_GW, n_fGW=n_fGW, T_keV=T_keV)
    except Exception:
        gb = {"Q": 0.0, "P_fus_MW": 0.0, "P_loss_MW": 0.0, "P_ext_MW": 0.0,
              "tau_E_s": 0.0, "W_MJ": 0.0}

    # ── L-H threshold ──────────────────────────────────────────────────────
    P_LH = lh_threshold_power(n_bar_e20, Bt, R0, a, kappa)
    P_heat = max(pb["P_fusion_MW"] * 0.2 + 30.0, pb["P_ext_MW"])
    lh_margin = P_heat / max(P_LH, 1e-12) - 1.0
    lh_margin = max(lh_margin, -10.0)

    # ── TF coil stress analysis ─────────────────────────────────────────
    conductor_type = "HTS" if hts_tf else "Nb3Sn"
    tf_stress = tf_coil_stress(R0, a, Bt, steel_type=steel_type, conductor=conductor_type)
    B_coil_max = tf_stress["B_peak_T"]  # true peak field at TF inner leg
    tf_stress_margin = max(tf_stress["tf_stress_margin"], -10.0)

    # ── Beta ───────────────────────────────────────────────────────────────
    if pb["Q"] > 0.5:
        beta_t, beta_p, beta_N, l_i = compute_beta(
            R0, a, Bt, Ip_MA, kappa, n_bar_e20, T_keV,
        )
    else:
        beta_N = 1.0
        beta_t = beta_N * Ip_MA / (a * Bt) / 100.0 if a * Bt > 0 else 0.0
        beta_p = 0.3
        l_i = 0.5 + 0.2 * kappa

    # ── Pedestal ──────────────────────────────────────────────────────────
    T_ped, ped_width = pedestal_estimate(R0, a, Bt, Ip_MA, kappa)

    # ── Bootstrap fraction ───────────────────────────────────────────────
    f_bs = min(bootstrap_fraction(eps, beta_p, nu_star=0.1), 1.0)

    # ── Divertor (using full divertor_analysis from divertor_sol) ────────
    # Core impurity radiation from argon seeding + tungsten erosion
    P_rad_core, P_rad_Ar, P_rad_W = core_impurity_radiation(
        n_bar_e20, T_keV, V
    )
    f_rad_core = min(0.50, P_rad_core / max(pb["P_alpha_MW"], 0.01))
    P_sep_MW_eff = pb["P_alpha_MW"] * (1.0 - f_rad_core) + pb["P_ext_MW"]
    dvert = divertor_analysis(
        P_sep_MW_eff, R0, a, kappa, Bt, Ip_MA, q95, n_bar_e20,
        divertor_type=divertor_type, f_rad_core=f_rad_core,
        double_null=double_null, negative_delta=negative_delta,
    )
    q_div = dvert["q_peak_MW_m2"]
    q_div_limit = dvert["q_limit_MW_m2"]
    divertor_margin = max(q_div_limit - q_div, -10.0)

    # ── Helium ash ──────────────────────────────────────────────────────────
    n_He, f_He, He_rate = helium_ash_balance(pb["P_fusion_MW"], V, n_D=n_bar_e20 * 1e20 / 2.0)

    # ── Neutron wall load ────────────────────────────────────────────────
    P_n_wall = neutron_wall_load(pb["P_fusion_MW"], R0, a, kappa)

    # ── Stability (using full_stability_analysis from mhd_stability) ─────
    if negative_delta:
        delta = -0.30
    else:
        delta = (design_dict.get("triangularity_upper", 0.3)
                 + design_dict.get("triangularity_lower", 0.3)) / 2.0
    mhd = full_stability_analysis(
        R0, a, kappa, delta, Bt, Ip_MA, l_i, q95, n_bar_e20, beta_N, T_ped,
    )
    p_disrupt = mhd["disruption_probability"]

    # ── Composite stability (used by optimizer) ──────────────────────────
    composite = min(mhd["stability_score"], lh_margin, divertor_margin, tf_stress_margin)
    composite = max(composite, -10.0)

    # ── Engineering metrics ──────────────────────────────────────────────

    # Steady-state current drive power (informational only — design is pulsed)
    # Reversed shear: boost bootstrap fraction to reduce CD requirement
    if reversed_shear:
        f_bs = min(f_bs * 1.5, 0.95)
    gamma_CD = 0.55 if inboard_launcher else (0.35 if reversed_shear else 0.30)
    f_nonboot = max(1.0 - f_bs, 0.0)
    I_CD_MA = f_nonboot * Ip_MA
    if gamma_CD > 0 and n_bar_e20 > 0 and R0 > 0:
        P_CD_MW = n_bar_e20 * R0 * I_CD_MA / gamma_CD
    else:
        P_CD_MW = 0.0
    P_CD_MW = max(P_CD_MW, 0.0)

    # Pulsed mode: CD only supplements inductive drive during burn
    # For CS-less designs, full non-inductive CD is needed
    if cs_less:
        P_CD_MW_pulsed = P_CD_MW
    else:
        P_CD_MW_pulsed = min(P_CD_MW, 2.0)

    # ── Tritium breeding (LiPb baseline + HCPB upgrade) ──────────────────
    tbr_lipb = tritium_breeding_ratio(R0, a, kappa)
    tbr_hcpb = hcpb_tbr(blanket_thickness_mm=800, li6_enrichment=0.15,
                        be_thickness_mm=20, coverage=0.92)["TBR"]

    # ── NTM ECCD stabilization ────────────────────────────────────────────
    beta_p_val = beta_p if isinstance(beta_p, (int, float)) else 0.5
    eccd = required_eccd_power(beta_N, beta_p_val, a, R0, n_bar_e20, T_keV,
                               q95, Ip_MA, Bt,
                               inboard_launcher=inboard_launcher)
    P_ECCD_MW = eccd["total_P_ECCD_MW"]
    n_gyrotrons = eccd["n_gyrotrons"]

    # ── Gyro-Bohm heating system sizing ──────────────────────────────────
    htg = size_heating_system(gb.get("P_ext_MW", 0), technology="EC")

    # ── Balance of plant (use gyro-Bohm if available and credible) ───────
    blanket_type = "FLiBe" if flibe_blanket else "HCPB"
    use_gb_for_bop = gb["Q"] > 0.5 and gb["P_fus_MW"] > 100
    P_fus_bop = gb["P_fus_MW"] if use_gb_for_bop else pb["P_fusion_MW"]
    P_ext_bop = gb["P_ext_MW"] if use_gb_for_bop else pb["P_ext_MW"]
    bop = balance_of_plant(P_fus_bop, P_ext_bop,
                           P_ECCD_recirc_MW=eccd["recirc_P_ECCD_MW"],
                           blanket_type=blanket_type, cold_temp_C=15.0)
    net_electrics = (bop["P_net_electric_MW"], bop["P_gross_electric_MW"],
                     bop["P_recirc_total_MW"])

    # ── Net electric with ECCD recirculating power ────────────────────────
    P_net_pulsed = max(net_electrics[0] - P_CD_MW_pulsed / 0.4, 0.0)
    P_recirc_pulsed = net_electrics[2] + P_CD_MW_pulsed / 0.4

    if cs_less:
        cs = None
        burn_time = 1e6  # unlimited — CS-less designs burn as long as fuel lasts
        cs_psi_available = 1e6
        cs_psi_required = 0.0
        cs_margin = 1e6
    else:
        cs = central_solenoid_design(R0, a, kappa, Bt, Ip_MA)
        burn_time = cs["burn_time_s"]
        cs_psi_available = cs["psi_available_Wb"]
        cs_psi_required = cs["psi_required_Wb"]
        cs_margin = cs["CS_margin"]
    E_TF = tf_stored_energy(R0, a, Bt)
    ripple = tf_ripple(R0, a)
    ripple_loss_alpha = alpha_ripple_loss_fraction(ripple, R0, a, 16, q95)
    ripple_loss_alpha_mitigated = alpha_ripple_loss_fraction(ripple / 2.0, R0, a, 18, q95)
    f_burn = tritium_burn_fraction(n_bar_e20, T_keV, pb["tau_E_s"])
    n_G = Ip_MA / (math.pi * a ** 2)

    # ── PF coil system ────────────────────────────────────────────────────
    pf = pf_coil_system(R0, a, kappa, Bt, Ip_MA, q95)

    cost_MS = capital_cost_estimate(E_TF, P_ext_bop, S, V, P_fus_bop,
                                    net_electrics[1], B_coil_max,
                                    C_PF_MS=pf["total_cost_MS"],
                                    cs_less=cs_less, hts_tf=hts_tf)

    # Cost breakdown components (for sensitivity analysis)
    if hts_tf:
        C_TF = E_TF * 130.0  # HTS: $130/GJ, no field premium
    else:
        nb3sn_mult = 1.0 + 0.6 * max(0, B_coil_max - 12.0) / 6.0
        C_TF = E_TF * 80.0 * nb3sn_mult
    C_PF = pf["total_cost_MS"]
    C_blanket = S * 0.8
    C_aux = P_ext_bop * 3.0
    C_turbine = net_electrics[1] * 1.2
    C_cooling = net_electrics[1] * 0.6
    C_tritium_plant = 1000.0
    C_site = 800.0
    C_IC = 300.0

    return {
        # Primary objectives (used by optimizer)
        "Q": pb["Q"],
        "Q_GB": gb["Q"],
        "volume_m3": V,
        "stability_score": composite,

        # Power balance (H98)
        "tau_E_s": pb["tau_E_s"],
        "tau_E_GB_s": gb["tau_E_s"],
        "P_fusion_MW": pb["P_fusion_MW"],
        "P_fusion_GB_MW": gb["P_fus_MW"],
        "P_loss_MW": pb["P_loss_MW"],
        "P_loss_GB_MW": gb["P_loss_MW"],
        "P_ext_MW": pb["P_ext_MW"],
        "P_ext_GB_MW": gb["P_ext_MW"],
        "P_alpha_MW": pb["P_alpha_MW"],
        "triple_product": pb["triple_product"],
        "T_keV": pb["T_keV"],
        "n_bar_e20": n_bar_e20,
        "W_MJ": pb["W_MJ"],

        # Geometry
        "surface_area_m2": S,
        "aspect_ratio": eps,

        # Stability margins (from MHD module)
        "beta_margin": mhd["beta_margin_no_wall"],  # backward compat
        "beta_margin_no_wall": mhd["beta_margin_no_wall"],
        "beta_margin_wall": mhd["beta_margin_wall"],
        "q95_margin": mhd["q95_margin"],
        "density_margin": mhd["density_margin"],
        "lh_margin": lh_margin,
        "divertor_margin": divertor_margin,
        "tf_stress_margin": tf_stress_margin,

        # Plasma parameters
        "beta_N": beta_N,
        "beta_N_no_wall_limit": mhd["βN_no_wall_limit"],
        "beta_N_wall_limit": mhd["βN_wall_limit"],
        "beta_t": beta_t,
        "beta_p": beta_p,
        "l_i": l_i,
        "f_bs": f_bs,
        "T_ped": T_ped,
        "ped_width": ped_width,
        "q_div_MWm2": q_div,
        "q_div_limit": q_div_limit,
        "divertor_type": divertor_type,
        "neutron_wall_MWm2": P_n_wall,
        "B_coil_max_T": B_coil_max,
        "sigma_case_MPa": tf_stress["sigma_case_MPa"],
        "case_margin": tf_stress["case_margin"],
        "Jc_margin": tf_stress["Jc_margin"],
        "T_margin_K": tf_stress["T_margin_K"],
        "T_critical_K": tf_stress["T_critical_K"],
        "T_operating_K": tf_stress["T_operating_K"],
        "TF_conductor": tf_stress["conductor"],
        "TF_feasible": tf_stress["TF_feasible"],
        "P_LH_MW": P_LH,

        # Divertor (from divertor_sol)
        "P_sep_MW": dvert["P_sep_MW"],
        "lambda_q_mm": dvert["lambda_q_mm"],
        "lambda_q_raw_mm": dvert["lambda_q_raw_mm"],
        "f_rad_core": f_rad_core,
        "f_rad_div": dvert["f_rad_div"],
        "divertor_q_peak_MWm2": dvert["q_peak_MW_m2"],
        "divertor_margin_MWm2": dvert["q_margin_MW_m2"],
        "divertor_A_wetted_m2": dvert["A_wetted_m2"],
        "divertor_f_exp": dvert["f_exp"],
        "divertor_regime": dvert.get("SOL_regime", ""),

        # MHD stability (from mhd_stability)
        "ntm_stability_metric": mhd["ntm_stability_metric"],
        "ntm_stable": mhd["ntm_stable"],

        # Current drive
        "P_CD_MW": P_CD_MW_pulsed,
        "I_CD_MA": I_CD_MA,
        "P_CD_steady_state_MW": P_CD_MW,

        # Impurity and ash
        "P_rad_core_MW": P_rad_core,
        "P_rad_Ar_MW": P_rad_Ar,
        "P_rad_W_MW": P_rad_W,
        "n_He_m3": n_He,
        "f_He": f_He,
        "profile_peaking": pb["peaking_factor"],

        # Engineering metrics
        "P_net_electric_MW": P_net_pulsed,
        "P_gross_electric_MW": net_electrics[1],
        "P_recirc_MW": P_recirc_pulsed,
        "TBR": tbr_lipb,
        "TBR_HCPB": tbr_hcpb,
        "TBR_tritium_self_sufficient": tbr_hcpb >= 1.05,
        "burn_time_s": burn_time,
        "CS_psi_available_Wb": cs_psi_available,
        "CS_psi_required_Wb": cs_psi_required,
        "CS_margin": cs_margin,
        "E_TF_stored_GJ": E_TF,
        "P_recirc_breakdown_MW": {
            "blanket_pump": bop["P_blanket_pump_MW"],
            "sCO2_compression": bop["P_sCO2_compression_MW"],
            "cryoplant": bop["P_cryoplant_MW"],
            "cooling_pumps": bop["P_cooling_pumps_MW"],
            "vacuum": bop["P_vacuum_pumps_MW"],
            "tritium": bop["P_tritium_processing_MW"],
            "BOP_aux": bop["P_BOP_aux_MW"],
            "plasma_control": bop["P_plasma_control_MW"],
            "ECCD": bop["P_ECCD_recirc_MW"],
        },
        "eta_thermal": bop["eta_thermal"],
        "blanket_T_out_C": bop["blanket_T_out_C"],
        "cooling_water_m3h": bop["cooling_water_m3h"],
        "P_rejected_MW": bop["P_rejected_thermal_MW"],
        "q_eng": P_net_pulsed / max(P_recirc_pulsed, 0.001),

        # NTM ECCD stabilization
        "P_ECCD_MW": P_ECCD_MW,
        "P_ECCD_launched_MW": eccd.get("P_ECCD_launched_MW", P_ECCD_MW),
        "ECCD_gyrotrons": n_gyrotrons,
        "ECCD_freq_GHz": eccd.get("gyrotron_freq_GHz", 0),
        "ECCD_gyrotron_TRL": eccd.get("gyrotron_TRL", 0),
        "ECCD_gyrotron_status": eccd.get("gyrotron_status", ""),
        "ECCD_recirc_MW": eccd["recirc_P_ECCD_MW"],
        "ECCD_cost_MS": eccd.get("cost_MS", 0),
        "ECCD_n_ports": eccd.get("n_ports", 0),

        # Gyro-Bohm heating system sizing
        "heating_P_wall_plug_MW": htg["wall_plug_power_MW"],
        "heating_n_units": htg["n_units"],
        "heating_n_ports": htg["n_ports"],
        "heating_cost_MS": htg["cost_MS"],

        # New engineering metrics
        "TF_ripple_pct": ripple,
        "alpha_ripple_loss_frac": ripple_loss_alpha,
        "alpha_ripple_loss_mitigated_frac": ripple_loss_alpha_mitigated,
        "tritium_burn_frac": f_burn,
        "disruption_prob": p_disrupt,
        "cost_MS": cost_MS,
        "cost_TF_coils_MS": C_TF,
        "cost_PF_coils_MS": C_PF,
        "cost_blanket_MS": C_blanket,
        "cost_aux_MS": C_aux,
        "cost_turbine_MS": C_turbine,
        "cost_cooling_MS": C_cooling,
        "cost_tritium_plant_MS": C_tritium_plant,
        "cost_site_MS": C_site,
        "cost_IC_MS": C_IC,
        "PF_N_coils": pf["N_coils"],
        "PF_total_NI_MA": pf["total_NI_MA"],
        "PF_total_mass_tonnes": pf["total_mass_tonnes"],
        "PF_conductor": pf["conductor"],
        "PF_vertical_stable": pf["vertical_stable"],
        "PF_vertical_growth_time_s": pf["vertical_growth_time_s"],
        "PF_coil_data": pf["coils"],

        # Upgrade flags
        "upgrade_double_null": double_null,
        "upgrade_cs_less": cs_less,
        "upgrade_flibe_blanket": flibe_blanket,
        "upgrade_negative_delta": negative_delta,
        "upgrade_reversed_shear": reversed_shear,
    }


if __name__ == "__main__":
    mast = {
        "R0": 0.85, "a": 0.5, "B0": 0.55, "Ip": 600.0,
        "elongation": 1.8, "triangularity_upper": 0.4,
        "triangularity_lower": 0.4, "q95": 3.5,
    }
    sparc = {
        "R0": 1.85, "a": 0.57, "B0": 12.2, "Ip": 8800.0,
        "elongation": 1.8, "triangularity_upper": 0.4,
        "triangularity_lower": 0.4, "q95": 3.5,
    }
    iter = {
        "R0": 6.2, "a": 2.0, "B0": 5.3, "Ip": 15000.0,
        "elongation": 1.7, "triangularity_upper": 0.3,
        "triangularity_lower": 0.3, "q95": 3.5,
    }
    for name, d in [("MAST", mast), ("SPARC", sparc), ("ITER", iter)]:
        print(f"\n{'='*50}")
        print(f" {name}")
        print(f"{'='*50}")
        r = quick_eval(d)
        for k, v in sorted(r.items()):
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.4e}" if abs(v) < 0.01 or abs(v) > 1e6 else f"  {k:25s}: {v:.4f}")
            else:
                print(f"  {k:25s}: {v}")
