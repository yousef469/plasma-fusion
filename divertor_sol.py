"""
Scrape-off layer (SOL) and divertor model for tokamak design.

Self-consistent two-point model with detachment physics.

References:
  - Stangeby, "The Plasma Boundary of Magnetic Fusion Devices" (2000)
  - Eich et al., Nucl. Fusion 53, 093031 (2013) — λ_q scaling
  - Pitcher & Stangeby, Plasma Phys. Control. Fusion 39, 779 (1997)
  - ITER Physics Basis, Nucl. Fusion 39, 2137 (1999) — Ch. 4 (divertor)
  - Kuang et al., J. Plasma Phys. 86, 865860505 (2020) — SPARC divertor
"""

import math
import numpy as np


# =============================================================================
# Physical constants
# =============================================================================
E_CHARGE = 1.602e-19
M_I = 2.5 * 1.66054e-27    # average D-T ion mass (kg)
M_E = 9.109e-31             # electron mass (kg)
MU_0 = 4 * math.pi * 1e-7


# =============================================================================
# Divertor technology types
# =============================================================================
DIVERTOR_TYPES = {
    "ITER": {
        "q_limit": 10.0,        # MW/m² (ITER W monoblock limit)
        "f_rad_div": 0.65,       # divertor radiation fraction
        "f_exp_mult": 1.0,      # flux expansion relative to ITER baseline
        "theta_deg": 2.0,       # target inclination angle (degrees)
        "desc": "ITER-like W monoblocks, CuCrZr cooling"
    },
    "X_DIVERTOR": {
        "q_limit": 15.0,
        "f_rad_div": 0.65,
        "f_exp_mult": 2.5,
        "theta_deg": 2.0,
        "desc": "X-divertor geometry, 2.5× flux expansion"
    },
    "SNOWFLAKE": {
        "q_limit": 20.0,
        "f_rad_div": 0.65,
        "f_exp_mult": 4.0,
        "theta_deg": 2.0,
        "desc": "Snowflake divertor, 4× flux expansion"
    },
    "LIQUID_LI": {
        "q_limit": 35.0,
        "f_rad_div": 0.65,
        "f_exp_mult": 1.5,
        "theta_deg": 2.0,
        "desc": "Liquid lithium CPS, evaporative cooling"
    },
    "LIQUID_LIPB": {
        "q_limit": 60.0,
        "f_rad_div": 0.65,
        "f_exp_mult": 1.5,
        "theta_deg": 2.0,
        "desc": "Flowing LiPb divertor, convective cooling"
    },
    "VAPOR_BOX": {
        "q_limit": 10.0,
        "f_rad_div": 0.85,
        "f_exp_mult": 1.0,
        "theta_deg": 2.0,
        "desc": "Vapor box divertor, very high rad fraction (85-90%)"
    },
}


# =============================================================================
# 1. Heat flux width scaling (Eich 2013)
# =============================================================================
def heat_flux_width(Bp, P_sep, R0):
    """
    SOL heat flux width λ_q (mm) from Eich 2013 multi-machine database.
    
    Eich 2013 (Nucl. Fusion 53, 093031):
      λ_q = 0.8 * B_pol^(-0.7) * P_sep^0.1 * R0^(-0.5)
    
    where B_pol is the poloidal field at the outer midplane (in T),
    P_sep in MW, R0 in m.
    
    Returns λ_q in mm.
    """
    if Bp <= 0:
        return 1.0
    # Avoid clamping — let the physics determine the value
    lambda_q_mm = 0.8 * Bp ** (-0.7) * max(P_sep, 0.1) ** 0.1 * R0 ** (-0.5)
    return max(lambda_q_mm, 0.1)  # soft floor at 0.1 mm


# =============================================================================
# 2. Poloidal field at the outer midplane
# =============================================================================
def poloidal_field(R0, a, kappa, Ip_MA):
    """
    Poloidal magnetic field at the outer midplane separatrix.
    
    B_pol = μ₀ I_p / (2π a √(κ/2))
    
    where the effective minor radius is a * sqrt(κ/2) for
    elongated plasmas in the outboard midplane.
    """
    Ip_A = Ip_MA * 1e6
    b_pol = MU_0 * Ip_A / (2 * math.pi * a * math.sqrt(kappa / 2.0))
    return max(b_pol, 0.01)


# =============================================================================
# 3. Connection length
# =============================================================================
def connection_length(R0, q95):
    """
    Parallel connection length from outer midplane to divertor target.
    L_∥ ≈ π * R0 * q95  (for the outboard leg)
    """
    return math.pi * R0 * q95


# =============================================================================
# 4. Flux expansion at the target
# =============================================================================
def flux_expansion(Bt, Bp, R_div, R0, f_exp_mult=1.0):
    """
    Flux expansion factor at the divertor target.
    
    f_exp = (B_t / B_p) * (R_target / R_midplane) * f_exp_mult
    
    Higher flux expansion reduces peak heat flux by spreading
    the power over a larger wetted area.
    """
    if Bp <= 0:
        return 10.0
    f_exp = (Bt / Bp) * (R_div / R0) * f_exp_mult
    return min(f_exp, 100.0)


# =============================================================================
# 5. Sound speed (for two-point model)
# =============================================================================
def sound_speed(T_eV):
    """Ion sound speed c_s = sqrt(2eT/m_i) [m/s]."""
    return math.sqrt(2 * E_CHARGE * max(T_eV, 0.1) / M_I)


# =============================================================================
# 6. Two-point model: upstream to target mapping
# =============================================================================
def two_point_model(n_u, T_u, L_par, P_sep, A_wetted):
    """
    Simple two-point model for divertor target conditions (informational).
    
    Returns: n_t, T_t, q_target (target density, temperature, peak flux)
    """
    if P_sep <= 0 or A_wetted <= 0:
        return 1e19, 1.0, 0.1

    gamma = 7.0  # sheath heat transmission coefficient
    q_par = P_sep * 1e6 / max(A_wetted, 0.01)  # W/m² (parallel heat flux density)

    # Target temperature from sheath heat transmission
    n_u_T_u = max(n_u, 1e18) * max(T_u, 0.1)
    if n_u_T_u > 0:
        T_t = ((q_par / (gamma * n_u_T_u)) ** 2 * M_I / (2 * E_CHARGE))
    else:
        T_t = 0.1

    T_t = max(T_t, 0.1)
    T_t = min(T_t, T_u)

    # Target density from pressure balance
    n_t = n_u * T_u / max(T_t, 0.1)

    # Target heat flux
    c_s = math.sqrt(2 * E_CHARGE * T_t / M_I)
    q_target = gamma * n_t * T_t * E_CHARGE * c_s  # W/m²

    return n_t, T_t, q_target / 1e6


# =============================================================================
# 7. Detachment criterion
# =============================================================================
def detachment_fraction(n_u, P_sep, R0):
    """
    Divertor detachment fraction based on upstream density.
    
    Detachment occurs when the upstream density exceeds a threshold:
      n_detach ≈ 3e19 * P_sep^0.5 / R0^0.5
    
    Beyond this threshold, the divertor radiates a significant fraction
    of the incident power.
    
    Reference: Stangeby 2000, ITER PB Ch. 4
    
    Returns: f_rad_div (0 to 1), fraction of P_sep radiated in divertor
    """
    if P_sep <= 0:
        return 0.65

    n_threshold = 3e19 * math.sqrt(max(P_sep, 0.1) / max(R0, 0.1))

    if n_u <= 0:
        return 0.0

    # Ratio of upstream density to detachment threshold
    r = n_u / max(n_threshold, 1e19)

    # Radiated fraction increases with detachment
    # Below threshold: low radiation (attached)
    # Above threshold: increasing radiation (detached)
    if r < 0.5:
        f_rad = 0.2  # weakly attached
    elif r < 1.0:
        f_rad = 0.2 + 0.45 * (r - 0.5) / 0.5  # transition
    else:
        f_rad = 0.65 + 0.2 * min(r - 1.0, 1.0)  # detached

    return min(f_rad, 0.85)


# =============================================================================
# 8. Main analysis function
# =============================================================================
def divertor_analysis(P_sep_MW, R0, a, kappa, Bt, Ip_MA, q95,
                      n_bar_e20, divertor_type="SNOWFLAKE", f_rad_core=0.20):
    """
    Complete SOL/divertor analysis for the reference design.
    
    Returns dict with all divertor parameters.
    """
    # Divertor technology parameters
    div = DIVERTOR_TYPES.get(divertor_type, DIVERTOR_TYPES["ITER"])
    f_exp_mult = div["f_exp_mult"]
    q_limit = div["q_limit"]
    f_rad_div = div["f_rad_div"]
    theta_deg = div["theta_deg"]

    # Poloidal field at outer midplane
    Bp = poloidal_field(R0, a, kappa, Ip_MA)

    # Connection length
    L_par = connection_length(R0, q95)

    # Heat flux width
    lambda_q_mm = heat_flux_width(Bp, P_sep_MW, R0)

    # Broadening factor for detached divertor
    # In a detached divertor (f_rad_div > 0.5), the heat flux is spread
    # by radiation and neutral friction, giving 2-5× broadening.
    # Reference: Kuang 2020 (SPARC), Pitts 2019 (ITER divertor)
    if f_rad_div > 0.5:
        f_broaden = 2.0 + 3.0 * (f_rad_div - 0.5) / 0.35  # 2-5× broadening
    else:
        f_broaden = 1.0 + 2.0 * f_rad_div / 0.5  # 1-3× during transition

    lambda_q_eff_mm = lambda_q_mm * min(f_broaden, 5.0)
    lambda_q = lambda_q_eff_mm / 1000.0

    # Power into SOL
    P_sep = max(P_sep_MW, 0.1)

    # Flux expansion
    R_div = R0 + a  # divertor radius (approximate)
    f_exp = flux_expansion(Bt, Bp, R_div, R0, f_exp_mult)

    # Target inclination
    theta_rad = theta_deg * math.pi / 180.0

    # Wetted area on target
    A_wetted = (2.0 * math.pi * R_div * lambda_q * f_exp
                / max(math.sin(theta_rad), 0.01))

    # Upstream SOL parameters (reactor-relevant)
    n_GW = Ip_MA / (math.pi * a ** 2) * 1e20  # m⁻³ (Greenwald)
    # For a reactor, upstream SOL density is a fraction of core density,
    # but increases with core density. In H-mode, n_u ≈ 0.1-0.5 * n_core.
    # Higher core density → higher SOL density → more detachment.
    n_u = max(0.25 * n_bar_e20 * 1e20, 1e19)
    T_u = 150.0  # upstream SOL temperature (eV, reactor-typical)
    # For high-density H-mode, SOL temperature is 100-200 eV

    # Detachment state
    # Use technology-specified radiated fraction as the design target
    # (detachment is achieved through impurity seeding and geometry design)
    f_rad_div = div["f_rad_div"]
    n_threshold = 8e19 * (max(P_sep, 0.1) / 100.0) ** 0.3 * (R0 / 6.0) ** -0.3

    # Target conditions from two-point model (informational)
    n_t, T_t, q_peak_from_sheath = two_point_model(n_u, T_u, L_par, P_sep, A_wetted)

    # Peak heat flux using the geometric approach (primary)
    # q_target = P_sep * (1 - f_rad_div) / A_wetted * (peak/avg ratio)
    q_avg = max(P_sep * (1.0 - f_rad_div) / max(A_wetted, 0.01), 0.01)
    q_peak = q_avg * 2.0  # peak-to-average ratio (typical for 2D profile)

    # SOL regime (based on divertor radiated fraction)
    # Thresholds from SOLPS database: Pitts 2019, Kukushkin 2013
    if f_rad_div < 0.3:
        regime = "sheath-limited"
    elif f_rad_div < 0.55:
        regime = "conduction-limited (partially detached)"
    else:
        regime = "detached"

    # Margin
    q_margin = q_limit - q_peak

    return {
        "lambda_q_mm": lambda_q_eff_mm,
        "lambda_q_raw_mm": lambda_q_mm,
        "P_sep_MW": P_sep,
        "f_exp": f_exp,
        "f_rad_div": f_rad_div,
        "Bp_T": Bp,
        "L_par_m": L_par,
        "A_wetted_m2": A_wetted,
        "q_peak_MW_m2": q_peak,
        "q_limit_MW_m2": q_limit,
        "q_margin_MW_m2": q_margin,
        "n_u_m3": n_u,
        "T_u_eV": T_u,
        "n_t_m3": n_t,
        "T_t_eV": T_t,
        "SOL_regime": regime,
        "n_threshold_m3": n_threshold,
        "divertor_type": divertor_type,
    }


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    # Reference design
    R0, a, kappa = 12.08, 0.96, 2.71
    Bt, Ip_MA, q95 = 11.7, 10.6, 3.1
    n_bar_e20 = 2.20
    P_alpha_MW = 1095.0
    P_ext_MW = 0.0
    f_rad_core = 0.21

    for dtype in ["ITER", "X_DIVERTOR", "SNOWFLAKE", "LIQUID_LI"]:
        P_sep = P_alpha_MW * (1.0 - f_rad_core) + P_ext_MW
        result = divertor_analysis(P_sep, R0, a, kappa, Bt, Ip_MA, q95,
                                   n_bar_e20, divertor_type=dtype,
                                   f_rad_core=f_rad_core)

        print(f"\n{'=' * 50}")
        print(f" DIVERTOR TYPE: {dtype}")
        print(f"{'=' * 50}")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.4f}" if abs(v) < 1000 else f"  {k:20s}: {v:.4e}")
            else:
                print(f"  {k:20s}: {v}")
