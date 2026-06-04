"""
Neoclassical tearing mode stabilization via ECCD.

Computes:
  - Rational surface locations (q = 3/2, q = 2)
  - EC current drive efficiency at each surface
  - Required driven current to stabilize NTM islands
  - Gyrotron count and power requirement

Physics:
  - NTM stability criterion (La Haye 2006, Phys. Plasmas 13, 055501):
    Δ' + Δ_bs + Δ_pol + Δ_CD = 0
  - Bootstrap drive: I_bs_drive ∝ β_p / w
  - ECCD stabilization: I_ECCD / I_p ≈ 0.3 × w / a × β_p / 2
  - EC current drive efficiency (Fisch-Boozer):
    γ_CD = n_e20 × R0 × I_CD_MW / P_CD [10^20 A/W/m²]

References:
  - La Haye 2006, Phys. Plasmas 13, 055501
  - Sweeney 2020, J. Plasma Phys. 86, 865860112
  - Poli 2018, Nucl. Fusion 58, 016009
  - ITER Physics Basis Ch. 6 (ECCD)
"""

import math

E_CHARGE = 1.602e-19


def rational_surface_radius(q_surface, q0, q95, a):
    """
    Estimate radius of a rational q-surface in a monotonic q-profile.
    Simple model: q(r) = q0 + (q95 - q0) × (r/a)²
    """
    if q_surface <= q0 or q_surface >= q95:
        return None
    r_over_a = math.sqrt((q_surface - q0) / max(q95 - q0, 0.01))
    if r_over_a > 1.0:
        return None
    return r_over_a * a, r_over_a


def eccd_efficiency(T_e_keV, n_e20, R0, harmonic=2, Bt=0.0):
    """
    EC current drive efficiency (Fisch-Boozer).
    γ_CD = n_e20 × R0 × I_CD / P_CD [10^20 A/W/m²]

    For 170 GHz, 2nd harmonic X-mode (ITER baseline):
      η ≈ 6.5 × T_e / (R0 × n_e20 × 20)  (simplified)
    Returns γ_CD.
    """
    if T_e_keV <= 0 or n_e20 <= 0:
        return 0.01

    # Simplified efficiency from ITER Physics Basis
    # γ_CD ≈ 0.04-0.3 for typical reactor conditions
    # Higher T_e → higher efficiency, higher n_e → lower current per watt
    gamma_cd = 0.04 + 0.016 * T_e_keV - 0.003 * n_e20
    gamma_cd = max(gamma_cd, 0.01)
    gamma_cd = min(gamma_cd, 0.30)
    return gamma_cd


def required_eccd_power(beta_N, beta_p, a, R0, n_e20, T_e_keV,
                        q95, Ip_MA, Bt):
    """
    Compute total ECCD power needed to stabilize NTMs.

    Stabilizes both q=3/2 and q=2 surfaces.
    Required driven current per surface:
      I_ECCD / I_p = 0.3 × (w/a) × (β_p/2)
    where w/a ≈ 0.05 is typical seed island width.

    Returns dict with power, gyrotron count, cost.
    """
    # Rational surfaces
    q0 = 0.8  # typical central safety factor
    results = []
    total_P_ECCD = 0.0

    for q_surf in [1.5, 2.0]:  # q = 3/2 and q = 2
        surf = rational_surface_radius(q_surf, q0, q95, a)
        if surf is None:
            continue
        r_surf, rho_surf = surf

        # Local temperature at rational surface (parabolic profile)
        alpha_T = 0.8
        T_local = T_e_keV * (1 + alpha_T) * (1 - rho_surf ** 2) ** alpha_T
        T_local = max(T_local, 1.0)

        # EC current drive efficiency at this location
        gamma_cd = eccd_efficiency(T_local, n_e20, R0, Bt=Bt)
        gamma_cd = max(gamma_cd, 0.02)

        # Required driven current (La Haye 2006 criterion)
        w_over_a = 0.05  # typical seed island width
        I_ECDD_per_Ip = 0.3 * w_over_a * max(beta_p, 0.1) / 2.0
        I_ECCD_kA = I_ECDD_per_Ip * Ip_MA * 1000.0

        # Required power
        if gamma_cd > 0 and n_e20 > 0 and R0 > 0:
            P_ECCD_surf = n_e20 * R0 * (I_ECCD_kA / 1000.0) / gamma_cd
        else:
            P_ECCD_surf = 0.0

        P_ECCD_surf = max(P_ECCD_surf, 0.0)
        total_P_ECCD += P_ECCD_surf

        results.append({
            "q_surface": q_surf,
            "r_surface_m": r_surf,
            "rho_surface": rho_surf,
            "T_local_keV": T_local,
            "gamma_CD": gamma_cd,
            "I_ECCD_kA": I_ECCD_kA,
            "P_ECCD_MW": P_ECCD_surf,
        })

    # Gyrotron count
    P_GYROTRON = 1.0  # MW per gyrotron (170 GHz, ITER-grade)
    n_gyrotrons = math.ceil(total_P_ECCD / P_GYROTRON)

    # Cost (Thumm 2020: ~$4M/MW for EC systems)
    cost_MS = total_P_ECCD * 4.0

    # Port requirement (8 gyrotrons per port)
    n_ports = max(1, math.ceil(n_gyrotrons / 8.0))
    port_area = n_ports * 0.8 * 1.2  # m²

    return {
        "surfaces": results,
        "total_P_ECCD_MW": total_P_ECCD,
        "n_gyrotrons": n_gyrotrons,
        "n_ports": n_ports,
        "port_area_m2": port_area,
        "cost_MS": cost_MS,
        "recirc_P_ECCD_MW": total_P_ECCD / 0.50,  # wall-plug at 50% gyrotron eff
    }


if __name__ == "__main__":
    # Reference design parameters
    R0, a = 12.08, 0.96
    Bt, Ip_MA = 11.7, 10.6
    q95 = 3.1
    n_e20 = 2.20
    T_e_keV = 15.0
    beta_N, beta_p = 3.08, 0.74

    print("=" * 60)
    print(" NTM STABILIZATION VIA ECCD")
    print("=" * 60)

    result = required_eccd_power(beta_N, beta_p, a, R0, n_e20, T_e_keV,
                                  q95, Ip_MA, Bt)

    for surf in result["surfaces"]:
        print(f"\n  q={surf['q_surface']:.1f} surface (r/a={surf['rho_surface']:.2f}):")
        print(f"    T_local = {surf['T_local_keV']:.1f} keV")
        print(f"    γ_CD = {surf['gamma_CD']:.3f} [10²⁰ A/W/m²]")
        print(f"    I_ECCD required = {surf['I_ECCD_kA']:.1f} kA")
        print(f"    P_ECCD required = {surf['P_ECCD_MW']:.1f} MW")

    print(f"\n  Total ECCD power: {result['total_P_ECCD_MW']:.1f} MW")
    print(f"  Gyrotrons (1 MW each): {result['n_gyrotrons']}")
    print(f"  Ports required: {result['n_ports']} ({result['port_area_m2']:.1f} m²)")
    print(f"  ECCD cost: ${result['cost_MS']:.0f}M")
    print(f"  Recirculating power (50% eff): {result['recirc_P_ECCD_MW']:.1f} MW")
