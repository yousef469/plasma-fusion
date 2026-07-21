"""
Neoclassical tearing mode stabilization via ECCD.

Physics basis:
  - NTM stability criterion (La Haye 2006):
    Δ' + Δ_bs + Δ_pol + Δ_CD = 0
  - Bootstrap drive: I_bs_drive ∝ β_p / w
  - ECCD stabilization: I_ECCD / I_p ≈ 0.3 × (w/a) × (β_p/2)

Frequency selection (high-field design, B₀ = 11.7 T):
  - Fundamental (1st harmonic O-mode): f = 28 GHz/T × B_res
  - For B₀ = 11.7 T: f_ce ≈ 328 GHz (on-axis fundamental)
  - No demonstrated CW gyrotron above 250 GHz
  - Recommended strategy: develop 330+ GHz gyrotrons (SPARC-like trajectory)
    or use 2nd harmonic X-mode at ~164 GHz (ITER-class gyrotrons, B_res ≈ 2.9 T)
  - This analysis quantifies power requirements independent of frequency choice;
    the frequency affects launcher design and absorption physics but not
    the Fisch-Boozer current drive efficiency scaling used here.

References:
  - La Haye 2006, Phys. Plasmas 13, 055501
  - Sweeney 2020, J. Plasma Phys. 86, 865860112
  - Poli 2018, Nucl. Fusion 58, 016009
  - ITER Physics Basis Ch. 6 (ECCD)
  - Thumm 2020, IEEE Trans. Plasma Sci. 48, 1438 (gyrotron roadmap)
"""

import math

E_CHARGE = 1.602e-19


def ec_frequency(Bt, harmonic=1):
    """EC resonance frequency for given toroidal field and harmonic."""
    f_GHz = harmonic * 28.0 * Bt
    return f_GHz


def gyrotron_availability(f_GHz):
    """
    Assess gyrotron technology readiness at a given frequency.
    Returns dict with status, TRL, and notes.
    """
    if f_GHz <= 170:
        return {"TRL": 9, "status": "commercial", "P_MW": 1.0,
                "eff": 0.50, "note": "ITER-grade, multiple suppliers"}
    elif f_GHz <= 250:
        return {"TRL": 6, "status": "demonstrated", "P_MW": 1.0,
                "eff": 0.45, "note": "DIII-D / SPARC R&D"}
    elif f_GHz <= 330:
        return {"TRL": 3, "status": "development", "P_MW": 0.5,
                "eff": 0.40, "note": "Requires R&D program (~5 yr)"}
    else:
        return {"TRL": 2, "status": "concept", "P_MW": 0.25,
                "eff": 0.35, "note": "Long-lead R&D needed (~10 yr)"}


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


def eccd_efficiency(T_e_keV, n_e20, R0, Bt=0.0, inboard_launcher=False):
    """
    EC current drive efficiency (Fisch-Boozer).
    γ_CD = n_e20 × R0 × I_CD / P_CD [10^20 A/W/m²]

    Based on ITER Physics Basis scaling with T_e and n_e.
    Inboard (high-field side) launchers achieve 1.5-2× higher efficiency
    due to stronger wave-particle coupling at higher B field.
    Reference: ARC 2015, MIT PSFC.
    """
    if T_e_keV <= 0 or n_e20 <= 0:
        return 0.01
    gamma_cd = 0.04 + 0.016 * T_e_keV - 0.003 * n_e20
    gamma_cd = max(gamma_cd, 0.01)
    limit = 0.55 if inboard_launcher else 0.30
    gamma_cd = min(gamma_cd, limit)
    return gamma_cd


def required_eccd_power(beta_N, beta_p, a, R0, n_e20, T_e_keV,
                        q95, Ip_MA, Bt, inboard_launcher=False):
    """
    Compute total ECCD power needed to stabilize NTMs.

    Stabilizes both q=3/2 and q=2 surfaces.
    Required driven current per surface:
      I_ECCD / I_p = 0.3 × (w/a) × (β_p/2)
    where w/a ≈ 0.05 is typical seed island width.

    inboard_launcher: use high-field-side launchers for 1.5-2×
                      higher CD efficiency (ARC 2015 concept).
    """
    q0 = 0.8
    results = []
    total_P_ECCD = 0.0

    for q_surf in [1.5, 2.0]:
        surf = rational_surface_radius(q_surf, q0, q95, a)
        if surf is None:
            continue
        r_surf, rho_surf = surf

        # Local temperature at rational surface (parabolic profile)
        alpha_T = 0.8
        T_local = T_e_keV * (1 + alpha_T) * (1 - rho_surf ** 2) ** alpha_T
        T_local = max(T_local, 1.0)

        # EC current drive efficiency at this location
        gamma_cd = eccd_efficiency(T_local, n_e20, R0, Bt=Bt,
                                   inboard_launcher=inboard_launcher)
        gamma_cd = max(gamma_cd, 0.02)

        # Required driven current (La Haye 2006 criterion)
        w_over_a = 0.05
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

    # Frequency selection for this machine
    # Inboard launchers use 2nd harmonic X-mode (HFS launch)
    # → ~170 GHz for 14 T, ITER-class gyrotrons, TRL 9
    # Outboard launchers use fundamental O-mode → 392 GHz for 14 T, TRL 2
    if inboard_launcher:
        # 2nd harmonic: f ≈ 28 * Bt / 2 → use 170 GHz ITER-class
        f_fund = 170.0  # ITER-class gyrotrons, regardless of Bt
        gyro_info = gyrotron_availability(f_fund)
    else:
        f_fund = ec_frequency(Bt, harmonic=1)
        gyro_info = gyrotron_availability(f_fund)
    P_GYROTRON = gyro_info["P_MW"]
    gyro_eff = gyro_info["eff"]
    n_gyrotrons = max(1, math.ceil(total_P_ECCD / P_GYROTRON))

    # Transmission line losses (waveguide + mirrors)
    # ITER baseline: ~75% transmission efficiency for 170 GHz
    # Higher frequencies incur more ohmic loss: ~70% at 330 GHz
    if f_fund <= 170:
        eta_trans = 0.75
    elif f_fund <= 250:
        eta_trans = 0.72
    elif f_fund <= 330:
        eta_trans = 0.68
    else:
        eta_trans = 0.60
    P_ECCD_launched = total_P_ECCD / eta_trans

    # Cost estimate
    # Gyrotron cost: Thumm 2020 ~$4/MW at 170 GHz, scales with TRL
    cost_per_MW = 4.0 * (170.0 / max(f_fund, 10)) ** 0.3
    cost_per_MW = max(cost_per_MW, 2.0)
    gyrotron_cost_MS = P_ECCD_launched * cost_per_MW
    # Launcher + transmission: ~$4M per port (steering mirrors, waveguides)
    n_ports = max(1, math.ceil(n_gyrotrons / 8.0))
    port_system_cost_MS = n_ports * 4.0
    # Auxiliary (power supplies, cooling, controls): ~$10M
    aux_cost_MS = 10.0
    total_cost_MS = gyrotron_cost_MS + port_system_cost_MS + aux_cost_MS

    port_area = n_ports * 0.8 * 1.2

    return {
        "surfaces": results,
        "total_P_ECCD_MW": total_P_ECCD,
        "P_ECCD_launched_MW": P_ECCD_launched,
        "f_ECCD_GHz": f_fund,
        "gyrotron_freq_GHz": f_fund,
        "gyrotron_TRL": gyro_info["TRL"],
        "gyrotron_status": gyro_info["status"],
        "gyrotron_note": gyro_info["note"],
        "P_gyrotron_MW": P_GYROTRON,
        "n_gyrotrons": n_gyrotrons,
        "gyrotron_efficiency": gyro_eff,
        "n_ports": n_ports,
        "port_area_m2": port_area,
        "transmission_efficiency": eta_trans,
        "cost_MS": total_cost_MS,
        "recirc_P_ECCD_MW": P_ECCD_launched / gyro_eff,
    }


if __name__ == "__main__":
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

    print(f"\n  Machine: R0={R0} m, a={a} m, Bt={Bt} T")
    print(f"  EC frequency (fundamental): {result['gyrotron_freq_GHz']:.0f} GHz")
    print(f"  Gyrotron TRL: {result['gyrotron_TRL']} ({result['gyrotron_status']})")
    print(f"  {result['gyrotron_note']}")

    for surf in result["surfaces"]:
        print(f"\n  q={surf['q_surface']:.1f} surface (r/a={surf['rho_surface']:.2f}):")
        print(f"    T_local = {surf['T_local_keV']:.1f} keV")
        print(f"    γ_CD = {surf['gamma_CD']:.3f}")
        print(f"    I_ECCD required = {surf['I_ECCD_kA']:.1f} kA")
        print(f"    P_ECCD required = {surf['P_ECCD_MW']:.1f} MW")

    print(f"\n  Total power at plasma: {result['total_P_ECCD_MW']:.1f} MW")
    print(f"  Transmission efficiency: {result['transmission_efficiency']:.0%}")
    print(f"  Launched power: {result['P_ECCD_launched_MW']:.1f} MW")
    print(f"  Gyrotrons: {result['n_gyrotrons']} × {result['P_gyrotron_MW']:.1f} MW  ({result['gyrotron_freq_GHz']:.0f} GHz)")
    print(f"  Ports: {result['n_ports']} ({result['port_area_m2']:.1f} m²)")
    print(f"  EC system cost: ${result['cost_MS']:.0f}M")
    print(f"  Wall-plug power: {result['recirc_P_ECCD_MW']:.1f} MW ({result['gyrotron_efficiency']:.0%} gyrotron eff)")
    print(f"  Recirc fraction of P_net: {result['recirc_P_ECCD_MW'] / 1762 * 100:.1f}%")
