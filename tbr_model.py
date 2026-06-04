"""
HCPB tritium breeding blanket model.
Calibrated to published MCNP results for EU DEMO HCPB.

Key calibration points:
  - Fischer 2020: 800mm, 15% Li6, 20mm Be → TBR = 1.17 (F4E)
  - Hernandez 2020: 800mm, 15% Li6, 20mm Be → TBR = 1.19 (KIT)
  - Chen 2021: parametric scaling validated against MCNP

Physics:
  - Be9(n,2n): 14 MeV → 2 neutrons at ~5 MeV; effective M ≈ 1.58 at 20mm
  - Li6(n,α)T: σ_th = 940 b → dominates thermal/epithermal captures
  - Li7(n,n'α)T: threshold > 2.5 MeV, σ_avg ≈ 0.15 b in fast spectrum
  - Li4SiO4 pebble bed, 40% packing, Eurofer structures

Returns:
  TBR = coverage × n_mult × f_breed × η_thickness
"""

import math


def hcpb_tbr(
    blanket_thickness_mm=800,
    li6_enrichment=0.15,
    be_thickness_mm=20,
    coverage=0.92,
):
    if blanket_thickness_mm <= 0:
        return {"TBR": 0.0, "T6": 0.0, "T7": 0.0, "n_mult": 1.0, "coverage": coverage}

    # ── 1. Neutron multiplication ──
    # Be9(n,2n): Σ_{n,2n} ≈ 0.31 cm⁻¹ at 14 MeV
    # Effective multiplication including secondary (n,2n) in Be:
    #   M(eff) = 1 + 0.58 for 20mm (calibrated to MCNP TBR=1.17)
    t_be_cm = max(be_thickness_mm / 10.0, 0.0)
    f_interact = 1.0 - math.exp(-0.31 * t_be_cm)
    # Secondary multiplication factor: multiplied n cause further (n,2n)
    n_mult = 1.0 + f_interact * 1.26
    n_mult = min(n_mult, 1.65)

    # ── 2. Lithium capture competition ──
    # Relative capture rates (calibrated to MCNP at 15% Li6):
    #   f_breed = 0.81 for reference → k_loss = 0.056
    k_li6 = 1.0
    k_li7 = 0.10
    k_loss = 0.056  # structural + leakage losses (Si, O, Fe, He)

    numerator = k_li6 * li6_enrichment + k_li7 * (1.0 - li6_enrichment)
    denominator = numerator + k_loss
    f_breed = numerator / denominator

    # ── 3. Thickness efficiency ──
    lambda_fast = 80.0  # mm
    tau = blanket_thickness_mm / lambda_fast
    eta_thick = 1.0 - math.exp(-tau)

    # ── 4. TBR components ──
    tbr_total = coverage * n_mult * f_breed * eta_thick

    f6 = k_li6 * li6_enrichment / max(numerator, 1e-12)
    f7 = k_li7 * (1.0 - li6_enrichment) / max(numerator, 1e-12)
    t6 = tbr_total * f6
    t7 = tbr_total * f7

    return {
        "TBR": tbr_total,
        "T6": t6,
        "T7": t7,
        "n_mult": n_mult,
        "f_breed": f_breed,
        "eta_thickness": eta_thick,
        "coverage": coverage,
        "li6_enrichment": li6_enrichment,
        "blanket_thickness_mm": blanket_thickness_mm,
        "be_thickness_mm": be_thickness_mm,
    }


def li6_sensitivity(enrichments=None):
    if enrichments is None:
        enrichments = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.90]
    return [(e, hcpb_tbr(li6_enrichment=e)) for e in enrichments]


def thickness_sensitivity(thicknesses=None):
    if thicknesses is None:
        thicknesses = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    return [(d, hcpb_tbr(blanket_thickness_mm=d)) for d in thicknesses]


if __name__ == "__main__":
    print("=" * 60)
    print(" HCPB TRITIUM BREEDING MODEL (calibrated to MCNP)")
    print("=" * 60)

    print("\nReference (HCPB, 800 mm @ 15% Li6, 20 mm Be, 92% cov):")
    ref = hcpb_tbr()
    print(f"  TBR = {ref['TBR']:.4f}   (expected: 1.17-1.19)")
    print(f"  T6 = {ref['T6']:.4f},  T7 = {ref['T7']:.4f}")
    print(f"  n_mult = {ref['n_mult']:.3f}")
    print(f"  f_breed = {ref['f_breed']:.3f}")
    print(f"  eta_thickness = {ref['eta_thickness']:.4f}")

    print(f"\nLi6 enrichment sensitivity:")
    print(f"  {'enrich':>6s}  {'TBR':>6s}  {'T6':>6s}  {'T7':>6s}  {'status':>10s}")
    for e, r in li6_sensitivity():
        ok = "OK" if r["TBR"] >= 1.05 else "SHORT"
        print(f"  {e*100:5.0f}%  {r['TBR']:.4f}  {r['T6']:.4f}  {r['T7']:.4f}  [{ok}]")

    print(f"\nBlanket thickness sensitivity (15% Li6):")
    print(f"  {'thick':>5s}  {'TBR':>6s}  {'status':>10s}")
    for d, r in thickness_sensitivity():
        ok = "OK" if r["TBR"] >= 1.05 else "SHORT"
        print(f"  {d:4d}mm  {r['TBR']:.4f}  [{ok}]")

    print(f"\nSelf-sufficiency (TBR ≥ 1.05) requires:")
    for e in [0.075, 0.10, 0.15, 0.20]:
        r = hcpb_tbr(li6_enrichment=e)
        print(f"  {e*100:.0f}% Li6 → TBR = {r['TBR']:.4f}  " +
              f"{'✓' if r['TBR'] >= 1.05 else '✗'}")
