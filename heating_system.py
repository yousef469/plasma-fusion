"""
Auxiliary heating system sizing for the reference design.

Computes:
  - Required wall-plug power for given P_ext
  - Port space requirements
  - Impact on net electric power
  - Heating technology options (NBI, EC, IC)

References:
  - ITER heating systems: Davis 2018, Nucl. Fusion
  - SPARC heating: Creely 2020, J. Plasma Phys.
  - NBI efficiency: Surrey 2018, Fusion Eng. Des.
  - Gyrotron efficiency: Thumm 2020, IEEE Trans. Plasma Sci.
"""

HEATING_TECHNOLOGIES = {
    "NBI": {
        "wall_plug_efficiency": 0.40,
        "power_per_unit_MW": 5.0,
        "power_per_port_MW": 20.0,  # 4 units per port
        "port_width_m": 1.5,
        "port_height_m": 2.0,
        "cost_per_MW_MS": 3.0,
        "desc": "Neutral beam injection (ITER-style, 1 MeV D-)"
    },
    "EC": {
        "wall_plug_efficiency": 0.50,
        "power_per_unit_MW": 1.0,   # 1 MW per gyrotron
        "power_per_port_MW": 8.0,   # 8 gyrotrons per port
        "port_width_m": 0.8,
        "port_height_m": 1.2,
        "cost_per_MW_MS": 4.0,
        "desc": "Electron cyclotron (170 GHz, 1 MW gyrotrons)"
    },
    "IC": {
        "wall_plug_efficiency": 0.45,
        "power_per_unit_MW": 3.0,
        "power_per_port_MW": 6.0,
        "port_width_m": 0.8,
        "port_height_m": 1.2,
        "cost_per_MW_MS": 3.5,
        "desc": "Ion cyclotron (40-55 MHz, 3 MW RF sources)"
    },
}


def size_heating_system(P_ext_MW, technology="EC"):
    """
    Size an auxiliary heating system for given P_ext.
    
    Returns dict with counts, costs, space, and net electric impact.
    """
    if P_ext_MW <= 0:
        return {
            "P_ext_MW": 0,
            "technology": technology,
            "wall_plug_power_MW": 0,
            "n_units": 0,
            "n_ports": 0,
            "port_area_m2": 0,
            "cost_MS": 0,
            "recirc_heating_MW": 0,
        }

    tech = HEATING_TECHNOLOGIES.get(technology, HEATING_TECHNOLOGIES["EC"])
    eta = tech["wall_plug_efficiency"]
    p_per_unit = tech["power_per_unit_MW"]
    p_per_port = tech["power_per_port_MW"]

    # Wall-plug power required
    P_wall_plug = P_ext_MW / eta

    # Number of individual units
    n_units = math.ceil(P_ext_MW / p_per_unit)
    n_ports = math.ceil(P_ext_MW / p_per_port)

    # Port area
    port_area = n_ports * tech["port_width_m"] * tech["port_height_m"]

    # Cost
    cost = P_ext_MW * tech["cost_per_MW_MS"]

    # Recirculating power from heating
    recirc = P_wall_plug

    # Total port area available (8 equatorial + 16 upper/lower)
    total_port_available = 8 * 1.5 * 2.0 + 16 * 0.8 * 1.2  # ~39 m²

    return {
        "P_ext_MW": P_ext_MW,
        "technology": technology,
        "technology_desc": tech["desc"],
        "wall_plug_efficiency": eta,
        "wall_plug_power_MW": P_wall_plug,
        "n_units": n_units,
        "n_ports": n_ports,
        "port_area_m2": port_area,
        "total_port_available_m2": total_port_available,
        "port_fraction": port_area / max(total_port_available, 1e-12),
        "cost_MS": cost,
        "recirc_heating_MW": recirc,
    }


def compare_technologies(P_ext_MW):
    """Return heating system specs for all three technologies."""
    results = {}
    for tech in ["NBI", "EC", "IC"]:
        results[tech] = size_heating_system(P_ext_MW, tech)
    return results


import math


if __name__ == "__main__":
    P_ext_GB = 750.6  # gyro-Bohm external heating requirement

    print("=" * 60)
    print(f" HEATING SYSTEM SIZING (P_ext = {P_ext_GB:.0f} MW)")
    print("=" * 60)

    print(f"\n{'Technology':<12s} {'η_wp':>6s} {'P_wall':>8s} {'n_units':>8s} "
          f"{'n_ports':>8s} {'ports_m²':>8s} {'port_frac':>9s} {'Cost M$':>8s}")
    for tech, r in compare_technologies(P_ext_GB).items():
        print(f"  {tech:<12s} {r['wall_plug_efficiency']:.2f}  "
              f"{r['wall_plug_power_MW']:8.0f} {r['n_units']:8d} "
              f"{r['n_ports']:8d} {r['port_area_m2']:8.2f} "
              f"{r['port_fraction']:.2%}  {r['cost_MS']:8.0f}")

    print(f"\nPreferred technology: EC (170 GHz gyrotrons)")
    ec = size_heating_system(P_ext_GB, "EC")
    print(f"  Required power to plasma: {ec['P_ext_MW']:.0f} MW")
    print(f"  Wall-plug power: {ec['wall_plug_power_MW']:.0f} MW ({ec['wall_plug_efficiency']:.0%} eff)")
    print(f"  Gyrotrons (1 MW each): {ec['n_units']}")
    print(f"  Ports required: {ec['n_ports']} ({ec['port_area_m2']:.1f} m² / {ec['total_port_available_m2']:.0f} m² avail)")
    print(f"  Heating cost: ${ec['cost_MS']:.0f}M")
    print(f"  Recirculating power: {ec['recirc_heating_MW']:.0f} MW")
    print()
    print(f"For comparison, primary H98 operation (P_ext = 0):")
    print(f"  No auxiliary heating needed during burn")
    print(f"  ~30 MW startup heating only, not continuous")
