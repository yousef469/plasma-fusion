# Plasma Fusion Power Plant Design — V3 (Gyro-Bohm)

An open-source, systems-level conceptual design of a tokamak fusion power plant, integrating 1.5D radial transport, profile-consistent MHD stability, SOL/divertor analysis, tritium breeding neutronics, TF coil structural analysis, and AI-based plasma control.

**Design philosophy**: SpaceX-style iterative improvement — V1 established a conservative baseline, V2 tested aggressive compact configurations, V3 converges on a stable design with all physics models within validated ranges.

## Key Result — V3 Gyro-Bohm Design

| Parameter | Value |
|-----------|-------|
| Major radius R₀ | **7.0 m** |
| Minor radius a | **1.2 m** |
| Aspect ratio A | **5.8** |
| Elongation κ | **3.0** |
| Triangularity δ | **-0.30** (negative) |
| Toroidal field B₀ | **13.0 T** (HTS REBCO) |
| Peak TF coil field | **22.8 T** |
| Plasma current Iₚ | **19.1 MA** |
| Safety factor q₉₅ | **3.3** |
| Density n/n_GW | **0.75** |
| **Fusion power P_fus** (gyro-Bohm) | **14,087 MW** |
| **Net electric power** | **5,250 MW** |
| Fusion gain Q | **> 100 (ignited)** |
| Equilibrium T | **17.0 keV** |
| Confinement time τ_E | **0.59 s** |
| Thermal efficiency | **40%** |
| Normalized beta β_N | **2.77** (no-wall limit: 3.67) |
| TF conductor | **REBCO HTS** (T_margin = 55 K) |
| **Total capital cost** | **$34.36B** |
| **Cost per kW** | **$6,545/kW** |
| **LCOE** | **$77/MWh** |
| Annual revenue at $90/MWh | **$3.52B** |
| Simple payback | **9.8 years** |

One plant powers **39.1 TWh/yr** — ~75% of New York City's annual electricity consumption.

### Engineering Margins

| Constraint | Value | Limit | Margin |
|------------|:-----:|:-----:|:------:|
| β_N (no-wall stability) | **2.77** | 3.67 | **24%** ✓ |
| TF coil peak field | **22.8 T** | 65 T (REBCO) | **55 K margin** ✓ |
| TF case stress (N50H) | **646 MPa** | 1,000 MPa | **1.55×** ✓ |
| Divertor peak flux | **34.5 MW/m²** | 60 MW/m² | **43%** ✓ |
| Neutron wall load | **12.3 MW/m²** | — | FLiBe mitigates to ~3 MW/m² |
| TBR (HCPB blanket) | **1.175** | > 1.0 | **17.5% surplus** ✓ |
| NTM stability (w/o ECCD) | **1.13** | < 1.0 | ECCD (12 MW) stabilizes |

### Gyro-Bohm vs H98

| Model | τ_E | P_fus | P_net |
|-------|:---:|:-----:|:-----:|
| **Gyro-Bohm** (primary) | **0.59 s** | **14.1 GW** | **5.25 GW** |
| H98(y,2) (reference) | 4.62 s | 11.4 GW | 4.30 GW |

Both models give similar net power despite 8× τ_E difference because the equilibrium solver finds different self-consistent temperatures (17.0 keV vs 15.0 keV).

### Cost Breakdown

| Component | Cost |
|-----------|:----:|
| TF coils (HTS REBCO) | $11.21B |
| Turbine hall | $6.78B |
| Cooling systems | $3.39B |
| PF coils | $0.93B |
| Blanket | $0.59B |
| Tritium plant | $1.00B |
| Site, buildings, I&C | $1.10B |
| Contingency (15%) | $4.36B |
| **Total** | **$34.36B** |

## Repository Contents

| File | Description |
|------|-------------|
| `physics_engine.py` | Main integration — calls all modules |
| `transport_solver.py` | Gyro-Bohm confinement solver + equilibrium T iteration |
| `mhd_stability.py` | Profile-consistent MHD (no-wall, wall, NTM, disruption) |
| `divertor_sol.py` | SOL/divertor two-point model + Eich λq |
| `tbr_model.py` | HCPB tritium breeding (calibrated to MCNP) |
| `heating_system.py` | Auxiliary heating sizing (NBI, EC, IC) |
| `ntm_stabilization.py` | ECCD for NTM control |
| `figures.py` | All publication-quality figures |
| `v3_model.py` | Parametric 3D model (STL generator) |
| `main.tex` | Paper source (LaTeX, 11 figures) |

## Methodology

All physics models from published peer-reviewed literature:

- **Fusion reactivity**: Bosch-Hale 1992
- **Confinement**: Gyro-Bohm (primary, C_GB=1.14 calibrated to JET DTE1) + ITER H98(y,2)
- **Equilibrium T**: Self-consistent power balance via binary search
- **MHD stability**: ITER Physics Basis, Sweeney 2020, La Haye 2006
- **Heat flux**: Eich 2013 multi-machine database
- **Divertor**: Stangeby 2000, Kuang 2020
- **Tritium breeding**: Fischer 2020, Hernandez 2020 (MCNP-calibrated)
- **ECCD**: Poli 2018, Fisch-Boozer
- **TF coils**: REBCO HTS model (Bc₂=65 T, Tc₀=93 K)
- **AI control**: Cross-entropy method

## Validation

Confinement validated against JET 1997 D-T record shot:

| Quantity | Predicted | Measured | Error |
|----------|:---------:|:--------:|:-----:|
| τE (H98) | 0.381 s | 0.40 s | 5% |
| Q (H98) | 0.62 | 0.67 | 7% |

## License

MIT License
