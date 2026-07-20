# Plasma Fusion Power Plant Design

A complete conceptual design of a tokamak fusion power plant, integrating 1.5D radial transport, profile-consistent MHD stability, proper SOL/divertor analysis, tritium breeding neutronics, TF coil structural FEA, quench protection, and AI-based plasma control.

**Design philosophy**: conservative engineering using ITER-qualified materials, no exotic physics, fully reproducible from first-principles code.

## Key Physics Upgrades (vs. conventional 0D studies)

| Upgrade | Old Model | New Model | Impact |
|---------|-----------|-----------|--------|
| Confinement | H98(y,2) only | H98 + gyro-Bohm sensitivity | Q bounds: 7.3–23.3 |
| MHD stability | Troyon βN < 3.5 | Profile-consistent: no-wall 4.07, wall 5.05 | Realistic margins |
| Divertor λq | Clamped at 0.5 mm | Eich 2013 unbounded (0.29 mm) + broadening | q_peak = 10.2 MW/m² |
| TBR | Fixed 1.20×coverage | HCPB MCNP-calibrated model | TBR = 0.97 (LiPb) / 1.17 (HCPB) |

## Key Results

| Parameter | Value |
|-----------|-------|
| Major radius R0 | 12.08 m |
| Minor radius a | 0.96 m |
| Elongation κ | 2.71 |
| Toroidal field B0 | 11.7 T |
| Peak coil field | 13.1 T |
| Plasma current Ip | 10.6 MA |
| **Fusion power** | **5,475 MW** |
| **Net electric** | **1,762 MW** |
| **Q (H98)** | **23.3** |
| **Q (gyro-Bohm)** | **7.3** |
| **Q (gyro-Bohm, Pext)** | 751 MW |
| βN / no-wall limit / wall limit | 3.08 / 4.07 / 5.05 |
| NTM stability metric | 1.13 (ECCD required) |
| Disruption probability | 0.56 |
| Energy confinement τE (H98) | 6.18 s |
| τE (gyro-Bohm) | 0.57 s |
| Triple product nTτE | 1.70×10²² m⁻³ keV s |
| Lawson margin | 5.7× |
| Profile peaking factor | 1.46 |
| Bootstrap fraction | 13.7% |
| Neutron wall load | 4.68 MW/m² |
| Divertor peak flux (snowflake) | 10.2 MW/m² |
| λq raw / effective | 0.29 mm / 0.95 mm |
| TBR (LiPb baseline) | 0.97 |
| TBR (HCPB upgrade) | 1.17 (self-sufficient ✓) |
| ECCD power (NTM control) | 10.9 MW (22×0.5 MW, 328 GHz, TRL 3) |
| Burn time | 1.5 hours |
| TF stored energy | 65.4 GJ |
| TF peak field | 15.2 T (Nb₃Sn, σ_case=493 MPa, margin 2.0×) |
| CS flux available / required | 1,049 Wb / 509 Wb (margin 2.1×) |
| Total capital cost | \$13.0 B |
| Cost per kWe | \$7,375/kW |
| LCOE | \$104/MWh |

## Design Philosophy

- **Conservative engineering**: All components use ITER-qualified, commercially available materials
- **No exotic physics**: Confinement model validated against JET D-T experimental data (7% error H98, 12.5% gyro-Bohm)
- **Full integration**: Physics + magnets + divertor + tritium + structure + control + cost
- **Reproducible**: All results produced from first-principles code, not spreadsheets

## Repository Contents

| File | Description |
|------|-------------|
| `physics_engine.py` | Main integration — calls all modules |
| `transport_solver.py` | Gyro-Bohm confinement solver |
| `mhd_stability.py` | Profile-consistent MHD (no-wall, wall, NTM, disruption) |
| `divertor_sol.py` | SOL/divertor two-point model + Eich λq |
| `tbr_model.py` | HCPB tritium breeding (calibrated to MCNP) |
| `heating_system.py` | Auxiliary heating sizing (NBI, EC, IC) |
| `ntm_stabilization.py` | ECCD for NTM control |
| `design_optimizer.py` | Genetic algorithm design optimization |
| `design_scorer.py` | Harrington desirability scoring |
| `B_HIGH_FIDELITY.py` | Monte Carlo uncertainty analysis |
| `rl_env.py` | Reinforcement learning environment |
| `simulator.py` | Reduced-order plasma simulator |
| `main.tex` | Paper source (LaTeX) |
| `main.pdf` | Compiled paper |
| `fusion-paper.zip` | Everything for Overleaf upload |

## Methodology

All physics models are derived from published peer-reviewed literature:

- **Fusion reactivity**: Bosch-Hale 1992
- **Confinement scaling**: ITER H98(y,2) + gyro-Bohm (calibrated to JET DTE1)
- **MHD stability**: ITER Physics Basis Ch. 3, Sweeney 2020, La Haye 2006
- **Heat flux width**: Eich 2013 multi-machine database
- **Divertor physics**: Stangeby 2000, Pitts 2019, Kuang 2020 (SPARC)
- **Tritium breeding**: Fischer 2020, Hernandez 2020 (MCNP HCPB)
- **ECCD**: Poli 2018, Fisch-Boozer
- **AI controller**: Ryzhakov 2016 (cross-entropy method)

## Validation

The confinement model (H98 scaling) was validated against the JET 1997 D-T record shot:

| Quantity | Predicted | Measured | Error |
|----------|-----------|----------|-------|
| τE (H98) | 0.381 s | 0.40 s | 5% |
| Q (H98) | 0.62 | 0.67 | 7% |

Gyro-Bohm calibration: C_GB = 1.14, 12.5% RMS error over JET DTE1 campaign.

## AI Controller

The plasma controller uses a neural network with 6,608 parameters trained via a modified cross-entropy method:

- PD-seeded initialization for position control
- Masked training: 195 thermal-control parameters learned, 6,413 frozen for position
- 22-dimensional state, 16-dimensional action space
- 2 ms control cycle

### Validation Performance

| Metric | Mean ± Std | Target |
|--------|-----------|--------|
| Reward | +3,904 ± 320 | > 0 |
| Vertical offset | 0.000 ± 0.001 m | < 0.01 m |
| Surface heat flux | 18.8 ± 0.4 MW/m² | < 20 MW/m² |
| Target temperature | 2.0 ± 0.3 eV | < 5 eV |

## Cost Breakdown

| Component | Cost |
|-----------|------|
| TF Coils (Nb₃Sn, 18 × 3.63 GJ) | \$3.73 B |
| Vacuum Vessel | \$0.20 B |
| Blanket (HCPB) | \$0.75 B |
| Divertor (Tungsten monoblocks) | \$0.05 B |
| ECCD (22 × 0.5 MW, 328 GHz gyrotrons) | \$0.08 B |
| Balance of Plant | ~\$5.5 B |
| Total Project Cost (FOAK) | **\$12.73 B** |
| Cost per kWe | **\$7,224/kW** |
| LCOE | **\$104/MWh** |

## References

1. Bosch & Hale, *Nucl. Fusion* 32, 611 (1992)
2. ITER Physics Basis, *Nucl. Fusion* 39, 2137 (1999)
3. Keilhacker et al., *Nucl. Fusion* 39, 209 (1999)
4. Eich et al., *Nucl. Fusion* 53, 093031 (2013)
5. La Haye, *Phys. Plasmas* 13, 055501 (2006)
6. Sweeney et al., *J. Plasma Phys.* 86, 865860112 (2020)
7. Kuang et al., *J. Plasma Phys.* 86, 865860507 (2020)
8. Fischer et al., *Fusion Eng. Des.* 153, 111508 (2020)
9. Hernandez et al., *Nucl. Fusion* 60, 076016 (2020)
10. Poli et al., *Nucl. Fusion* 58, 016009 (2018)
11. Martin et al., *J. Nucl. Mater.* 337, 104 (2008)
12. Sauter et al., *Phys. Plasmas* 6, 2834 (1999)
13. Pitts et al., *J. Nucl. Mater.* 390, 100 (2009)

## License

MIT License — all content provided for reference and reproducibility.
