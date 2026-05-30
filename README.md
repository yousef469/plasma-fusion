# Plasma Fusion Power Plant Design

A complete conceptual design of a self-sustaining tokamak fusion power plant, integrating plasma physics, magnet engineering, divertor analysis, tritium breeding, structural mechanics, quench protection, and AI-based plasma control.

## Key Results

| Parameter | Value |
|-----------|-------|
| Fusion Power | 2,735 MW (thermal) |
| Net Electric Output | 1,713 MW |
| Toroidal Field | 11.7 T |
| Major Radius | 12.08 m |
| Triple Product (Lawson Criterion) | 7× above ignition |
| Energy Confinement Time | 10.4 s |
| Tritium Breeding Ratio | 1.54 |
| Peak Coil Stress (Safety Factor) | 32 MPa (SF > 28) |
| Divertor Heat Flux | 6.7 MW/m² (3× margin) |
| AI Controller Reward | +3,904 |
| Capital Cost | \$12.66 B (\$7,324/kW) |
| LCOE | \$104/MWh |

## Design Philosophy

- **Conservative engineering**: All components use ITER-qualified, commercially available materials
- **No exotic physics**: Confinement model validated against JET D-T experimental data (7% error)
- **Full integration**: Physics + magnets + divertor + tritium + structure + control + cost
- **Reproducible**: All results produced from first-principles code, not spreadsheets

## Contents

- `paper/` — LaTeX source for the research paper
- `physics/` — Plasma power balance, stability, and confinement models
- `engineering/` — Structural FEA, quench protection, vacuum vessel design
- `divertor/` — SOLPS 2-point model and detachment analysis
- `tritium/` — Breeding blanket neutronics and fuel cycle
- `control/` — AI plasma controller training and validation
- `results/` — Output parameters, sensitivity analysis, Pareto front

## Methodology

All physics models are derived from published peer-reviewed literature:

- **Fusion reactivity**: Bosch-Hale 1992
- **Confinement scaling**: ITER H98(y,2)
- **Heat flux width**: Eich 2013 multi-machine database
- **Tritium breeding**: Abdou 2021 parametric fits
- **Vertical stability**: Walker 2015, Albanese 2021
- **Divertor physics**: Stangeby 2000, Pitts 2009

## Validation

The confinement model (H98 scaling) was validated against the JET 1997 D-T record shot:

| Quantity | Predicted | Measured | Error |
|----------|-----------|----------|-------|
| $\tau_E$ | 0.381 s | 0.40 s | 5% |
| $Q$ | 0.62 | 0.67 | 7% |

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
| TF Coils (Nb₃Sn) | \$1.72 B |
| Vacuum Vessel | \$0.20 B |
| Blanket (LiPb) | \$0.40 B |
| Divertor (Tungsten) | \$0.05 B |
| Heating (Gyrotrons) | \$0.18 B |
| Materials Subtotal | ~\$2.4 B |
| Direct Cost | ~\$5.5 B |
| Total Project Cost (FOAK) | **\$12.66 B** |
| Cost per kWe | **\$7,324/kW** |

## References

1. Bosch & Hale, *Nucl. Fusion* 32, 611 (1992)
2. ITER Physics Basis, *Nucl. Fusion* 39, 2137 (1999)
3. Keilhacker et al., *Nucl. Fusion* 39, 209 (1999)
4. Eich et al., *Nucl. Fusion* 53, 093031 (2013)
5. Abdou et al., *Fusion Eng. Des.* 171, 112546 (2021)
6. Martin et al., *J. Nucl. Mater.* 337, 104 (2008)
7. Sauter et al., *Phys. Plasmas* 6, 2834 (1999)
8. Pitts et al., *J. Nucl. Mater.* 390, 100 (2009)
9. Walker et al., *Fusion Eng. Des.* (2015)
10. Menard et al., *Nucl. Fusion* 56, 106023 (2016)

## License

All content is provided for reference and reproducibility purposes.
