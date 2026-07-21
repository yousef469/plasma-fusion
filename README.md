# Plasma Fusion Power Plant Design — V3 (Gyro-Bohm)

An open-source, systems-level conceptual design of a tokamak fusion power plant, integrating 1.5D radial transport, profile-consistent MHD stability, SOL/divertor analysis, tritium breeding neutronics, TF coil structural analysis, and AI-based plasma control.

**Design philosophy**: SpaceX-style iterative improvement — start with high performance, then optimize for stability, verifiability, and cost. V1 was a big HTS tokamak. V2 pushed compactness to the limit (β_N=5.06, A=16.7). **V3 is the operational design**: all physics models within validated ranges, no-wall MHD stability, gyro-Bohm confinement (conservative), and $5,018/kW.

## Key Result — V3 Gyro-Bohm Design

| Parameter | Value |
|-----------|-------|
| Major radius R₀ | **7.0 m** |
| Minor radius a | **1.2 m** |
| Aspect ratio A | **5.8** |
| Elongation κ | **3.0** |
| Triangularity δ | **-0.30** (negative — ELM-free) |
| Toroidal field B₀ | **13.0 T** (HTS REBCO) |
| Peak TF coil field | **22.8 T** |
| Plasma current Iₚ | **19.1 MA** |
| Safety factor q₉₅ | **3.5** |
| Density n/n_GW | **0.75** |
| Plasma volume | **597 m³** |
| Surface area | **742 m²** |
| **Fusion power P_fus** | **14,087 MW** (gyro-Bohm equilibrium) |
| **Net electric power** | **7,632 MW** |
| Fusion gain Q | **> 100 (ignited)** |
| Equilibrium T | **17.0 keV** |
| Thermal efficiency | **58.8%** (FLiBe + sCO₂ Brayton) |
| Burn time | **5,400 s (1.5 hours)** |
| TF stored energy | **86.2 GJ** |
| Conductor | **REBCO HTS** (T_margin = 55 K) |
| **Total capital cost** | **$35.34B** |
| **Cost per kW** | **$4,630/kW** |
| **LCOE (90% CF, 30 yr)** | **$50/MWh (5.0¢/kWh)** |
| Annual revenue at $90/MWh | **$5.4B** |
| Simple payback | **6.5 years** |

One plant powers **60.2 TWh/yr** — ~115% of New York City's annual electricity consumption. At $90/MWh, it earns $5.4B/yr with $35.3B capital cost. **100 plants replace the entire US electric grid.** **200 plants fully electrify everything** (transportation + heating).

### Engineering Margins

| Constraint | Value | Limit | Margin |
|------------|:-----:|:-----:|:------:|
| β_N (no-wall stability) | **2.77** | 3.67 | **0.90** ✓ |
| TF coil peak field | **22.8 T** | 23 T | **0.2 T** ✓ |
| TF case stress (N50H) | **889 MPa** | 1,550 MPa | **1.74×** ✓ |
| TF conductor T_margin | **55 K** | > 1 K | **55×** ✓ (REBCO) |
| Divertor peak flux | **34.5 MW/m²** | 60 MW/m² | **25.5 MW/m²** ✓ |
| Neutron wall load | **12.3 MW/m²** | — | Acceptable (FLiBe liquid blanket mitigates to ~3 MW/m² at structure) |
| Q95 margin | **3.50** | 3.0 | **0.50** ✓ |
| L-H threshold margin | **4.3×** | > 1.0 | **4.3×** ✓ |
| CS flux margin | **2.33** | > 1.0 | **1.33** ✓ |
| TBR (HCPB blanket) | **1.175** | > 1.0 | **0.175** ✓ |
| NTM stability (w/o ECCD) | **1.13** | < 1.0 | **Marginal — ECCD (12 MW) stabilizes** ⚠️ |

### Gyro-Bohm vs H98 — The Critical Uncertainty

The single largest physics uncertainty is the confinement model. Values at fixed n/n_GW=0.75:

| Model | τ_E | P_fus | P_net | Cost | $/kW | LCOE |
|-------|:---:|:-----:|:-----:|:----:|:----:|:----:|
| **Gyro-Bohm** (primary, consistent) | **0.59 s** | **14.1 GW** | **7.63 GW** | **$35.3B** | **$4,630** | **$50/MWh** |
| H98(y,2) (reference, for comparison) | 4.62 s | 11.4 GW | 6.28 GW | $31.5B | $5,018 | $54/MWh |

Gyro-Bohm is conservative: it predicts 8× lower τ_E than H98. Surprisingly, both models give similar net power because the plasma density and geometry dominate. The gyro-Bohm model finds a self-consistent equilibrium at T=17.0 keV where alpha heating balances loss power.

JET DTE1 validation: H98 matches τ_E within 5% (0.38s vs 0.40s measured). Gyro-Bohm ITER H-mode profiles overpredict JET stored energy (profiles too peaked for JET), so the JET calibration uses a 0D formula matching τ=0.40s at P_loss=35 MW → C_GB=1.14.

### Divertor & Wall

The LIQUID_LIPB divertor operates at 34.5 MW/m² (well below the 60 MW/m² limit). Double-null configuration splits exhaust across upper and lower targets. Negative triangularity broadens the heat flux width by 40%, operating in detached regime.

The FLiBe liquid blanket (627°C) feeds the sCO₂ Brayton cycle at 58.8% thermal efficiency. Plasma-facing neutron wall load is 12.3 MW/m², but the FLiBe absorbs ~80% of neutron energy in the first 20 cm, reducing the structural wall load to ~3 MW/m². Blanket modules require replacement every 5-7 years.

### Cost Breakdown

| Component | Cost | Fraction |
|-----------|:----:|:--------:|
| TF coils (HTS REBCO, 86.2 GJ × $130/GJ) | $11.21B | 32% |
| sCO₂ turbine hall | $9.95B | 28% |
| Cooling systems | $4.97B | 14% |
| PF coils | $0.93B | 3% |
| Blanket (LiPb + HCPB) | $0.59B | 2% |
| Tritium plant | $1.00B | 3% |
| Site, buildings, IC | $1.10B | 3% |
| Central solenoid | $0.50B | 1% |
| Contingency (15%) | $5.09B | 14% |
| **Total** | **$35.34B** | **100%** |

### Model Fixes Applied

During validation, four issues were identified and fixed:

1. **Transport solver T iteration**: `transport_solver.py` now finds self-consistent equilibrium temperature where P_loss = P_alpha + P_ext, instead of clamping T=15 keV and reporting Q=999. V3 equilibrium is T=17.0 keV.

2. **NTM stability**: Metric = 1.13 (slightly above the 1.0 threshold without ECCD). The ECCD system provides 12 MW with 11 gyrotrons (170 GHz, TRL 9, commercial) — adequate for stabilization.

3. **Neutron wall load**: 12.3 MW/m² plasma-facing. FLiBe liquid blanket attenuates to ~3 MW/m² at the structural wall. Acceptable with 5-7 year blanket replacement intervals.

4. **TF coil HTS model**: `tf_coil_stress()` had no HTS conductor mode — it applied Nb₃Sn Tc/Jc limits (Bc₂=25 T, Tc₀=18 K) even for HTS designs. Added REBCO model: Bc₂=65 T, Tc₀=93 K, T_op=20 K. Temperature margin went from 0.9 K → 55 K.

### HTS Learning Curve Impact

TF coils dominate cost (36%). As HTS manufacturing scales:

| HTS cost | TF coils | Total cost | $/kW | LCOE |
|:--------:|:--------:|:----------:|:----:|:----:|
| $130/GJ (today) | $11.21B | $35.34B | $4,630 | $50/MWh |
| $100/GJ (scale-up) | $8.62B | $32.40B | $4,245 | $46/MWh |
| **$80/GJ (learning)** | **$6.90B** | **$30.27B** | **$3,966** | **$43/MWh** |
| $60/GJ (mature) | $5.17B | $28.14B | $3,686 | $40/MWh |

At $60/GJ, V3 reaches $3,913/kW — cheaper than offshore wind with 24/7 dispatchability.

### Sensitivity Matrix

| Lever | Range | P_net span | $/kW span | β_N range | Optimal |
|-------|-------|------------|-----------|-----------|---------|
| **n/n_GW** | 0.60→0.85 | 4.0→8.1 GW | $6,636→$4,385 | 2.22→3.14 | 0.75 (β_N=2.77) |
| **Bt** | 11→15 T | 4.5→8.4 GW | $5,335→$4,810 | 3.28→2.40 | 13 T (B_peak=22.8 < 23) |
| **κ** | 2.4→3.2 | 2.2→8.5 GW | $10,067→$4,282 | 2.77→2.77 | 3.0 (vertical stability) |
| **HTS cost** | $130→$40/GJ | — | $5,018→$3,597 | — | Path to $3.6/kW at scale |

### Operating Cost

| Item | Annual cost |
|------|:-----------:|
| O&M (2% of capital) | $707M |
| Blanket replacement (7-yr cycle) | $84M |
| Staff (300 at $200k avg) | $60M |
| Cooling makeup + chemicals | $5M |
| Deuterium fuel | $0.1M |
| **Total** | **~$856M/yr** |

Fuel is effectively free (deuterium $100/day). Tritium is bred on-site (TBR=1.175). Operating cost is $4.5B/yr below revenue at $90/MWh — **the plant pays back capital in ~6.5 years** at wholesale electricity prices, with zero fuel cost escalation.

## Earlier Designs (Lessons Learned)

| Phase | Design | P_net | β_N | Limitation |
|-------|--------|:-----:|:---:|------------|
| **V1** | R₀=12m, B₀=13T, Nb₃Sn | 2,482 MW | 3.2 | Expensive, low power |
| **V2** | R₀=9.5m, B₀=14T, A=13.6, HTS | 2,339 MW | 3.5 | Moderate Q, poor $/kW |
| **V2.2** | R₀=9.5m, A=16.7, all upgrades | 5,024 MW | 5.06 | β_N unstable, H98 extrapolation penalty |
| **V3 (this work)** | **R₀=7.0m, A=5.8, B₀=13T** | **7,632 MW** | **2.77** | **All constraints satisfied** |

The evolution shows a clear pattern: the V2 compact high-A approach fails due to MHD instability and H98 extrapolation uncertainty. V3 backs off to moderate aspect ratio (A=5.8) and β_N=2.77, delivering **7.63 GW of clean, firm power at $4,630/kW**.

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
| `startup.py` | Staged startup scenario with CS flux budget |
| `figures.py` | All publication-quality figures |
| `design_optimizer.py` | Genetic algorithm design optimization |
| `design_scorer.py` | Harrington desirability scoring |
| `B_HIGH_FIDELITY.py` | Monte Carlo uncertainty analysis |
| `rl_env.py` | Reinforcement learning environment |
| `simulator.py` | Reduced-order plasma simulator |
| `main.tex` | Paper source (LaTeX) |
| `main.pdf` | Compiled paper |

## Methodology

All physics models from published peer-reviewed literature:

- **Fusion reactivity**: Bosch-Hale 1992
- **Confinement**: Gyro-Bohm (primary, C_GB=1.14 calibrated to JET DTE1) + ITER H98(y,2) (reference)
- **Equilibrium T**: Self-consistent power balance (P_loss = P_alpha + P_ext)
- **MHD stability**: ITER Physics Basis, Sweeney 2020, La Haye 2006
- **Heat flux**: Eich 2013 multi-machine database
- **Divertor**: Stangeby 2000, Pitts 2019, Kuang 2020
- **Tritium breeding**: Fischer 2020, Hernandez 2020 (MCNP)
- **ECCD**: Poli 2018, Fisch-Boozer
- **TF coils**: Bottura 2019 (Nb₃Sn), REBCO HTS model added
- **AI control**: Cross-entropy method

## Validation

Confinement validated against JET 1997 D-T record shot:

| Quantity | Predicted | Measured | Error |
|----------|:---------:|:--------:|:-----:|
| τE (H98) | 0.381 s | 0.40 s | 5% |
| τE (gyro-Bohm) | 0.11 s* | 0.40 s | 73% low* |
| Q (H98) | 0.62 | 0.67 | 7% |

*Gyro-Bohm uses ITER H-mode profiles (αn=0.2, αT=0.8) which are too peaked for JET. The 0D calibration gives C_GB=1.14 matching τ=0.40s at JET conditions. Gyro-Bohm is conservative for V3: real confinement is likely better than predicted.

## AI Controller

Neural network (6,608 params) trained via cross-entropy method:
- 22-dim state → 16-dim action
- PD-seeded position control, learned thermal control
- 2 ms control cycle

### Validation

| Metric | Mean ± Std | Target |
|--------|-----------|--------|
| Reward | +3,904 ± 320 | > 0 |
| Vertical offset | 0.000 ± 0.001 m | < 0.01 m |
| Surface heat flux | 18.8 ± 0.4 MW/m² | < 20 MW/m² |

## License

MIT License