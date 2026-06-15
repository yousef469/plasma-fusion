"""
Plasma response simulator for AI control training.
Models RZIp + thermal + divertor dynamics for our tokamak design.

Physics basis: ITER plasma control design approach (Walker 2015, 
Ambrosino 2020, Albanese 2021). Uses linearized plasma response
around equilibrium + nonlinear divertor model.

All equations from published fusion control literature.
"""
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional, Callable
import json

# ============================================================
# Plasma parameters (from SNOWFLAKE_SPEC.md)
# ============================================================
@dataclass
class PlasmaParams:
    R0: float = 12.08       # m
    a: float = 0.96          # m
    kappa: float = 2.71      # elongation
    delta: float = 0.5       # triangularity (approx)
    BT: float = 11.7         # T
    Ip: float = 20e6         # A
    Bp_avg: float = 1.8      # T (poloidal field ~ mu0*Ip/(2*pi*a*sqrt((1+kappa^2)/2)))
    volume: float = 660      # m^3
    
    # Derived
    @property
    def R_self_inductance(self) -> float:
        """Plasma internal inductance ~ mu0*R0*(ln(8R0/a) - 2 + li/2)"""
        li = 0.8  # internal inductance
        return 4e-7 * np.pi * self.R0 * (np.log(8*self.R0/self.a) - 2 + li/2)
    
    @property
    def tau_E(self) -> float:
        """Energy confinement time from H98 scaling (s)"""
        return 3.5  # validated from our 0D engine
    
    @property
    def W_th(self) -> float:
        """Thermal stored energy (J) = 3/2 * <n> * <T> * V"""
        return 1200e6  # 1.2 GJ from our design
    
    @property
    def L_p(self) -> float:
        """Plasma inductance (H)"""
        return self.R_self_inductance
    
    @property
    def tau_growth_vertical(self) -> float:
        """Vertical instability growth time (s).
        For kappa=2.71, typical growth time ~50-200ms for ITER-class.
        Larger R0 gives slower growth.
        """
        return 0.15  # 150 ms
    
    @property
    def tau_position(self) -> float:
        """Radial position response time (s)"""
        return 0.05  # 50 ms
    
    @property
    def tau_current(self) -> float:
        """Plasma current diffusion time (s)"""
        return 5.0  # L/R time


@dataclass
class CoilParams:
    """PF coil positions and electrical parameters"""
    # Positions from Module 1 refined design
    positions: list = field(default_factory=lambda: [
        {"name": "PF1U", "R": 2.5, "Z": 2.5, "turns": 100},
        {"name": "PF2U", "R": 4.5, "Z": 5.0, "turns": 100},
        {"name": "PF3U", "R": 7.0, "Z": 6.5, "turns": 100},
        {"name": "PF4U", "R": 10.0, "Z": 5.5, "turns": 100},
        {"name": "PF5U", "R": 12.5, "Z": 3.0, "turns": 100},
        {"name": "PF6U", "R": 14.0, "Z": 1.0, "turns": 100},
        {"name": "PF1L", "R": 2.5, "Z": -2.5, "turns": 100},
        {"name": "PF2L", "R": 4.5, "Z": -5.0, "turns": 100},
        {"name": "PF3L", "R": 7.0, "Z": -6.5, "turns": 100},
        {"name": "PF4L", "R": 10.0, "Z": -5.5, "turns": 100},
        {"name": "PF5L", "R": 12.5, "Z": -3.0, "turns": 100},
        {"name": "PF6L", "R": 14.0, "Z": -1.0, "turns": 100},
    ])
    
    # Electrical parameters (per turn)
    R_turn: float = 1e-3      # Ohm/turn (NbTi at 4K)
    L_turn: float = 1e-6      # H/turn
    
    @property
    def n_coils(self) -> int:
        return len(self.positions)
    
    @property
    def names(self) -> list:
        return [c["name"] for c in self.positions]


@dataclass
class VSCoilParams:
    """Fast in-vessel vertical stabilization (VS) coils.
    
    Two small coil pairs at Z = ±1.5m, inside the vessel, close to the plasma.
    Powered differentially: upper gets +V, lower gets -V.
    Provides ~0.5ms response, 300× faster than vertical growth time (150ms).
    
    Design: 20 turns, CuCrZr conductor, 0.3m coil radius.
    Same approach as ITER in-vessel VS coils (Albanese 2021, Walker 2015).
    """
    L_vs: float = 50e-6       # H (total inductance)
    R_vs: float = 0.1         # Ohm (limits I_ss = 20kA at 2kV, matches supply)
    tau_mult: float = 1.0     # optional tau multiplier for tuning
    V_max: float = 2000.0     # V (±2 kV supply)
    I_max: float = 20000.0    # A (±20 kA supply)
    eta_vs: float = 5.0e-6    # m/s/A (stabilization efficiency)
    
    # Position: inside vessel, just above/below midplane
    coil_R: float = 11.5      # m (coil center radius)
    Z_upper: float = 1.5      # m (upper coil)
    Z_lower: float = -1.5     # m (lower coil)
    
    @property
    def tau(self) -> float:
        """L/R time constant (s)"""
        return self.L_vs / self.R_vs * self.tau_mult


# ============================================================
# Plasma Response Model (RZIp + β + divertor)
# ============================================================
class PlasmaSimulator:
    """Nonlinear ODE model of plasma dynamics for control design.
    
    State vector:
        [0]   δR    - Radial position offset (m)
        [1]   δZ    - Vertical position (m)
        [2]   δIp   - Plasma current offset (A)
        [3]   δβp   - Poloidal beta offset
        [4]   δne   - Density offset (m^-3)
        [5-16] I_coils - PF coil currents (A) for 12 coils
    
    Control inputs:
        PF coil voltages (V) for each coil
        NB/ICRF heating power offset (W)
        Gas fueling rate (atoms/s)
        Impurity seeding rate (Ar atoms/s)
    """
    
    def __init__(self, plasma: PlasmaParams = None, coils: CoilParams = None, vs_coils: VSCoilParams = None, has_vs: bool = True):
        self.plasma = plasma or PlasmaParams()
        self.coils = coils or CoilParams()
        self.vs_coils = vs_coils or VSCoilParams()
        self.has_vs = has_vs
        self.n_coils = self.coils.n_coils
        self.n_vs = 1 if has_vs else 0  # one differential VS pair
        self.n_states = 5 + self.n_coils + self.n_vs  # R, Z, Ip, βp, ne, + PF coils + VS coil
        
    def compute_mutual_inductance(self, Rc, Zc):
        """Mutual inductance between plasma and a PF coil at (Rc, Zc).
        Approximate: M = mu0*R0 * (ln(8R0/a) - 2) * (Rc/R0)**0.5 * ... etc.
        From standard tokamak circuit theory.
        """
        R0, a = self.plasma.R0, self.plasma.a
        dR = Rc - R0
        dZ = Zc
        
        # Simplified mutual inductance
        r = np.sqrt(dR**2 + dZ**2)
        if r < a:
            return 4e-7 * np.pi * R0 * np.sqrt(Rc/R0) * (np.log(8*R0/a) - 2) * np.exp(-r/a)
        else:
            return 4e-7 * np.pi * R0 * np.sqrt(Rc/R0) * (np.log(8*R0/a) - 2) * (a/r)**2
    
    def equilibrium_bfield(self, R, Z):
        """Equilibrium poloidal field (T) from PF coils at operating point.
        Returns (Br, Bz) at (R, Z).
        """
        # This is a linearized model around the equilibrium.
        # The equilibrium Bz needed for radial force balance:
        Bz_equil = 4e-7 * np.pi * self.plasma.Ip / (2 * np.pi * R) * \
                   (np.log(8*R/self.plasma.a) + self.plasma.kappa/2 - 1.5)
        return 0.0, Bz_equil
    
    def derivatives(self, t, state, controls, noise=None, vs_dt=None):
        """Compute state derivatives.
        
        controls: dict with keys 'V_coils' (n_coils array), 'V_vs' (scalar),
                  'P_heat', 'fuel', 'seed'
        vs_dt: if provided, uses analytic L/R circuit solution for VS coil
               (required when vs_dt > vs.tau/2 for numerical stability)
        """
        R0, a, kappa, BT, Ip0 = self.plasma.R0, self.plasma.a, self.plasma.kappa, self.plasma.BT, self.plasma.Ip
        tau_E = self.plasma.tau_E
        W_th = self.plasma.W_th
        
        # Unpack state
        dR = state[0]      # radial offset
        dZ = state[1]      # vertical offset  
        dIp = state[2]     # plasma current offset
        dbp = state[3]     # beta_p offset
        dne = state[4]     # density offset
        I_coils = state[5:5+self.n_coils] # PF coil currents
        if self.has_vs:
            I_vs = state[5+self.n_coils]  # VS differential current
        else:
            I_vs = 0.0
        
        # Unpack controls
        V_coils = controls.get('V_coils', np.zeros(self.n_coils))
        V_vs = controls.get('V_vs', 0.0)
        P_heat = controls.get('P_heat', 0.0)      # MW
        fuel = controls.get('fuel', 0.0)  # atoms/s
        seed = controls.get('seed', 0.0)  # Ar atoms/s
        
        # === COIL CIRCUIT EQUATIONS ===
        R_plasma = 1e-8 * self.plasma.R0 * 2 * np.pi / (np.pi * a**2)
        L_p = self.plasma.L_p
        
        dI_coils = np.zeros(self.n_coils)
        for i in range(self.n_coils):
            R_i = self.coils.R_turn * self.coils.positions[i]['turns']
            L_i = self.coils.L_turn * self.coils.positions[i]['turns']
            dI_coils[i] = (V_coils[i] - R_i * I_coils[i]) / (L_i + 1e-10)
        
        # VS coil circuit: fast L/R circuit
        # Differential pair: upper +V, lower -V
        # Uses analytic solution for numerical stability when vs_dt is large
        vs = self.vs_coils
        if self.has_vs and vs_dt is not None and vs_dt > 0:
            # Exact L/R circuit solution for constant V over timestep
            I_ss = np.clip(V_vs / vs.R_vs, -vs.I_max, vs.I_max)
            I_vs_new = I_ss + (I_vs - I_ss) * np.exp(-vs_dt / vs.tau)
            dI_vs = (I_vs_new - I_vs) / vs_dt
        else:
            # Explicit Euler (OK for small dt/tau)
            dI_vs_clamped = (V_vs - vs.R_vs * I_vs) / (vs.L_vs + 1e-12)
            if abs(I_vs) >= vs.I_max and I_vs * dI_vs_clamped > 0:
                dI_vs = 0.0
            else:
                dI_vs = dI_vs_clamped
        
        V_loop = controls.get('V_loop', 0.0)
        
        # === RADIAL POSITION (first-order response) ===
        tau_R = self.plasma.tau_position
        gamma_R = 1.0 / tau_R
        k_R = 4e-7 * np.pi * Ip0**2 / (2 * np.pi * R0**2)
        Bz_equil = self.equilibrium_bfield(R0, 0)[1]
        dR_dt = -gamma_R * dR
        
        # === VERTICAL POSITION (unstable mode) ===
        tau_v = 0.15  # vertical growth time (150ms)
        gamma_v = 1.0 / tau_v  # ~6.67 s⁻¹
        
        # Stabilizing force from differential PF coil current (upper-lower)
        eta_stab = 1.5e-6  # m/s/A (50kA → 0.075 m/s)
        I_diff = np.mean(I_coils[:6]) - np.mean(I_coils[6:])
        
        # + additional stabilization from fast VS coils
        # VS coils at Z=±1.5m provide faster but limited authority control
        eta_vs = vs.eta_vs if self.has_vs else 0.0
        
        # Combined stabilization: PF differential + VS fast
        dZ_dt = gamma_v * dZ - eta_stab * I_diff - eta_vs * I_vs
        
        # === PLASMA CURRENT ===
        dIp_dt = (V_loop - R_plasma * (Ip0 + dIp)) / L_p
        
        # === BETA POLOIDAL ===
        dp_heat = P_heat * 1e6  # MW to W
        dp_loss = (W_th / tau_E) * dbp
        dbp_dt = (dp_heat - dp_loss) / W_th
        
        # === DENSITY ===
        tau_p = tau_E / 2
        n0 = 1.2e20  # m^-3 (design density)
        dne_dt = (fuel / self.plasma.volume - (n0 + dne) / tau_p)
        
        # === ADD NOISE ===
        if noise is not None:
            dR_dt += noise[0]
            dZ_dt += noise[1]
            dIp_dt += noise[2]
            dbp_dt += noise[3]
            dne_dt += noise[4]
            dI_coils += noise[5:5+self.n_coils]
        
        # === Assemble derivative vector ===
        derivs = [dR_dt, dZ_dt, dIp_dt / 1e6, dbp_dt, dne_dt]
        derivs.extend(dI_coils)
        if self.has_vs:
            derivs.append(dI_vs)
        
        return np.array(derivs)
    
    def compute_Bz(self, R, Z, I_coils):
        """Total Bz at (R, Z) from all PF coils"""
        Bz = 0.0
        for i, c in enumerate(self.coils.positions):
            Rc, Zc = c['R'], c['Z']
            dz = Z - Zc
            dr = R - Rc
            r = np.sqrt(dr**2 + dz**2)
            if r < 0.01:
                continue
            # Simplified: dipole approximation for r >> Rc
            Bz += 4e-7 * np.pi * I_coils[i] * Rc**2 / (2 * (Rc**2 + dz**2)**1.5)
        return Bz
    
    def compute_dBzdZ(self, R, Z, I_coils):
        """Vertical derivative of Bz at (R, Z)"""
        dBzdZ = 0.0
        for i, c in enumerate(self.coils.positions):
            Rc, Zc = c['R'], c['Z']
            dz = Z - Zc
            dr = R - Rc
            dist2 = dr**2 + dz**2
            if dist2 < 0.01:
                continue
            # dBz/dZ = -3*mu0*I*Rc^2*dz / (2*(Rc^2+dz^2)^(5/2))
            dBzdZ += -3 * 4e-7 * np.pi * I_coils[i] * Rc**2 * dz / (2 * (Rc**2 + dz**2)**2.5)
        return dBzdZ
    
    def compute_divertor(self, state, controls):
        """Compute divertor conditions from current state.
        Returns dict with q_surf, T_target, n_target, f_rad.
        """
        R0, a, kappa = self.plasma.R0, self.plasma.a, self.plasma.kappa
        dR = state[0]
        dZ = state[1]
        dIp = state[2]
        dbp = state[3]
        dne = state[4]
        
        seed = controls.get('seed', 0.0)
        
        # Design: q_parallel = 549 MW/m^2 (from Module 3 SOLPS analysis)
        # For P_SOL = 200 MW, R0 = 12.08m → effective SOL area = 0.364 m^2
        q_parallel_design = 549.0  # MW/m^2 (at outer midplane)
        
        # Flux expansion factor (snowflake: ~10× for our design)
        # At operating point: q_par=549, f_rad=0.65, f_flux=10 → q_surf=19.2
        f_flux_base = 10.0
        f_flux = max(3.0, f_flux_base - 5.0 * abs(dZ) / 0.1)
        
        # Radiation fraction: baseline 30% (core) + seeded from divertor
        # Ar seeding: seed ~ 2e22 atoms/s gives f_rad_extra ~ 0.35 → f_rad ≈ 0.65
        f_rad_base = 0.3
        f_rad_seeded = seed * 5e-24  # 1e22 Ar/s → f_rad +0.05
        f_rad = min(0.85, f_rad_base + f_rad_seeded)
        
        # Surface heat flux with radiation + flux expansion
        q_surf = q_parallel_design * (1 - f_rad) / f_flux
        
        # 2-point model calibrated to Module 3 results
        # Attached (f_rad=0.3): T_target ≈ 100-200 eV
        # Detached (f_rad=0.65): T_target ≈ 2 eV
        T_u = 200.0  # eV upstream
        T_det = 2.0  # eV at detachment target
        f_det = 0.65  # f_rad at detachment onset
        
        # Effective detachment fraction
        f_det_eff = max(0, min(1, (f_rad - 0.3) / (f_det - 0.3)))
        T_target = T_u * (1 - f_det_eff)**3 + T_det
        T_target = max(1.5, min(T_target, T_u))
        
        # Density at target (from particle conservation)
        n_u = 1e20  # m^-3 upstream
        n_target = n_u * np.sqrt(T_u / (T_target + 0.1))
        
        # Connection length (snowflake: longer)
        B_pol = self.plasma.Bp_avg * (1 + dbp)
        L_c = 2 * np.pi * R0 * self.plasma.BT / (B_pol + 1e-3) * 1.5
        
        # Parallel heat flux at target
        q_par_target = q_parallel_design * (1 - f_rad)
        
        return {
            'q_parallel': q_parallel_design,
            'q_surf': q_surf,
            'T_target': T_target,
            'n_target': n_target,
            'f_rad': f_rad,
            'f_flux': f_flux,
        }
    
    def simulate(self, t_span, dt, initial_state, control_policy, noise_std=None):
        """Run simulation with a given control policy.
        
        control_policy: function(state, t) -> controls dict
        """
        n_steps = int((t_span[1] - t_span[0]) / dt)
        state = np.array(initial_state, dtype=float)
        
        trajectory = []
        for step in range(n_steps):
            t = t_span[0] + step * dt
            
            # Get controls
            controls = control_policy(state, t)
            # Ensure V_vs exists even for policies that don't use it
            if 'V_vs' not in controls:
                controls['V_vs'] = 0.0
            
            # Compute noise
            noise = None
            if noise_std is not None:
                noise = np.random.randn(self.n_states) * noise_std
            
            # Step ODE (use analytic VS update for stability)
            dstate = self.derivatives(t, state, controls, noise, vs_dt=dt)
            state = state + dstate * dt
            
            # Get divertor state
            divertor = self.compute_divertor(state, controls)
            
            trajectory.append({
                't': t,
                'state': state.copy(),
                'controls': controls,
                'divertor': divertor.copy(),
            })
        
        return trajectory


# ============================================================
# Equilibrium operating point
# ============================================================
class EquilibriumPoint:
    """Defines the desired operating point for the controller."""
    
    def __init__(self, plasma: PlasmaParams = None):
        self.p = plasma or PlasmaParams()
        
        # Target values
        self.R_target = self.p.R0
        self.Z_target = 0.0
        self.Ip_target = self.p.Ip
        self.beta_p_target = 1.5  # design beta_p
        self.ne_target = 1.2e20   # m^-3
        
        # Divertor targets
        self.T_target_min = 2.0   # eV (below this = detached, OK)
        self.q_surf_max = 20.0    # MW/m^2
        self.f_rad_target = 0.65  # detachment level
        
        # Limits (for safety layer)
        self.Z_max = 0.3   # m before hitting wall
        self.R_min = self.p.R0 - 0.5 * self.p.a
        self.R_max = self.p.R0 + 0.5 * self.p.a
        self.Ip_max = self.p.Ip * 1.2
        self.beta_N_max = 3.5  # Troyon limit


# ============================================================
# Random disturbance generator for training
# ============================================================
class DisturbanceGenerator:
    """Generates random disturbances to train the RL controller."""
    
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
    
    def random_initial_state(self, plasma: PlasmaParams):
        """Random initial offset from equilibrium (startup scenario)."""
        return np.array([
            self.rng.uniform(-0.1, 0.1),  # dR (m)
            self.rng.uniform(-0.05, 0.05),  # dZ (m)
            self.rng.uniform(-1e6, 1e6),  # dIp (A)
            self.rng.uniform(-0.1, 0.1),  # dβp
            self.rng.uniform(-0.1e20, 0.1e20),  # dne (m^-3)
        ])
    
    def random_disturbance(self):
        """Random disturbance (ELM, sawtooth, etc.)"""
        return {
            'type': self.rng.choice(['ELM', 'sawtooth', 'density_puff', 'none']),
            'magnitude': self.rng.uniform(0.01, 0.1),
            'duration': self.rng.uniform(0.001, 0.1),
        }


# Quick test
if __name__ == '__main__':
    plasma = PlasmaParams()
    sim = PlasmaSimulator(plasma)
    
    def make_pole_placement(sim):
        """Factory: control policy with equilibrium voltage offset.
        
        Physics: vertical instability has growth rate γ = 6.67 s⁻¹.
        At dZ=1cm, dZ/dt = 0.067 m/s. Need I_diff ≈ 44.5kA differential
        for PF stabilization (η=1.5e-6).
        
        With R_coil = 0.1Ω, I = V/R, so V_diff_needed = 44.5kA * 0.1Ω = 4.45kV.
        Set V_eq ≈ 500V (for 5kA nominal) to leave ±4.5kV headroom for control.
        """
        R_coil = sim.coils.R_turn * sim.coils.positions[0]['turns']  # 0.1Ω
        I_nominal = 5000.0  # 5kA nominal (gives V_eq = 500V)
        V_eq = R_coil * I_nominal
        
        # Gains: at 1cm offset, produce ~45kA differential
        Kz = 2.5e5    # V/m (at 1cm: 2500V → 25kA diff per coil → 50kA total diff)
        Kr = 1e5      # V/m (radial control)
        
        def control(state, t):
            dR, dZ = state[0], state[1]
            
            V_ctrl = np.zeros(12)
            V_ctrl[:6] = Kz * dZ + Kr * dR
            V_ctrl[6:] = -Kz * dZ + Kr * dR
            V_ctrl = np.clip(V_ctrl, -4500, 4500)  # leave room for V_eq
            
            V_coils = np.clip(V_eq + V_ctrl, -5000, 5000)
            
            # VS coil: full-throttle proportional (200kV/m, saturates at 1cm)
            K_vs = 2e5    # V/m (at 1cm: 2000V → 20kA → η=0.1 m/s)
            V_vs = np.clip(K_vs * dZ, -2000, 2000)
            
            return {
                'V_coils': V_coils,
                'V_vs': V_vs,
                'P_heat': 0.0,
                'fuel': 0.0,
                'seed': 0.0,
                'V_loop': 0.0,
            }
        return control
    
    pole_placement = make_pole_placement(sim)
    
    print(f"VS coils: tau={sim.vs_coils.tau*1e3:.2f}ms, "
          f"I_max={sim.vs_coils.I_max/1000:.0f}kA, "
          f"eta={sim.vs_coils.eta_vs:.1e} m/s/A")
    print(f"PF coils: R={sim.coils.R_turn * sim.coils.positions[0]['turns']}Ω, "
          f"V_supply=±5000V, V_eq=500V (I_nom=5kA)")
    
    # Test 1: Vertical perturbation with combined PF + VS feedback
    print("\n" + "=" * 60)
    print("Test 1: Vertical perturbation (1cm) with PF + VS feedback")
    print("=" * 60)
    
    I_nom = 5000.0  # sustainable PF coil current (A, at V_eq=500V)
    initial = np.zeros(sim.n_states)
    initial[5:5+12] = I_nom
    if sim.has_vs:
        initial[5+12] = 0.0
    initial[1] = 0.01
    
    traj = sim.simulate([0, 5.0], 0.001, initial, pole_placement)
    
    z_vals = np.array([s['state'][1] for s in traj])
    vs_vals = np.array([s['state'][17] for s in traj]) if sim.has_vs else np.array([])
    vs_vals_hist = [s['state'][17] for s in traj] if sim.has_vs else []
    final_z = z_vals[-1]
    max_z = np.max(np.abs(z_vals))
    max_vs = np.max(np.abs(vs_vals)) / 1000 if len(vs_vals) > 0 else 0
    
    # Convergence check: last 10% of samples
    n_check = max(10, len(z_vals) // 10)
    final_segment = z_vals[-n_check:]
    converged = np.std(final_segment) < 1e-4 and np.abs(np.mean(final_segment)) < 0.001
    
    print(f"Initial Z offset: 0.010 m")
    print(f"Final Z offset: {final_z:.6f} m")
    print(f"Max Z deviation: {max_z:.6f} m")
    print(f"Max VS current: {max_vs:.1f} kA")
    print(f"Final VS current: {vs_vals_hist[-1]/1000:.1f} kA" if vs_vals_hist else "", "")
    print(f"Stable: {'YES' if converged else 'NO'} (std={np.std(final_segment):.6f})")
    
    # Test 2: VS-only stabilization (no PF differential)
    print("\n" + "=" * 60)
    print("Test 2: VS-only vertical stabilization (no PF differential)")
    print("=" * 60)
    
    def vs_only_control(state, t):
        dZ = state[1]
        R_coil = sim.coils.R_turn * sim.coils.positions[0]['turns']
        V_coils = np.full(12, R_coil * I_nom)  # same voltage on all = no diff
        V_vs = np.clip(2e5 * dZ, -2000, 2000)
        return {'V_coils': V_coils, 'V_vs': V_vs, 'P_heat': 0.0, 'fuel': 0.0, 'seed': 0.0, 'V_loop': 0.0}
    
    initial_vs = np.zeros(sim.n_states)
    initial_vs[5:5+12] = I_nom
    initial_vs[1] = 0.01
    if sim.has_vs:
        initial_vs[5+12] = 0.0
    
    traj_vs = sim.simulate([0, 2.0], 0.001, initial_vs, vs_only_control)
    z_vs = np.array([s['state'][1] for s in traj_vs])
    z_vs_seg = z_vs[-max(10, len(z_vs)//10):]
    vs_stable = 'YES' if np.std(z_vs_seg) < 1e-4 and abs(np.mean(z_vs_seg)) < 0.001 else 'NO'
    print(f"VS-only final Z: {z_vs[-1]:.6f} m, max: {np.max(np.abs(z_vs)):.6f} m, "
          f"stable: {vs_stable}")
    
    # Test 3: Radial perturbation
    print("\n" + "=" * 60)
    print("Test 3: Radial perturbation (5cm) with feedback")
    print("=" * 60)
    
    initial2 = np.zeros(sim.n_states)
    initial2[5:5+12] = I_nom
    initial2[0] = 0.05
    
    traj2 = sim.simulate([0, 5.0], 0.001, initial2, pole_placement)
    r_vals = [s['state'][0] for s in traj2]
    print(f"Initial R offset: 0.050 m")
    print(f"Final R offset: {r_vals[-1]:.6f} m")
    
    # Save params
    with open('SIMULATOR_PARAMS.json', 'w') as f:
        json.dump({
            'R0': plasma.R0, 'a': plasma.a, 'kappa': plasma.kappa,
            'BT': plasma.BT, 'Ip': plasma.Ip, 'volume': plasma.volume,
            'tau_E': plasma.tau_E, 'tau_growth': plasma.tau_growth_vertical,
            'W_th': plasma.W_th, 'n_states': sim.n_states,
            'has_vs': sim.has_vs, 'vs_tau': sim.vs_coils.tau,
            'vs_I_max': sim.vs_coils.I_max, 'vs_V_max': sim.vs_coils.V_max,
        }, f, indent=2)
    print(f"VS coils tau={sim.vs_coils.tau*1e3:.2f}ms, I_max={sim.vs_coils.I_max/1000:.0f}kA")
    print("\nSaved SIMULATOR_PARAMS.json")
