"""
RL Gymnasium environment for tokamak plasma control.
Wraps the PlasmaSimulator for reinforcement learning.

State:  [dR, dZ, dIp, dbeta_p, dne, I_coils(12), I_vs, q_surf, T_target, f_rad]  (22 dims)
Action: [V_coils(12), V_vs, P_heat, fuel_rate, seed_rate]  (16 dims)
Reward: negative weighted error + penalties for limit violations

Based on: DeepMind TCV control (Degrave et al. 2022, Nature)
"""
import numpy as np
from simulator import PlasmaSimulator, PlasmaParams, EquilibriumPoint
from gymnasium import Env, spaces
from gymnasium.envs.registration import register
from typing import Optional
import json, os

# ============================================================
# Plasma Control Environment
# ============================================================
class PlasmaControlEnv(Env):
    """Tokamak plasma control RL environment."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, render_mode=None, seed=42):
        super().__init__()
        
        self.plasma = PlasmaParams()
        self.sim = PlasmaSimulator(self.plasma)
        self.eq = EquilibriumPoint(self.plasma)
        self.rng = np.random.default_rng(seed)
        
        self.render_mode = render_mode
        
        # State space bounds (includes VS coil current)
        self.n_coils = self.sim.n_coils
        self.has_vs = self.sim.has_vs
        self.n_vs = self.sim.n_vs
        I_vs_max = self.sim.vs_coils.I_max if self.has_vs else 0
        n_states = 5 + self.n_coils + self.n_vs + 3  # plasma + PF + VS + divertor
        
        high = np.array([1.0, 0.5, 10e6, 2.0, 5e20] + 
                        [10e6]*self.n_coils +
                        [I_vs_max]*self.n_vs +
                        [50, 200, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, shape=(n_states,), dtype=np.float32)
        
        # Action space: PF coil voltages + VS voltage + heating power + fuel + Ar seed
        V_vs_max = self.sim.vs_coils.V_max if self.has_vs else 0
        n_actions = self.n_coils + self.n_vs + 3
        self.action_space = spaces.Box(
            np.array([-5000]*self.n_coils + [-V_vs_max]*self.n_vs + [-50, 0, 0], dtype=np.float32),
            np.array([+5000]*self.n_coils + [+V_vs_max]*self.n_vs + [+50, +1e23, +1e23], dtype=np.float32),
            dtype=np.float32
        )
        
        # Time step
        self.dt = 0.002  # 2ms control cycle
        
        # Episode tracking
        self.state = None
        self.step_count = 0
        self.max_steps = 5000  # 10s at 2ms
        
        # Reward weights
        self.w = {
            'R': 50.0,
            'Z': 200.0,
            'Ip': 0.05,
            'beta': 5.0,
            'ne': 0.001,
            'q_surf': 1.0,
            'T_target': 2.0,
            'V_coils': 0.001,
        }
        
        # Divertor targets (snowflake: q_surf < 20 MW/m^2, T_target 2-15 eV)
        self.q_surf_design = 19.2
        self.q_surf_warn = 22.0
        self.q_surf_hard = 45.0  # allow agent time to seed and radiate
        self.T_target_min = 1.5
        self.T_target_max = 20.0
    
    def _get_obs(self, state_plasma, divertor):
        """Build observation from simulator state + divertor.
        State_plasma includes PF coils + VS coil (if has_vs).
        """
        return np.concatenate([
            state_plasma[:5],                     # dR, dZ, dIp, dbp, dne
            state_plasma[5:5+self.n_coils],        # I_coils (PF)
            state_plasma[5+self.n_coils:],         # I_vs (if has_vs)
            [divertor['q_surf'], divertor['T_target'], divertor['f_rad']]
        ]).astype(np.float32)
    
    def _compute_reward(self, state_plasma, divertor):
        """Compute reward signal.
        
        Positive reward for being within operating limits.
        Negative reward for violations.
        """
        dR, dZ, dIp, dbp, dne = state_plasma[:5]
        
        reward = 0.0
        
        # Position tracking (want zero offset)
        reward -= self.w['R'] * dR**2
        reward -= self.w['Z'] * dZ**2
        reward -= self.w['Ip'] * (dIp / 1e6)**2
        reward -= self.w['beta'] * dbp**2
        reward -= self.w['ne'] * (dne / 1e20)**2
        
        # Divertor health
        q_surf = divertor['q_surf']
        T_target = divertor['T_target']
        
        # Penalize deviation from design q_surf (soft, allows transient high q)
        reward -= self.w['q_surf'] * min(1, (q_surf - self.q_surf_design)**2 / self.q_surf_design**2)
        
        if q_surf > self.q_surf_warn:
            reward -= self.w['q_surf'] * (q_surf - self.q_surf_warn)**2 / self.q_surf_design
        
        # Target temperature health
        if T_target < self.T_target_min:
            reward -= self.w['T_target'] * (self.T_target_min - T_target)**2
        elif T_target > self.T_target_max:
            reward -= self.w['T_target'] * (T_target - self.T_target_max)**2
        
        if abs(dR) < 0.01 and abs(dZ) < 0.01 and \
           q_surf < self.q_surf_warn and T_target < self.T_target_max:
            reward += 2.0
        
        return reward
    
    def _check_termination(self, state_plasma, divertor):
        """Check if episode should end."""
        dZ = state_plasma[1]
        q_surf = divertor['q_surf']
        
        # Vertical position limit (hit the wall)
        if abs(dZ) > 0.5:
            return True
        
        if q_surf > self.q_surf_hard:
            return True
        
        # Disruption (simplified: large position excursion)
        if abs(state_plasma[0]) > 0.3:
            return True
        
        return False
    
    def reset(self, seed=None, options=None):
        """Reset to a random initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.step_count = 0
        
        # Random initial perturbation from equilibrium
        dR = self.rng.uniform(-0.05, 0.05)
        dZ = self.rng.uniform(-0.02, 0.02)
        dIp = self.rng.uniform(-1e6, 1e6)
        dbp = self.rng.uniform(-0.05, 0.05)
        dne = self.rng.uniform(-0.1e20, 0.1e20)
        
        # PF coil currents: near sustainable equilibrium (25kA nominal)
        # R_coil=0.1Ω, V_eq=2500V for 25kA, leaving ±2500V headroom
        I_nom = 25000.0
        I_coils = np.full(self.n_coils, I_nom) + self.rng.uniform(-5000, 5000, self.n_coils)
        
        # VS coil: starts at zero
        I_vs = 0.0
        
        state_parts = [dR, dZ, dIp, dbp, dne] + list(I_coils)
        if self.has_vs:
            state_parts.append(I_vs)
        state_plasma = np.array(state_parts, dtype=float)
        
        # Zero controls at t=0
        controls = {
            'V_coils': np.zeros(self.n_coils),
            'P_heat': 0.0, 'fuel': 0.0, 'seed': 0.0, 'V_loop': 0.0,
            'V_vs': 0.0,
        }
        
        divertor = self.sim.compute_divertor(state_plasma, controls)
        
        self.state = state_plasma
        obs = self._get_obs(state_plasma, divertor)
        
        if self.render_mode == 'human':
            print(f"Reset: dR={dR:.3f}, dZ={dZ:.3f}, q_surf={divertor['q_surf']:.1f}")
        
        return obs, {'divertor': divertor}
    
    def step(self, action):
        """Take action, return next state, reward, terminated, truncated, info."""
        # Unpack action: [V_coils(12), V_vs(1), P_heat, fuel, seed]
        V_vs_max = self.sim.vs_coils.V_max if self.has_vs else 0
        V_coils = np.clip(action[:self.n_coils], -5000, 5000)
        vs_idx = self.n_coils
        V_vs = float(np.clip(action[vs_idx], -V_vs_max, V_vs_max)) if self.has_vs else 0.0
        P_heat = float(np.clip(action[vs_idx + self.n_vs], -50, 50))
        fuel = float(max(0, action[vs_idx + self.n_vs + 1]))
        seed = float(max(0, action[vs_idx + self.n_vs + 2]))
        
        controls = {
            'V_coils': V_coils,
            'V_vs': V_vs,
            'P_heat': P_heat,
            'fuel': fuel,
            'seed': seed,
            'V_loop': 0.0,
        }
        
        # Simulate one step (analytic VS update for numerical stability)
        dstate = self.sim.derivatives(0, self.state, controls, vs_dt=self.dt)
        new_state = self.state + dstate * self.dt
        
        # Divertor calculation
        divertor = self.sim.compute_divertor(new_state, controls)
        
        # Reward
        reward = self._compute_reward(new_state, divertor)
        
        # Termination check
        terminated = self._check_termination(new_state, divertor)
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        self.state = new_state
        obs = self._get_obs(new_state, divertor)
        
        info = {
            'divertor': divertor,
            'step': self.step_count,
            'dR': new_state[0],
            'dZ': new_state[1],
        }
        
        if self.render_mode == 'human' and (self.step_count % 100 == 0 or terminated):
            print(f"Step {self.step_count}: dR={new_state[0]:+.4f}, dZ={new_state[1]:+.4f}, "
                  f"q_surf={divertor['q_surf']:.1f}, T_target={divertor['T_target']:.1f}, "
                  f"reward={reward:.1f}")
        
        return obs, reward, terminated, truncated, info


# ============================================================
# Training data generation
# ============================================================
def generate_trajectories(n_episodes=100, max_steps=5000, noise_std=0.0, seed=42):
    """Generate training data by running random control policies.
    
    Returns list of {state, action, reward, next_state, done} transitions.
    """
    env = PlasmaControlEnv(seed=seed)
    transitions = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Random action with structure
            n = env.action_space.shape[0]
            action = np.random.randn(n).astype(np.float32)
            action[:env.n_coils] *= 1000  # ±1kV PF
            if env.has_vs:
                action[env.n_coils] *= 500  # ±500V VS
                h_idx = env.n_coils + env.n_vs
            else:
                h_idx = env.n_coils
            action[h_idx] *= 10     # ±10 MW
            action[h_idx + 1] = abs(action[h_idx + 1]) * 1e22  # fuel 0-1e22
            action[h_idx + 2] = abs(action[h_idx + 2]) * 1e22  # seed 0-1e22
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            transitions.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'done': terminated or truncated,
                'info': info,
            })
            
            obs = next_obs
            done = terminated or truncated
            step += 1
        
        if ep % 10 == 0:
            print(f"Episode {ep}: {step} steps, final reward={reward:.1f}")
    
    return transitions


# ============================================================
# Save/load training data
# ============================================================
def save_trajectories(trajectories, filepath='training_data.npz'):
    """Save trajectories to compressed numpy file."""
    n = len(trajectories)
    obs_dim = len(trajectories[0]['obs'])
    act_dim = len(trajectories[0]['action'])
    
    data = {
        'obs': np.array([t['obs'] for t in trajectories]),
        'action': np.array([t['action'] for t in trajectories]),
        'reward': np.array([t['reward'] for t in trajectories]),
        'next_obs': np.array([t['next_obs'] for t in trajectories]),
        'done': np.array([t['done'] for t in trajectories]),
    }
    np.savez_compressed(filepath, **data)
    print(f"Saved {n} transitions to {filepath}")


# ============================================================
# Test
# ============================================================
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    # Test environment
    print("=" * 60)
    print("Testing Plasma Control Environment")
    print("=" * 60)
    
    env = PlasmaControlEnv(render_mode='human')
    
    # Test 1: Zero action = just drift
    print("\nTest 1: Zero action (no control)")
    obs, info = env.reset()
    total_reward = 0
    for _ in range(200):
        action = np.zeros(env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"  Episode ended at step {env.step_count}: dZ={info['dZ']:.4f}m")
            break
    print(f"  Total reward: {total_reward:.1f}")
    
    # Test 2: Simple PD controller as action (with VS)
    print("\nTest 2: Hand-tuned PD + VS controller")
    obs, info = env.reset()
    total_reward = 0
    dt = env.dt
    prev_dZ = 0.0
    for _ in range(2000):
        dR, dZ = obs[0], obs[1]
        dZ_dot = (dZ - prev_dZ) / dt if _ > 0 else 0
        prev_dZ = dZ
        
        R_coil = env.sim.coils.R_turn * env.sim.coils.positions[0]['turns']
        I_nom = 25000.0
        V_eq = R_coil * I_nom  # 2500V for 25kA
        Kz = 2.5e5
        Kr = 1e5
        
        action = np.zeros(env.action_space.shape[0])
        vs_offset = env.has_vs
        action[:env.n_coils] = V_eq
        action[:6] += Kz * dZ + Kr * dR
        action[6:12] += -Kz * dZ + Kr * dR
        action[:env.n_coils] = np.clip(action[:env.n_coils], -5000, 5000)
        
        if env.has_vs:
            K_vs = 2e5
            action[env.n_coils] = float(np.clip(K_vs * dZ, -2000, 2000))
        
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"  Steps: {env.step_count}, total_reward: {total_reward:.1f}")
    print(f"  Terminal: dR={info['dR']:.4f}, dZ={info['dZ']:.4f}")
    
    # Generate some training data
    print("\n" + "=" * 60)
    print("Generating training trajectories...")
    traj = generate_trajectories(n_episodes=5, max_steps=500)
    save_trajectories(traj, 'training_data_test.npz')
    print("Done!")
