# spherical_pn_sim.py
# Minimal spherical PN simulation (leveled, outward frame).
# - Drone constrained to a sphere of radius R (moves tangentially)
# - Target (Shahed) follows a configurable linear trajectory
# - Aimpoint = predicted impact point of target on the sphere
# - Guidance = PN on tangent plane (heading-rate form) + timing term

from dataclasses import dataclass, field
import numpy as np
from numpy.linalg import norm
from typing import Optional, Tuple, List
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (import side-effects)

# ----------------------------
# math3d: small 3-D utilities
# ----------------------------

EPS = 1e-9

def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n < EPS:
        return v * 0.0
    return v / n

def safe_unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n < EPS:
        return unit(fallback)
    return v / n

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def angle_wrap(a: float) -> float:
    # Wrap to [-pi, pi)
    return (a + math.pi) % (2*math.pi) - math.pi

def angle_unwrap(prev: float, current: float) -> float:
    """Return current adjusted so that the jump from prev is minimal."""
    d = angle_wrap(current - prev)
    return prev + d

def rodrigues_rotate(vec: np.ndarray, axis_unit: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vec around axis_unit by 'angle' (right-hand rule)."""
    a = axis_unit
    c = math.cos(angle)
    s = math.sin(angle)
    return vec * c + np.cross(a, vec) * s + a * (np.dot(a, vec)) * (1 - c)

def tangent_basis(u: np.ndarray, up=np.array([0.0, 0.0, 1.0])) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a leveled local basis at u on the sphere:
      e = (up × u) / ||up × u||   (east-like)
      n = u × e                   (north-like)
    Falls back to a different reference near poles.
    """
    cross_zu = np.cross(up, u)
    if norm(cross_zu) < 1e-6:
        # Near pole: use X-axis as reference
        ref = np.array([1.0, 0.0, 0.0])
        cross_ru = np.cross(ref, u)
        e = unit(cross_ru)
    else:
        e = unit(cross_zu)
    n = unit(np.cross(u, e))
    return e, n

def project_to_tangent(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project v onto tangent plane at u and normalize to unit length."""
    # Remove radial component along u
    t = v - np.dot(v, u) * u
    return safe_unit(t, fallback=np.array([0.0, 0.0, 0.0]))

def central_angle(u: np.ndarray, u_star: np.ndarray) -> float:
    """γ = atan2(||u×u_*||, u·u_*) with u, u_* unit."""
    s = norm(np.cross(u, u_star))
    c = float(np.dot(u, u_star))
    return math.atan2(s, c)

# --------------------------------
# models: target & interceptor
# --------------------------------

@dataclass
class TargetLinear:
    r0: np.ndarray        # initial position (3,)
    v:  np.ndarray        # constant velocity (3,)

    def state(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        return (self.r0 + self.v * t, self.v)

@dataclass
class InterceptorOnSphere:
    R: float                  # sphere radius
    u: np.ndarray             # current unit position vector on sphere (3,)
    chi: float                # current track/bearing angle on tangent plane (rad)
    v_max: float              # max tangential speed (m/s)

    def move_along_heading(self, dt: float, v_cmd: float):
        """
        Advance u along the great-circle direction defined by chi at speed v_cmd.
        Heading vector in tangent plane: d = cos(chi)*n + sin(chi)*e
        Rotate u about axis (u × d) by angle theta = (v_cmd/R)*dt
        """
        e, n = tangent_basis(self.u)
        d = math.cos(self.chi) * n + math.sin(self.chi) * e
        d = project_to_tangent(self.u, d)  # ensure tangent & unit
        omega = v_cmd / self.R                 # angular speed (rad/s)
        theta = omega * dt
        axis = unit(np.cross(self.u, d))       # rotation axis
        if norm(axis) < EPS or abs(theta) < 1e-12:
            return
        self.u = unit(rodrigues_rotate(self.u, axis, theta))

# --------------------------------
# aimpoint predictor
# --------------------------------

@dataclass
class ImpactSolution:
    hit: bool
    tau: float
    u_star: np.ndarray

def predict_impact_or_closest(r_t: np.ndarray, v_t: np.ndarray, R: float) -> ImpactSolution:
    """
    Solve |r + v tau| = R for the smallest tau>0 (linear motion).
    If no positive root, use closest-approach projection to sphere.
    """
    a = np.dot(v_t, v_t)
    b = 2.0 * np.dot(r_t, v_t)
    c = np.dot(r_t, r_t) - R**2

    hit = False
    tau_hit = np.inf

    if a > EPS:
        disc = b*b - 4*a*c
        if disc >= 0.0:
            sqrt_disc = math.sqrt(disc)
            tau1 = (-b - sqrt_disc) / (a*2.0)
            tau2 = (-b + sqrt_disc) / (a*2.0)
            candidates = [t for t in (tau1, tau2) if t > 0.0]
            if candidates:
                tau_hit = min(candidates)
                hit = True

    if hit:
        p = r_t + v_t * tau_hit
        u_star = unit(p)
        return ImpactSolution(True, tau_hit, u_star)

    # Fallback: closest approach time in Euclidean sense (clamped >= 0)
    if a > EPS:
        tau_ca = max(0.0, -b / (2.0 * a))
    else:
        tau_ca = 0.0
    p = r_t + v_t * tau_ca
    if norm(p) < EPS:
        p = np.array([R, 0.0, 0.0])  # arbitrary
    u_star = unit(p)
    return ImpactSolution(False, tau_ca, u_star)

# --------------------------------
# guidance: PN on tangent plane
# --------------------------------

@dataclass
class PNParams:
    N: float = 3.0               # navigation constant
    tgo_mode: str = "impact"     # "impact" or "distance"
    chi_rate_max: float = math.radians(60.0)  # max turn rate (rad/s)

class SphericalPNGuidance:
    def __init__(self, R: float, v_max: float, params: PNParams):
        self.R = R
        self.v_max = v_max
        self.params = params
        self._prev_chi_los: Optional[float] = None

    def compute_commands(
        self,
        u: np.ndarray,
        chi: float,
        u_star: np.ndarray,
        tau_hit: float,
        dt: float
    ) -> Tuple[float, float, float, float]:
        """
        Returns:
          chi_los, chi_los_rate, chi_dot_cmd, v_cmd
        """
        # Build leveled tangent frame
        e, n = tangent_basis(u)
        # Tangential bearing toward aimpoint (project u_star into tangent plane)
        b = project_to_tangent(u, u_star)
        # LOS azimuth in the sheet
        chi_los = math.atan2(np.dot(b, e), np.dot(b, n))

        # LOS rate (finite difference + unwrap)
        if self._prev_chi_los is None:
            chi_los_rate = 0.0
        else:
            chi_los_u = angle_unwrap(self._prev_chi_los, chi_los)
            chi_los_rate = (chi_los_u - self._prev_chi_los) / max(dt, 1e-6)
            chi_los = chi_los_u
        self._prev_chi_los = chi_los

        # Geodesic separation
        gamma = central_angle(u, u_star)

        # Time-to-go
        if self.params.tgo_mode == "impact":
            t_go = max(1e-3, tau_hit)  # synchronize to predicted impact
        else:
            t_go = max(1e-3, self.R * gamma / max(self.v_max, 1e-6))

        # PN + timing augmentation (APN)
        chi_dot_cmd = gamma / t_go + self.params.N * chi_los_rate
        chi_dot_cmd = clamp(chi_dot_cmd, -self.params.chi_rate_max, self.params.chi_rate_max)

        # Speed schedule: try to arrive on time, subject to v_max
        v_sched = self.R * gamma / t_go
        v_cmd = clamp(v_sched, 0.0, self.v_max)

        return chi_los, chi_los_rate, chi_dot_cmd, v_cmd

# --------------------------------
# simulation
# --------------------------------

@dataclass
class SimConfig:
    R: float = 1000.0
    dt: float = 0.05
    T: float  = 60.0
    v_max: float = 60.0          # interceptor max tangential speed (m/s)
    pn: "PNParams" = field(default_factory=PNParams)

class Simulator:
    def __init__(self, cfg: SimConfig, target: TargetLinear, interceptor: InterceptorOnSphere):
        self.cfg = cfg
        self.target = target
        self.interceptor = interceptor
        self.guidance = SphericalPNGuidance(cfg.R, interceptor.v_max, cfg.pn)

        self.t_hist: List[float] = []
        self.u_hist: List[np.ndarray] = []
        self.r_t_hist: List[np.ndarray] = []
        self.u_star_hist: List[np.ndarray] = []
        self.gamma_hist: List[float] = []
        self.chi_los_hist: List[float] = []
        self.chi_los_rate_hist: List[float] = []

        # Add step-by-step simulation state
        self.current_time: float = 0.0
        self.current_step: int = 0
        self.is_running: bool = False

    def step(self, t: float):
        R = self.cfg.R
        dt = self.cfg.dt

        # Target state now
        r_t, v_t = self.target.state(t)

        # Aimpoint prediction
        sol = predict_impact_or_closest(r_t, v_t, R)

        # Guidance
        u = self.interceptor.u
        chi = self.interceptor.chi
        chi_los, chi_los_rate, chi_dot_cmd, v_cmd = self.guidance.compute_commands(
            u=u, chi=chi, u_star=sol.u_star, tau_hit=sol.tau, dt=dt
        )

        # Integrate heading
        self.interceptor.chi = angle_wrap(self.interceptor.chi + chi_dot_cmd * dt)
        # Move along current heading
        self.interceptor.move_along_heading(dt=dt, v_cmd=v_cmd)

        # Log
        self.t_hist.append(t)
        self.u_hist.append(self.interceptor.u.copy())
        self.r_t_hist.append(r_t.copy())
        self.u_star_hist.append(sol.u_star.copy())
        self.gamma_hist.append(central_angle(self.interceptor.u, sol.u_star))
        self.chi_los_hist.append(chi_los)
        self.chi_los_rate_hist.append(chi_los_rate)

    def run(self):
        """Run complete simulation automatically"""
        self.current_time = 0.0
        self.current_step = 0
        while self.current_time <= self.cfg.T:
            self.step(self.current_time)
            self.current_time += self.cfg.dt
            self.current_step += 1

    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.current_time = 0.0
        self.current_step = 0
        self.is_running = False
        
        # Clear history
        self.t_hist.clear()
        self.u_hist.clear()
        self.r_t_hist.clear()
        self.u_star_hist.clear()
        self.gamma_hist.clear()
        self.chi_los_hist.clear()
        self.chi_los_rate_hist.clear()

    def run_single_step(self) -> bool:
        """
        Run a single simulation step.
        Returns True if simulation continues, False if finished.
        """
        if self.current_time > self.cfg.T:
            self.is_running = False
            return False
            
        self.step(self.current_time)
        self.current_time += self.cfg.dt
        self.current_step += 1
        self.is_running = True
        
        return self.current_time <= self.cfg.T

    def run_interactive(self):
        """Run simulation step-by-step with user interaction"""
        print("=== Interactive Spherical PN Simulation ===")
        print("Commands:")
        print("  ENTER/SPACE: Next step")
        print("  'r': Reset simulation")
        print("  'q': Quit")
        print("  'p': Plot current state")
        print("  'a': Auto-run remaining steps")
        print("=" * 45)
        
        self.reset_simulation()
        
        while True:
            # Display current state
            print(f"\nStep: {self.current_step:4d} | Time: {self.current_time:6.2f}s | ", end="")
            if len(self.gamma_hist) > 0:
                print(f"Central Angle: {math.degrees(self.gamma_hist[-1]):6.1f}°")
            else:
                print("Starting...")
            
            # Check if simulation is complete
            if self.current_time > self.cfg.T:
                print("\n🎯 Simulation Complete!")
                print("Type 'p' to plot results, 'r' to restart, or 'q' to quit.")
                
            # Get user input
            try:
                cmd = input(">>> ").strip().lower()
            except KeyboardInterrupt:
                print("\nSimulation interrupted.")
                break
                
            if cmd in ['q', 'quit', 'exit']:
                break
            elif cmd in ['r', 'reset']:
                self.reset_simulation()
                print("Simulation reset.")
            elif cmd in ['p', 'plot']:
                if len(self.t_hist) > 0:
                    self.plot_3d_interactive()
                else:
                    print("No data to plot yet.")
            elif cmd in ['a', 'auto']:
                print("Auto-running remaining steps...")
                while self.run_single_step():
                    if self.current_step % 20 == 0:  # Progress update every 20 steps
                        print(f"Step {self.current_step}, Time: {self.current_time:.2f}s")
                print("Auto-run complete!")
            elif cmd in ['', ' ']:  # ENTER or SPACE
                if not self.run_single_step():
                    print("Simulation finished!")
            else:
                print("Unknown command. Use ENTER for next step, 'q' to quit.")

    def plot_3d(self):
        """Original plotting method - fixed sphere rendering"""
        # Prepare arrays
        U = np.array(self.u_hist)                    # (N,3)
        RT = np.array(self.r_t_hist)
        US = np.array(self.u_star_hist)
        R = self.cfg.R

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create properly round sphere mesh
        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        xs = R * np.outer(np.cos(u), np.sin(v))
        ys = R * np.outer(np.sin(u), np.sin(v))
        zs = R * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.4, color='gray')

        # Paths
        ax.plot(RT[:,0], RT[:,1], RT[:,2], linewidth=1.2, label="Target (Shahed)", color='red')
        ax.plot(R*U[:,0], R*U[:,1], R*U[:,2], linewidth=1.8, label="Interceptor (Drone)", color='blue')
        ax.plot(R*US[:,0], R*US[:,1], R*US[:,2], linewidth=1.0, linestyle='--', 
                label="Aimpoint direction", color='orange', alpha=0.7)

        # Mark start & end of interceptor trajectory
        start_pt = R * U[0]
        end_pt   = R * U[-1]

        ax.scatter(*start_pt, s=40, color='limegreen', edgecolors='black',
                linewidths=0.6, label='Drone start', zorder=5)
        ax.scatter(*end_pt,   s=40, color='crimson',  edgecolors='black',
                linewidths=0.6, label='Drone end',   zorder=5)

        # Labels offset outward from the surface
        ax.text(*(start_pt + 0.05 * R * unit(start_pt)), "START", fontsize=10, fontweight='bold')
        ax.text(*(end_pt   + 0.05 * R * unit(end_pt)),   "END",   fontsize=10, fontweight='bold')

        # CRITICAL: Force equal aspect ratio for round sphere
        max_range = R * 1.2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_box_aspect([1,1,1])  # Equal aspect ratio

        # Labels and formatting
        ax.set_xlabel("X [m]", fontsize=12)
        ax.set_ylabel("Y [m]", fontsize=12)
        ax.set_zlabel("Z [m]", fontsize=12)
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
        
        # Add simulation info to title
        final_time = self.t_hist[-1] if self.t_hist else 0
        final_gamma = math.degrees(self.gamma_hist[-1]) if self.gamma_hist else 0
        plt.title(f"Spherical PN Defense System\nFinal Time: {final_time:.1f}s, "
                 f"Final Separation: {final_gamma:.1f}°", fontsize=14)
        
        plt.tight_layout()
        plt.show()

    def plot_3d_interactive(self):
        """Interactive plotting method with current simulation state display"""
        if len(self.u_hist) == 0:
            print("No simulation data to plot.")
            return
            
        # Prepare arrays
        U = np.array(self.u_hist)
        RT = np.array(self.r_t_hist)
        US = np.array(self.u_star_hist)
        R = self.cfg.R

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Create properly round sphere mesh
        u = np.linspace(0, 2*np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        xs = R * np.outer(np.cos(u), np.sin(v))
        ys = R * np.outer(np.sin(u), np.sin(v))
        zs = R * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, linewidth=0.2, alpha=0.3, color='lightgray')

        # Plot trajectories up to current point
        if len(U) > 1:
            ax.plot(RT[:, 0], RT[:, 1], RT[:, 2], 'r-', linewidth=2, label="Shahed trajectory")
            ax.plot(R*U[:, 0], R*U[:, 1], R*U[:, 2], 'b-', linewidth=2, label="Drone trajectory")
            ax.plot(R*US[:, 0], R*US[:, 1], R*US[:, 2], 'orange', linestyle='--', 
                   linewidth=1, alpha=0.6, label="Aimpoint direction")

        # Current positions (larger markers)
        current_drone_pos = R * U[-1]
        current_target_pos = RT[-1]
        current_aimpoint = R * US[-1]

        ax.scatter(*current_drone_pos, s=200, color='blue', edgecolors='white', 
                  linewidths=2, label='Drone (current)', zorder=10, marker='o')
        ax.scatter(*current_target_pos, s=200, color='red', edgecolors='white', 
                  linewidths=2, label='Shahed (current)', zorder=10, marker='^')
        ax.scatter(*current_aimpoint, s=150, color='orange', edgecolors='black', 
                  linewidths=1, label='Aimpoint (current)', zorder=8, marker='*')

        # Start position
        if len(U) > 0:
            start_pt = R * U[0]
            ax.scatter(*start_pt, s=100, color='green', edgecolors='black',
                      linewidths=1, label='Drone start', zorder=7, marker='s')

        # CRITICAL: Force equal aspect ratio for round sphere
        max_range = R * 1.3
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_box_aspect([1,1,1])

        # Labels and formatting
        ax.set_xlabel("X [m]", fontsize=11)
        ax.set_ylabel("Y [m]", fontsize=11)
        ax.set_zlabel("Z [m]", fontsize=11)
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)

        # Enhanced title with current state
        current_gamma = math.degrees(self.gamma_hist[-1]) if self.gamma_hist else 0
        plt.title(f"Step {self.current_step} | Time: {self.current_time:.2f}s | "
                 f"Separation: {current_gamma:.1f}°\nSpherical PN Defense System", 
                 fontsize=13, pad=20)
        
        plt.tight_layout()
        plt.show(block=False)  # Non-blocking for interactive use
        plt.pause(0.1)

    def plot_time_series(self):
        t = np.array(self.t_hist)
        gamma = np.array(self.gamma_hist)
        chi_los = np.unwrap(np.array(self.chi_los_hist))
        chi_los_rate = np.array(self.chi_los_rate_hist)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Central angle
        ax1.plot(t, np.degrees(gamma), 'b-', linewidth=2)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Central Angle γ [°]")
        ax1.set_title("Separation Angle Between Drone and Aimpoint")
        ax1.grid(True, alpha=0.3)

        # LOS azimuth and rate
        ax2.plot(t, np.degrees(chi_los), 'g-', linewidth=2, label="χ_LOS [°]")
        ax2.plot(t, np.degrees(chi_los_rate), 'r-', linewidth=2, label="χ̇_LOS [°/s]")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Angle / Rate")
        ax2.set_title("Line-of-Sight Azimuth and Angular Rate")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# --------------------------------
# Example scenario
# --------------------------------

def example():
    # ---- Parameters you can tweak easily ----
    R = 1000.0                           # protection sphere radius [m]
    dt = 0.05                            # sim step [s]
    T  = 50.0                            # total time [s]
    v_max = 80.0                         # interceptor max tangential speed [m/s]

    # PN tuning
    pn = PNParams(
        N=3.5,
        tgo_mode="impact",               # or "distance"
        chi_rate_max=math.radians(90.0)
    )
    cfg = SimConfig(R=R, dt=dt, T=T, v_max=v_max, pn=pn)

    # ---- Target (Shahed) trajectory ----
    # Approaches the sphere roughly radially toward the origin (center)
    r0 = np.array([0.0, -3500.0, 1500.0])   # initial position [m]
    v  = np.array([0.0, +60.0, -10.0])      # constant velocity [m/s]
    target = TargetLinear(r0=r0, v=v)

    # ---- Interceptor initial state on sphere ----
    # Place drone on the sphere at some longitude/latitude
    u0 = unit(np.array([0.3, -0.95, 0.1]))  # unit direction
    interceptor = InterceptorOnSphere(R=R, u=u0, chi=0.0, v_max=v_max)

    # Initialize heading to current LOS (optional, helps initial transient)
    r_t0, _ = target.state(0.0)
    sol0 = predict_impact_or_closest(r_t0, v, R)
    e0, n0 = tangent_basis(u0)
    b0 = project_to_tangent(u0, sol0.u_star)
    interceptor.chi = math.atan2(np.dot(b0, e0), np.dot(b0, n0))

    # ---- Run simulation ----
    sim = Simulator(cfg, target, interceptor)
    
    # Choose simulation mode:
    print("Choose simulation mode:")
    print("1. Interactive step-by-step")
    print("2. Full auto-run with plots")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        sim.run_interactive()
    else:
        sim.run()
        sim.plot_3d()
        sim.plot_time_series()

if __name__ == "__main__":
    example()
