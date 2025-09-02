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
class TargetCurved:
    """
    Curved trajectory target with helical/spiral motion.
    Combines linear motion with circular/helical components for evasive maneuvers.
    """
    r0: np.ndarray        # initial position (3,)
    v_linear: np.ndarray  # linear velocity component (3,)
    orbit_center: np.ndarray  # center of circular motion (3,)
    orbit_radius: float   # radius of circular motion [m]
    omega: float          # angular frequency [rad/s]
    phase_offset: float = 0.0  # initial phase [rad]

    def state(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns position and velocity at time t.
        Motion = linear_motion + circular_motion_in_plane
        """
        # Linear component
        r_linear = self.r0 + self.v_linear * t
        
        # Circular/helical component (in XY plane by default)
        phase = self.omega * t + self.phase_offset
        
        # Circular motion vectors (perpendicular to linear velocity direction)
        if norm(self.v_linear) > 1e-6:
            # Create orthogonal basis for circular motion
            v_unit = unit(self.v_linear)
            # Use Z-axis as reference for cross product
            ref_axis = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(v_unit, ref_axis)) > 0.9:  # nearly parallel to Z
                ref_axis = np.array([1.0, 0.0, 0.0])  # use X-axis instead
            
            u1 = unit(np.cross(v_unit, ref_axis))  # first perpendicular direction
            u2 = unit(np.cross(v_unit, u1))       # second perpendicular direction
        else:
            # Default to XY plane if no linear velocity
            u1 = np.array([1.0, 0.0, 0.0])
            u2 = np.array([0.0, 1.0, 0.0])
        
        # Circular displacement and velocity
        r_circle = self.orbit_radius * (np.cos(phase) * u1 + np.sin(phase) * u2)
        v_circle = self.orbit_radius * self.omega * (-np.sin(phase) * u1 + np.cos(phase) * u2)
        
        # Total motion
        position = r_linear + r_circle
        velocity = self.v_linear + v_circle
        
        return (position, velocity)

@dataclass
class TargetSinusoidal:
    """
    Sinusoidal trajectory target for weaving/serpentine motion.
    """
    r0: np.ndarray        # initial position (3,)
    v_base: np.ndarray    # base velocity direction (3,)
    amplitude: float      # oscillation amplitude [m]
    frequency: float      # oscillation frequency [Hz]
    
    def state(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns position and velocity for sinusoidal weaving motion.
        """
        # Base linear motion
        r_base = self.r0 + self.v_base * t
        
        # Create perpendicular direction for oscillation
        if norm(self.v_base) > 1e-6:
            v_unit = unit(self.v_base)
            # Use Z-axis as reference
            ref_axis = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(v_unit, ref_axis)) > 0.9:
                ref_axis = np.array([1.0, 0.0, 0.0])
            perp_dir = unit(np.cross(v_unit, ref_axis))
        else:
            perp_dir = np.array([1.0, 0.0, 0.0])
        
        # Sinusoidal oscillation
        omega = 2.0 * math.pi * self.frequency
        phase = omega * t
        
        r_osc = self.amplitude * math.sin(phase) * perp_dir
        v_osc = self.amplitude * omega * math.cos(phase) * perp_dir
        
        position = r_base + r_osc
        velocity = self.v_base + v_osc
        
        return (position, velocity)

@dataclass
class TargetTurning:
    """
    Simple turning trajectory - Shahed gradually turns right while maintaining speed.
    Creates a gentle arc/spiral trajectory.
    """
    r0: np.ndarray        # initial position (3,)
    v0: np.ndarray        # initial velocity (3,)
    turn_rate: float      # turn rate [rad/s] - positive for right turn
    
    def state(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns position and velocity for constant turn rate trajectory.
        The target maintains constant speed but gradually changes direction.
        """
        speed = norm(self.v0)
        if speed < 1e-6:
            return (self.r0, self.v0)
        
        # If no turn rate, just linear motion
        if abs(self.turn_rate) < 1e-6:
            return (self.r0 + self.v0 * t, self.v0)
        
        # For constant turn rate, we have circular motion in the horizontal plane
        # The trajectory is a helix if there's vertical velocity component
        
        # Separate horizontal and vertical components
        v0_horizontal = np.array([self.v0[0], self.v0[1], 0.0])
        v0_vertical = np.array([0.0, 0.0, self.v0[2]])
        
        speed_horizontal = norm(v0_horizontal)
        
        if speed_horizontal < 1e-6:
            # Pure vertical motion - no turning possible
            position = self.r0 + self.v0 * t
            velocity = self.v0
        else:
            # Calculate circular motion parameters
            radius = speed_horizontal / abs(self.turn_rate)
            
            # Initial heading angle in horizontal plane
            initial_heading = math.atan2(v0_horizontal[1], v0_horizontal[0])
            
            # Center of circular motion (perpendicular to initial velocity)
            # For right turn (positive turn_rate), center is to the right of initial velocity
            center_direction = np.array([-math.sin(initial_heading), math.cos(initial_heading), 0.0])
            if self.turn_rate < 0:  # Left turn
                center_direction = -center_direction
            
            center = self.r0 + radius * center_direction
            
            # Current angle around the circle
            angle = initial_heading + self.turn_rate * t
            
            # Position on the circle
            circle_pos = center + radius * np.array([math.cos(angle), math.sin(angle), 0.0])
            
            # Add vertical motion
            vertical_pos = self.r0[2] + v0_vertical[2] * t
            position = np.array([circle_pos[0], circle_pos[1], vertical_pos])
            
            # Velocity is tangent to the circle plus vertical component
            horizontal_velocity = speed_horizontal * np.array([-math.sin(angle), math.cos(angle), 0.0])
            velocity = horizontal_velocity + v0_vertical
        
        return (position, velocity)

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
    N: float = 5               # navigation constant
    tgo_mode: str = "impact"     # "impact" or "distance"
    pn_mode: str = "apn"     # "apn" or "pn_only"
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

        if self.params.pn_mode == "apn":
            # PN + timing augmentation (APN)
            chi_dot_cmd = gamma / t_go + self.params.N * chi_los_rate
        else:
            # PN only (works better for little curvatures)
            chi_dot_cmd = self.params.N * chi_los_rate

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
        self.distance_hist: List[float] = []

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

        # Calculate direct distance between interceptor and Shahed
        interceptor_pos = R * self.interceptor.u  # Interceptor position in 3D space
        direct_distance = norm(r_t - interceptor_pos)  # Euclidean distance

        # Log
        self.t_hist.append(t)
        self.u_hist.append(self.interceptor.u.copy())
        self.r_t_hist.append(r_t.copy())
        self.u_star_hist.append(sol.u_star.copy())
        self.gamma_hist.append(central_angle(self.interceptor.u, sol.u_star))
        self.chi_los_hist.append(chi_los)
        self.chi_los_rate_hist.append(chi_los_rate)
        self.distance_hist.append(direct_distance)

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
        self.distance_hist.clear()

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
                print(f"Central Angle: {math.degrees(self.gamma_hist[-1]):6.1f}° | ", end="")
                print(f"Distance: {self.distance_hist[-1]:6.0f}m")
            else:
                print("Starting...")
            
            # Check if simulation is complete
            if self.current_time > self.cfg.T:
                print("\n[COMPLETE] Simulation Complete!")
                if len(self.distance_hist) > 0:
                    min_distance = min(self.distance_hist)
                    min_time_idx = self.distance_hist.index(min_distance)
                    min_time = self.t_hist[min_time_idx]
                    print(f"[INFO] Minimum distance: {min_distance:.0f}m at t={min_time:.1f}s")
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
                        print(f"Step {self.current_step}, Time: {self.current_time:.2f}s, Distance: {self.distance_hist[-1]:.0f}m")
                print("Auto-run complete!")
            elif cmd in ['', ' ']:  # ENTER or SPACE
                if not self.run_single_step():
                    print("Simulation finished!")
            else:
                print("Unknown command. Use ENTER for next step, 'q' to quit.")

    def plot_3d(self):
        """Original plotting method - fixed sphere rendering with vector arrows"""
        # Prepare arrays
        U = np.array(self.u_hist)                   
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

        # Draw coordinate frame vectors at final drone position
        if len(U) > 0:
            final_u = U[-1]  # Final unit position vector
            final_u_star = US[-1]  # Final aimpoint unit vector
            
            # Calculate local tangent frame vectors
            e, n = tangent_basis(final_u)
            b = project_to_tangent(final_u, final_u_star)  # Bearing vector to aimpoint
            
            # Vector scaling for visibility
            vector_scale = R * 0.12
            
            # Position vector u (radial, from origin to drone)
            ax.quiver(0, 0, 0, 
                     final_u[0] * vector_scale, 
                     final_u[1] * vector_scale, 
                     final_u[2] * vector_scale,
                     color='purple', linewidth=3, arrow_length_ratio=0.1, 
                     label='u (radial)')
            
            # At final drone position, draw tangent frame vectors
            final_drone_pos = end_pt
            
            # East vector e (tangent to sphere)
            ax.quiver(final_drone_pos[0], final_drone_pos[1], final_drone_pos[2],
                     e[0] * vector_scale, 
                     e[1] * vector_scale, 
                     e[2] * vector_scale,
                     color='cyan', linewidth=2, arrow_length_ratio=0.15,
                     label='e (east)')
            
            # North vector n (tangent to sphere)
            ax.quiver(final_drone_pos[0], final_drone_pos[1], final_drone_pos[2],
                     n[0] * vector_scale, 
                     n[1] * vector_scale, 
                     n[2] * vector_scale,
                     color='magenta', linewidth=2, arrow_length_ratio=0.15,
                     label='n (north)')
            
            # Bearing vector b (tangent projection toward aimpoint)
            ax.quiver(final_drone_pos[0], final_drone_pos[1], final_drone_pos[2],
                     b[0] * vector_scale, 
                     b[1] * vector_scale, 
                     b[2] * vector_scale,
                     color='gold', linewidth=2, arrow_length_ratio=0.15,
                     label='b (bearing to aimpoint)')

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
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=9)
        
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

        # Draw coordinate frame vectors at current drone position
        if len(U) > 0:
            current_u = U[-1]  # Current unit position vector
            current_u_star = US[-1]  # Current aimpoint unit vector
            
            # Calculate local tangent frame vectors
            e, n = tangent_basis(current_u)
            b = project_to_tangent(current_u, current_u_star)  # Bearing vector to aimpoint
            
            # Vector scaling for visibility
            vector_scale = R * 0.15
            
            # Position vector u (radial, from origin to drone)
            ax.quiver(0, 0, 0, 
                     current_u[0] * vector_scale, 
                     current_u[1] * vector_scale, 
                     current_u[2] * vector_scale,
                     color='purple', linewidth=3, arrow_length_ratio=0.1, 
                     label='u (radial)')
            
            # At drone position, draw tangent frame vectors
            drone_pos = current_drone_pos
            
            # East vector e (tangent to sphere)
            ax.quiver(drone_pos[0], drone_pos[1], drone_pos[2],
                     e[0] * vector_scale, 
                     e[1] * vector_scale, 
                     e[2] * vector_scale,
                     color='cyan', linewidth=2, arrow_length_ratio=0.15,
                     label='e (east)')
            
            # North vector n (tangent to sphere)
            ax.quiver(drone_pos[0], drone_pos[1], drone_pos[2],
                     n[0] * vector_scale, 
                     n[1] * vector_scale, 
                     n[2] * vector_scale,
                     color='magenta', linewidth=2, arrow_length_ratio=0.15,
                     label='n (north)')
            
            # Bearing vector b (tangent projection toward aimpoint)
            ax.quiver(drone_pos[0], drone_pos[1], drone_pos[2],
                     b[0] * vector_scale, 
                     b[1] * vector_scale, 
                     b[2] * vector_scale,
                     color='gold', linewidth=2, arrow_length_ratio=0.15,
                     label='b (bearing to aimpoint)')

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
        """Enhanced time series plots including direct distance"""
        t = np.array(self.t_hist)
        gamma = np.array(self.gamma_hist)
        chi_los = np.unwrap(np.array(self.chi_los_hist))
        chi_los_rate = np.array(self.chi_los_rate_hist)
        distance = np.array(self.distance_hist)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

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

        # Direct distance plot
        ax3.plot(t, distance, 'm-', linewidth=2, label="Direct Distance")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Distance [m]")
        ax3.set_title("Direct Distance Between Interceptor and Shahed")
        ax3.grid(True, alpha=0.3)
        
        # Highlight minimum distance
        if len(distance) > 0:
            min_dist = np.min(distance)
            min_idx = np.argmin(distance)
            min_time = t[min_idx]
            
            ax3.scatter(min_time, min_dist, color='red', s=100, zorder=5, 
                       label=f'Min: {min_dist:.0f}m @ {min_time:.1f}s')
            ax3.axhline(y=min_dist, color='red', linestyle='--', alpha=0.5)
            ax3.legend()
            
            # Add interception assessment
            if min_dist < 5:  # Within 5m considered successful intercept
                success_text = "[SUCCESS] INTERCEPT"
                text_color = 'green'
            elif min_dist < 10:  # Within 10m considered close
                success_text = "[CLOSE] APPROACH"
                text_color = 'orange'
            else:
                success_text = "[MISS] TARGET ESCAPED"
                text_color = 'red'
                
            ax3.text(0.02, 0.98, success_text, transform=ax3.transAxes, 
                    fontsize=12, fontweight='bold', color=text_color,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def get_interception_statistics(self) -> dict:
        """
        Calculate interception performance statistics.
        
        Returns:
            Dictionary with interception metrics
        """
        if not self.distance_hist:
            return {}
            
        distances = np.array(self.distance_hist)
        times = np.array(self.t_hist)
        
        min_distance = np.min(distances)
        min_time_idx = np.argmin(distances)
        min_time = times[min_time_idx]
        
        # Calculate closure rate at minimum distance
        if min_time_idx > 0 and min_time_idx < len(distances) - 1:
            # Use finite difference around minimum
            dt = times[min_time_idx + 1] - times[min_time_idx - 1]
            dd = distances[min_time_idx + 1] - distances[min_time_idx - 1]
            closure_rate = -dd / dt  # Negative because we want positive for approaching
        else:
            closure_rate = 0.0
        
        # Determine interception success
        if min_distance < 50:
            success_level = "SUCCESS"
        elif min_distance < 200:
            success_level = "CLOSE"
        else:
            success_level = "MISS"
            
        return {
            'min_distance': min_distance,
            'min_time': min_time,
            'closure_rate': closure_rate,
            'success_level': success_level,
            'final_distance': distances[-1],
            'initial_distance': distances[0]
        }

    def print_interception_summary(self):
        """Print summary of interception performance"""
        stats = self.get_interception_statistics()
        
        if not stats:
            print("No interception data available.")
            return
            
        print(f"\n[SUMMARY] INTERCEPTION RESULTS:")
        print(f"  Minimum distance: {stats['min_distance']:.0f} m")
        print(f"  Time of closest approach: {stats['min_time']:.1f} s")
        print(f"  Closure rate at closest approach: {stats['closure_rate']:.1f} m/s")
        print(f"  Initial distance: {stats['initial_distance']:.0f} m")
        print(f"  Final distance: {stats['final_distance']:.0f} m")
        print(f"  Result: {stats['success_level']}")
        
        # Performance assessment
        if stats['success_level'] == "SUCCESS":
            print("  [SUCCESS] Successful interception!")
        elif stats['success_level'] == "CLOSE":
            print("  [CLOSE] Close approach - potential damage radius")
        else:
            print("  [MISS] Target escaped")

# --------------------------------
# Example scenario
# --------------------------------

def get_speed_configuration():
    """
    Interactive speed configuration for testing different scenarios.
    Returns (shahed_speed, drone_max_speed, speed_ratio)
    """
    print("\n=== SPEED CONFIGURATION ===")
    print("Configure target speeds to test interception feasibility:")
    print("1. Realistic (Shahed: 60 m/s, Drone: 30 m/s) - Challenging")
    print("2. Balanced (Shahed: 40 m/s, Drone: 35 m/s) - Fair fight")  
    print("3. Drone advantage (Shahed: 30 m/s, Drone: 45 m/s) - Easy intercept")
    print("4. Custom speeds")
    
    while True:
        try:
            choice = input("\nChoose speed preset (1-4): ").strip()
            
            if choice == "1":
                shahed_speed, drone_max_speed = 60.0, 30.0
                break
            elif choice == "2":
                shahed_speed, drone_max_speed = 40.0, 35.0
                break
            elif choice == "3":
                shahed_speed, drone_max_speed = 30.0, 45.0
                break
            elif choice == "4":
                shahed_speed = float(input("Enter Shahed speed [m/s]: "))
                drone_max_speed = float(input("Enter Drone max speed [m/s]: "))
                break
            else:
                print("Invalid choice. Please enter 1-4.")
                continue
                
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Using default speeds.")
            shahed_speed, drone_max_speed = 60.0, 30.0
            break
    
    speed_ratio = shahed_speed / drone_max_speed
    print(f"\n[CONFIG] Speed Configuration:")
    print(f"  Shahed speed: {shahed_speed:.1f} m/s")
    print(f"  Drone max speed: {drone_max_speed:.1f} m/s") 
    print(f"  Speed ratio: {speed_ratio:.2f} {'(Shahed faster)' if speed_ratio > 1 else '(Drone faster)' if speed_ratio < 1 else '(Equal speed)'}")
    
    return shahed_speed, drone_max_speed, speed_ratio

def create_normalized_velocity(direction: np.ndarray, speed: float) -> np.ndarray:
    """
    Create a velocity vector with specified speed in the given direction.
    
    Args:
        direction: Desired direction vector (will be normalized)
        speed: Desired speed magnitude [m/s]
    
    Returns:
        Velocity vector with specified speed
    """
    return speed * unit(direction)

def example_curved_trajectory():
    """
    Example scenario with curved/evasive Shahed trajectory.
    The target performs helical motion while approaching the sphere.
    """
    print("\n=== CURVED TRAJECTORY SCENARIO ===")
    
    # ---- Parameters ----
    R = 1000.0                           # protection sphere radius [m]
    dt = 0.05                            # sim step [s]
    T  = 45.0                            # total time [s]
    v_max = 30.0                         # interceptor max tangential speed [m/s]

    # PN tuning - slightly more aggressive for evasive target
    pn = PNParams(
        N=5,                           # higher navigation gain
        tgo_mode="impact",                # or "distance"
        pn_mode="apn",                    # or "pn_only"
        chi_rate_max=math.radians(120.0)  # higher turn rate capability
    )
    cfg = SimConfig(R=R, dt=dt, T=T, v_max=v_max, pn=pn)

    # ---- Curved Target (Shahed) trajectory ----
    # Base approach trajectory
    r0 = np.array([1200.0, -3000.0, 800.0])     # initial position [m]
    v_linear = np.array([-15.0, +55.0, -12.0])  # base velocity [m/s]
    
    # Helical evasive maneuver parameters
    orbit_center = r0  # helical motion around initial position
    orbit_radius = 200.0  # [m] - radius of evasive circles
    omega = 0.15  # [rad/s] - evasive maneuver frequency
    
    target = TargetCurved(
        r0=r0,
        v_linear=v_linear,
        orbit_center=orbit_center,
        orbit_radius=orbit_radius,
        omega=omega,
        phase_offset=0.0
    )

    # ---- Interceptor initial state ----
    # Position drone strategically to intercept curved trajectory
    u0 = unit(np.array([0.8, -0.5, 0.3]))
    interceptor = InterceptorOnSphere(R=R, u=u0, chi=0.0, v_max=v_max)

    # Initialize heading toward predicted aimpoint
    r_t0, v_t0 = target.state(0.0)
    sol0 = predict_impact_or_closest(r_t0, v_t0, R)
    e0, n0 = tangent_basis(u0)
    b0 = project_to_tangent(u0, sol0.u_star)
    interceptor.chi = math.atan2(np.dot(b0, e0), np.dot(b0, n0))

    # ---- Run simulation ----
    sim = Simulator(cfg, target, interceptor)
    
    print(f"Target initial position: {r0}")
    print(f"Target linear velocity: {v_linear} m/s")
    print(f"Evasive maneuver: radius={orbit_radius}m, frequency={omega:.2f} rad/s")
    print(f"Interceptor max speed: {v_max} m/s")
    
    # Choose simulation mode
    print("\nChoose simulation mode:")
    print("1. Interactive step-by-step")
    print("2. Full auto-run with plots")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        sim.run_interactive()
    else:
        sim.run()
        sim.plot_3d()
        sim.plot_time_series()

def example_sinusoidal_trajectory():
    """
    Example scenario with sinusoidal weaving Shahed trajectory.
    """
    print("\n=== SINUSOIDAL WEAVING SCENARIO ===")
    
    # ---- Parameters ----
    R = 1000.0
    dt = 0.05
    T = 45.0
    v_max = 30.0

    # PN tuning
    pn = PNParams(
        N=5,
        tgo_mode="impact",                  # or "distance"
        pn_mode="apn",                      # or "pn_only"
        chi_rate_max=math.radians(100.0)
    )
    cfg = SimConfig(R=R, dt=dt, T=T, v_max=v_max, pn=pn)

    # ---- Sinusoidal Target ----
    r0 = np.array([-500.0, -2800.0, 600.0])
    v_base = np.array([8.0, +65.0, -15.0])  # base approach velocity
    amplitude = 300.0  # [m] weaving amplitude
    frequency = 0.08   # [Hz] weaving frequency
    
    target = TargetSinusoidal(
        r0=r0,
        v_base=v_base,
        amplitude=amplitude,
        frequency=frequency
    )

    # ---- Interceptor ----
    u0 = unit(np.array([-0.4, -0.8, 0.4]))
    interceptor = InterceptorOnSphere(R=R, u=u0, chi=0.0, v_max=v_max)

    # Initialize heading
    r_t0, v_t0 = target.state(0.0)
    sol0 = predict_impact_or_closest(r_t0, v_t0, R)
    e0, n0 = tangent_basis(u0)
    b0 = project_to_tangent(u0, sol0.u_star)
    interceptor.chi = math.atan2(np.dot(b0, e0), np.dot(b0, n0))

    # ---- Run simulation ----
    sim = Simulator(cfg, target, interceptor)
    
    print(f"Target initial position: {r0}")
    print(f"Target base velocity: {v_base} m/s")
    print(f"Weaving: amplitude={amplitude}m, frequency={frequency:.3f} Hz")
    
    print("\nChoose simulation mode:")
    print("1. Interactive step-by-step")
    print("2. Full auto-run with plots")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        sim.run_interactive()
    else:
        sim.run()
        sim.plot_3d()
        sim.plot_time_series()

def example():
    """
    Original linear trajectory scenario.
    """
    print("\n=== LINEAR TRAJECTORY SCENARIO ===")
    
    # ---- Parameters you can tweak easily ----
    R = 1000.0                           # protection sphere radius [m]
    dt = 0.05                            # sim step [s]
    T  = 45.0                            # total time [s]
    v_max = 30.0                         # interceptor max tangential speed [m/s]

    # PN tuning
    pn = PNParams(
        N=5,
        tgo_mode="impact",               # or "distance"
        pn_mode="apn",                   # or "pn_only"
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

def example_simple_turn():
    """
    Example scenario with simple turning Shahed - gradually turns right.
    """
    print("\n=== SIMPLE TURNING SCENARIO ===")
    
    # ---- Parameters ----
    R = 1000.0                           # protection sphere radius [m]
    dt = 0.05                            # sim step [s]
    T  = 45.0                            # total time [s] - longer to see the turn
    v_max = 30.0                         # interceptor max tangential speed [m/s]

    # PN tuning - standard settings for gentle turn
    pn = PNParams(
        N=5,
        tgo_mode="distance",                  # or "distance"
        pn_mode="pn_only",                      # or "pn_only"
        chi_rate_max=math.radians(80.0)
    )
    cfg = SimConfig(R=R, dt=dt, T=T, v_max=v_max, pn=pn)

    # ---- Simple Turning Target ----
    r0 = np.array([-800.0, -2500.0, 400.0])     # initial position [m]
    v0 = np.array([25.0, +55.0, -8.0])          # initial velocity [m/s]
    turn_rate = math.radians(2.0)                # 2 degrees/second right turn
    
    target = TargetTurning(
        r0=r0,
        v0=v0,
        turn_rate=turn_rate
    )

    # ---- Interceptor initial state ----
    # Position drone to intercept the turning trajectory
    u0 = unit(np.array([-0.6, -0.7, 0.4]))
    interceptor = InterceptorOnSphere(R=R, u=u0, chi=0.0, v_max=v_max)

    # Initialize heading toward predicted aimpoint
    r_t0, v_t0 = target.state(0.0)
    sol0 = predict_impact_or_closest(r_t0, v_t0, R)
    e0, n0 = tangent_basis(u0)
    b0 = project_to_tangent(u0, sol0.u_star)
    interceptor.chi = math.atan2(np.dot(b0, e0), np.dot(b0, n0))

    # ---- Run simulation ----
    sim = Simulator(cfg, target, interceptor)
    
    print(f"Target initial position: {r0}")
    print(f"Target initial velocity: {v0} m/s")
    print(f"Turn rate: {math.degrees(turn_rate):.1f} °/s (right turn)")
    print(f"Initial speed: {norm(v0):.1f} m/s")
    
    # Calculate and display turn radius for reference
    horizontal_speed = norm(v0[:2])
    if abs(turn_rate) > 1e-6:
        turn_radius = horizontal_speed / abs(turn_rate)
        print(f"Turn radius: {turn_radius:.0f} m")
    
    print("\nChoose simulation mode:")
    print("1. Interactive step-by-step")
    print("2. Full auto-run with plots")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        sim.run_interactive()
    else:
        sim.run()
        sim.plot_3d()
        sim.plot_time_series()

def example_with_speed_config():
    """
    Enhanced linear trajectory scenario with configurable speeds.
    """
    print("\n=== LINEAR TRAJECTORY WITH SPEED CONFIG ===")
    
    # Get speed configuration
    shahed_speed, drone_max_speed, speed_ratio = get_speed_configuration()
    
    # ---- Parameters ----
    R = 1000.0                           # protection sphere radius [m]
    dt = 0.05                            # sim step [s]
    T  = 60.0                            # longer simulation time [s]

    # PN tuning - adjust aggressiveness based on speed ratio
    if speed_ratio > 1.5:
        # Shahed much faster - need aggressive PN
        pn = PNParams(
            N=5,
            tgo_mode="impact",               
            pn_mode="apn",                   
            chi_rate_max=math.radians(120.0)  # high turn rate
        )
    elif speed_ratio > 1.1:
        # Shahed somewhat faster - moderate PN
        pn = PNParams(
            N=5,
            tgo_mode="impact",               
            pn_mode="apn",                   
            chi_rate_max=math.radians(90.0)
        )
    else:
        # Drone equal or faster
        pn = PNParams(
            N=5,
            tgo_mode="distance",               
            pn_mode="apn",                   
            chi_rate_max=math.radians(60.0)
        )
    
    cfg = SimConfig(R=R, dt=dt, T=T, v_max=drone_max_speed, pn=pn)

    # ---- Target (Shahed) trajectory with configured speed ----
    r0 = np.array([0.0, -3500.0, 1500.0])   # initial position [m]
    # Create velocity vector with desired speed toward sphere center
    approach_direction = np.array([0.0, +1.0, -0.15])  # slightly downward approach
    v = create_normalized_velocity(approach_direction, shahed_speed)
    target = TargetLinear(r0=r0, v=v)

    # ---- Interceptor initial state on sphere ----
    u0 = unit(np.array([0.3, -0.95, 0.1]))  # unit direction
    interceptor = InterceptorOnSphere(R=R, u=u0, chi=0.0, v_max=drone_max_speed)

    # Initialize heading to current LOS 
    r_t0, _ = target.state(0.0)
    sol0 = predict_impact_or_closest(r_t0, v, R)
    e0, n0 = tangent_basis(u0)
    b0 = project_to_tangent(u0, sol0.u_star)
    interceptor.chi = math.atan2(np.dot(b0, e0), np.dot(b0, n0))

    # ---- Feasibility Analysis ----
    print(f"\n[ANALYSIS] INTERCEPT FEASIBILITY:")
    
    # Calculate time to impact for straight-line approach
    impact_sol = predict_impact_or_closest(r0, v, R)
    if impact_sol.hit:
        time_to_impact = impact_sol.tau
        impact_point = r0 + v * time_to_impact
        print(f"  Time to sphere impact: {time_to_impact:.1f} s")
        print(f"  Impact point: [{impact_point[0]:.0f}, {impact_point[1]:.0f}, {impact_point[2]:.0f}] m")
        
        # Calculate required drone travel distance
        drone_start_pos = R * u0
        drone_impact_distance = norm(impact_point - drone_start_pos)
        required_drone_speed = drone_impact_distance / time_to_impact
        
        print(f"  Drone required average speed: {required_drone_speed:.1f} m/s")
        print(f"  Drone max speed: {drone_max_speed:.1f} m/s")
        
        if required_drone_speed <= drone_max_speed:
            print(f"  [FEASIBLE] INTERCEPT POSSIBLE (margin: {drone_max_speed - required_drone_speed:.1f} m/s)")
        else:
            print(f"  [CHALLENGING] INTERCEPT DIFFICULT (deficit: {required_drone_speed - drone_max_speed:.1f} m/s)")
    else:
        print(f"  [WARNING] No direct sphere impact predicted")

    # ---- Run simulation ----
    sim = Simulator(cfg, target, interceptor)
    
    print(f"\nTarget configured speed: {shahed_speed:.1f} m/s")
    print(f"Target actual speed: {norm(v):.1f} m/s")
    print(f"Speed ratio (Shahed/Drone): {speed_ratio:.2f}")
    
    # Choose simulation mode:
    print("\nChoose simulation mode:")
    print("1. Interactive step-by-step")
    print("2. Full auto-run with plots")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        sim.run_interactive()
    else:
        sim.run()
        sim.plot_3d()
        sim.plot_time_series()

# Add after the existing dataclasses, before the simulation functions

@dataclass
class StandardShahedConfig:
    """
    Standardized Shahed configuration for consistent scenarios.
    All scenarios use the same base parameters with trajectory-specific modifications.
    """
    # Standard starting position and approach
    base_position: np.ndarray = field(default_factory=lambda: np.array([0.0, -3500.0, 1200.0]))
    base_speed: float = 60.0  # m/s - realistic Shahed-136 speed
    approach_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, -0.2]))  # toward sphere center, slightly downward
    
    # Standard drone configuration  
    drone_max_speed: float = 30.0  # m/s
    drone_start_bearing: np.ndarray = field(default_factory=lambda: np.array([0.3, -0.9, 0.3]))  # strategic intercept position
    
    # Standard simulation parameters
    protection_radius: float = 1000.0  # m
    simulation_time: float = 50.0  # s
    time_step: float = 0.05  # s

def create_standard_shahed_trajectory(config: StandardShahedConfig, trajectory_type: str, **kwargs):
    """
    Create standardized Shahed trajectory with consistent base parameters.
    
    Args:
        config: Standard configuration
        trajectory_type: "linear", "curved", "sinusoidal", "turning"
        **kwargs: Trajectory-specific parameters
    """
    # Create base velocity vector with standard speed and direction
    base_velocity = create_normalized_velocity(config.approach_direction, config.base_speed)
    
    if trajectory_type == "linear":
        return TargetLinear(r0=config.base_position, v=base_velocity)
    
    elif trajectory_type == "curved":
        # Extract evasive maneuver parameters
        evasion_radius = kwargs.get('evasion_radius', 150.0)  # m
        evasion_frequency = kwargs.get('evasion_frequency', 0.12)  # rad/s
        
        return TargetCurved(
            r0=config.base_position,
            v_linear=base_velocity,
            orbit_center=config.base_position,
            orbit_radius=evasion_radius,
            omega=evasion_frequency,
            phase_offset=0.0
        )
    
    elif trajectory_type == "sinusoidal":
        # Extract weaving parameters
        weave_amplitude = kwargs.get('weave_amplitude', 200.0)  # m
        weave_frequency = kwargs.get('weave_frequency', 0.08)  # Hz
        
        return TargetSinusoidal(
            r0=config.base_position,
            v_base=base_velocity,
            amplitude=weave_amplitude,
            frequency=weave_frequency
        )
    
    elif trajectory_type == "turning":
        # Extract turning parameters
        turn_rate = kwargs.get('turn_rate', math.radians(1.5))  # rad/s
        
        return TargetTurning(
            r0=config.base_position,
            v0=base_velocity,
            turn_rate=turn_rate
        )
    
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

def create_standard_interceptor(config: StandardShahedConfig) -> InterceptorOnSphere:
    """Create standardized interceptor with consistent positioning."""
    u0 = unit(config.drone_start_bearing)
    return InterceptorOnSphere(
        R=config.protection_radius, 
        u=u0, 
        chi=0.0, 
        v_max=config.drone_max_speed
    )

def get_adaptive_pn_params(trajectory_type: str, speed_ratio: float) -> PNParams:
    """
    Get PN parameters adapted to trajectory type and speed ratio.
    
    Args:
        trajectory_type: Type of target trajectory
        speed_ratio: shahed_speed / drone_speed
    """
    if trajectory_type == "linear":
        if speed_ratio > 1.8:
            return PNParams(N=5, tgo_mode="impact", pn_mode="apn", chi_rate_max=math.radians(120.0))
        elif speed_ratio > 1.3:
            return PNParams(N=5, tgo_mode="impact", pn_mode="apn", chi_rate_max=math.radians(90.0))
        else:
            return PNParams(N=5, tgo_mode="impact", pn_mode="apn", chi_rate_max=math.radians(60.0))
            
    elif trajectory_type in ["curved", "sinusoidal"]:
        # Evasive trajectories need more aggressive PN
        return PNParams(N=5, tgo_mode="impact", pn_mode="apn", chi_rate_max=math.radians(150.0))
        
    elif trajectory_type == "turning":
        # Turning trajectories need moderate PN with good prediction
        return PNParams(N=5, tgo_mode="impact", pn_mode="apn", chi_rate_max=math.radians(100.0))
    
    else:
        # Default aggressive settings
        return PNParams(N=5, tgo_mode="impact", pn_mode="apn", chi_rate_max=math.radians(120.0))

# Updated example functions with consistent parameters
def example_standardized():
    """Standardized linear trajectory scenario."""
    print("\n=== STANDARDIZED LINEAR TRAJECTORY ===")
    
    config = StandardShahedConfig()
    print_scenario_info(config, "linear")
    
    target = create_standard_shahed_trajectory(config, "linear")
    interceptor = create_standard_interceptor(config)
    
    # Initialize interceptor heading
    initialize_interceptor_heading(interceptor, target, config)
    
    # Adaptive PN parameters
    speed_ratio = config.base_speed / config.drone_max_speed
    pn_params = get_adaptive_pn_params("linear", speed_ratio)
    
    cfg = SimConfig(
        R=config.protection_radius, 
        dt=config.time_step, 
        T=config.simulation_time, 
        v_max=config.drone_max_speed, 
        pn=pn_params
    )
    
    run_simulation_with_choice(cfg, target, interceptor)

def example_standardized_curved():
    """Standardized curved/evasive trajectory scenario."""
    print("\n=== STANDARDIZED CURVED TRAJECTORY ===")
    
    config = StandardShahedConfig()
    
    # Evasive maneuver parameters
    evasion_radius = 150.0  # m - radius of evasive circles
    evasion_frequency = 0.12  # rad/s - how fast it spirals
    
    print_scenario_info(config, "curved", 
                       evasion_radius=evasion_radius, 
                       evasion_frequency=evasion_frequency)
    
    target = create_standard_shahed_trajectory(
        config, "curved", 
        evasion_radius=evasion_radius,
        evasion_frequency=evasion_frequency
    )
    interceptor = create_standard_interceptor(config)
    
    # Initialize interceptor heading
    initialize_interceptor_heading(interceptor, target, config)
    
    # Adaptive PN parameters for evasive target
    speed_ratio = config.base_speed / config.drone_max_speed
    pn_params = get_adaptive_pn_params("curved", speed_ratio)
    
    cfg = SimConfig(
        R=config.protection_radius, 
        dt=config.time_step, 
        T=config.simulation_time, 
        v_max=config.drone_max_speed, 
        pn=pn_params
    )
    
    run_simulation_with_choice(cfg, target, interceptor)

def example_standardized_sinusoidal():
    """Standardized sinusoidal weaving trajectory scenario."""
    print("\n=== STANDARDIZED SINUSOIDAL TRAJECTORY ===")
    
    config = StandardShahedConfig()
    
    # Weaving parameters
    weave_amplitude = 200.0  # m
    weave_frequency = 0.08   # Hz
    
    print_scenario_info(config, "sinusoidal",
                       weave_amplitude=weave_amplitude,
                       weave_frequency=weave_frequency)
    
    target = create_standard_shahed_trajectory(
        config, "sinusoidal",
        weave_amplitude=weave_amplitude,
        weave_frequency=weave_frequency
    )
    interceptor = create_standard_interceptor(config)
    
    # Initialize interceptor heading
    initialize_interceptor_heading(interceptor, target, config)
    
    # Adaptive PN parameters
    speed_ratio = config.base_speed / config.drone_max_speed
    pn_params = get_adaptive_pn_params("sinusoidal", speed_ratio)
    
    cfg = SimConfig(
        R=config.protection_radius, 
        dt=config.time_step, 
        T=config.simulation_time, 
        v_max=config.drone_max_speed, 
        pn=pn_params
    )
    
    run_simulation_with_choice(cfg, target, interceptor)

def example_standardized_turning():
    """Standardized turning trajectory scenario."""
    print("\n=== STANDARDIZED TURNING TRAJECTORY ===")
    
    config = StandardShahedConfig()
    
    # Turning parameters
    turn_rate = math.radians(1.5)  # 1.5 degrees/second
    
    print_scenario_info(config, "turning", turn_rate=turn_rate)
    
    target = create_standard_shahed_trajectory(
        config, "turning",
        turn_rate=turn_rate
    )
    interceptor = create_standard_interceptor(config)
    
    # Initialize interceptor heading
    initialize_interceptor_heading(interceptor, target, config)
    
    # Adaptive PN parameters
    speed_ratio = config.base_speed / config.drone_max_speed
    pn_params = get_adaptive_pn_params("turning", speed_ratio)
    
    cfg = SimConfig(
        R=config.protection_radius, 
        dt=config.time_step, 
        T=config.simulation_time, 
        v_max=config.drone_max_speed, 
        pn=pn_params
    )
    
    run_simulation_with_choice(cfg, target, interceptor)

# Helper functions
def print_scenario_info(config: StandardShahedConfig, trajectory_type: str, **kwargs):
    """Print standardized scenario information."""
    print(f"[PARAMS] STANDARDIZED SCENARIO PARAMETERS:")
    print(f"  Shahed starting position: {config.base_position}")
    print(f"  Shahed speed: {config.base_speed:.1f} m/s")
    print(f"  Drone max speed: {config.drone_max_speed:.1f} m/s")
    print(f"  Speed ratio: {config.base_speed/config.drone_max_speed:.2f}")
    print(f"  Protection radius: {config.protection_radius:.0f} m")
    
    if trajectory_type == "curved":
        print(f"  Evasive circles radius: {kwargs.get('evasion_radius', 150):.0f} m")
        print(f"  Evasive frequency: {kwargs.get('evasion_frequency', 0.12):.3f} rad/s")
    elif trajectory_type == "sinusoidal":
        print(f"  Weaving amplitude: {kwargs.get('weave_amplitude', 200):.0f} m")
        print(f"  Weaving frequency: {kwargs.get('weave_frequency', 0.08):.3f} Hz")
    elif trajectory_type == "turning":
        turn_rate = kwargs.get('turn_rate', math.radians(1.5))
        print(f"  Turn rate: {math.degrees(turn_rate):.1f} °/s")

def initialize_interceptor_heading(interceptor: InterceptorOnSphere, target, config: StandardShahedConfig):
    """Initialize interceptor heading toward predicted aimpoint."""
    r_t0, v_t0 = target.state(0.0)
    sol0 = predict_impact_or_closest(r_t0, v_t0, config.protection_radius)
    e0, n0 = tangent_basis(interceptor.u)
    b0 = project_to_tangent(interceptor.u, sol0.u_star)
    interceptor.chi = math.atan2(np.dot(b0, e0), np.dot(b0, n0))

def run_simulation_with_choice(cfg: SimConfig, target, interceptor: InterceptorOnSphere):
    """Run simulation with user choice of mode."""
    sim = Simulator(cfg, target, interceptor)
    
    print("\nChoose simulation mode:")
    print("1. Interactive step-by-step")
    print("2. Full auto-run with plots")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        sim.run_interactive()
    else:
        sim.run()
        sim.plot_3d()
        sim.plot_time_series()

# Updated main function
def main():
    """
    Main function with standardized scenarios.
    """
    print("=== Spherical Proportional Navigation Simulator ===")
    print("Choose a scenario (all use STANDARDIZED parameters):")
    print("1. Linear trajectory")
    print("2. Curved/helical trajectory (evasive circles)") 
    print("3. Sinusoidal weaving trajectory")
    print("4. Simple turning trajectory")
    print("5. Linear with custom speed configuration")
    print("6. Original (legacy) examples")
    print("7. Exit")
    
    while True:
        try:
            choice = input("\nEnter scenario number (1-7): ").strip()
            
            if choice == "1":
                example_standardized()
                break
            elif choice == "2":
                example_standardized_curved()
                break
            elif choice == "3":
                example_standardized_sinusoidal()
                break
            elif choice == "4":
                example_standardized_turning()
                break
            elif choice == "5":
                example_with_speed_config()
                break
            elif choice == "6":
                # Sub-menu for legacy examples
                print("\nLegacy examples (inconsistent parameters):")
                print("a. Linear (original)")
                print("b. Curved (original)")
                print("c. Sinusoidal (original)")
                print("d. Turning (original)")
                legacy_choice = input("Choose legacy example (a-d): ").strip().lower()
                
                if legacy_choice == "a":
                    example()
                elif legacy_choice == "b":
                    example_curved_trajectory()
                elif legacy_choice == "c":
                    example_sinusoidal_trajectory()
                elif legacy_choice == "d":
                    example_simple_turn()
                break
            elif choice == "7":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
