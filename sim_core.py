"""
MSSDPPG ULTRA-REALISTIC SIMULATION ENGINE
==========================================
Complete bidirectional system simulation with:
- Container geometry constraints (swing angle limits)
- Heat generation and thermal effects
- Friction (bearing, air, mechanical)
- Wind resistance and drag
- Electromagnetic damping
- Lock-release controller
- Real-world efficiency losses

Author: Based on validated system physics
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.signal import find_peaks
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

g = 9.81              # Gravity (m/s²)
rho_air = 1.225       # Air density (kg/m³)
nu_air = 1.5e-5       # Kinematic viscosity (m²/s)
T_ambient = 298.15    # Ambient temperature (K) = 25°C

# =============================================================================
# SYSTEM CONFIGURATIONS
# =============================================================================

@dataclass
class SystemConfig:
    """Complete system specification"""
    name: str

    # Geometry
    L1: float                    # Upper arm length (m)
    L2: float                    # Lower arm length (m)
    vane_width: float           # Vane width (m)
    vane_height: float          # Vane height (m)

    # Masses
    m_upper_arm: float          # Upper arm distributed mass (kg)
    m_middle: float             # Middle hinge concentrated mass (kg)
    m_lower_arm: float          # Lower arm distributed mass (kg)
    m_tip: float                # Tip mass (kg)

    # System layout
    num_pendulums: int          # Total pendulums
    num_shafts: int             # Number of shafts
    num_hinge_gens: int         # Hinge generators
    num_alternators: int        # Alternators

    # Container constraints
    container_width: float      # Internal width (m)
    container_height: float     # Internal height (m)
    max_swing_angle: float      # Maximum safe swing angle (rad)

    # Coefficients (scale with size)
    bearing_friction: float     # Bearing friction coefficient
    air_drag_coeff: float       # Vane drag coefficient
    mechanical_loss: float      # Mechanical transmission loss

    # Expected performance
    expected_power_6ms: float   # kW @ 6 m/s
    color: str = '#4ECDC4'

# Define all 4 scenarios
SCENARIOS = {
    '4x40ft': SystemConfig(
        name='4×40ft Container (Main Product)',
        L1=2.0,
        L2=2.0,
        vane_width=1.0,
        vane_height=2.0,
        m_upper_arm=5.0,
        m_middle=30.0,      # 30kg middle mass
        m_lower_arm=3.0,
        m_tip=5.0,
        num_pendulums=48,   # 48 total (24 per shaft)
        num_shafts=2,
        num_hinge_gens=96,  # 96 hinge generators (2 per pendulum)
        num_alternators=4,
        container_width=2.44,   # Standard ISO container internal width
        container_height=2.59,  # Standard ISO container internal height
        max_swing_angle=np.deg2rad(55),  # Limited by container
        bearing_friction=0.015,
        air_drag_coeff=1.2,
        mechanical_loss=0.03,
        expected_power_6ms=19.6,
        color='#4ECDC4'
    ),

    '1x20ft': SystemConfig(
        name='1×20ft Container (Magic 20ft)',
        L1=1.4,
        L2=1.4,
        vane_width=0.7,
        vane_height=1.4,
        m_upper_arm=2.45,
        m_middle=14.7,
        m_lower_arm=1.47,
        m_tip=2.45,
        num_pendulums=24,
        num_shafts=1,
        num_hinge_gens=48,
        num_alternators=2,
        container_width=2.44,   # Standard ISO container internal width
        container_height=2.59,  # Standard ISO container internal height
        max_swing_angle=np.deg2rad(60),
        bearing_friction=0.012,
        air_drag_coeff=1.2,
        mechanical_loss=0.025,
        expected_power_6ms=1.79,
        color='#95E1D3'
    ),

    'tower': SystemConfig(
        name='Tower Cantilever (Building Facade)',
        L1=0.75,
        L2=0.75,
        vane_width=0.4,
        vane_height=0.75,
        m_upper_arm=0.28,
        m_middle=4.2,
        m_lower_arm=0.17,
        m_tip=0.7,
        num_pendulums=8,
        num_shafts=1,
        num_hinge_gens=16,
        num_alternators=2,
        container_width=1.5,
        container_height=2.5,
        max_swing_angle=np.deg2rad(65),
        bearing_friction=0.010,
        air_drag_coeff=1.2,
        mechanical_loss=0.02,
        expected_power_6ms=0.684,
        color='#F38181'
    ),

    'mega': SystemConfig(
        name='15m Mega-Pendulum (Utility Scale)',
        L1=6.0,
        L2=6.0,
        vane_width=3.0,
        vane_height=6.0,
        m_upper_arm=45.0,
        m_middle=120.0,     # 120kg middle mass
        m_lower_arm=27.0,
        m_tip=30.0,
        num_pendulums=1,
        num_shafts=1,
        num_hinge_gens=2,
        num_alternators=1,
        container_width=8.0,
        container_height=15.0,
        max_swing_angle=np.deg2rad(45),
        bearing_friction=0.020,
        air_drag_coeff=1.2,
        mechanical_loss=0.04,
        expected_power_6ms=77.2,
        color='#AA96DA'
    )
}

# =============================================================================
# ELECTROMAGNETIC GENERATOR MODEL
# =============================================================================

class HingeGenerator:
    """Realistic hinge generator with thermal model"""

    def __init__(self, size_scale=1.0):
        self.size_scale = size_scale

        # Scale parameters with size
        self.k_t = 0.75 * size_scale          # Torque constant (Nm/A)
        self.R_coil_25C = 0.45 * size_scale   # Coil resistance at 25°C (Ω)
        self.gen_efficiency = 0.85             # Electrical efficiency

        # Thermal model
        self.T_coil = T_ambient
        self.C_thermal = 250.0 * size_scale    # Thermal capacitance (J/K)
        self.R_thermal = 1.5 / size_scale      # Thermal resistance (K/W)
        self.T_max = 423.15                    # 150°C max

        # Lock-release controller
        self.i_high = 6.0 * size_scale         # Lock current (A)
        self.i_low = 1.5 * size_scale          # Release current (A)
        self.i_current = self.i_low

    def get_coil_resistance(self):
        """Temperature-dependent coil resistance"""
        alpha_cu = 0.00393  # Copper temp coefficient (1/K)
        R = self.R_coil_25C * (1 + alpha_cu * (self.T_coil - 298.15))
        return R

    def update_thermal(self, P_loss, dt):
        """Update coil temperature from losses"""
        dT = (P_loss * self.R_thermal - (self.T_coil - T_ambient)) * dt / (self.R_thermal * self.C_thermal)
        self.T_coil = np.clip(self.T_coil + dT, T_ambient, self.T_max)

    def get_torque_and_power(self, omega, locked=False):
        """Calculate EM torque and electrical power"""
        # Set current based on lock state
        self.i_current = self.i_high if locked else self.i_low

        # Thermal derating
        if self.T_coil > 373.15:  # Start derating at 100°C
            derate_factor = 1.0 - (self.T_coil - 373.15) / (self.T_max - 373.15)
            self.i_current *= max(0.5, derate_factor)

        # Torque (opposes motion)
        T_em = -self.k_t * self.i_current * np.sign(omega) if abs(omega) > 0.01 else 0.0

        # Power calculations
        P_mech = abs(T_em * omega)
        R = self.get_coil_resistance()
        P_copper = self.i_current**2 * R
        P_elec = max(0.0, (P_mech - P_copper) * self.gen_efficiency)

        return T_em, P_elec, P_copper

# =============================================================================
# ULTRA-REALISTIC DOUBLE PENDULUM
# =============================================================================

class RealisticDoublePendulum:
    """Double pendulum with ALL physical effects"""

    def __init__(self, config: SystemConfig, wind_speed=6.0):
        self.config = config
        self.wind_speed = wind_speed

        # Initial state [theta1, omega1, theta2, omega2]
        self.state = [0.15, 0.0, 0.0, 0.0]

        # Generators
        size_scale = config.L1 / 2.0  # Scale relative to 2m reference
        self.gen_upper = HingeGenerator(size_scale)
        self.gen_lower = HingeGenerator(size_scale)

        # Lock-release state
        self.last_zero_cross_time = 0.0
        self.lock_active = False

        # Bearing thermal state
        self.T_bearing = T_ambient

        # Power tracking
        self.power_hinge_upper = 0.0
        self.power_hinge_lower = 0.0
        self.power_shaft = 0.0
        self.heat_loss = 0.0

        # History for plotting
        self.time_history = []
        self.theta1_history = []
        self.omega1_history = []
        self.power_history = []
        self.lock_history = []
        self.temp_history = []
        self.max_angle_reached = 0.0

    def check_container_collision(self, theta1, theta2):
        """Check if pendulum would hit container walls"""
        c = self.config

        # Calculate tip position
        x_tip = c.L1 * np.sin(theta1) + c.L2 * np.sin(theta2)
        y_tip = -c.L1 * np.cos(theta1) - c.L2 * np.cos(theta2)

        # Check against container bounds
        half_width = c.container_width / 2.0

        collision = (abs(x_tip) > half_width or
                    y_tip > 0 or
                    abs(theta1) > c.max_swing_angle)

        return collision

    def mass_matrix(self, theta1, theta2):
        """Mass matrix with middle mass configuration"""
        c = self.config
        cos_delta = np.cos(theta1 - theta2)

        # Upper pendulum inertia
        I1_arm = (1/3) * c.m_upper_arm * c.L1**2
        I1_middle = c.m_middle * c.L1**2  # Middle mass
        I1_lower = (c.m_lower_arm + c.m_tip) * c.L1**2
        M11 = I1_arm + I1_middle + I1_lower

        # Lower pendulum inertia
        I2_arm = (1/3) * c.m_lower_arm * c.L2**2
        I2_tip = c.m_tip * c.L2**2
        M22 = I2_arm + I2_tip

        # Coupling
        M12 = (c.m_lower_arm * 0.5 + c.m_tip) * c.L1 * c.L2 * cos_delta

        return np.array([[M11, M12], [M12, M22]])

    def wind_torque(self, theta, omega):
        """Wind force with realistic drag and venturi effect"""
        c = self.config

        # Venturi boost from container channeling
        venturi = 1.15 if c.num_pendulums > 8 else 1.0
        v_wind_eff = self.wind_speed * venturi

        # Relative velocity (subtract vane motion)
        v_vane = abs(omega) * c.L1
        v_rel = max(0.1, v_wind_eff - v_vane)

        # Drag force
        A_vane = c.vane_width * c.vane_height
        F_drag = 0.5 * rho_air * c.air_drag_coeff * A_vane * v_rel**2

        # Torque (perpendicular component)
        T = F_drag * c.L1 * abs(np.sin(theta))

        # Direction: assists motion in wind direction
        return T if omega >= 0 else -T

    def bearing_friction_torque(self, omega):
        """Realistic bearing friction (viscous + Coulomb)"""
        c = self.config

        # Temperature-dependent viscosity
        T_factor = (self.T_bearing - T_ambient) / 50.0
        visc_red = max(0.7, 1.0 - 0.3 * T_factor)  # Viscosity drops with temp

        # Viscous friction
        T_visc = -c.bearing_friction * omega * visc_red

        # Coulomb friction
        T_coulomb = -0.3 * np.sign(omega) if abs(omega) > 0.01 else 0.0

        return T_visc + T_coulomb

    def update_bearing_thermal(self, P_loss, dt):
        """Update bearing temperature"""
        R_th = 0.25  # K/W
        C_th = 8000.0  # J/K
        dT = (P_loss * R_th - (self.T_bearing - T_ambient)) * dt / (R_th * C_th)
        self.T_bearing = np.clip(self.T_bearing + dT, T_ambient, 373.15)

    def clutch_torque_bidirectional(self, omega, available_power):
        """Bidirectional clutch with realistic efficiency"""
        c = self.config

        if abs(omega) < 0.1:
            return 0.0, 0.0, 0.0

        # Clutch torque limit (150 Nm scaled)
        size_scale = c.L1 / 2.0
        T_limit = 150.0 * size_scale

        # Power-limited torque
        T_available = min(available_power / abs(omega), T_limit)

        # Clutch efficiency (97% average)
        clutch_eff = 0.97
        T_clutch = T_available * clutch_eff

        # Heat generation from clutch slip
        P_clutch_loss = T_available * abs(omega) * (1 - clutch_eff)

        # Resistive torque on arm
        T_resistive = -T_clutch * np.sign(omega)
        P_output = T_clutch * abs(omega)

        return T_resistive, P_output, P_clutch_loss

    def lock_release_controller(self, theta1, omega1, t):
        """Electromagnetic lock-release control"""
        # Zero-crossing detection
        if abs(omega1) < 0.05 and t > self.last_zero_cross_time + 0.3:
            self.last_zero_cross_time = t

        # Lock window (during high-angle swing)
        lock_theta_min = np.deg2rad(15)
        lock_theta_max = min(np.deg2rad(55), self.config.max_swing_angle)

        in_lock_window = (abs(omega1) > 0.1 and
                         lock_theta_min <= abs(theta1) <= lock_theta_max)

        # Release after zero-crossing
        time_since_cross = t - self.last_zero_cross_time
        in_release_window = time_since_cross < 0.12

        # Set lock state
        self.lock_active = in_lock_window and not in_release_window

        return self.lock_active

    def equations_of_motion(self, state, t):
        """Complete physics with ALL effects"""
        theta1, omega1, theta2, omega2 = state
        c = self.config

        # Check for collision
        if self.check_container_collision(theta1, theta2):
            # Emergency damping to prevent collision
            return [omega1, -50*omega1, omega2, -50*omega2]

        # Track maximum angle
        self.max_angle_reached = max(self.max_angle_reached, abs(theta1))

        # Mass matrix
        M = self.mass_matrix(theta1, theta2)

        # Gravity torques (with middle mass)
        T_g1_upper = -c.m_upper_arm * g * (c.L1/2) * np.sin(theta1)
        T_g1_middle = -c.m_middle * g * c.L1 * np.sin(theta1)
        T_g1_lower = -(c.m_lower_arm + c.m_tip) * g * c.L1 * np.sin(theta1)
        T_g1 = T_g1_upper + T_g1_middle + T_g1_lower

        T_g2_arm = -c.m_lower_arm * g * (c.L2/2) * np.sin(theta2)
        T_g2_tip = -c.m_tip * g * c.L2 * np.sin(theta2)
        T_g2 = T_g2_arm + T_g2_tip

        # Wind torques
        T_wind1 = self.wind_torque(theta1, omega1)
        T_wind2 = self.wind_torque(theta2, omega2) * 0.7  # Lower arm less effective

        # Bearing friction
        T_bearing1 = self.bearing_friction_torque(omega1)
        T_bearing2 = self.bearing_friction_torque(omega2)

        # Available wind power
        P_wind_upper = abs(T_wind1 * omega1)
        P_wind_lower = abs(T_wind2 * omega2)

        # Lock-release control
        lock_state = self.lock_release_controller(theta1, omega1, t)

        # Generator torques and power
        T_em_upper, P_hinge_upper, P_cu_upper = self.gen_upper.get_torque_and_power(omega1, lock_state)
        T_em_lower, P_hinge_lower, P_cu_lower = self.gen_lower.get_torque_and_power(omega2, False)

        # Clutch torque (bidirectional, upper hinge only)
        P_available_clutch = max(0, P_wind_upper - P_hinge_upper)
        T_clutch, P_shaft, P_clutch_loss = self.clutch_torque_bidirectional(omega1, P_available_clutch)

        # Coupling term (Coriolis)
        h = (c.m_lower_arm * 0.5 + c.m_tip) * c.L1 * c.L2 * omega1 * omega2 * np.sin(theta1 - theta2)
        h = np.clip(h, -5000, 5000)

        # Total torques
        T1 = T_wind1 + T_g1 + T_bearing1 + T_em_upper + T_clutch + h
        T2 = T_wind2 + T_g2 + T_bearing2 + T_em_lower - h

        # Solve for accelerations
        T_vec = np.array([T1, T2])
        try:
            alpha = np.linalg.solve(M, T_vec)
        except:
            alpha = np.array([0.0, 0.0])

        # Clip accelerations for stability
        alpha = np.clip(alpha, -500, 500)

        # Update thermal models
        self.gen_upper.update_thermal(P_cu_upper, 0.01)
        self.gen_lower.update_thermal(P_cu_lower, 0.01)
        P_bearing_loss = abs(T_bearing1 * omega1) + abs(T_bearing2 * omega2)
        self.update_bearing_thermal(P_bearing_loss + P_clutch_loss, 0.01)

        # Store power values
        self.power_hinge_upper = P_hinge_upper
        self.power_hinge_lower = P_hinge_lower
        self.power_shaft = P_shaft
        self.heat_loss = P_cu_upper + P_cu_lower + P_bearing_loss + P_clutch_loss

        return [omega1, alpha[0], omega2, alpha[1]]

# =============================================================================
# SYSTEM SIMULATOR
# =============================================================================

def simulate_system(config: SystemConfig, wind_speeds=[4, 6, 8], duration=20.0):
    """Simulate complete system at multiple wind speeds"""

    results_by_speed = {}

    for wind_speed in wind_speeds:
        # Create representative pendulum
        pendulum = RealisticDoublePendulum(config, wind_speed)

        # Time array
        dt = 0.01
        time = np.arange(0, duration, dt)

        # Solve ODE
        solution = odeint(pendulum.equations_of_motion, pendulum.state, time,
                         full_output=False)

        theta1 = solution[:, 0]
        omega1 = solution[:, 1]
        theta2 = solution[:, 2]
        omega2 = solution[:, 3]

        # Calculate positions
        x1 = config.L1 * np.sin(theta1)
        y1 = -config.L1 * np.cos(theta1)
        x2 = x1 + config.L2 * np.sin(theta2)
        y2 = y1 - config.L2 * np.cos(theta2)

        # Cycle analysis
        peaks, _ = find_peaks(theta1, height=0, distance=int(0.5/dt))
        if len(peaks) > 1:
            periods = np.diff(time[peaks])
            cycle_time = np.mean(periods)
            frequency = 1.0 / cycle_time
        else:
            cycle_time = np.nan
            frequency = np.nan

        # Power calculation (scale to full system)
        avg_power_per_pendulum = (pendulum.power_hinge_upper +
                                  pendulum.power_hinge_lower +
                                  pendulum.power_shaft) / 1000  # kW

        # Total system power
        hinge_power_total = avg_power_per_pendulum * config.num_pendulums * 0.5
        shaft_power_total = avg_power_per_pendulum * config.num_pendulums * 0.5

        # Alternator conversion
        alternator_power = shaft_power_total * 0.95 * 0.88

        total_power = hinge_power_total + alternator_power

        # Store results
        results_by_speed[wind_speed] = {
            'time': time,
            'theta1': theta1,
            'omega1': omega1,
            'theta2': theta2,
            'omega2': omega2,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'cycle_time': cycle_time,
            'frequency': frequency,
            'power_per_pendulum': avg_power_per_pendulum,
            'hinge_power': hinge_power_total,
            'alternator_power': alternator_power,
            'total_power': total_power,
            'max_angle': np.rad2deg(pendulum.max_angle_reached),
            'avg_temp': (pendulum.gen_upper.T_coil + pendulum.gen_lower.T_coil)/2,
            'pendulum': pendulum
        }

    return results_by_speed
