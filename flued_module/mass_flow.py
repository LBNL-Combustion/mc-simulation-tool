"""
Mass flow and pressure calculations for chimney simulation
Based on algorithms from FLUESIM5.BAS mass.flow and pres.param subroutines

Extended with:
- Colebrook-White friction factor (optional)
- Distributed leakage (VENT-II-style)
- Acceleration/deceleration loss terms
"""
import numpy as np
from .physics import PhysicalProperties


# =============================================================================
# FRICTION FACTOR CALCULATIONS (Part C)
# =============================================================================

def friction_factor(Re: float, rel_rough: float = 0.0, method: str = "current") -> float:
    """Calculate Darcy friction factor

    Supports two methods:
    - "current": Existing Blasius (smooth) / Swamee-Jain (rough) correlations
    - "colebrook": Iterative Colebrook-White solution (more accurate for turbulent flow)

    Args:
        Re: Reynolds number (dimensionless)
        rel_rough: Relative roughness ε/D (dimensionless), default 0.0 for smooth
        method: "current" or "colebrook"

    Returns:
        Darcy friction factor (dimensionless)
    """
    # Laminar flow: always use f = 64/Re regardless of method
    if Re < 2300:
        return 64.0 / max(Re, 1.0)

    # Turbulent flow
    if method == "colebrook":
        return _friction_factor_colebrook(Re, rel_rough)
    else:
        # Current method: Blasius (smooth) or Swamee-Jain (rough)
        return _friction_factor_current(Re, rel_rough)


def _friction_factor_current(Re: float, rel_rough: float) -> float:
    """Current friction factor method: Blasius (smooth) / Swamee-Jain (rough)

    Args:
        Re: Reynolds number (turbulent, Re >= 2300)
        rel_rough: Relative roughness ε/D

    Returns:
        Darcy friction factor
    """
    if rel_rough < 1e-6:
        # Blasius correlation for smooth pipes
        return 0.316 / Re**0.25
    else:
        # Swamee-Jain explicit approximation for rough pipes
        return 0.25 / (np.log10(rel_rough/3.7 + 5.74/Re**0.9))**2


def _friction_factor_colebrook(Re: float, rel_rough: float) -> float:
    """Colebrook-White iterative friction factor solution

    Solves: 1/sqrt(f) = -2.0*log10( (ε/D)/3.7 + 2.51/(Re*sqrt(f)) )

    Args:
        Re: Reynolds number (turbulent, Re >= 2300)
        rel_rough: Relative roughness ε/D

    Returns:
        Darcy friction factor
    """
    # Initial guess
    f = 0.02

    # Iterative solution
    for _ in range(30):
        f_old = f
        # Colebrook-White equation rearranged for f_new
        # 1/sqrt(f) = -2*log10( rel_rough/3.7 + 2.51/(Re*sqrt(f)) )
        sqrt_f = np.sqrt(f)
        rhs = -2.0 * np.log10(rel_rough/3.7 + 2.51/(Re * sqrt_f))
        f_new = 1.0 / (rhs**2) if rhs != 0 else f

        # Clamp to physical range
        f = np.clip(f_new, 1e-4, 1.0)

        # Check convergence
        if abs(f - f_old) < 1e-8:
            break

    return f


# =============================================================================
# DISTRIBUTED LEAKAGE CALCULATIONS (Part A)
# =============================================================================

def calculate_leakage_mass_flow(P_gauge: float, A_leak_m2: float, Cd_leak: float,
                                 rho_ambient: float, rho_gas: float) -> float:
    """Calculate leakage mass flow at a segment joint/location

    Leakage model:
    - If P_gauge > 0: gas pressure above ambient → leaking OUT (uses rho_ambient)
    - If P_gauge < 0: gas pressure below ambient → leaking IN (uses rho_gas)

    Args:
        P_gauge: Local gauge pressure P_segment - P_ambient [Pa]
        A_leak_m2: Leakage area [m²]
        Cd_leak: Discharge coefficient for leakage (default 0.61)
        rho_ambient: Ambient air density [kg/m³]
        rho_gas: Local gas density [kg/m³]

    Returns:
        Signed leakage mass flow [kg/s]:
        - Negative: leakage OUT (reduces downstream mass flow)
        - Positive: leakage IN (adds to downstream mass flow)
    """
    if A_leak_m2 <= 0:
        return 0.0

    # Select reference density based on pressure difference direction (VENT-II logic)
    if P_gauge > 0:
        # Leaking OUT: use ambient density (gas expands to ambient)
        rho_ref = rho_ambient
    else:
        # Leaking IN: use gas density (ambient air enters and heats up)
        rho_ref = rho_gas

    # Orifice flow: m_dot = Cd * A * sqrt(2 * rho * |ΔP|)
    dP_abs = abs(P_gauge)
    m_leak_magnitude = Cd_leak * A_leak_m2 * np.sqrt(2.0 * rho_ref * dP_abs)

    # Sign convention:
    # +ΔP (above ambient) → outflow → subtract from downstream → negative
    # -ΔP (below ambient) → inflow → add to downstream → positive
    m_leak_signed = -np.sign(P_gauge) * m_leak_magnitude

    return m_leak_signed


# =============================================================================
# ACCELERATION/DECELERATION LOSS (Part B)
# =============================================================================

def calculate_accel_pressure_loss(rho_upstream: float, v_upstream: float,
                                   rho_downstream: float, v_downstream: float,
                                   C_accel: float = 1.0) -> float:
    """Calculate pressure loss due to velocity head change between segments

    Steady loss term representing acceleration/deceleration:
        dp_accel = C_accel * (q_downstream - q_upstream)
    where q = 0.5 * rho * v²

    Sign convention:
    - If velocity increases downstream → positive dp_accel → additional loss
    - If velocity decreases downstream → negative dp_accel → recovery

    Args:
        rho_upstream: Gas density at upstream segment [kg/m³]
        v_upstream: Gas velocity at upstream segment [m/s]
        rho_downstream: Gas density at downstream segment [kg/m³]
        v_downstream: Gas velocity at downstream segment [m/s]
        C_accel: Acceleration loss coefficient (default 1.0)

    Returns:
        Pressure loss [Pa] (positive = loss opposing flow)
    """
    # Velocity heads (dynamic pressure)
    q_upstream = 0.5 * rho_upstream * v_upstream**2
    q_downstream = 0.5 * rho_downstream * v_downstream**2

    # Acceleration loss: change in velocity head
    dp_accel = C_accel * (q_downstream - q_upstream)

    return dp_accel

class MassFlow:
    """Mass flow and pressure calculations for flue systems"""
    
    def __init__(self):
        self.physics = PhysicalProperties()
    
    def buoyancy_pressure(self, height, density_cold, density_hot):
        """Calculate buoyancy pressure difference
        
        Args:
            height: Height difference (m)
            density_cold: Density of cold fluid (kg/m³)
            density_hot: Density of hot fluid (kg/m³)
            
        Returns:
            Buoyancy pressure (Pa)
        """
        return (density_cold - density_hot) * self.physics.GRAVITY * height
    
    def pressure_loss_friction(self, friction_factor, length, diameter, density, velocity):
        """Calculate pressure loss due to friction
        
        Args:
            friction_factor: Darcy friction factor (dimensionless)
            length: Pipe length (m)
            diameter: Pipe diameter (m)
            density: Fluid density (kg/m³)
            velocity: Flow velocity (m/s)
            
        Returns:
            Pressure loss (Pa)
        """
        return friction_factor * (length / diameter) * (density * velocity**2) / 2
    
    def pressure_loss_minor(self, loss_coefficient, density, velocity):
        """Calculate minor pressure losses
        
        Args:
            loss_coefficient: Minor loss coefficient (dimensionless)
            density: Fluid density (kg/m³)
            velocity: Flow velocity (m/s)
            
        Returns:
            Pressure loss (Pa)
        """
        return loss_coefficient * (density * velocity**2) / 2
    
    def calculate_mass_flow_rate(self, driving_pressure, total_resistance):
        """Calculate mass flow rate from driving pressure and system resistance
        
        Args:
            driving_pressure: Net driving pressure (Pa)
            total_resistance: Total system flow resistance (Pa·s²/kg²)
            
        Returns:
            Mass flow rate (kg/s)
        """
        if driving_pressure <= 0 or total_resistance <= 0:
            return 0.001  # Minimum flow to avoid division by zero
        
        # From pressure balance: dp = resistance * (mass_flow / area)²
        # Solve for mass flow rate
        mass_flow = np.sqrt(abs(driving_pressure) / total_resistance)
        
        return mass_flow
    
    def system_resistance_analysis(self, segments_data):
        """Calculate total system flow resistance
        
        Args:
            segments_data: List of dictionaries with segment properties
            Each dict should contain: length, diameter, friction_factor, minor_losses
            
        Returns:
            Total flow resistance (Pa·s²/kg²)
        """
        total_resistance = 0.0
        
        for segment in segments_data:
            length = segment['length']
            diameter = segment['diameter']
            friction_factor = segment['friction_factor']
            minor_losses = segment.get('minor_losses', [])
            
            # Area
            area = np.pi * (diameter / 2) ** 2
            
            # Friction resistance
            friction_resistance = friction_factor * length / (diameter * area**2)
            
            # Minor loss resistance
            minor_resistance = sum(minor_losses) / area**2
            
            # Total segment resistance
            segment_resistance = friction_resistance + minor_resistance
            total_resistance += segment_resistance
        
        return total_resistance
    
    def draft_calculation(self, chimney_height, T_gas_avg, T_ambient, pressure_ambient):
        """Calculate natural draft pressure
        
        Args:
            chimney_height: Total chimney height (m)
            T_gas_avg: Average gas temperature in chimney (K)
            T_ambient: Ambient temperature (K)
            pressure_ambient: Ambient pressure (Pa)
            
        Returns:
            Draft pressure (Pa)
        """
        # Densities
        rho_ambient = self.physics.air_density(T_ambient, pressure_ambient)
        rho_gas = self.physics.air_density(T_gas_avg, pressure_ambient)
        
        # Draft pressure
        draft = self.buoyancy_pressure(chimney_height, rho_ambient, rho_gas)
        
        return draft