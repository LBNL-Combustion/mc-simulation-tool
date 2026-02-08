"""
Comprehensive heat transfer model for chimney/flue systems
Improved physics with stable numerics and proper coupling
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Literal
from dataclasses import dataclass
from .condensation_physics import CondensationPhysics, CondensationState
from .stable_triple_wall import StableTripleWallSolver, WallLayerProperties
from .adaptive_triple_wall import AdaptiveTripleWallSolver
#from .energy_diagnostics import EnergyDiagnostics
import logging
import warnings

@dataclass
class FlowProperties:
    """Properties for flow calculations"""
    velocity: float  # m/s
    density: float   # kg/m³
    viscosity: float # Pa·s
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    temperature: float  # K
    pressure: float  # Pa
    
class ComprehensiveHeatTransfer:
    """
    Comprehensive heat transfer model including:
    - Natural convection (vertical/horizontal) - from temperature-driven density differences
    - Forced convection (laminar/turbulent/transition) - from flow velocity
    - Combined natural and forced convection - both effects together
    - External radiation - heat loss to surroundings via electromagnetic waves
    - Condensation and mass transfer - water vapor condensing on cold surfaces
    - Transient wall analysis - how wall temperature changes over time
    
    This is the workhorse that calculates all the heat transfer coefficients
    and temperature changes for each segment at each time step.
    """
    
    # PHYSICAL CONSTANTS
    STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴) - for radiation calculations
    
    def __init__(self, include_interlayer_radiation: bool = True):
        """Initialize heat transfer calculator

        Args:
            include_interlayer_radiation: If True (default), adaptive triple-wall
                solver includes inter-layer radiation. Passed through to
                AdaptiveTripleWallSolver.
        """
        from .physics import PhysicalProperties
        self.physics = PhysicalProperties()
        self.condensation = CondensationPhysics()
        self.triple_wall_solver = StableTripleWallSolver()
        self.adaptive_solver = AdaptiveTripleWallSolver(
            include_interlayer_radiation=include_interlayer_radiation
        )
        #self.energy_diagnostics = EnergyDiagnostics()
        self.logger = logging.getLogger(__name__)

        # Track condensation state for each segment
        self.condensation_states = {}
    
    
    def gas_properties(self, temperature_K: float, pressure_Pa: float = 98457) -> FlowProperties:
        """Calculate flue gas properties at given temperature and pressure
        
        These properties are essential for heat transfer and flow calculations.
        All correlations are for air (flue gas is approximated as air).
        
        Args:
            temperature_K: Gas temperature in Kelvin
            pressure_Pa: Gas pressure in Pascals (default is atmospheric)
            
        Returns:
            FlowProperties object with density, viscosity, conductivity, etc.
        """
        
        # Use PhysicalProperties for all calculations
        density = self.physics.air_density(temperature_K, pressure_Pa)
        viscosity = self.physics.air_viscosity(temperature_K)
        thermal_conductivity = self.physics.air_thermal_conductivity(temperature_K)
        specific_heat = self.physics.air_specific_heat(temperature_K)
        
        # Velocity will be calculated separately based on mass flow
        velocity = 1.0  # Placeholder
        
        return FlowProperties(
            velocity=velocity,
            density=density,
            viscosity=viscosity,
            thermal_conductivity=thermal_conductivity,
            specific_heat=specific_heat,
            temperature=temperature_K,
            pressure=pressure_Pa
        )
    
    def reynolds_number(self, velocity: float, diameter: float, props: FlowProperties) -> float:
        """Calculate Reynolds number with bounds checking
        
        Reynolds number = (inertial forces) / (viscous forces)
        Re = ρ * v * D / μ
        
        Low Re (<2300) = laminar flow (smooth, layered)
        High Re (>4000) = turbulent flow (chaotic, mixed)
        Between = transition region
        
        Args:
            velocity: Flow velocity (m/s)
            diameter: Pipe diameter (m)
            props: Gas properties
            
        Returns:
            Reynolds number (dimensionless)
        """
        # Protect against division by zero and invalid inputs
        if props.viscosity <= 0 or diameter <= 0:
            return 1.0  # Minimum Re to avoid division by zero
        if abs(velocity) < 1e-10:
            return 1.0  # Near-zero velocity means laminar
            
        # Standard Reynolds number calculation
        Re = props.density * abs(velocity) * diameter / props.viscosity
        
        # Check for numerical issues
        if np.isnan(Re) or np.isinf(Re):
            return 1.0  # Fallback for numerical issues
        return max(Re, 1.0)  # Ensure positive
    
    def prandtl_number(self, props: FlowProperties) -> float:
        """Calculate Prandtl number"""
        return props.viscosity * props.specific_heat / props.thermal_conductivity
    
    def grashof_number(self, diameter: float, T_surface: float, T_fluid: float, props: FlowProperties) -> float:
        """Calculate Grashof number for natural convection"""
        beta = 1 / props.temperature  # thermal expansion coefficient for ideal gas
        delta_T = abs(T_surface - T_fluid)
        return self.physics.GRAVITY * beta * delta_T * diameter**3 * props.density**2 / props.viscosity**2
    
    def rayleigh_number(self, grashof: float, prandtl: float) -> float:
        """Calculate Rayleigh number"""
        return grashof * prandtl
    
    def nusselt_forced_convection_internal(self, reynolds: float, prandtl: float, 
                                         length_diameter_ratio: Optional[float] = None) -> float:
        """
        Calculate Nusselt number for forced convection inside tubes
        
        Nusselt number = (convective heat transfer) / (conductive heat transfer)
        Nu = h * D / k
        
        Higher Nu means more convective heat transfer relative to conduction.
        
        Args:
            reynolds: Reynolds number (flow regime indicator)
            prandtl: Prandtl number (fluid property ratio)
            length_diameter_ratio: L/D ratio for entrance effects
            
        Returns:
            Nusselt number (dimensionless)
        """
        if reynolds < 2300:  # LAMINAR FLOW - smooth, predictable
            if length_diameter_ratio and length_diameter_ratio < 60:
                # DEVELOPING FLOW - still forming velocity/temperature profiles
                # Hausen equation accounts for entrance effects
                term = reynolds * prandtl * (1/length_diameter_ratio)
                if term > 1000:  # Limit to prevent numerical overflow
                    term = 1000
                Nu_dev = 3.66 + (0.0668 * term) / (1 + 0.04 * (term)**(2/3))
                return max(Nu_dev, 3.66)
            else:
                # FULLY DEVELOPED LAMINAR - stable profiles
                # 4.36 for constant heat flux (chimney has varying wall temp)
                # 3.66 would be for constant wall temperature
                return 4.36
        
        elif reynolds < 4000:  # TRANSITION REGION - unstable, changing
            # Linear interpolation between laminar and turbulent
            # Real transition is complex and unsteady
            Nu_lam = 4.36  # Laminar value
            Nu_turb = 0.023 * (4000)**0.8 * prandtl**0.4  # Turbulent at Re=4000
            factor = (reynolds - 2300) / (4000 - 2300)  # 0 to 1 across transition
            return Nu_lam + factor * (Nu_turb - Nu_lam)
        
        else:  # Turbulent flow
            # Dittus-Boelter equation for heating
            # Add bounds checking to prevent NaN
            if reynolds <= 0 or prandtl <= 0:
                return 3.66  # Return laminar value as fallback
            if reynolds > 1e6:  # Very high Re
                reynolds = 1e6  # Cap to prevent overflow
            return 0.023 * reynolds**0.8 * prandtl**0.4
    
    def nusselt_natural_convection_external(self, rayleigh: float, orientation: str, 
                                          height_diameter_ratio: float = 1.0) -> float:
        """
        Calculate Nusselt number for natural convection on external surfaces
        
        Args:
            rayleigh: Rayleigh number
            orientation: 'vertical' or 'horizontal'
            height_diameter_ratio: H/D ratio for vertical cylinders
        """
        if orientation.lower() == 'vertical':
            # Vertical cylinder correlations (Churchill and Chu)
            if rayleigh < 1e4:
                return 1.0  # Conduction dominated
            elif rayleigh < 1e9:
                # Laminar flow
                # Use Prandtl number = 0.7 for air (typical)
                Pr = 0.7
                return 0.68 + 0.67 * rayleigh**0.25 / (1 + (0.492/Pr)**(9/16))**(4/9)
            else:
                # Turbulent flow
                Pr = 0.7
                return (0.825 + 0.387 * rayleigh**(1/6) / (1 + (0.492/Pr)**(9/16))**(8/27))**2
        
        else:  # horizontal
            # Horizontal cylinder correlations (Churchill and Chu)
            if rayleigh < 1e-5:
                return 1.0  # Conduction dominated
            elif rayleigh < 1e12:
                # Full range correlation for horizontal cylinders
                # Use Prandtl number = 0.7 for air (typical)
                Pr = 0.7
                term1 = 0.6 + 0.387 * rayleigh**(1/6) / (1 + (0.559/Pr)**(9/16))**(8/27)
                return term1**2
            else:
                # Very high Rayleigh number
                return 0.15 * rayleigh**(1/3)
    
    def nusselt_natural_convection_internal(self, rayleigh: float, orientation: str,
                                           length_diameter_ratio: float = 10.0) -> float:
        """
        Calculate Nusselt number for natural convection inside tubes
        
        Args:
            rayleigh: Rayleigh number
            orientation: 'vertical' or 'horizontal'
            length_diameter_ratio: L/D ratio for the tube
        """
        if orientation.lower() == 'vertical':
            # Vertical tube internal natural convection
            if rayleigh < 1e3:
                return 1.0  # Conduction only
            elif rayleigh < 1e6:
                # Laminar natural convection in vertical tubes
                # Based on correlations for enclosed vertical channels
                return 1.0 + 0.105 * rayleigh**0.3 * (1 / (1 + (1/length_diameter_ratio)**0.5))
            else:
                # Turbulent natural convection
                return 0.046 * rayleigh**(1/3)
        else:  # horizontal
            # Horizontal tube internal natural convection
            if rayleigh < 1e3:
                return 1.0  # Conduction only
            elif rayleigh < 1e5:
                # Laminar flow with crescents
                return 1.0 + 0.25 * (rayleigh / 1e3)**0.25
            else:
                # Turbulent natural convection
                return 0.14 * rayleigh**(1/3)
    
    def combined_convection_coefficient(self, h_forced: float, h_natural: float) -> float:
        """
        Calculate combined convection coefficient for mixed convection
        Uses Churchill-Usagi method
        """
        n = 3.5  # Typical exponent for mixed convection
        return (h_forced**n + h_natural**n)**(1/n)
    
    def radiation_heat_transfer_coefficient(self, T_surface: float, T_ambient: float, 
                                          emissivity: float) -> float:
        """Calculate radiation heat transfer coefficient"""
        T_avg = (T_surface + T_ambient) / 2
        return 4 * emissivity * self.STEFAN_BOLTZMANN * T_avg**3
    
    def dew_point_temperature(self, pressure_Pa: float, humidity_ratio: float) -> float:
        """
        Calculate dew point temperature
        
        Args:
            pressure_Pa: Total pressure in Pa
            humidity_ratio: kg water / kg dry air
            
        Returns:
            Dew point temperature in K
        """
        # Partial pressure of water vapor
        p_vapor = pressure_Pa * humidity_ratio / (0.622 + humidity_ratio)
        
        # Use condensation physics to find temperature where saturation pressure equals vapor pressure
        # Binary search for temperature
        T_min, T_max = 173.15, 373.15  # -100°C to 100°C
        
        for _ in range(20):  # 20 iterations gives ~1e-6 precision
            T_mid = (T_min + T_max) / 2
            p_sat = self.condensation.saturation_pressure(T_mid)
            
            if p_sat < p_vapor:
                T_min = T_mid
            else:
                T_max = T_mid
                
        return T_mid
    
    
    def condensation_heat_transfer_coefficient(self, props: FlowProperties, 
                                             wall_temp: float, vapor_temp: float) -> float:
        """
        Calculate condensation heat transfer coefficient
        Uses Nusselt's laminar film condensation theory
        """
        if wall_temp >= vapor_temp:
            return 0.0  # No condensation
        
        # Properties of liquid water at film temperature
        T_film = (wall_temp + vapor_temp) / 2
        
        # Latent heat of vaporization (approximate)
        h_fg = 2.26e6  # J/kg at 100°C, approximate
        
        # Liquid water properties (approximate)
        rho_liquid = 1000.0  # kg/m³
        mu_liquid = 1e-3   # Pa·s
        k_liquid = 0.6     # W/(m·K)
        
        # Nusselt number for condensation on vertical surface
        delta_T = vapor_temp - wall_temp
        Nu_cond = 0.943 * ((rho_liquid * self.physics.GRAVITY * h_fg) / 
                          (mu_liquid * k_liquid * delta_T))**0.25
        
        return Nu_cond * k_liquid  # Simplified, should include length scale
    
    def mass_transfer_coefficient(self, props: FlowProperties, diameter: float, 
                                velocity: float) -> float:
        """
        Calculate mass transfer coefficient using heat/mass transfer analogy
        """
        Re = self.reynolds_number(velocity, diameter, props)
        Pr = self.prandtl_number(props)
        Sc = Pr  # Schmidt number ≈ Prandtl for gases (approximation)
        
        # Sherwood number using Chilton-Colburn analogy
        if Re < 2300:
            Sh = 3.66  # Laminar
        else:
            Sh = 0.023 * Re**0.8 * Sc**(1/3)  # Turbulent
        
        # Mass diffusivity (water vapor in air, approximate)
        D_AB = 2.6e-5 * (props.temperature / 298)**1.8  # m²/s
        
        return Sh * D_AB / diameter
    
    def pressure_drop(self, props: FlowProperties, diameter: float, length: float, 
                     velocity: float, roughness: float = 0.0, 
                     fittings_factor: float = 1.0,
                     elbows: Optional[List] = None,
                     elbow_method: Literal["K_sum", "equivalent_length"] = "K_sum") -> float:
        """
        Calculate pressure drop due to friction and fittings
        
        Args:
            props: Flow properties
            diameter: Pipe diameter in m
            length: Pipe length in m
            velocity: Flow velocity in m/s
            roughness: Surface roughness in m
            fittings_factor: Factor for additional losses from fittings (deprecated)
            elbows: Optional list of ElbowSpec objects for physics-based elbow losses
            elbow_method: Method for calculating elbow losses ("K_sum" or "equivalent_length")
            
        Returns:
            Pressure drop in Pa
        """
        Re = self.reynolds_number(velocity, diameter, props)
        
        # Friction factor with protection against zero Reynolds number
        if Re < 1e-6:  # Near zero flow
            f = 64.0  # Use maximum laminar value
        elif Re < 2300:  # Laminar
            f = 64 / Re
        else:  # Turbulent - Colebrook equation (simplified)
            if roughness == 0:
                f = 0.316 / Re**0.25  # Smooth pipe
            else:
                # Approximate for rough pipes
                f = 0.25 / (np.log10(roughness / (3.7 * diameter) + 5.74 / Re**0.9))**2
        
        # Effective length for friction calculation
        effective_length = length
        K_total_elbows = 0.0
        
        # Process physics-based elbow losses if provided
        if elbows:
            from .flow.elbows import elbow_losses
            
            # Calculate volumetric flow rate
            area = np.pi * (diameter / 2)**2
            volumetric_flow = velocity * area
            
            for elbow_spec in elbows:
                # Calculate losses for this elbow
                elbow_loss = elbow_losses(
                    elbow_spec, 
                    rho=props.density,
                    mu=props.viscosity,
                    volumetric_flow_m3_s=volumetric_flow
                )
                
                if elbow_method == "equivalent_length":
                    # Add equivalent length to the straight pipe length
                    effective_length += elbow_loss.Le_m
                else:  # K_sum method
                    # Accumulate K values
                    K_total_elbows += elbow_loss.K
        
        # Calculate friction pressure drop using effective length
        dp_friction = f * (effective_length / diameter) * 0.5 * props.density * velocity**2
        
        # Add minor losses from K values (if using K_sum method)
        dp_minor = K_total_elbows * 0.5 * props.density * velocity**2
        
        # Apply legacy fittings_factor if no elbows specified (backward compatibility)
        if not elbows and fittings_factor != 1.0:
            warnings.warn(
                "fittings_factor is deprecated. Use elbows parameter with ElbowSpec objects instead.",
                DeprecationWarning,
                stacklevel=2
            )
            dp_total = dp_friction * fittings_factor
        else:
            dp_total = dp_friction + dp_minor
        
        return dp_total
    
    def buoyancy_pressure(self, height: float, rho_ambient: float, rho_gas: float) -> float:
        """
        Calculate buoyancy pressure difference for draft calculations
        
        Args:
            height: Height difference in m
            rho_ambient: Ambient air density in kg/m³
            rho_gas: Flue gas density in kg/m³
            
        Returns:
            Pressure difference in Pa (positive = upward draft)
        """
        return self.physics.GRAVITY * height * (rho_ambient - rho_gas)
    
    def air_gap_heat_transfer_coefficient(self, T_hot: float, T_cold: float, 
                                        gap_width: float, gap_height: float) -> float:
        """
        Calculate heat transfer coefficient across an air gap including
        conduction, convection, and radiation
        
        Args:
            T_hot: Hot surface temperature in K
            T_cold: Cold surface temperature in K  
            gap_width: Air gap width in m
            gap_height: Height of air gap in m
            
        Returns:
            Equivalent heat transfer coefficient in W/(m²·K)
        """
        if gap_width <= 0:
            return 1e6  # Very high value for no gap (direct contact)
            
        # Air properties at mean temperature
        T_mean = (T_hot + T_cold) / 2
        air_props = self.gas_properties(T_mean)
        
        # Conduction through air
        h_cond = air_props.thermal_conductivity / gap_width
        
        # Natural convection in enclosed space
        delta_T = abs(T_hot - T_cold)
        if delta_T > 0.1:  # Minimum temperature difference
            # Rayleigh number for vertical enclosure
            # Ra = Gr × Pr, where:
            #   Gr = g × β × ΔT × L³ / ν²  (Grashof number)
            #   Pr = ν / α = μ × cp / k    (Prandtl number)
            #   ν = μ / ρ                  (kinematic viscosity)
            #   α = k / (ρ × cp)           (thermal diffusivity)
            #
            # Expanding: Ra = g × β × ΔT × L³ × ρ² × cp / (μ × k)
            # where L = gap_width (characteristic length for enclosed gap)
            #
            # Reference: Incropera & DeWitt, Fundamentals of Heat and Mass Transfer
            beta = 1 / T_mean  # thermal expansion coefficient for ideal gas [1/K]
            g = self.physics.GRAVITY
            L = gap_width
            rho = air_props.density
            mu = air_props.viscosity
            k = air_props.thermal_conductivity
            cp = air_props.specific_heat

            Ra = (g * beta * delta_T * L**3 * rho**2 * cp) / (mu * k)

            # Nusselt number for vertical enclosures
            # Hollands/Catton correlation for 10^4 < Ra < 10^7, 10 < H/W < 40
            # Reference: Catton, I. (1978) "Natural Convection in Enclosures"
            if Ra < 1e4:
                Nu = 1.0  # Pure conduction
            elif Ra < 1e10:
                Nu = 0.42 * Ra**0.25 * (gap_height/gap_width)**(-0.3)
            else:
                Nu = 0.046 * Ra**(1/3)
                
            h_conv = Nu * air_props.thermal_conductivity / gap_width
        else:
            h_conv = 0.0
            
        # Radiation between surfaces (assuming emissivity ~0.8 for both)
        emissivity_eff = 1 / (1/0.8 + 1/0.8 - 1)  # Effective emissivity
        h_rad = emissivity_eff * self.STEFAN_BOLTZMANN * (T_hot**2 + T_cold**2) * (T_hot + T_cold)
        
        # Combined heat transfer coefficient
        h_total = h_cond + h_conv + h_rad
        
        return h_total
    
    def transient_wall_temperature(self, T_wall_old: float, T_gas: float, T_ambient: float,
                                  wall_properties: Dict, geometry: Dict, 
                                  h_inner: float, h_outer: float, dt: float) -> float:
        """
        Calculate transient wall temperature for single wall using stable semi-implicit method
        Prevents oscillations at early times by using proper numerical scheme
        
        Args:
            T_wall_old: Previous wall temperature in K
            T_gas: Gas temperature in K
            T_ambient: Ambient temperature in K
            wall_properties: Dict with 'density', 'specific_heat', 'thickness', 'thermal_conductivity'
            geometry: Dict with 'inner_diameter', 'outer_diameter', 'length'
            h_inner: Inner convection coefficient in W/(m²·K)
            h_outer: Outer convection coefficient in W/(m²·K)
            dt: Time step in s
            
        Returns:
            New wall temperature in K
        """
        # Wall properties
        rho_w = wall_properties['density']  # kg/m³
        cp_w = wall_properties['specific_heat']  # J/(kg·K)
        k_w = wall_properties.get('thermal_conductivity', 50.0)  # W/(m·K)
        
        # Geometry
        D_inner = geometry['inner_diameter']  # m
        D_outer = geometry['outer_diameter']  # m
        length = geometry['length']  # m
        
        # Cylindrical radii
        r_inner = D_inner / 2
        r_outer = D_outer / 2
        
        # Cylindrical heat transfer areas
        A_inner = np.pi * D_inner * length  # Inner surface area
        A_outer = np.pi * D_outer * length  # Outer surface area
        A_mean = np.pi * (D_inner + D_outer) / 2 * length  # Mean area for lumped model
        
        # Cylindrical wall thermal mass
        V_wall = np.pi * length * (D_outer**2 - D_inner**2) / 4
        C_wall = rho_w * cp_w * V_wall  # Thermal capacitance (J/K)
        
        # CYLINDRICAL thermal resistance through wall (K/W)
        R_wall = np.log(r_outer / r_inner) / (2 * np.pi * k_w * length)
        
        # Overall conductances (W/K)
        UA_inner = h_inner * A_inner  # Inner surface
        UA_outer = h_outer * A_outer  # Outer surface
        
        # Semi-implicit time integration for stability
        # (C/dt + UA_inner + UA_outer) * T_new = C/dt * T_old + UA_inner * T_gas + UA_outer * T_ambient
        
        # Coefficients
        a = C_wall / dt + UA_inner + UA_outer
        b = C_wall / dt * T_wall_old + UA_inner * T_gas + UA_outer * T_ambient
        
        # Solve for new temperature
        T_wall_new = b / a
        
        # Apply damping ONLY for extremely small timesteps to prevent oscillations
        # But don't over-damp for normal timesteps like 0.5s
        if dt < 0.01:  # Only for very small timesteps (< 10ms)
            damping = 0.7  # Mild damping
            T_wall_new = damping * T_wall_new + (1 - damping) * T_wall_old
        
        # Physical bounds based on thermodynamics
        # In a passive system (no refrigeration), nothing can go below ambient
        # Wall temperature will be between gas and ambient
        if T_gas > T_ambient:
            # Normal operation: hot gas heats wall
            T_wall_new = np.clip(T_wall_new, T_ambient, T_gas)
        else:
            # Gas at or below ambient: wall approaches ambient
            T_wall_new = max(T_wall_new, T_ambient)
        
        return T_wall_new
    
    def transient_triple_wall_temperatures(self, T_liner_old: float, T_middle_old: float, 
                                         T_outer_old: float, T_gas: float, T_ambient: float,
                                         wall_properties: Dict, geometry: Dict, 
                                         h_inner: float, h_outer: float, dt: float) -> Tuple[float, float, float]:
        """
        Calculate transient temperatures for triple wall system using stable solver
        
        Args:
            T_liner_old: Previous liner temperature in K
            T_middle_old: Previous middle layer temperature in K  
            T_outer_old: Previous outer wall temperature in K
            T_gas: Gas temperature in K
            T_ambient: Ambient temperature in K
            wall_properties: Dict with properties for each layer
            geometry: Dict with geometry information
            h_inner: Inner convection coefficient in W/(m²·K)
            h_outer: Outer convection coefficient in W/(m²·K)
            dt: Time step in s
            
        Returns:
            Tuple of (T_liner_new, T_middle_new, T_outer_new) in K
        """
        # Convert wall properties to WallLayerProperties objects
        liner_props = WallLayerProperties(
            thickness=wall_properties['liner']['thickness'],
            thermal_conductivity=wall_properties['liner']['thermal_conductivity'],
            density=wall_properties['liner']['density'],
            specific_heat=wall_properties['liner']['specific_heat'],
            emissivity=wall_properties['liner'].get('emissivity', 0.3)
        )
        
        middle_props = WallLayerProperties(
            thickness=wall_properties['middle']['thickness'],
            thermal_conductivity=wall_properties['middle']['thermal_conductivity'],
            density=wall_properties['middle']['density'],
            specific_heat=wall_properties['middle']['specific_heat'],
            emissivity=0.9  # High emissivity for insulation
        )
        
        outer_props = WallLayerProperties(
            thickness=wall_properties['outer']['thickness'],
            thermal_conductivity=wall_properties['outer']['thermal_conductivity'],
            density=wall_properties['outer']['density'],
            specific_heat=wall_properties['outer']['specific_heat'],
            emissivity=wall_properties['outer'].get('emissivity', 0.8)
        )
        
        # Package properties
        properties = {
            'liner': liner_props,
            'middle': middle_props,
            'outer': outer_props
        }
        
        # Determine if we need adaptive sub-stepping
        # Check if liner is very thin (< 1mm) which causes stiffness
        use_adaptive = liner_props.thickness < 0.001 or dt > 0.05
        
        if use_adaptive:
            # Use adaptive solver for stiff systems
            return self.adaptive_solver.solve_with_adaptive_dt(
                T_gas, T_ambient, h_inner, h_outer,
                T_liner_old, T_middle_old, T_outer_old,
                geometry, properties, dt
            )
        else:
            # Use standard solver for well-conditioned systems
            return self.triple_wall_solver.solve_triple_wall_stable(
                T_gas, T_ambient, h_inner, h_outer,
                T_liner_old, T_middle_old, T_outer_old,
                geometry, properties, dt
            )
    
    def _create_default_elbow_from_fittings(self, has_fittings: bool, diameter: float) -> Optional[List]:
        """
        Create default ElbowSpec objects for backward compatibility.
        
        Args:
            has_fittings: Legacy boolean flag indicating presence of fittings
            diameter: Pipe diameter in meters
            
        Returns:
            List with one default ElbowSpec if has_fittings is True, else None
        """
        if not has_fittings:
            return None
            
        from .flow.elbows import ElbowSpec
        
        # Create a default 90° short-radius elbow as approximation
        # This maintains similar pressure drop behavior to the old 1.5x factor
        default_elbow = ElbowSpec(
            angle_deg=90.0,
            diameter_m=diameter,
            R_over_D=1.0,  # Short radius
            elbow_type="short_radius",
            smoothness="commercial_steel",
            vane_factor=1.0
        )
        
        self.logger.debug("Created default elbow for backward compatibility with has_fittings=True")
        warnings.warn(
            "has_fittings flag is deprecated. Define explicit ElbowSpec objects for accurate modeling.",
            DeprecationWarning,
            stacklevel=3
        )
        
        return [default_elbow]
    
    def calculate_segment_heat_transfer(self, segment_data: Dict, dt: float) -> Dict:
        """
        Calculate complete heat transfer for a chimney segment with iterative condensation coupling
        
        This is the MAIN CALCULATION that determines how much heat flows from the
        hot gas to the walls and then to ambient. It handles:
        1. Convection from gas to inner wall
        2. Conduction through wall(s)
        3. Convection and radiation from outer wall to ambient
        4. Condensation effects if gas cools below dew point
        
        Args:
            segment_data: Dictionary containing all segment properties
            dt: Time step in seconds
            
        Returns:
            Dictionary with updated temperatures and heat transfer coefficients
        """
        # EXTRACT INPUT DATA
        T_gas = segment_data['gas_temperature']  # Hot gas temperature (K)
        T_ambient = segment_data['ambient_temperature']  # Outside air temperature (K)
        pressure = segment_data['pressure']  # Gas pressure (Pa)
        velocity = segment_data['velocity']  # Gas velocity (m/s)
        diameter = segment_data['diameter']  # Inner diameter (m)
        length = segment_data['length']  # Segment length (m)
        orientation = segment_data.get('orientation', 'vertical')  # Affects natural convection
        has_fittings = segment_data.get('has_fittings', False)  # Turbulence enhancement
        wall_props = segment_data['wall_properties']  # Material properties
        wall_type = segment_data.get('wall_type', 'single')  # 'single' or 'triple'
        humidity_ratio = segment_data.get('humidity_ratio', 0.01)  # Water content (kg/kg)
        segment_id = segment_data.get('segment_id', 0)  # For tracking state
        
        # CALCULATE GAS PROPERTIES
        # Temperature-dependent properties are crucial for accuracy
        gas_props = self.gas_properties(T_gas, pressure)
        gas_props.velocity = velocity
        
        # CONDENSATION STATE TRACKING
        # Each segment remembers its condensate film thickness
        if segment_id not in self.condensation_states:
            self.condensation_states[segment_id] = CondensationState()
        
        # DIMENSIONLESS NUMBERS
        # These characterize the flow and heat transfer regime
        Re = self.reynolds_number(velocity, diameter, gas_props)  # Flow regime
        Pr = self.prandtl_number(gas_props)  # Fluid property ratio
        
        # FORCED CONVECTION (from flow velocity)
        # Calculate heat transfer coefficient from gas flow
        Nu_forced = self.nusselt_forced_convection_internal(Re, Pr, length/diameter)
        h_forced = Nu_forced * gas_props.thermal_conductivity / diameter
        
        # FITTINGS ENHANCEMENT
        # Elbows, tees create turbulence that enhances heat transfer
        if has_fittings:
            h_forced *= 1.2  # 20% increase is typical for fittings

        # Iterative solution for coupled heat transfer and condensation
        max_iterations = 5
        tolerance = 1e-4

        # Initialize temperatures
        if wall_type == 'single':
            T_wall_current = segment_data['wall_temperature']
        else:  # triple
            T_liner_current = segment_data['liner_temperature']
            T_middle_current = segment_data['middle_temperature']
            T_outer_current = segment_data['outer_temperature']
            T_wall_current = T_liner_current  # Use liner as reference

        # ITERATIVE COUPLING LOOP
        # Wall temperature affects condensation which affects heat transfer
        # Need to iterate to find consistent solution
        for iteration in range(max_iterations):
            # Store previous values to check convergence
            T_wall_prev = T_wall_current
            humidity_ratio_prev = humidity_ratio

            # NATURAL CONVECTION (from temperature differences)
            # Hot wall creates buoyancy-driven flow along the surface
            Gr = self.grashof_number(diameter, T_wall_current, T_gas, gas_props)
            Ra = self.rayleigh_number(Gr, Pr)  # Combines Gr and Pr
            
            # Calculate natural convection heat transfer
            length_diameter_ratio = length / diameter
            Nu_natural = self.nusselt_natural_convection_internal(Ra, orientation, length_diameter_ratio)
            h_natural = Nu_natural * gas_props.thermal_conductivity / diameter
            
            # COMBINE FORCED AND NATURAL CONVECTION
            # Both mechanisms act together - use appropriate correlation
            h_inner = self.combined_convection_coefficient(h_forced, h_natural)
            
            # ENGINEERING MINIMUMS
            # Real chimneys have turbulence, roughness that prevent very low h
            # The low calculated h values don't match experimental observations
            # because correlations don't account for:
            # - Surface roughness and imperfections
            # - Entrance/exit effects
            # - Turbulent mixing from fittings and bends
            # - Buoyancy-enhanced mixing in vertical sections
            h_min_chimney = 10.0  # W/(m²·K) - reduced from 20 to prevent over-predicted heat loss
            h_min_general = 5.0   # W/(m²·K) - minimum for other orientations

            # Apply minimum based on orientation and gas temperature
            # Use absolute temperature threshold (100°C) instead of relative difference
            # This prevents feedback loop where low h → low ΔT → minimum not applied
            T_gas_C = T_gas - 273.15
            is_hot_gas = T_gas_C > 100  # Gas above 100°C is considered "hot"

            if orientation.lower() == 'vertical' and is_hot_gas:
                # Hot gas in vertical chimney should have good heat transfer
                if h_inner < h_min_chimney:
                    self.logger.debug(f"Increasing h_inner from {h_inner:.1f} to {h_min_chimney} W/(m²·K)")
                h_inner = max(h_inner, h_min_chimney)
            else:
                # Apply general minimum to prevent numerical issues
                h_inner = max(h_inner, h_min_general)
            
            # Update condensation state
            area = np.pi * diameter * length
            condensation_state = self.condensation_states[segment_id]

            # Modify heat transfer coefficient for condensate film
            h_inner_modified = self.condensation.film_heat_transfer_modification(
                h_inner, condensation_state.film_thickness
            )

            # Update condensation with current wall temperature
            condensation_state = self.condensation.update_condensation_state(
                condensation_state, T_gas, T_wall_current, pressure,
                humidity_ratio, h_inner_modified, area, dt
            )
            self.condensation_states[segment_id] = condensation_state

            # Add latent heat effect to inner heat transfer
            if condensation_state.latent_heat_rate != 0:
                # Effective heat transfer coefficient including latent heat
                q_sensible = h_inner_modified * area * (T_gas - T_wall_current)
                q_total = q_sensible + condensation_state.latent_heat_rate
                h_inner_effective = q_total / (area * (T_gas - T_wall_current)) if T_gas != T_wall_current else h_inner_modified
            else:
                h_inner_effective = h_inner_modified

            # Update humidity ratio based on condensation
            if segment_data.get('mass_flow_rate', 0) > 0:
                gas_volume = np.pi * (diameter/2)**2 * length
                humidity_ratio = self.condensation.gas_humidity_change(
                    humidity_ratio, condensation_state.net_rate,
                    segment_data['mass_flow_rate'], gas_volume,
                    gas_props.density, dt
                )
            
            # Wall temperature update based on wall type
            if wall_type == 'single':
                # Single wall calculation
                outer_diameter = diameter + 2 * wall_props['thickness']
                ambient_props = self.gas_properties(T_ambient, pressure)
                
                # Outer convection
                Gr_outer = self.grashof_number(outer_diameter, T_wall_current, T_ambient, ambient_props)
                Ra_outer = self.rayleigh_number(Gr_outer, self.prandtl_number(ambient_props))
                Nu_outer = self.nusselt_natural_convection_external(Ra_outer, orientation)
                h_conv_outer = Nu_outer * ambient_props.thermal_conductivity / outer_diameter
                
                # Radiation to ambient
                emissivity = wall_props.get('emissivity', 0.8)
                h_rad_outer = self.radiation_heat_transfer_coefficient(T_wall_current, T_ambient, emissivity)
                h_outer = h_conv_outer + h_rad_outer
                
                # Apply minimum h_outer for numerical stability
                h_outer = max(h_outer, 1.0)  # W/(m²·K)
                
                # Geometry
                geometry = {
                    'inner_diameter': diameter,
                    'outer_diameter': outer_diameter,
                    'length': length
                }
                
                # Update wall temperature with effective h_inner
                T_wall_new = self.transient_wall_temperature(
                    T_wall_current, T_gas, T_ambient, wall_props, geometry, 
                    h_inner_effective, h_outer, dt
                )
                
                T_wall_current = T_wall_new
                
            else:  # triple wall
                # Outer surface properties
                outer_diameter = (diameter + 2 * (wall_props['liner']['thickness'] + 
                                wall_props.get('air_gap_inner', 0) +
                                wall_props['middle']['thickness'] + 
                                wall_props.get('air_gap_outer', 0) +
                                wall_props['outer']['thickness']))
                
                ambient_props = self.gas_properties(T_ambient, pressure)
                Gr_outer = self.grashof_number(outer_diameter, T_outer_current, T_ambient, ambient_props)
                Ra_outer = self.rayleigh_number(Gr_outer, self.prandtl_number(ambient_props))
                Nu_outer = self.nusselt_natural_convection_external(Ra_outer, orientation)
                h_conv_outer = Nu_outer * ambient_props.thermal_conductivity / outer_diameter
                
                # Radiation to ambient
                emissivity_outer = wall_props['outer'].get('emissivity', 0.8)
                h_rad_outer = self.radiation_heat_transfer_coefficient(T_outer_current, T_ambient, emissivity_outer)
                h_outer = h_conv_outer + h_rad_outer
                
                # Apply minimum h_outer for numerical stability
                h_outer = max(h_outer, 1.0)  # W/(m²·K)
                
                # Geometry
                geometry = {
                    'inner_diameter': diameter,
                    'length': length
                }
                
                # Update triple wall temperatures
                T_liner_new, T_middle_new, T_outer_new = self.transient_triple_wall_temperatures(
                    T_liner_current, T_middle_current, T_outer_current, 
                    T_gas, T_ambient, wall_props, geometry, 
                    h_inner_effective, h_outer, dt
                )
                
                T_liner_current = T_liner_new
                T_middle_current = T_middle_new
                T_outer_current = T_outer_new
                T_wall_current = T_liner_current
            
            # Check convergence
            residual = abs(T_wall_current - T_wall_prev) / (T_wall_current + 273.15)
            humidity_residual = abs(humidity_ratio - humidity_ratio_prev) / (humidity_ratio + 1e-6)
            
            if residual < tolerance and humidity_residual < tolerance:
                break

        # Apply spiral corrugation heat transfer enhancement (v2.0)
        # This multiplier is pre-computed during preprocessing and stored in part.corrugation_effects
        # It accounts for swirl-induced enhancement of convective heat transfer
        corrugation = segment_data.get('corrugation', None)
        if corrugation is not None and corrugation.get('type') == 'spiral':
            # Check if HT enhancement is enabled
            if corrugation.get('enable_spiral_HT_enhancement', False):
                # Get the multiplier from the part's corrugation_effects
                # This is set during preprocessing in chimney.py
                Mh = segment_data.get('Mh', 1.0)
                h_inner_effective *= Mh
                self.logger.debug(f"Applied spiral HT multiplier Mh={Mh:.3f} to h_inner")

        # Prepare results
        if wall_type == 'single':
            results = {
                'wall_temperature_new': T_wall_current,
                'h_inner': h_inner_effective,
                'h_outer': h_outer,
            }
        else:
            results = {
                'liner_temperature_new': T_liner_current,
                'middle_temperature_new': T_middle_current,
                'outer_temperature_new': T_outer_current,
                'wall_temperature_new': T_liner_current,
                'h_inner': h_inner_effective,
                'h_outer': h_outer,
            }
        
        # Common calculations
        # Check for explicit elbow specifications first
        elbows = segment_data.get('elbows', None)
        elbow_method = segment_data.get('elbow_method', 'K_sum')
        
        # If no explicit elbows but has_fittings is True, create default for backward compatibility
        if elbows is None and has_fittings:
            elbows = self._create_default_elbow_from_fittings(has_fittings, diameter)
        
        # Calculate pressure drop with elbow support
        dp = self.pressure_drop(
            gas_props, diameter, length, velocity,
            roughness=segment_data.get('roughness', 0.0),
            elbows=elbows,
            elbow_method=elbow_method
        )
        
        ambient_density = self.gas_properties(T_ambient, pressure).density
        dp_buoyancy = self.buoyancy_pressure(length, ambient_density, gas_props.density)
        
        # Dew point
        T_dew = self.dew_point_temperature(pressure, humidity_ratio)
        
        # Add all results
        results.update({
            'humidity_ratio_new': humidity_ratio,
            'condensation_rate': condensation_state.condensation_rate,
            'evaporation_rate': condensation_state.evaporation_rate,
            'film_thickness': condensation_state.film_thickness,
            'film_mass': condensation_state.film_mass,
            'latent_heat_rate': condensation_state.latent_heat_rate,
            'reynolds_number': Re,
            'nusselt_forced': Nu_forced,
            'nusselt_natural': Nu_natural,
            'pressure_drop': dp,
            'buoyancy_pressure': dp_buoyancy,
            'dew_point': T_dew,
            'condensation_flag': T_wall_current < T_dew,
            'wall_type': wall_type,
            'convergence_iterations': iteration + 1
        })
        
        return results