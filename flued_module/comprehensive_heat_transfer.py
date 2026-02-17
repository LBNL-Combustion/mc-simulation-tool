"""Comprehensive heat transfer model for chimney/flue systems."""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Literal
from dataclasses import dataclass
from .condensation_physics import CondensationPhysics, CondensationState
from .stable_triple_wall import StableTripleWallSolver, WallLayerProperties
from .adaptive_triple_wall import AdaptiveTripleWallSolver
#from .energy_diagnostics import EnergyDiagnostics
import logging
import warnings
from ._perf_counters import COUNTERS as _PC  # TEMP: profiling

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
    """Heat transfer model: convection, radiation, condensation, and transient wall analysis."""
    
    STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
    
    def __init__(self, include_interlayer_radiation: bool = True):
        """Args:
            include_interlayer_radiation: enable inter-layer radiation in adaptive triple-wall solver.
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

        self.condensation_states = {}

        # Internal radiation (gas <-> inner wall) always enabled;
        # per-segment emissivity read from wall_properties.
        self.include_internal_radiation = True
    
    
    def gas_properties(self, temperature_K: float, pressure_Pa: float = 98457) -> FlowProperties:
        """Flue gas properties (air approximation) at given T and P."""
        density = self.physics.air_density(temperature_K, pressure_Pa)
        viscosity = self.physics.air_viscosity(temperature_K)
        thermal_conductivity = self.physics.air_thermal_conductivity(temperature_K)
        specific_heat = self.physics.air_specific_heat(temperature_K)

        velocity = 1.0  # placeholder; set by caller
        
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
        """Reynolds number, clamped to >= 1.0."""
        if props.viscosity <= 0 or diameter <= 0:
            return 1.0
        if abs(velocity) < 1e-10:
            return 1.0

        Re = props.density * abs(velocity) * diameter / props.viscosity

        if np.isnan(Re) or np.isinf(Re):
            return 1.0
        return max(Re, 1.0)
    
    def prandtl_number(self, props: FlowProperties) -> float:
        """Calculate Prandtl number"""
        return props.viscosity * props.specific_heat / props.thermal_conductivity
    
    def grashof_number(self, diameter: float, T_surface: float, T_fluid: float, props: FlowProperties) -> float:
        """Grashof number for natural convection."""
        beta = 1 / props.temperature  # ideal-gas thermal expansion coeff [1/K]
        delta_T = abs(T_surface - T_fluid)
        return self.physics.GRAVITY * beta * delta_T * diameter**3 * props.density**2 / props.viscosity**2
    
    def rayleigh_number(self, grashof: float, prandtl: float) -> float:
        """Calculate Rayleigh number"""
        return grashof * prandtl
    
    def nusselt_forced_convection_internal(self, reynolds: float, prandtl: float,
                                         length_diameter_ratio: Optional[float] = None) -> float:
        """Nusselt number for forced convection in tubes."""
        if reynolds < 2300:  # Laminar
            if length_diameter_ratio and length_diameter_ratio < 60:
                # Hausen developing-flow correlation
                term = reynolds * prandtl * (1/length_diameter_ratio)
                if term > 1000:
                    term = 1000
                Nu_dev = 3.66 + (0.0668 * term) / (1 + 0.04 * (term)**(2/3))
                return max(Nu_dev, 3.66)
            else:
                return 4.36  # constant heat flux, fully developed

        elif reynolds < 4000:  # Transition: linear interp
            Nu_lam = 4.36
            Nu_turb = 0.023 * (4000)**0.8 * prandtl**0.4
            factor = (reynolds - 2300) / (4000 - 2300)
            return Nu_lam + factor * (Nu_turb - Nu_lam)

        else:  # Turbulent — Dittus-Boelter (heating)
            if reynolds <= 0 or prandtl <= 0:
                return 3.66
            if reynolds > 1e6:
                reynolds = 1e6
            return 0.023 * reynolds**0.8 * prandtl**0.4
    
    def nusselt_natural_convection_external(self, rayleigh: float, orientation: str,
                                          height_diameter_ratio: float = 1.0) -> float:
        """Nusselt for external natural convection (Churchill & Chu)."""
        if orientation.lower() == 'vertical':
            if rayleigh < 1e4:
                return 1.0
            elif rayleigh < 1e9:
                Pr = 0.7  # air
                return 0.68 + 0.67 * rayleigh**0.25 / (1 + (0.492/Pr)**(9/16))**(4/9)
            else:
                Pr = 0.7
                return (0.825 + 0.387 * rayleigh**(1/6) / (1 + (0.492/Pr)**(9/16))**(8/27))**2

        else:  # horizontal
            if rayleigh < 1e-5:
                return 1.0
            elif rayleigh < 1e12:
                Pr = 0.7
                term1 = 0.6 + 0.387 * rayleigh**(1/6) / (1 + (0.559/Pr)**(9/16))**(8/27)
                return term1**2
            else:
                return 0.15 * rayleigh**(1/3)
    
    def nusselt_natural_convection_internal(self, rayleigh: float, orientation: str,
                                           length_diameter_ratio: float = 10.0) -> float:
        """Nusselt for internal natural convection in tubes."""
        if orientation.lower() == 'vertical':
            if rayleigh < 1e3:
                return 1.0
            elif rayleigh < 1e6:
                return 1.0 + 0.105 * rayleigh**0.3 * (1 / (1 + (1/length_diameter_ratio)**0.5))
            else:
                return 0.046 * rayleigh**(1/3)
        else:  # horizontal
            if rayleigh < 1e3:
                return 1.0
            elif rayleigh < 1e5:
                return 1.0 + 0.25 * (rayleigh / 1e3)**0.25
            else:
                return 0.14 * rayleigh**(1/3)
    
    def combined_convection_coefficient(self, h_forced: float, h_natural: float) -> float:
        """Churchill-Usagi mixed convection combination."""
        n = 3.5
        return (h_forced**n + h_natural**n)**(1/n)
    
    def radiation_heat_transfer_coefficient(self, T_surface: float, T_ambient: float,
                                          emissivity: float) -> float:
        """Linearized radiation h_rad [W/(m2*K)]."""
        T_avg = (T_surface + T_ambient) / 2
        return 4 * emissivity * self.STEFAN_BOLTZMANN * T_avg**3
    
    def dew_point_temperature(self, pressure_Pa: float, humidity_ratio: float) -> float:
        """Dew point via binary search on saturation pressure.

        Args:
            pressure_Pa: total pressure [Pa]
            humidity_ratio: kg water / kg dry air

        Returns:
            Dew point temperature [K]
        """
        p_vapor = pressure_Pa * humidity_ratio / (0.622 + humidity_ratio)

        T_min, T_max = 173.15, 373.15  # -100 to 100 degC
        
        for _ in range(20):  # ~1e-6 K precision
            T_mid = (T_min + T_max) / 2
            p_sat = self.condensation.saturation_pressure(T_mid)
            
            if p_sat < p_vapor:
                T_min = T_mid
            else:
                T_max = T_mid
                
        return T_mid
    
    
    def condensation_heat_transfer_coefficient(self, props: FlowProperties,
                                             wall_temp: float, vapor_temp: float) -> float:
        """Nusselt laminar film condensation h [W/m2K]."""
        if wall_temp >= vapor_temp:
            return 0.0

        T_film = (wall_temp + vapor_temp) / 2

        h_fg = 2.26e6       # J/kg (approx at 100 degC)
        rho_liquid = 1000.0  # kg/m3
        mu_liquid = 1e-3     # Pa*s
        k_liquid = 0.6       # W/(m*K)

        delta_T = vapor_temp - wall_temp
        Nu_cond = 0.943 * ((rho_liquid * self.physics.GRAVITY * h_fg) /
                          (mu_liquid * k_liquid * delta_T))**0.25

        return Nu_cond * k_liquid  # simplified; omits length scale
    
    def mass_transfer_coefficient(self, props: FlowProperties, diameter: float,
                                velocity: float) -> float:
        """Mass transfer coefficient via Chilton-Colburn analogy."""
        Re = self.reynolds_number(velocity, diameter, props)
        Pr = self.prandtl_number(props)
        Sc = Pr  # Sc ~ Pr for gases

        if Re < 2300:
            Sh = 3.66
        else:
            Sh = 0.023 * Re**0.8 * Sc**(1/3)

        D_AB = 2.6e-5 * (props.temperature / 298)**1.8  # m2/s, H2O in air

        return Sh * D_AB / diameter
    
    def pressure_drop(self, props: FlowProperties, diameter: float, length: float,
                     velocity: float, roughness: float = 0.0,
                     fittings_factor: float = 1.0,
                     elbows: Optional[List] = None,
                     elbow_method: Literal["K_sum", "equivalent_length"] = "K_sum") -> float:
        """Friction + fittings pressure drop [Pa]."""
        Re = self.reynolds_number(velocity, diameter, props)

        # Darcy friction factor
        if Re < 1e-6:
            f = 64.0
        elif Re < 2300:
            f = 64 / Re
        else:
            if roughness == 0:
                f = 0.316 / Re**0.25  # Blasius (smooth)
            else:
                f = 0.25 / (np.log10(roughness / (3.7 * diameter) + 5.74 / Re**0.9))**2

        effective_length = length
        K_total_elbows = 0.0

        if elbows:
            from .flow.elbows import elbow_losses

            area = np.pi * (diameter / 2)**2
            volumetric_flow = velocity * area

            for elbow_spec in elbows:
                elbow_loss = elbow_losses(
                    elbow_spec, 
                    rho=props.density,
                    mu=props.viscosity,
                    volumetric_flow_m3_s=volumetric_flow
                )
                
                if elbow_method == "equivalent_length":
                    effective_length += elbow_loss.Le_m
                else:
                    K_total_elbows += elbow_loss.K

        dp_friction = f * (effective_length / diameter) * 0.5 * props.density * velocity**2
        dp_minor = K_total_elbows * 0.5 * props.density * velocity**2

        # Legacy fittings_factor path (deprecated)
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
        """Buoyancy dP [Pa] (positive = upward draft)."""
        return self.physics.GRAVITY * height * (rho_ambient - rho_gas)
    
    def air_gap_heat_transfer_coefficient(self, T_hot: float, T_cold: float,
                                        gap_width: float, gap_height: float) -> float:
        """Combined conduction + convection + radiation across an air gap [W/(m2*K)]."""
        if gap_width <= 0:
            return 1e6  # direct contact

        T_mean = (T_hot + T_cold) / 2
        air_props = self.gas_properties(T_mean)

        h_cond = air_props.thermal_conductivity / gap_width

        delta_T = abs(T_hot - T_cold)
        if delta_T > 0.1:
            beta = 1 / T_mean  # ideal-gas thermal expansion [1/K]
            g = self.physics.GRAVITY
            L = gap_width
            rho = air_props.density
            mu = air_props.viscosity
            k = air_props.thermal_conductivity
            cp = air_props.specific_heat

            Ra = (g * beta * delta_T * L**3 * rho**2 * cp) / (mu * k)

            # Catton (1978) vertical enclosure correlation
            if Ra < 1e4:
                Nu = 1.0
            elif Ra < 1e10:
                Nu = 0.42 * Ra**0.25 * (gap_height/gap_width)**(-0.3)
            else:
                Nu = 0.046 * Ra**(1/3)
                
            h_conv = Nu * air_props.thermal_conductivity / gap_width
        else:
            h_conv = 0.0
            
        # Radiation between parallel surfaces (emissivity ~0.8 each)
        emissivity_eff = 1 / (1/0.8 + 1/0.8 - 1)
        h_rad = emissivity_eff * self.STEFAN_BOLTZMANN * (T_hot**2 + T_cold**2) * (T_hot + T_cold)
        
        h_total = h_cond + h_conv + h_rad
        
        return h_total
    
    def transient_wall_temperature(self, T_wall_old: float, T_gas: float, T_ambient: float,
                                  wall_properties: Dict, geometry: Dict,
                                  h_inner: float, h_outer: float, dt: float) -> float:
        """Single-wall transient temperature via semi-implicit lumped capacitance."""
        rho_w = wall_properties['density']        # kg/m3
        cp_w = wall_properties['specific_heat']   # J/(kg*K)
        k_w = wall_properties.get('thermal_conductivity', 50.0)  # W/(m*K)

        D_inner = geometry['inner_diameter']  # m
        D_outer = geometry['outer_diameter']  # m
        length = geometry['length']           # m

        r_inner = D_inner / 2
        r_outer = D_outer / 2

        A_inner = np.pi * D_inner * length
        A_outer = np.pi * D_outer * length
        A_mean = np.pi * (D_inner + D_outer) / 2 * length

        V_wall = np.pi * length * (D_outer**2 - D_inner**2) / 4
        C_wall = rho_w * cp_w * V_wall  # J/K
        C_wall += geometry.get('extra_wall_Cj', 0.0)  # absorbed fitting mass

        R_wall = np.log(r_outer / r_inner) / (2 * np.pi * k_w * length)  # K/W

        UA_inner = h_inner * A_inner  # W/K
        UA_outer = h_outer * A_outer  # W/K

        # Semi-implicit: (C/dt + UA_in + UA_out)*T_new = C/dt*T_old + UA_in*T_gas + UA_out*T_amb
        a = C_wall / dt + UA_inner + UA_outer
        b = C_wall / dt * T_wall_old + UA_inner * T_gas + UA_outer * T_ambient

        T_wall_new = b / a

        # Mild damping only for very small timesteps (< 10 ms) to suppress oscillations
        if dt < 0.01:
            damping = 0.7
            T_wall_new = damping * T_wall_new + (1 - damping) * T_wall_old

        # Passive system: wall stays between gas and ambient
        if T_gas > T_ambient:
            T_wall_new = np.clip(T_wall_new, T_ambient, T_gas)
        else:
            T_wall_new = max(T_wall_new, T_ambient)
        
        return T_wall_new
    
    def transient_triple_wall_temperatures(self, T_liner_old: float, T_middle_old: float,
                                         T_outer_old: float, T_gas: float, T_ambient: float,
                                         wall_properties: Dict, geometry: Dict,
                                         h_inner: float, h_outer: float, dt: float) -> Tuple[float, float, float]:
        """Triple-wall transient solve; returns (T_liner, T_middle, T_outer) [K]."""
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
        
        properties = {
            'liner': liner_props,
            'middle': middle_props,
            'outer': outer_props
        }
        
        # Thin liner (< 1 mm) or large dt causes stiffness -> use adaptive sub-stepping
        use_adaptive = liner_props.thickness < 0.001 or dt > 0.05

        if use_adaptive:
            return self.adaptive_solver.solve_with_adaptive_dt(
                T_gas, T_ambient, h_inner, h_outer,
                T_liner_old, T_middle_old, T_outer_old,
                geometry, properties, dt
            )
        else:
            return self.triple_wall_solver.solve_triple_wall_stable(
                T_gas, T_ambient, h_inner, h_outer,
                T_liner_old, T_middle_old, T_outer_old,
                geometry, properties, dt
            )
    
    def _create_default_elbow_from_fittings(self, has_fittings: bool, diameter: float) -> Optional[List]:
        """Create default 90-deg elbow for legacy has_fittings=True compatibility."""
        if not has_fittings:
            return None
            
        from .flow.elbows import ElbowSpec

        # 90-deg short-radius elbow approximates old 1.5x fittings_factor
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
        """Full heat-transfer kernel for one chimney segment (convection, condensation, wall transient)."""
        _PC['ht_kernel_calls'] += 1  # TEMP: profiling

        T_gas = segment_data['gas_temperature']       # K
        T_ambient = segment_data['ambient_temperature'] # K
        pressure = segment_data['pressure']             # Pa
        velocity = segment_data['velocity']             # m/s
        diameter = segment_data['diameter']             # m (inner)
        length = segment_data['length']                 # m (pipe length, for dP)
        thermal_length = segment_data.get('thermal_length', length)  # may include absorbed fitting area
        extra_wall_Cj = segment_data.get('extra_wall_Cj', 0.0)
        orientation = segment_data.get('orientation', 'vertical')
        has_fittings = segment_data.get('has_fittings', False)
        wall_props = segment_data['wall_properties']
        wall_type = segment_data.get('wall_type', 'single')
        humidity_ratio = segment_data.get('humidity_ratio', 0.01)  # kg/kg
        segment_id = segment_data.get('segment_id', 0)
        
        gas_props = self.gas_properties(T_gas, pressure)
        gas_props.velocity = velocity

        if segment_id not in self.condensation_states:
            self.condensation_states[segment_id] = CondensationState()

        Re = self.reynolds_number(velocity, diameter, gas_props)
        Pr = self.prandtl_number(gas_props)

        Nu_forced = self.nusselt_forced_convection_internal(Re, Pr, length/diameter)
        h_forced = Nu_forced * gas_props.thermal_conductivity / diameter

        # Fittings turbulence enhancement (~20%)
        if has_fittings:
            h_forced *= 1.2

        max_iterations = 5
        tolerance = 1e-4

        if wall_type == 'single':
            T_wall_current = segment_data['wall_temperature']
        else:  # triple
            T_liner_current = segment_data['liner_temperature']
            T_middle_current = segment_data['middle_temperature']
            T_outer_current = segment_data['outer_temperature']
            T_wall_current = T_liner_current

        # Iterative coupling: wall temp <-> condensation <-> h
        for iteration in range(max_iterations):
            T_wall_prev = T_wall_current
            humidity_ratio_prev = humidity_ratio

            Gr = self.grashof_number(diameter, T_wall_current, T_gas, gas_props)
            Ra = self.rayleigh_number(Gr, Pr)

            length_diameter_ratio = length / diameter
            Nu_natural = self.nusselt_natural_convection_internal(Ra, orientation, length_diameter_ratio)
            h_natural = Nu_natural * gas_props.thermal_conductivity / diameter

            h_inner = self.combined_convection_coefficient(h_forced, h_natural)

            # Engineering minimums — smooth-pipe correlations under-predict h in real
            # chimneys (roughness, entrance effects, fitting-induced turbulence)
            h_min_chimney = 10.0  # W/(m2*K)
            h_min_general = 5.0   # W/(m2*K)

            # Use absolute T threshold to avoid feedback loop (low h -> low dT -> no min)
            T_gas_C = T_gas - 273.15
            is_hot_gas = T_gas_C > 100

            if orientation.lower() == 'vertical' and is_hot_gas:
                if h_inner < h_min_chimney:
                    self.logger.debug(f"Increasing h_inner from {h_inner:.1f} to {h_min_chimney} W/(m2*K)")
                h_inner = max(h_inner, h_min_chimney)
            else:
                h_inner = max(h_inner, h_min_general)

            h_inner *= getattr(self, 'h_inner_multiplier', 1.0)  # DIAG: sensitivity
            # Internal radiation (gas <-> inner wall)
            if wall_type == 'triple':
                _eps_ir = wall_props.get('liner', {}).get('emissivity', 0.3)
            else:
                _eps_ir = wall_props.get('emissivity', 0.8)
            _T_avg_ir = (T_gas + T_wall_current) / 2.0
            h_inner += 4.0 * _eps_ir * self.STEFAN_BOLTZMANN * _T_avg_ir**3

            area = np.pi * diameter * thermal_length
            condensation_state = self.condensation_states[segment_id]

            h_inner_modified = self.condensation.film_heat_transfer_modification(
                h_inner, condensation_state.film_thickness
            )

            condensation_state = self.condensation.update_condensation_state(
                condensation_state, T_gas, T_wall_current, pressure,
                humidity_ratio, h_inner_modified, area, dt
            )
            self.condensation_states[segment_id] = condensation_state

            if condensation_state.latent_heat_rate != 0:
                q_sensible = h_inner_modified * area * (T_gas - T_wall_current)
                q_total = q_sensible + condensation_state.latent_heat_rate
                h_inner_effective = q_total / (area * (T_gas - T_wall_current)) if T_gas != T_wall_current else h_inner_modified
            else:
                h_inner_effective = h_inner_modified

            if segment_data.get('mass_flow_rate', 0) > 0:
                gas_volume = np.pi * (diameter/2)**2 * length
                humidity_ratio = self.condensation.gas_humidity_change(
                    humidity_ratio, condensation_state.net_rate,
                    segment_data['mass_flow_rate'], gas_volume,
                    gas_props.density, dt
                )
            
            if wall_type == 'single':
                outer_diameter = diameter + 2 * wall_props['thickness']
                ambient_props = self.gas_properties(T_ambient, pressure)

                Gr_outer = self.grashof_number(outer_diameter, T_wall_current, T_ambient, ambient_props)
                Ra_outer = self.rayleigh_number(Gr_outer, self.prandtl_number(ambient_props))
                Nu_outer = self.nusselt_natural_convection_external(Ra_outer, orientation)
                h_conv_outer = Nu_outer * ambient_props.thermal_conductivity / outer_diameter

                emissivity = wall_props.get('emissivity', 0.8)
                h_rad_outer = self.radiation_heat_transfer_coefficient(T_wall_current, T_ambient, emissivity)
                h_outer = h_conv_outer + h_rad_outer
                h_outer *= getattr(self, 'h_outer_multiplier', 1.0)  # DIAG: sensitivity
                h_outer = max(h_outer, 1.0)  # W/(m2*K)

                geometry = {
                    'inner_diameter': diameter,
                    'outer_diameter': outer_diameter,
                    'length': thermal_length,
                    'extra_wall_Cj': extra_wall_Cj,
                }

                T_wall_new = self.transient_wall_temperature(
                    T_wall_current, T_gas, T_ambient, wall_props, geometry,
                    h_inner_effective, h_outer, dt
                )
                
                T_wall_current = T_wall_new
                
            else:  # triple wall
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
                
                emissivity_outer = wall_props['outer'].get('emissivity', 0.8)
                h_rad_outer = self.radiation_heat_transfer_coefficient(T_outer_current, T_ambient, emissivity_outer)
                h_outer = h_conv_outer + h_rad_outer
                h_outer *= getattr(self, 'h_outer_multiplier', 1.0)  # DIAG: sensitivity
                h_outer = max(h_outer, 1.0)  # W/(m2*K)

                geometry = {
                    'inner_diameter': diameter,
                    'length': thermal_length,
                    'extra_wall_Cj': extra_wall_Cj,
                }

                T_liner_new, T_middle_new, T_outer_new = self.transient_triple_wall_temperatures(
                    T_liner_current, T_middle_current, T_outer_current, 
                    T_gas, T_ambient, wall_props, geometry, 
                    h_inner_effective, h_outer, dt
                )
                
                T_liner_current = T_liner_new
                T_middle_current = T_middle_new
                T_outer_current = T_outer_new
                T_wall_current = T_liner_current
            
            residual = abs(T_wall_current - T_wall_prev) / (T_wall_current + 273.15)
            humidity_residual = abs(humidity_ratio - humidity_ratio_prev) / (humidity_ratio + 1e-6)
            
            if residual < tolerance and humidity_residual < tolerance:
                break

        _PC['condensation_iters_total'] += iteration + 1  # TEMP: profiling

        # Spiral corrugation swirl-induced HT enhancement (Mh pre-computed in chimney.py)
        corrugation = segment_data.get('corrugation', None)
        if corrugation is not None and corrugation.get('type') == 'spiral':
            if corrugation.get('enable_spiral_HT_enhancement', False):
                Mh = segment_data.get('Mh', 1.0)
                h_inner_effective *= Mh
                self.logger.debug(f"Applied spiral HT multiplier Mh={Mh:.3f} to h_inner")

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
        
        elbows = segment_data.get('elbows', None)
        elbow_method = segment_data.get('elbow_method', 'K_sum')

        if elbows is None and has_fittings:
            elbows = self._create_default_elbow_from_fittings(has_fittings, diameter)
        
        dp = self.pressure_drop(
            gas_props, diameter, length, velocity,
            roughness=segment_data.get('roughness', 0.0),
            elbows=elbows,
            elbow_method=elbow_method
        )
        
        ambient_density = self.gas_properties(T_ambient, pressure).density
        dp_buoyancy = self.buoyancy_pressure(length, ambient_density, gas_props.density)

        T_dew = self.dew_point_temperature(pressure, humidity_ratio)

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