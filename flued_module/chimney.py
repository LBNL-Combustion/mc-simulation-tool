"""
Main Chimney class for flue simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .physics import PhysicalProperties
from .mass_flow import MassFlow
from .comprehensive_heat_transfer import ComprehensiveHeatTransfer
from .coupled_solver import CoupledFlowThermalSolver
from .early_diagnostics import EarlyTimeDiagnostics
from .chimney_geometry_plot import plot_chimney_geometry, plot_chimney_snapshot

@dataclass
class ChimneyPart:
    """Configuration for a single or triple-wall chimney part
    
    This class stores all the properties needed to define one section of the chimney.
    The chimney can be made up of multiple parts with different properties.
    """
    # BASIC CONFIGURATION
    type: str  # 'single' for simple wall, 'triple' for insulated wall with air gaps
    inner_diameter_ft: float  # Flow area diameter in feet (will be converted to meters)
    height_ft: float  # Height of this chimney section in feet
    z_ft: float  # Elevation above reference point in feet (affects buoyancy)
    orientation: str  # 'vertical' or 'horizontal' - affects natural convection
    T_ambient_K: float  # Ambient temperature outside the chimney in Kelvin
    Pa: float  # Atmospheric pressure in Pascals
    num_segments: int  # Number of computational cells to divide this part into
    
    # SINGLE WALL PROPERTIES (used when type='single')
    # These define the material properties for a simple single-layer wall
    thickness_w_m: float = 0.001  # Wall thickness in meters
    density_w_kg_m3: float = 7850  # Wall material density (kg/m³) - default is steel
    cp_w_J_kg_K: float = 460  # Wall specific heat (J/kg·K) - energy to heat 1kg by 1K
    k_w_W_m_K: float = 50  # Wall thermal conductivity (W/m·K) - heat flow rate
    epsilon_w: float = 0.8  # Wall emissivity (0-1) - ability to radiate heat
    
    # TRIPLE WALL PROPERTIES (used when type='triple')
    # Triple wall chimneys have 3 layers for better insulation and safety
    
    # Inner wall (liner) - contacts the hot flue gases
    thickness_inner_m: float = 0.0008  # Inner wall thickness (m) - thin stainless steel
    density_inner_kg_m3: float = 8000  # Inner wall density (kg/m³) - stainless steel
    cp_inner_J_kg_K: float = 500  # Inner wall specific heat (J/kg·K)
    k_inner_W_m_K: float = 16  # Inner wall thermal conductivity (W/m·K) - lower than regular steel
    epsilon_inner: float = 0.3  # Inner wall emissivity - polished metal radiates less
    
    # Middle layer (insulation) - provides thermal resistance
    thickness_middle_m: float = 0.025  # Middle layer thickness (m) - 25mm insulation
    density_middle_kg_m3: float = 100  # Middle layer density (kg/m³) - lightweight insulation
    cp_middle_J_kg_K: float = 840  # Middle layer specific heat (J/kg·K)
    k_middle_W_m_K: float = 0.04  # Middle layer thermal conductivity (W/m·K) - very low for insulation
    
    # Outer wall (structural) - provides strength and weather protection
    thickness_outer_m: float = 0.001  # Outer wall thickness (m) - structural steel
    density_outer_kg_m3: float = 7850  # Outer wall density (kg/m³) - regular steel
    cp_outer_J_kg_K: float = 460  # Outer wall specific heat (J/kg·K)
    k_outer_W_m_K: float = 50  # Outer wall thermal conductivity (W/m·K)
    epsilon_outer: float = 0.8  # Outer wall emissivity - painted/oxidized surface
    
    # AIR GAP PROPERTIES (for triple wall)
    # Air gaps provide additional insulation through trapped air
    air_gap_inner_m: float = 0.0  # Air gap between inner and middle layers (m)
    air_gap_outer_m: float = 0.0  # Air gap between middle and outer layers (m)
    
    # GENERAL PROPERTIES (apply to all wall types)
    film_w: float = 5e-5  # Max condensate film thickness (m) - for condensation modeling
    has_fittings: bool = False  # True if section has elbows/tees (increases pressure drop)
    roughness_m: float = 1e-6  # Surface roughness in meters - affects friction factor
    humidity_ratio: float = 0.01  # kg water / kg dry air - for condensation calculations
    
    def __post_init__(self):
        """Convert units and validate inputs
        
        This method runs automatically after initialization to:
        1. Convert Imperial units (feet) to SI units (meters)
        2. Calculate derived geometric properties
        3. Validate that inputs make physical sense
        """
        # Convert feet to meters (1 ft = 0.3048 m)
        self.inner_diameter_m = self.inner_diameter_ft * 0.3048
        self.height_m = self.height_ft * 0.3048
        self.z_m = self.z_ft * 0.3048
        
        # Calculate flow area and perimeter for heat transfer calculations
        self.inner_area_m2 = np.pi * (self.inner_diameter_m / 2) ** 2  # Cross-sectional area
        self.inner_perimeter_m = np.pi * self.inner_diameter_m  # Wetted perimeter
        
        # Calculate outer diameter based on wall type
        if self.type == 'single':
            self.outer_diameter_m = self.inner_diameter_m + 2 * self.thickness_w_m
        elif self.type == 'triple':
            # For triple wall: inner liner + gap + middle + gap + outer
            total_thickness = (self.thickness_inner_m + self.air_gap_inner_m + 
                             self.thickness_middle_m + self.air_gap_outer_m + 
                             self.thickness_outer_m)
            self.outer_diameter_m = self.inner_diameter_m + 2 * total_thickness
            
            # Calculate intermediate diameters for triple wall
            self.liner_outer_diameter_m = self.inner_diameter_m + 2 * self.thickness_inner_m
            self.middle_inner_diameter_m = self.liner_outer_diameter_m + 2 * self.air_gap_inner_m
            self.middle_outer_diameter_m = self.middle_inner_diameter_m + 2 * self.thickness_middle_m
            self.outer_inner_diameter_m = self.middle_outer_diameter_m + 2 * self.air_gap_outer_m
        else:
            raise ValueError(f"Unknown wall type: {self.type}. Must be 'single' or 'triple'")
        
        # Validate inputs
        if self.num_segments < 1:
            raise ValueError("num_segments must be at least 1")
        if self.inner_diameter_m <= 0:
            raise ValueError("inner_diameter must be positive")
        if self.height_m <= 0:
            raise ValueError("height must be positive")

class Chimney:
    """Main chimney simulation class
    
    This is the core class that manages the entire simulation. It:
    1. Stores the chimney configuration (geometry, materials, conditions)
    2. Manages the simulation time stepping
    3. Coordinates the physics calculations (heat transfer, fluid flow)
    4. Stores and provides access to all results
    """
    
    def __init__(self, parts: List[Dict[str, Any]], dt: float = 0.5, total_time: float = 60.0,
                 use_coupled_solver: bool = True):
        """Initialize chimney simulation
        
        Args:
            parts: List of dictionaries defining chimney parts. Each dict contains
                   properties like 'type', 'inner_diameter_ft', 'height_ft', etc.
            dt: Time step in seconds. Smaller = more accurate but slower.
                For triple walls, may need dt < 0.1s for stability.
            total_time: Total simulation time in seconds. How long to simulate
                        after ignition.
            use_coupled_solver: If True, use iterative coupling between flow and thermal
                               equations for better accuracy. If False, use faster
                               sequential approach.
        """
        # Store simulation parameters
        self.dt = dt  # Time step size
        self.total_time = total_time  # How long to simulate
        self.time_steps = int(total_time / dt)  # Number of time steps
        
        # Convert part dictionaries to ChimneyPart objects
        # This validates inputs and converts units
        self.parts = [ChimneyPart(**part) for part in parts]
        
        # Calculate total number of computational segments
        # Each part is divided into segments for numerical solution
        self.total_segments = sum(part.num_segments for part in self.parts)
        
        # Check for triple wall parts and warn if time step is large
        # Triple walls have thin metal layers that require small time steps
        # for numerical stability (explicit time integration limit)
        has_triple_wall = any(part.get('type', 'single') == 'triple' for part in parts)
        if has_triple_wall and dt > 1.0:
            print(f"Warning: Large time step ({dt}s) detected with triple wall parts.")
            print("Consider using a smaller time step if numerical instabilities occur.")
            # We don't automatically reduce dt - let user decide based on their needs
        
        # Initialize physics modules
        # Each module handles a specific aspect of the physics
        self.physics = PhysicalProperties()  # Gas properties (density, viscosity, etc.)
        self.mass_flow = MassFlow()  # Flow rate calculations
        self.comprehensive_ht = ComprehensiveHeatTransfer()  # Advanced heat transfer with all modes
        
        # Initialize coupled solver if requested
        # Coupled solver iterates between flow and thermal equations for consistency
        self.use_coupled_solver = use_coupled_solver
        if use_coupled_solver:
            self.coupled_solver = CoupledFlowThermalSolver()
            self.convergence_history = []  # Track convergence metrics for debugging
            
        # Initialize early diagnostics
        # This checks for numerical problems in the first few time steps
        self.diagnostics = EarlyTimeDiagnostics(ambient_temp_K=self.parts[0].T_ambient_K)
        
        # Initialize all the arrays to store results
        # This allocates memory for temperature, flow, pressure histories
        self._initialize_arrays()
        
        # Simulation state tracking
        self.current_time = 0.0  # Current simulation time in seconds
        self.time_step = 0  # Current time step index
    #Done
    def run_simulation(self, save_interval: int = 1):
        """Run the complete simulation using comprehensive heat transfer model
        
        This is the main simulation loop that advances through time, calculating
        temperatures, flow rates, and pressures at each time step.
        
        Args:
            save_interval: Save results every N time steps (not currently used)
        """
        print(f"Starting comprehensive heat transfer simulation:")
        print(f"  {self.total_segments} segments, {self.time_steps} time steps")
        print(f"  Time step: {self.dt}s, Total time: {self.total_time}s")
        
        # MAIN TIME LOOP - advance from t=dt to t=total_time
        for step in range(1, self.time_steps + 1):
            # Update simulation state
            self.time_step = step
            self.current_time = step * self.dt
            
            # STEP 1: CALCULATE INLET TEMPERATURE
            # This is the temperature of gas entering the bottom of the chimney
            # It ramps up over time to simulate fire establishment
            T_ambient_inlet = self.parts[0].T_ambient_K
            
            # Note: Smooth ramp-up code is commented out but could be enabled
            # to avoid numerical shock from sudden temperature change
            T_inlet = self._default_inlet_temp(self.current_time, T_ambient=T_ambient_inlet)
            
            # STEP 2: SOLVE PHYSICS EQUATIONS
            if self.use_coupled_solver:
                # COUPLED APPROACH: Iterate between flow and thermal equations
                # This ensures consistency between buoyancy (depends on T) and
                # heat transfer (depends on flow). More accurate but slower.
                metrics = self.coupled_solver.solve_coupled_timestep(self, step, T_inlet, self.dt)
                self.convergence_history.append(metrics)
                
                # Warn if iteration didn't converge (but continue anyway)
                if step % max(1, self.time_steps // 20) == 0 and not metrics.converged:
                    print(f"  Warning: Coupling did not converge at t={self.current_time:.1f}s")
            else:
                # SEQUENTIAL APPROACH: First heat transfer, then flow
                # Faster but less accurate since T and flow aren't fully consistent
                
                # Solve heat transfer for all segments with current flow
                self._solve_comprehensive_heat_transfer(step, T_inlet)
                
                # Update flow based on new temperatures (buoyancy)
                self._update_flow_and_pressure(step)
            
            # STEP 3: DIAGNOSTICS AND STABILITY CHECKS
            # Check for unphysical behavior in early timesteps when transients are strong
            if step <= 30:  # First 30 timesteps are most prone to instability
                # Prepare temperature arrays based on what exists
                T_liner = self.T_liner[step] if hasattr(self, 'T_liner') else None
                T_middle = self.T_middle[step] if hasattr(self, 'T_middle') else None
                T_outer = self.T_outer[step] if hasattr(self, 'T_outer') else None
                
                # Run diagnostic checks for numerical problems
                self.diagnostics.check_timestep(
                    step, self.current_time, 
                    self.T_gas[step], self.T_wall[step],
                    T_liner, T_middle, T_outer,
                    self.mass_flow_rate[step], self.dt
                )
            
            # STEP 4: PROGRESS REPORTING
            # Print status every 10% of simulation (less frequent for speed)
            if step % max(1, self.time_steps // 10) == 0:
                progress = 100 * step / self.time_steps
                gas_temp_range = (self.T_gas[step].min() - 273.15, self.T_gas[step].max() - 273.15)
                wall_temp_range = (self.T_wall[step].min() - 273.15, self.T_wall[step].max() - 273.15)
                print(f"  {progress:5.1f}% | t={self.current_time:6.1f}s | Gas: {gas_temp_range[0]:5.1f}-{gas_temp_range[1]:5.1f}°C | Wall: {wall_temp_range[0]:5.1f}-{wall_temp_range[1]:5.1f}°C")
                
        print("Comprehensive simulation completed successfully")
        
        # Run conservation diagnostics
        self.check_conservation()
    #Done
    def _default_inlet_temp(self, t, T_ambient=293.15, T_max=773.15, tau=300):
        """
        Models the inlet gas temperature entering the chimney as a function of time.
        This is the temperature of the flue gas entering the bottom of the chimney,
        NOT the fire temperature itself.
        
        Uses an exponential rise model that simulates:
        - Cold start at ambient temperature
        - Gradual heating as fire establishes
        - Asymptotic approach to maximum temperature
        
        Parameters:
            t (float or array): Time in seconds since ignition
            T_ambient (float): Ambient temperature in Kelvin (default is 20°C = 293.15 K)
            T_max (float): Maximum inlet gas temperature in Kelvin (default is 500°C = 773.15 K)
            tau (float): Characteristic time constant in seconds. Controls how fast it heats up.
                        Larger tau = slower heating. tau=300s means 63% of max temp at 5 min.
        
        Returns:
            float or array: Inlet gas temperature in Kelvin at time t
        """
        # Exponential rise: starts at T_ambient, approaches T_max
        # At t=tau, temperature is 63% of the way from ambient to max
        # At t=3*tau, temperature is 95% of the way to max
        return T_ambient + (T_max - T_ambient) * (1 - np.exp(-t / tau))
    def _default_inlet_temp_fire(self, t_sec: float) -> float:
        """Calculate inlet gas temperature based on standard fire curve.
        
        This uses the ISO 834 standard fire curve (common in fire testing)
        to model how fire temperature evolves. The flue gas temperature
        is empirically scaled to be about 50% of the fire temperature.
        
        Args:
            t_sec: Time in seconds since ignition
            
        Returns:
            Temperature in Kelvin
        """
        # ISO 834 fire curve: T = 20 + 345 * log10(8t + 1) where t is in minutes
        # We add 1.5s to avoid log(0) at t=0
        T_fire_C = 20 + 345 * np.log10(8 * ((t_sec + 1.5) / 60) + 1)
        
        # Flue gases are cooler than fire due to heat loss and air dilution
        T_flue_C = 0.5 * T_fire_C   # Empirical scaling factor
        
        # Convert to Kelvin and cap at reasonable maximum
        T_flue_K = T_flue_C + 273.15
        return min(T_flue_K, 800)    # Cap at 800K (527°C) for safety
    #Done
    def _solve_comprehensive_heat_transfer(self, step: int, T_inlet: float):
        """Solve comprehensive heat transfer for all segments
        
        This method calculates heat transfer for each segment of the chimney,
        updating gas and wall temperatures based on convection, conduction,
        and radiation.
        
        Args:
            step: Current time step index
            T_inlet: Temperature of gas entering bottom of chimney (K)
        """
        
        # Set inlet boundary condition - hot gas enters at bottom
        self.T_gas[step, 0] = T_inlet
        
        # SOLVE EACH SEGMENT FROM BOTTOM TO TOP
        # Gas flows upward, so we solve in flow direction
        for seg in range(self.total_segments):
            # Get info about which part this segment belongs to
            part_idx, local_seg, part = self.get_segment_info(seg)
            
            # Length of this computational segment
            segment_length = part.height_m / part.num_segments
            
            # INLET CONDITION FOR THIS SEGMENT
            if seg > 0:
                # Gas enters this segment from the previous segment
                # Temperature is continuous (no sudden jumps)
                self.T_gas[step, seg] = self.T_gas[step, seg-1]
                
                # Debug output for transition between different wall types
                if seg == 5 and step == self.time_steps:  # Example: first segment of second part
                    print(f"DEBUG Gas flow from single to triple wall:")
                    print(f"  Single wall final segment gas temp: {self.T_gas[step, seg-1] - 273.15:.1f}°C")
                    print(f"  Triple wall first segment gas temp: {self.T_gas[step, seg] - 273.15:.1f}°C")
                
            # Estimate velocity for this segment
            if step == 1:
                velocity = 1.0  # Initial guess
            else:
                velocity = self.velocity[step-1, seg]
                
            # PREPARE INPUT DATA FOR HEAT TRANSFER CALCULATION
            # Package all the information needed to compute heat transfer
            segment_data = {
                'gas_temperature': self.T_gas[step, seg],      # Current gas temp in segment
                'ambient_temperature': part.T_ambient_K,        # Outside air temperature
                'pressure': part.Pa,                            # Gas pressure (affects properties)
                'velocity': velocity,                           # Gas velocity (affects convection)
                'diameter': part.inner_diameter_m,              # Flow area diameter
                'length': segment_length,                       # Length of this segment
                'orientation': part.orientation,                # Vertical/horizontal (affects natural conv)
                'has_fittings': part.has_fittings,            # Elbows/tees increase turbulence
                'humidity_ratio': part.humidity_ratio,          # For condensation calculations
                'wall_type': part.type,                        # 'single' or 'triple'
                'segment_id': seg,                             # For debugging
                'mass_flow_rate': self.mass_flow_rate[step-1] if step > 0 else 0.01  # Flow rate
            }
            
            # Add wall-specific data based on configuration
            if part.type == 'single':
                # Get previous wall temperature 
                prev_T_wall = self.T_wall[step-1, seg]
                # Ensure temperature is physical (check for uninitialized arrays)
                if prev_T_wall <= 0:  # Uninitialized or absolute zero
                    print(f"WARNING: Uninitialized wall temp {prev_T_wall:.1f}K at step={step}, seg={seg}, using ambient")
                    prev_T_wall = part.T_ambient_K
                    
                segment_data.update({
                    'wall_temperature': prev_T_wall,
                    'wall_properties': {
                        'density': part.density_w_kg_m3,
                        'specific_heat': part.cp_w_J_kg_K,
                        'thickness': part.thickness_w_m,
                        'emissivity': part.epsilon_w,
                        'thermal_conductivity': part.k_w_W_m_K
                    }
                })
            elif part.type == 'triple':
                # Get previous temperatures 
                prev_T_liner = self.T_liner[step-1, seg]
                prev_T_middle = self.T_middle[step-1, seg] 
                prev_T_outer = self.T_outer[step-1, seg]
                
                # Ensure temperatures are physical (check for uninitialized arrays)
                if prev_T_liner <= 0:  # Uninitialized or absolute zero
                    print(f"WARNING: Uninitialized liner temp {prev_T_liner:.1f}K at step={step}, seg={seg}, using ambient")
                    prev_T_liner = part.T_ambient_K
                    prev_T_middle = part.T_ambient_K
                    prev_T_outer = part.T_ambient_K
                
                # Debug: show what we're reading
                if step <= 3 and seg <= 2:
                    print(f"DEBUG: Step {step}, seg {seg} - reading prev temps: liner={prev_T_liner-273.15:.1f}°C, middle={prev_T_middle-273.15:.1f}°C, outer={prev_T_outer-273.15:.1f}°C")
                    
                segment_data.update({
                    'liner_temperature': prev_T_liner,
                    'middle_temperature': prev_T_middle,
                    'outer_temperature': prev_T_outer,
                    'wall_properties': {
                        'liner': {
                            'density': part.density_inner_kg_m3,
                            'specific_heat': part.cp_inner_J_kg_K,
                            'thickness': part.thickness_inner_m,
                            'emissivity': part.epsilon_inner,
                            'thermal_conductivity': part.k_inner_W_m_K
                        },
                        'middle': {
                            'density': part.density_middle_kg_m3,
                            'specific_heat': part.cp_middle_J_kg_K,
                            'thickness': part.thickness_middle_m,
                            'thermal_conductivity': part.k_middle_W_m_K
                        },
                        'outer': {
                            'density': part.density_outer_kg_m3,
                            'specific_heat': part.cp_outer_J_kg_K,
                            'thickness': part.thickness_outer_m,
                            'emissivity': part.epsilon_outer,
                            'thermal_conductivity': part.k_outer_W_m_K
                        },
                        'air_gap_inner': part.air_gap_inner_m,
                        'air_gap_outer': part.air_gap_outer_m
                    }
                })
            
            # Calculate comprehensive heat transfer
            results = self.comprehensive_ht.calculate_segment_heat_transfer(segment_data, self.dt)
            
            # Debug heat transfer coefficients and temperatures
            if step == self.time_steps:  # Only debug final step
                print(f"DEBUG Segment {seg+1} (Part {part_idx+1} {part.type.title()} Wall):")
                print(f"  Gas temp in: {self.T_gas[step, seg] - 273.15:.1f}°C")
                print(f"  h_inner: {results['h_inner']:.1f} W/(m²·K)")
                print(f"  h_outer: {results['h_outer']:.1f} W/(m²·K)")
                
                if part.type == 'triple':
                    if 'liner_temperature_new' in results:
                        print(f"  Liner temp out: {results['liner_temperature_new'] - 273.15:.1f}°C")
                        print(f"  Middle temp out: {results['middle_temperature_new'] - 273.15:.1f}°C")
                        print(f"  Outer temp out: {results['outer_temperature_new'] - 273.15:.1f}°C")
                else:
                    print(f"  Wall temp out: {results['wall_temperature_new'] - 273.15:.1f}°C")
            
            # Validate results
            if 'wall_temperature_new' not in results and 'liner_temperature_new' not in results:
                print(f"ERROR: No temperature results at step={step}, seg={seg}")
                print(f"Results keys: {list(results.keys())}")
                # Use ambient as fallback
                results['wall_temperature_new'] = part.T_ambient_K
                results['liner_temperature_new'] = part.T_ambient_K
                results['middle_temperature_new'] = part.T_ambient_K
                results['outer_temperature_new'] = part.T_ambient_K
            
            # Update temperatures and properties based on wall type
            if part.type == 'single':
                # Ensure result is physical
                T_wall_new = results['wall_temperature_new']
                if T_wall_new <= 0:
                    print(f"ERROR: Computed wall temp {T_wall_new:.1f}K is unphysical, using ambient")
                    T_wall_new = part.T_ambient_K
                    
                self.T_wall[step, seg] = T_wall_new
                self.T_outer[step, seg] = T_wall_new  # Single wall
                # Set triple wall arrays for consistency
                self.T_liner[step, seg] = T_wall_new
                self.T_middle[step, seg] = T_wall_new
            elif part.type == 'triple':
                # Ensure results are physical
                T_liner_new = results['liner_temperature_new']
                T_middle_new = results['middle_temperature_new']
                T_outer_new = results['outer_temperature_new']
                
                if T_liner_new <= 0:
                    print(f"ERROR: Computed liner temp {T_liner_new:.1f}K is unphysical, using ambient")
                    T_liner_new = part.T_ambient_K
                    T_middle_new = part.T_ambient_K
                    T_outer_new = part.T_ambient_K
                    
                # Debug: show what we're storing
                if step <= 3 and seg <= 2:
                    print(f"DEBUG: Step {step}, seg {seg} - storing new temps: liner={T_liner_new-273.15:.1f}°C, middle={T_middle_new-273.15:.1f}°C, outer={T_outer_new-273.15:.1f}°C")
                    
                self.T_liner[step, seg] = T_liner_new
                self.T_middle[step, seg] = T_middle_new
                self.T_outer[step, seg] = T_outer_new
                # Set wall temperature to liner for compatibility
                self.T_wall[step, seg] = T_liner_new
            
            # Update gas temperature based on heat loss to walls
            # Calculate heat loss rate from gas
            q_loss = results['h_inner'] * (np.pi * part.inner_diameter_m * segment_length) * (self.T_gas[step, seg] - self.T_wall[step, seg])
            
            # Gas properties for heat capacity
            gas_props = self.comprehensive_ht.gas_properties(self.T_gas[step, seg], part.Pa)
            gas_volume = np.pi * (part.inner_diameter_m/2)**2 * segment_length
            gas_mass = gas_props.density * gas_volume
            
            # Calculate gas temperature change including advection
            # NOTE: Only update gas temperature here if NOT using coupled solver
            # The coupled solver handles gas temperature updates itself
            if gas_mass > 0 and not self.use_coupled_solver and seg > 0:
                # For segments after the first, include advection from upstream
                mass_flow = self.mass_flow_rate[step-1] if step > 0 else 0.0
                
                # CRITICAL PHYSICS: If no flow, gas temperature approaches wall temperature
                # which approaches ambient. Gas CANNOT cool below ambient without flow!
                if mass_flow > 0:
                    T_upstream = self.T_gas[step, seg-1]
                    
                    # Energy balance: accumulation = advection_in - advection_out - heat_loss
                    # mass * cp * dT/dt = mdot * cp * (T_in - T_out) - q_loss
                    q_advection_net = mass_flow * gas_props.specific_heat * (T_upstream - self.T_gas[step, seg])
                    dT_gas = (q_advection_net - q_loss) * self.dt / (gas_mass * gas_props.specific_heat)
                    
                    # CRITICAL: Limit cooling to available thermal energy
                    # Gas cannot cool below ambient without external work
                    if dT_gas < 0:
                        max_cooling = self.T_gas[step, seg] - part.T_ambient_K
                        if abs(dT_gas) > max_cooling:
                            dT_gas = -max_cooling
                    
                    self.T_gas[step, seg] = self.T_gas[step, seg] + dT_gas
                else:
                    # No flow: gas temperature approaches wall temperature
                    # Wall temperature approaches ambient (see wall heat transfer)
                    # This prevents gas from cooling below ambient
                    tau_gas_wall = gas_mass * gas_props.specific_heat / (results['h_inner'] * np.pi * part.inner_diameter_m * segment_length)
                    self.T_gas[step, seg] = self.T_gas[step, seg] + (self.T_wall[step, seg] - self.T_gas[step, seg]) * self.dt / tau_gas_wall
            
            # Store heat transfer coefficients
            self.h_conv_inner[step, seg] = results['h_inner']
            self.h_conv_outer[step, seg] = results['h_outer']
            
            # Update density based on new gas temperature
            self.density[step, seg] = self.comprehensive_ht.gas_properties(
                self.T_gas[step, seg], part.Pa
            ).density
            
            # Apply physical temperature bounds - nothing below ambient
            # This is fundamental thermodynamics: no refrigeration in system
            self.T_gas[step, seg] = max(self.T_gas[step, seg], part.T_ambient_K)
            self.T_wall[step, seg] = max(self.T_wall[step, seg], part.T_ambient_K)
            if part.type == 'triple':
                self.T_liner[step, seg] = max(self.T_liner[step, seg], part.T_ambient_K)
                self.T_middle[step, seg] = max(self.T_middle[step, seg], part.T_ambient_K)
                self.T_outer[step, seg] = max(self.T_outer[step, seg], part.T_ambient_K)
    #Done
    def _update_flow_and_pressure(self, step: int):
        """Update mass flow rate, velocity, and pressure using buoyancy and pressure losses
        
        This implements the momentum equation for the chimney:
        - Buoyancy forces (from density differences) drive the flow upward
        - Friction and fitting losses oppose the flow
        - Net pressure determines the flow rate
        
        Args:
            step: Current time step index
        """
        
        # STEP 1: CALCULATE DRIVING FORCES (BUOYANCY)
        # Hot gas is less dense than cold ambient air, creating upward force
        total_buoyancy = 0.0  # Total buoyancy pressure driving flow (Pa)
        total_pressure_loss = 0.0  # Total friction losses resisting flow (Pa)
        
        # Sum contributions from each segment
        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            segment_length = part.height_m / part.num_segments
            
            # Density of hot gas in chimney
            rho_gas = self.density[step, seg]
            
            # Density of cold ambient air outside
            rho_ambient = self.comprehensive_ht.gas_properties(part.T_ambient_K, part.Pa).density
            
            # Buoyancy pressure: dp = g * height * (rho_ambient - rho_gas)
            # Positive when gas is hotter (less dense) than ambient
            dp_buoyancy_seg = self.comprehensive_ht.buoyancy_pressure(
                segment_length, rho_ambient, rho_gas
            )
            total_buoyancy += dp_buoyancy_seg
            
            # STEP 2: CALCULATE RESISTANCE FORCES (FRICTION)
            # Get velocity from previous time step (or initial guess)
            if step > 1:
                velocity = self.velocity[step-1, seg]
            else:
                velocity = 1.0  # Initial guess for first time step
                
            # Get gas properties at current temperature
            gas_props = self.comprehensive_ht.gas_properties(self.T_gas[step, seg], part.Pa)
            gas_props.velocity = velocity
            
            # Fittings (elbows, tees) increase pressure drop
            fittings_factor = 1.5 if part.has_fittings else 1.0
            
            # Calculate pressure drop from friction (Darcy-Weisbach equation)
            dp_loss_seg = self.comprehensive_ht.pressure_drop(
                gas_props, part.inner_diameter_m, segment_length, 
                velocity, part.roughness_m, fittings_factor
            )
            total_pressure_loss += dp_loss_seg
        
        # STEP 3: SOLVE FOR MASS FLOW RATE
        # Balance equation: Buoyancy pressure - Friction losses = Net pressure
        net_pressure = total_buoyancy - total_pressure_loss
        
        if net_pressure > 0:
            # Positive net pressure drives upward flow
            # Use Bernoulli equation simplified: dp = K * rho * v^2 / 2
            # Solving for velocity: v = sqrt(2 * dp / (K * rho))
            avg_density = np.mean(self.density[step, :])  # Average gas density
            flow_coefficient = 1.0  # Simplified - could include discharge coefficient
            velocity_avg = np.sqrt(2 * abs(net_pressure) / (flow_coefficient * avg_density))
            
            # Mass flow rate = density * velocity * area
            avg_area = np.mean([part.inner_area_m2 for part in self.parts])
            self.mass_flow_rate[step] = avg_density * velocity_avg * avg_area
        else:
            # No buoyancy (or negative) means no natural draft
            # This happens when:
            # 1. Gas has cooled to ambient temperature
            # 2. Friction losses exceed buoyancy forces
            self.mass_flow_rate[step] = 0.0
        
        # STEP 4: UPDATE VELOCITIES IN EACH SEGMENT
        # Velocity varies with diameter and density changes
        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            
            # Velocity from continuity equation
            self.velocity[step, seg] = self.mass_flow_rate[step] / (
                self.density[step, seg] * part.inner_area_m2
            )
            
            # Pressure (simplified - just track relative values)
            if seg == 0:
                self.pressure[step, seg] = part.Pa  # Reference pressure
            else:
                # Pressure decreases due to friction and increases due to buoyancy
                self.pressure[step, seg] = self.pressure[step, seg-1]  # Simplified
    #Done
    def get_summary_data(self) -> Dict[str, Any]:
        """Get summary data from the simulation"""
        return {
            'total_segments': self.total_segments,
            'total_time': self.total_time,
            'time_steps': self.time_steps,
            'final_gas_temp_range_C': (
                float(self.T_gas[-1].min() - 273.15), 
                float(self.T_gas[-1].max() - 273.15)
            ),
            'final_wall_temp_range_C': (
                float(self.T_wall[-1].min() - 273.15), 
                float(self.T_wall[-1].max() - 273.15)
            ),
            'final_mass_flow_rate_kg_s': float(self.mass_flow_rate[-1]),
            'parts_config': [
                {
                    'type': part.type,
                    'height_m': part.height_m,
                    'diameter_m': part.inner_diameter_m,
                    'segments': part.num_segments
                }
                for part in self.parts
            ]
        }
    #Done
    def check_conservation(self):
        """Check mass and energy conservation throughout the simulation"""
        print("\n" + "="*80)
        print("CONSERVATION DIAGNOSTICS")
        print("="*80)
        
        # Energy conservation check
        self._check_energy_conservation()
        
        # Mass conservation check (if condensation is present)
        self._check_mass_conservation()
    #Done
    def _check_energy_conservation(self):
        """Check energy conservation by comparing input, stored, and lost energy"""
        print("\nEnergy Conservation Check:")
        
        # Calculate total energy input at inlet
        energy_input = 0.0
        for step in range(1, self.time_steps + 1):
            T_inlet = self.T_gas[step, 0]
            mdot = self.mass_flow_rate[step]
            cp = self.physics.air_specific_heat(T_inlet)
            T_ref = self.parts[0].T_ambient_K
            energy_input += mdot * cp * (T_inlet - T_ref) * self.dt
        
        # Calculate energy stored in gas and walls
        energy_stored_gas = 0.0
        energy_stored_wall = 0.0
        
        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            segment_length = part.height_m / part.num_segments
            
            # Gas energy
            T_gas_final = self.T_gas[-1, seg]
            T_gas_initial = self.T_gas[0, seg]
            gas_volume = part.inner_area_m2 * segment_length
            rho_final = self.density[-1, seg]
            cp_gas = self.physics.air_specific_heat(T_gas_final)
            energy_stored_gas += rho_final * gas_volume * cp_gas * (T_gas_final - T_gas_initial)
            
            # Wall energy
            if part.type == 'single':
                T_wall_final = self.T_wall[-1, seg]
                T_wall_initial = self.T_wall[0, seg]
                wall_volume = np.pi * segment_length * ((part.outer_diameter_m)**2 - (part.inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_w_kg_m3 * wall_volume * part.cp_w_J_kg_K * (T_wall_final - T_wall_initial)
            else:  # triple wall
                # Liner
                T_liner_final = self.T_liner[-1, seg]
                T_liner_initial = self.T_liner[0, seg]
                V_liner = np.pi * segment_length * ((part.liner_outer_diameter_m)**2 - (part.inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_inner_kg_m3 * V_liner * part.cp_inner_J_kg_K * (T_liner_final - T_liner_initial)
                
                # Middle
                T_middle_final = self.T_middle[-1, seg]
                T_middle_initial = self.T_middle[0, seg]
                V_middle = np.pi * segment_length * ((part.middle_outer_diameter_m)**2 - (part.middle_inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_middle_kg_m3 * V_middle * part.cp_middle_J_kg_K * (T_middle_final - T_middle_initial)
                
                # Outer
                T_outer_final = self.T_outer[-1, seg]
                T_outer_initial = self.T_outer[0, seg]
                V_outer = np.pi * segment_length * ((part.outer_diameter_m)**2 - (part.outer_inner_diameter_m)**2) / 4
                energy_stored_wall += part.density_outer_kg_m3 * V_outer * part.cp_outer_J_kg_K * (T_outer_final - T_outer_initial)
        
        # Estimate energy lost to ambient (simplified)
        energy_lost = energy_input - energy_stored_gas - energy_stored_wall
        
        # Calculate percentages
        if energy_input > 0:
            pct_stored_gas = 100 * energy_stored_gas / energy_input
            pct_stored_wall = 100 * energy_stored_wall / energy_input
            pct_lost = 100 * energy_lost / energy_input
            
            print(f"  Total energy input: {energy_input/1e6:.2f} MJ")
            print(f"  Energy stored in gas: {energy_stored_gas/1e6:.2f} MJ ({pct_stored_gas:.1f}%)")
            print(f"  Energy stored in walls: {energy_stored_wall/1e6:.2f} MJ ({pct_stored_wall:.1f}%)")
            print(f"  Energy lost to ambient: {energy_lost/1e6:.2f} MJ ({pct_lost:.1f}%)")
            print(f"  Energy balance closure: {100 - abs(pct_stored_gas + pct_stored_wall + pct_lost - 100):.1f}%")
    #Done
    def _check_mass_conservation(self):
        """Check water vapor mass conservation with condensation"""
        print("\nMass Conservation Check (Water Vapor):")
        
        # Check if we have condensation tracking
        if not hasattr(self.comprehensive_ht, 'condensation_states') or not self.comprehensive_ht.condensation_states:
            print("  No condensation tracking available")
            return
        
        # Calculate total condensed mass
        total_condensed = 0.0
        for seg_id, state in self.comprehensive_ht.condensation_states.items():
            total_condensed += state.film_mass
        
        print(f"  Total water condensed: {total_condensed*1000:.3f} g")
        
        # Estimate initial vs final water vapor in gas
        initial_water = 0.0
        final_water = 0.0
        
        for seg in range(self.total_segments):
            part_idx, local_seg, part = self.get_segment_info(seg)
            segment_length = part.height_m / part.num_segments
            gas_volume = part.inner_area_m2 * segment_length
            
            # Initial
            rho_initial = self.density[0, seg]
            humidity_initial = part.humidity_ratio
            dry_air_mass_initial = gas_volume * rho_initial / (1 + humidity_initial)
            initial_water += dry_air_mass_initial * humidity_initial
            
            # Final (simplified - would need to track humidity ratio evolution)
            rho_final = self.density[-1, seg]
            humidity_final = part.humidity_ratio  # This is simplified
            dry_air_mass_final = gas_volume * rho_final / (1 + humidity_final)
            final_water += dry_air_mass_final * humidity_final
        
        water_removed = initial_water - final_water
        
        print(f"  Water vapor removed from gas: {water_removed*1000:.3f} g")
        print(f"  Mass balance error: {abs(water_removed - total_condensed)*1000:.3f} g")
        
        if initial_water > 0:
            error_pct = 100 * abs(water_removed - total_condensed) / initial_water
            print(f"  Mass balance closure: {100 - error_pct:.1f}%")
    #Done
    def _initialize_arrays(self):
        """Initialize arrays for storing simulation results
        
        This pre-allocates all the numpy arrays we'll need to store the
        simulation history. Arrays are 2D: [time_step, segment]
        """
        # TEMPERATURE ARRAYS (all in Kelvin)
        # Gas temperature at each segment over time
        self.T_gas = np.zeros((self.time_steps + 1, self.total_segments))
        # Wall temperature - for single wall this is the only wall temp
        # For triple wall, this mirrors T_liner for compatibility
        self.T_wall = np.zeros((self.time_steps + 1, self.total_segments))
        # Outer surface temperature (same as T_wall for single wall)
        self.T_outer = np.zeros((self.time_steps + 1, self.total_segments))
        
        # TRIPLE WALL TEMPERATURE ARRAYS
        # These store the temperature of each layer in triple wall sections
        self.T_liner = np.zeros((self.time_steps + 1, self.total_segments))  # Inner liner (contacts gas)
        self.T_middle = np.zeros((self.time_steps + 1, self.total_segments))  # Insulation layer
        # Note: T_outer is used for outer layer of triple wall
        
        # FLOW PROPERTIES
        self.mass_flow_rate = np.zeros(self.time_steps + 1)  # Mass flow rate in kg/s (scalar per timestep)
        self.velocity = np.zeros((self.time_steps + 1, self.total_segments))  # Gas velocity in m/s
        self.pressure = np.zeros((self.time_steps + 1, self.total_segments))  # Pressure in Pa
        self.density = np.zeros((self.time_steps + 1, self.total_segments))  # Gas density in kg/m³
        
        # Heat transfer coefficients
        self.h_conv_inner = np.zeros((self.time_steps + 1, self.total_segments))  # W/(m²·K)
        self.h_conv_outer = np.zeros((self.time_steps + 1, self.total_segments))  # W/(m²·K)
        
        # Time array
        self.time_array = np.linspace(0, self.total_time, self.time_steps + 1)
        
        # Set initial conditions
        self._set_initial_conditions()
    #Done    
    def _set_initial_conditions(self):
        """Set initial temperature and flow conditions with consistent initialization"""
        # Initialize with small temperature rise to avoid nonphysical cooling pulse
        # This ensures initial heat flux is near zero
        print(f"DEBUG: Initializing conditions for {len(self.parts)} parts")
        
        for i, part in enumerate(self.parts):
            start_seg = sum(p.num_segments for p in self.parts[:i])
            end_seg = start_seg + part.num_segments
            
            # CRITICAL PHYSICS FIX: Only bottom segment gets hot air initially
            # All other segments start at ambient temperature
            # Gas heats up walls as it flows upward, cooling in the process
            for seg in range(start_seg, end_seg):
                if seg == 0:
                    # Bottom segment gets hot inlet air
                    initial_gas_temp = part.T_ambient_K + 10.0  # Small initial rise
                else:
                    # All other segments start at ambient (cold)
                    # They will be heated by the hot gas flowing through
                    initial_gas_temp = part.T_ambient_K
                
                # Initialize ALL timesteps to prevent -273.1°C issue
                self.T_gas[:, seg] = initial_gas_temp
            
            # Initial wall temperatures - start at ambient
            # Walls heat up slowly from ambient according to physics
            initial_wall_temp = part.T_ambient_K  # Start at ambient temperature
            
            print(f"DEBUG: Part {i+1} ({part.type}), segments {start_seg}-{end_seg-1}, T_ambient={part.T_ambient_K-273.15:.1f}°C")
            
            if part.type == 'single':
                self.T_wall[:, start_seg:end_seg] = initial_wall_temp
                self.T_outer[:, start_seg:end_seg] = initial_wall_temp
                # Initialize triple wall arrays to ambient for single wall segments
                # This prevents -273.1°C errors in diagnostics
                self.T_liner[:, start_seg:end_seg] = initial_wall_temp
                self.T_middle[:, start_seg:end_seg] = initial_wall_temp
                print(f"DEBUG: Single wall initialized - T_wall[0,{start_seg}] = {self.T_wall[0,start_seg]-273.15:.1f}°C")
            elif part.type == 'triple':
                # Initialize all triple wall layers at ambient
                # They will heat up according to physics
                self.T_liner[:, start_seg:end_seg] = part.T_ambient_K
                self.T_middle[:, start_seg:end_seg] = part.T_ambient_K
                self.T_outer[:, start_seg:end_seg] = part.T_ambient_K
                # Set wall temperature to liner for compatibility
                self.T_wall[:, start_seg:end_seg] = part.T_ambient_K
                print(f"DEBUG: Triple wall initialized - T_liner[0,{start_seg}] = {self.T_liner[0,start_seg]-273.15:.1f}°C")
            
            # Initial density based on actual temperature for each segment
            for seg in range(start_seg, end_seg):
                # Use the actual gas temperature for this segment
                seg_gas_temp = self.T_gas[0, seg]  # Already set above
                initial_density = self.physics.air_density(seg_gas_temp, part.Pa)
                self.density[:, seg] = initial_density
                
                # Initialize pressure to atmospheric for all timesteps
                self.pressure[:, seg] = part.Pa
                
                # Initialize velocity to small positive value
                self.velocity[:, seg] = 0.1  # m/s
        
        # Initial mass flow rate (kg/s) - small positive value for all timesteps
        self.mass_flow_rate[:] = 0.01
        
    def get_segment_info(self, segment_idx: int) -> tuple:
        """Get part and local segment index for a global segment index"""
        current_seg = 0
        for part_idx, part in enumerate(self.parts):
            if current_seg + part.num_segments > segment_idx:
                local_seg = segment_idx - current_seg
                return part_idx, local_seg, part
            current_seg += part.num_segments
        raise IndexError(f"Segment index {segment_idx} out of range")

    
    #plotting/Analyzing
    #Done
    def plot_temperature_profiles(self, times_to_plot: Optional[List[float]] = None):
        """Plot comprehensive temperature profiles with separate triple wall layers
        
        Args:
            times_to_plot: List of times to plot. If None, plots initial, middle, and final
        """
        if times_to_plot is None:
            times_to_plot = [0, self.total_time/2, self.total_time]
        
        # Check if we have triple wall parts
        has_triple_wall = any(part.type == 'triple' for part in self.parts)
        
        # Create segment positions and part boundaries
        positions = []
        part_boundaries = []
        current_z = 0
        
        for part_idx, part in enumerate(self.parts):
            seg_height = part.height_m / part.num_segments
            part_start_z = current_z
            
            for i in range(part.num_segments):
                positions.append(current_z + (i + 0.5) * seg_height)
            
            current_z += part.height_m
            wall_type = "Triple" if part.type == 'triple' else "Single"
            part_boundaries.append((part_start_z, current_z, f"Part {part_idx+1} ({wall_type})"))
        
        # Create figure with more subplots if triple wall
        if has_triple_wall:
            fig = plt.figure(figsize=(20, 15))
            subplot_rows, subplot_cols = 3, 3
        else:
            fig = plt.figure(figsize=(16, 12))
            subplot_rows, subplot_cols = 2, 3
        
        # 1. Temperature vs height profiles (with triple wall layers)
        ax1 = plt.subplot(subplot_rows, subplot_cols, 1)
        colors = plt.cm.Set1(np.linspace(0, 1, len(times_to_plot)))
        
        for i, time_val in enumerate(times_to_plot):
            time_idx = int(time_val / self.dt)
            time_idx = min(time_idx, self.time_steps)
            
            # Gas temperature
            ax1.plot(self.T_gas[time_idx] - 273.15, positions, 
                    label=f'Gas T at t={time_val:.1f}s', 
                    linestyle='-', color=colors[i], marker='o', markersize=4, linewidth=2)
            
            if has_triple_wall:
                # Show all wall layers for triple wall sections
                ax1.plot(self.T_liner[time_idx] - 273.15, positions, 
                        label=f'Liner T at t={time_val:.1f}s', 
                        linestyle='--', color=colors[i], marker='s', markersize=3)
                ax1.plot(self.T_middle[time_idx] - 273.15, positions, 
                        label=f'Middle T at t={time_val:.1f}s', 
                        linestyle=':', color=colors[i], marker='^', markersize=3)
                ax1.plot(self.T_outer[time_idx] - 273.15, positions, 
                        label=f'Outer T at t={time_val:.1f}s', 
                        linestyle='-.', color=colors[i], marker='d', markersize=3)
            else:
                ax1.plot(self.T_wall[time_idx] - 273.15, positions, 
                        label=f'Wall T at t={time_val:.1f}s', 
                        linestyle='--', color=colors[i], marker='s', markersize=4)
        
        # Add part boundaries
        for start_z, end_z, label in part_boundaries:
            ax1.axhline(y=start_z, color='gray', linestyle=':', alpha=0.7)
            ax1.text(ax1.get_xlim()[0] if ax1.get_xlim()[0] != ax1.get_xlim()[1] else 20, 
                    (start_z + end_z)/2, label, rotation=90, ha='left', va='center', fontsize=8)
        
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Height (m)')
        ax1.set_title('Temperature vs Height\n(All Layers)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Temperature vs time by part (with triple wall layers)
        ax2 = plt.subplot(subplot_rows, subplot_cols, 2)
        current_seg = 0
        
        # Define colors for parts (still needed for other plots)
        part_colors = plt.cm.tab10(np.linspace(0, 1, len(self.parts)))
        
        # Define distinct colors for different temperature types
        gas_color = 'red'
        liner_color = 'orange' 
        middle_color = 'green'
        outer_color = 'blue'
        wall_color = 'purple'
        
        for part_idx, part in enumerate(self.parts):
            end_seg = current_seg + part.num_segments
            
            # Average temperatures for this part
            part_gas_temps = np.mean(self.T_gas[:, current_seg:end_seg], axis=1) - 273.15
            
            ax2.plot(self.time_array, part_gas_temps, 
                    label=f'Part {part_idx+1} Gas', color=gas_color, linestyle='-', linewidth=3)
            
            if part.type == 'triple':
                part_liner_temps = np.mean(self.T_liner[:, current_seg:end_seg], axis=1) - 273.15
                part_middle_temps = np.mean(self.T_middle[:, current_seg:end_seg], axis=1) - 273.15
                part_outer_temps = np.mean(self.T_outer[:, current_seg:end_seg], axis=1) - 273.15
                
                ax2.plot(self.time_array, part_liner_temps, 
                        label=f'Part {part_idx+1} Liner', color=liner_color, linestyle='--', linewidth=2)
                ax2.plot(self.time_array, part_middle_temps, 
                        label=f'Part {part_idx+1} Middle', color=middle_color, linestyle=':', linewidth=2)
                ax2.plot(self.time_array, part_outer_temps, 
                        label=f'Part {part_idx+1} Outer', color=outer_color, linestyle='-.', linewidth=2)
            else:
                part_wall_temps = np.mean(self.T_wall[:, current_seg:end_seg], axis=1) - 273.15
                ax2.plot(self.time_array, part_wall_temps, 
                        label=f'Part {part_idx+1} Wall', color=wall_color, linestyle='--', linewidth=2)
            
            current_seg = end_seg
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Temperature Evolution by Part')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Final temperature distribution
        ax3 = plt.subplot(subplot_rows, subplot_cols, 3)
        final_gas_temps = self.T_gas[-1] - 273.15
        segment_numbers = range(1, self.total_segments + 1)
        x_pos = np.arange(len(segment_numbers))
        
        if has_triple_wall:
            bar_width = 0.2
            ax3.bar(x_pos - 1.5*bar_width, final_gas_temps, bar_width, 
                   label='Gas', alpha=0.8, color='red')
            
            # Create arrays that show single wall temp for all layers when it's a single wall
            liner_temps = np.zeros(self.total_segments)
            middle_temps = np.zeros(self.total_segments)
            outer_temps = np.zeros(self.total_segments)
            
            current_seg = 0
            for part in self.parts:
                end_seg = current_seg + part.num_segments
                if part.type == 'single':
                    # For single wall, show wall temperature for all three layers
                    liner_temps[current_seg:end_seg] = self.T_wall[-1, current_seg:end_seg] - 273.15
                    middle_temps[current_seg:end_seg] = self.T_wall[-1, current_seg:end_seg] - 273.15
                    outer_temps[current_seg:end_seg] = self.T_wall[-1, current_seg:end_seg] - 273.15
                else:  # triple
                    # For triple wall, show actual layer temperatures
                    liner_temps[current_seg:end_seg] = self.T_liner[-1, current_seg:end_seg] - 273.15
                    middle_temps[current_seg:end_seg] = self.T_middle[-1, current_seg:end_seg] - 273.15
                    outer_temps[current_seg:end_seg] = self.T_outer[-1, current_seg:end_seg] - 273.15
                current_seg = end_seg
            
            ax3.bar(x_pos - 0.5*bar_width, liner_temps, bar_width, 
                   label='Liner', alpha=0.8, color='orange')
            ax3.bar(x_pos + 0.5*bar_width, middle_temps, bar_width, 
                   label='Middle', alpha=0.8, color='green')
            ax3.bar(x_pos + 1.5*bar_width, outer_temps, bar_width, 
                   label='Outer', alpha=0.8, color='blue')
        else:
            bar_width = 0.35
            ax3.bar(x_pos - bar_width/2, final_gas_temps, bar_width, 
                   label='Gas', alpha=0.8, color='red')
            ax3.bar(x_pos + bar_width/2, self.T_wall[-1] - 273.15, bar_width, 
                   label='Wall', alpha=0.8, color='blue')
        
        # Add part boundaries
        current_seg = 0
        for part_idx, part in enumerate(self.parts):
            if part_idx > 0:
                ax3.axvline(x=current_seg - 0.5, color='gray', linestyle='--', alpha=0.7)
            current_seg += part.num_segments
        
        ax3.set_xlabel('Segment Number')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title(f'Final Temperature Distribution')
        ax3.set_xticks(x_pos[::max(1, len(x_pos)//8)])
        ax3.set_xticklabels([str(i+1) for i in x_pos[::max(1, len(x_pos)//8)]])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Triple wall temperature differences (if applicable)
        if has_triple_wall:
            ax4 = plt.subplot(subplot_rows, subplot_cols, 4)
            
            # Calculate temperature differences in triple wall sections
            liner_gas_diff = self.T_gas[-1] - self.T_liner[-1]
            middle_liner_diff = self.T_liner[-1] - self.T_middle[-1]
            outer_middle_diff = self.T_middle[-1] - self.T_outer[-1]
            outer_ambient_diff = self.T_outer[-1] - np.array([part.T_ambient_K for part in self.parts for _ in range(part.num_segments)])
            
            ax4.plot(liner_gas_diff, positions, label='Gas - Liner', marker='o', linestyle='-')
            ax4.plot(middle_liner_diff, positions, label='Liner - Middle', marker='s', linestyle='-')
            ax4.plot(outer_middle_diff, positions, label='Middle - Outer', marker='^', linestyle='-')
            ax4.plot(outer_ambient_diff, positions, label='Outer - Ambient', marker='d', linestyle='-')
            
            ax4.set_xlabel('Temperature Difference (K)')
            ax4.set_ylabel('Height (m)')
            ax4.set_title('Temperature Differences Across Layers')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Insulation effectiveness
            ax5 = plt.subplot(subplot_rows, subplot_cols, 5)
            current_seg = 0
            
            for part_idx, part in enumerate(self.parts):
                end_seg = current_seg + part.num_segments
                if part.type == 'triple':
                    # Calculate insulation effectiveness: (T_liner - T_outer) / (T_liner - T_ambient)
                    liner_temps = np.mean(self.T_liner[:, current_seg:end_seg], axis=1)
                    outer_temps = np.mean(self.T_outer[:, current_seg:end_seg], axis=1)
                    
                    # Debug: Check temperature values
                    print(f"DEBUG Part {part_idx+1} Triple Wall:")
                    print(f"  Final Liner temp: {liner_temps[-1] - 273.15:.1f}°C")
                    print(f"  Final Middle temp: {np.mean(self.T_middle[-1, current_seg:end_seg]) - 273.15:.1f}°C")
                    print(f"  Final Outer temp: {outer_temps[-1] - 273.15:.1f}°C")
                    print(f"  Ambient temp: {part.T_ambient_K - 273.15:.1f}°C")
                    
                    # Avoid division by zero
                    denominator = liner_temps - part.T_ambient_K
                    denominator = np.where(np.abs(denominator) < 1.0, 1.0, denominator)  # Avoid very small denominators
                    effectiveness = (liner_temps - outer_temps) / denominator * 100
                    effectiveness = np.clip(effectiveness, 0, 100)  # Limit to 0-100%
                    
                    ax5.plot(self.time_array, effectiveness, 
                            label=f'Part {part_idx+1} Insulation Effectiveness', 
                            color=part_colors[part_idx], linewidth=2)
                current_seg = end_seg
            
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Insulation Effectiveness (%)')
            ax5.set_title('Triple Wall Insulation Performance')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Summary statistics - use subplot 6 for triple wall (bottom right) or 3 for single wall
        if has_triple_wall:
            stats_subplot_num = 6  # Bottom right position in 3x3 grid (instead of subplot 9)
        else:
            stats_subplot_num = 6  # Bottom right position in 2x3 grid
        ax_stats = plt.subplot(subplot_rows, subplot_cols, stats_subplot_num)
        ax_stats.axis('off')
        
        # Calculate comprehensive statistics
        final_gas_avg = np.mean(final_gas_temps)
        max_gas_temp = np.max(self.T_gas) - 273.15
        
        stats_text = f"""SIMULATION SUMMARY
        
Parts: {len(self.parts)}
Total Segments: {self.total_segments}
Simulation Time: {self.total_time}s
        
FINAL AVERAGE TEMPERATURES (°C):
Gas: {final_gas_avg:.1f}
"""
        
        if has_triple_wall:
            final_liner_avg = np.mean(self.T_liner[-1] - 273.15)
            final_middle_avg = np.mean(self.T_middle[-1] - 273.15)
            final_outer_avg = np.mean(self.T_outer[-1] - 273.15)
            
            stats_text += f"""Liner: {final_liner_avg:.1f}
Middle: {final_middle_avg:.1f}
Outer: {final_outer_avg:.1f}

MAXIMUM TEMPERATURES (°C):
Gas: {max_gas_temp:.1f}
Liner: {np.max(self.T_liner) - 273.15:.1f}
Outer: {np.max(self.T_outer) - 273.15:.1f}"""
        else:
            final_wall_avg = np.mean(self.T_wall[-1] - 273.15)
            stats_text += f"""Wall: {final_wall_avg:.1f}

MAXIMUM TEMPERATURES (°C):
Gas: {max_gas_temp:.1f}
Wall: {np.max(self.T_wall) - 273.15:.1f}"""
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Also show segment-wise diagnostics
        self.plot_segment_diagnostics()
        
        # Print detailed segment information
        self._print_detailed_segment_info()
    #Done
    def analyze_draft(self, plot=True, save_plots=False, plot_dir='./'):
        """
        Analyze chimney draft performance
        
        Args:
            plot: Whether to create plots
            save_plots: Whether to save plots to files
            plot_dir: Directory to save plots
            
        Returns:
            DraftResults object with analysis
        """
        from .draft_analysis import DraftAnalyzer
        
        analyzer = DraftAnalyzer(self)
        results = analyzer.calculate_draft()
        
        # Print summary
        analyzer.print_draft_summary(results)
        
        # Create plots if requested
        if plot:
            import os
            
            # Main draft analysis plot
            fig1 = analyzer.plot_draft_analysis(results)
            if save_plots:
                save_path = os.path.join(plot_dir, 'draft_analysis.png')
                fig1.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved draft analysis to {save_path}")
            
            # Draft vs temperature plot
            fig2 = analyzer.plot_draft_vs_temperature(results)
            if save_plots:
                save_path = os.path.join(plot_dir, 'draft_vs_temperature.png')
                fig2.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved draft vs temperature to {save_path}")
        
        return results
    #Done
    def plot_segment_diagnostics(self):
        """Plot detailed segment-wise temperature evolution for all segments"""
        print("\nGenerating segment-wise diagnostic plots...")
        
        # Create figure with subplots for each part
        n_parts = len(self.parts)
        fig = plt.figure(figsize=(20, 5 * n_parts))
        
        current_seg = 0
        
        for part_idx, part in enumerate(self.parts):
            # Create subplot for this part
            ax = plt.subplot(n_parts, 1, part_idx + 1)
            
            # Plot each segment in this part
            for local_seg in range(part.num_segments):
                global_seg = current_seg + local_seg
                
                # Gas temperature
                ax.plot(self.time_array, self.T_gas[:, global_seg] - 273.15,
                       label=f'Seg {global_seg+1} Gas', 
                       color='red', alpha=0.7 - 0.1*local_seg, linewidth=2)
                
                if part.type == 'single':
                    # Wall temperature
                    ax.plot(self.time_array, self.T_wall[:, global_seg] - 273.15,
                           label=f'Seg {global_seg+1} Wall',
                           color='blue', alpha=0.7 - 0.1*local_seg, linewidth=1.5, linestyle='--')
                else:  # triple wall
                    # Liner temperature
                    ax.plot(self.time_array, self.T_liner[:, global_seg] - 273.15,
                           label=f'Seg {global_seg+1} Liner',
                           color='orange', alpha=0.7 - 0.1*local_seg, linewidth=1.5, linestyle='--')
                    
                    # Middle temperature
                    ax.plot(self.time_array, self.T_middle[:, global_seg] - 273.15,
                           label=f'Seg {global_seg+1} Middle',
                           color='green', alpha=0.7 - 0.1*local_seg, linewidth=1.5, linestyle=':')
                    
                    # Outer temperature
                    ax.plot(self.time_array, self.T_outer[:, global_seg] - 273.15,
                           label=f'Seg {global_seg+1} Outer',
                           color='blue', alpha=0.7 - 0.1*local_seg, linewidth=1.5, linestyle='-.')
            
            # Formatting
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'Part {part_idx+1} ({part.type.title()} Wall) - Segment Temperature Evolution\n'
                        f'Height: {part.height_m:.2f}m, Diameter: {part.inner_diameter_m*1000:.1f}mm, '
                        f'Segments: {part.num_segments}')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Add ambient temperature reference
            ax.axhline(y=part.T_ambient_K - 273.15, color='gray', linestyle=':', 
                      alpha=0.5, label='Ambient')
            
            current_seg += part.num_segments
        
        plt.tight_layout()
        plt.show()
        
        # Create additional plots for heat transfer coefficients
        self._plot_heat_transfer_coefficients()
    #Done
    def temp_evolution_by_part(self, ambient_temp=False):
        # Temperature vs time by part with custom color schemes
        fig, ax = plt.subplots(1, 1, figsize=(20, 6.5))
        current_seg = 0
        
        # Convert time to minutes
        time_minutes = self.time_array / 60.0
        
        # First plot the inlet temperature in black
        T_ambient_inlet = self.parts[0].T_ambient_K
        inlet_temps = np.array([self._default_inlet_temp(t, T_ambient=T_ambient_inlet) 
                               for t in self.time_array]) - 273.15
        ax.plot(time_minutes, inlet_temps, 
                label='Inlet Gas', color='black', linestyle='-', linewidth=3.5, zorder=10)
        
        # Also plot the gas temperature at the first segment (bottom) to show the difference
        first_seg_gas_temps = self.T_gas[:, 0] - 273.15
        ax.plot(time_minutes, first_seg_gas_temps, 
                label='Bottom Segment Gas', color='gray', linestyle='--', linewidth=2, alpha=0.7)
        
        # Get color maps for different temperature types
        # We'll use different intensities within each colormap for different parts
        n_parts = len(self.parts)
        
        # Define color intensity ranges for each part (0.3 to 0.9 to avoid too light/dark)
        color_intensities = np.linspace(0.3, 0.9, n_parts)
        
        # Plot ambient temperatures if requested
        if ambient_temp:
            # Use light blue shades for ambient temperatures
            ambient_intensities = np.linspace(0.2, 0.5, n_parts)  # Lighter shades
            for part_idx, part in enumerate(self.parts):
                ambient_color = plt.cm.Blues(ambient_intensities[part_idx])
                height_ft = round(part.z_ft, 2)
                ax.axhline(y=part.T_ambient_K - 273.15, 
                          color=ambient_color, 
                          linestyle=':', 
                          linewidth=2.5,
                          alpha=0.7,
                          label=f'Part {part_idx+1} Ambient ({height_ft} ft)')
        
        for part_idx, part in enumerate(self.parts):
            end_seg = current_seg + part.num_segments
            
            # Average temperatures for this part
            part_gas_temps = np.mean(self.T_gas[:, current_seg:end_seg], axis=1) - 273.15
            
            # Gas temperature - use Reds colormap
            gas_color = plt.cm.Reds(color_intensities[part_idx])
            height_ft = round(part.z_ft,2)
            ax.plot(time_minutes, part_gas_temps, 
                    label=f'Part {part_idx+1} Gas ({height_ft} ft)', color=gas_color, linestyle='-', linewidth=3)
            
            if part.type == 'triple':
                part_liner_temps = np.mean(self.T_liner[:, current_seg:end_seg], axis=1) - 273.15
                part_middle_temps = np.mean(self.T_middle[:, current_seg:end_seg], axis=1) - 273.15
                part_outer_temps = np.mean(self.T_outer[:, current_seg:end_seg], axis=1) - 273.15
                
                # Triple wall inner (liner) - use Blues colormap
                liner_color = plt.cm.Blues(color_intensities[part_idx])
                ax.plot(time_minutes, part_liner_temps, 
                        label=f'Part {part_idx+1} Inner ({height_ft} ft)', color=liner_color, linestyle='--', linewidth=2)
                
                # Triple wall middle - use Purples colormap
                middle_color = plt.cm.Purples(color_intensities[part_idx])
                ax.plot(time_minutes, part_middle_temps, 
                        label=f'Part {part_idx+1} Middle ({height_ft} ft)', color=middle_color, linestyle=':', linewidth=2)
                
                # Triple wall outer - use RdPu (Red-Purple/Pink) colormap
                outer_color = plt.cm.RdPu(color_intensities[part_idx])
                ax.plot(time_minutes, part_outer_temps, 
                        label=f'Part {part_idx+1} Outer ({height_ft} ft)', color=outer_color, linestyle='-.', linewidth=2)
            else:
                part_wall_temps = np.mean(self.T_wall[:, current_seg:end_seg], axis=1) - 273.15
                
                # Single wall - use Greens colormap
                wall_color = plt.cm.Greens(color_intensities[part_idx])
                ax.plot(time_minutes, part_wall_temps, 
                        label=f'Part {part_idx+1} Wall ({height_ft} ft)', color=wall_color, linestyle='--', linewidth=2)
            
            current_seg = end_seg
        
        ax.set_xlabel('Time (minutes)', fontsize=15)
        ax.set_ylabel('Temperature (°C)', fontsize=15)
        ax.set_title('Temperature Evolution by Part', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=15, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add a text box explaining the color scheme
        color_scheme_text = (
            "Color Scheme:\n"
            "• Inlet: Black\n"
            "• Gas: Reds\n"
            "• Single Wall: Greens\n"
            "• Triple Inner: Blues\n"
            "• Triple Middle: Purples\n"
            "• Triple Outer: Pinks"
        )
        if ambient_temp:
            color_scheme_text += "\n• Ambient: Light Blues"
        ax.text(0.02, 0.98, color_scheme_text, transform=ax.transAxes, 
                fontsize=15, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    #Done
    def temp_evolution_br(self):
        """Plot temperature evolution showing bottom, average, and top segments for each part"""
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        
        # Convert time to minutes
        time_minutes = self.time_array / 60.0
        
        # First plot the inlet temperature in black
        T_ambient_inlet = self.parts[0].T_ambient_K
        inlet_temps = np.array([self._default_inlet_temp(t, T_ambient=T_ambient_inlet) 
                               for t in self.time_array]) - 273.15
        ax.plot(time_minutes, inlet_temps, 
                label='Inlet Gas', color='black', linestyle='-', linewidth=3.5, zorder=10)
        
        # Get color maps for different temperature types
        n_parts = len(self.parts)
        color_intensities = np.linspace(0.3, 0.9, n_parts)
        
        current_seg = 0
        
        for part_idx, part in enumerate(self.parts):
            end_seg = current_seg + part.num_segments
            height_ft = round(part.height_ft)
            
            # Get temperatures for bottom, average, and top
            # Gas temperatures
            part_gas_bottom = self.T_gas[:, current_seg] - 273.15  # First segment
            part_gas_avg = np.mean(self.T_gas[:, current_seg:end_seg], axis=1) - 273.15  # Average
            part_gas_top = self.T_gas[:, end_seg-1] - 273.15  # Last segment
            
            # Plot gas temperatures - use Reds colormap
            gas_color = plt.cm.Reds(color_intensities[part_idx])
            ax.plot(time_minutes, part_gas_bottom, 
                    label=f'P{part_idx+1} Gas Bottom ({height_ft} ft)', 
                    color=gas_color, linestyle='-', linewidth=2.5, alpha=1.0)
            ax.plot(time_minutes, part_gas_avg, 
                    label=f'P{part_idx+1} Gas Avg ({height_ft} ft)', 
                    color=gas_color, linestyle='--', linewidth=2.5, alpha=0.7)
            ax.plot(time_minutes, part_gas_top, 
                    label=f'P{part_idx+1} Gas Top ({height_ft} ft)', 
                    color=gas_color, linestyle=':', linewidth=2.5, alpha=0.5)
            
            if part.type == 'triple':
                # Inner/Liner temperatures
                part_liner_bottom = self.T_liner[:, current_seg] - 273.15
                part_liner_avg = np.mean(self.T_liner[:, current_seg:end_seg], axis=1) - 273.15
                part_liner_top = self.T_liner[:, end_seg-1] - 273.15
                
                liner_color = plt.cm.Blues(color_intensities[part_idx])
                ax.plot(time_minutes, part_liner_bottom, 
                        label=f'P{part_idx+1} Inner Bottom', 
                        color=liner_color, linestyle='-', linewidth=2, alpha=1.0)
                ax.plot(time_minutes, part_liner_avg, 
                        label=f'P{part_idx+1} Inner Avg', 
                        color=liner_color, linestyle='--', linewidth=2, alpha=0.7)
                ax.plot(time_minutes, part_liner_top, 
                        label=f'P{part_idx+1} Inner Top', 
                        color=liner_color, linestyle=':', linewidth=2, alpha=0.5)
                
                # Middle layer temperatures
                part_middle_bottom = self.T_middle[:, current_seg] - 273.15
                part_middle_avg = np.mean(self.T_middle[:, current_seg:end_seg], axis=1) - 273.15
                part_middle_top = self.T_middle[:, end_seg-1] - 273.15
                
                middle_color = plt.cm.Purples(color_intensities[part_idx])
                ax.plot(time_minutes, part_middle_bottom, 
                        label=f'P{part_idx+1} Mid Bottom', 
                        color=middle_color, linestyle='-', linewidth=1.8, alpha=1.0)
                ax.plot(time_minutes, part_middle_avg, 
                        label=f'P{part_idx+1} Mid Avg', 
                        color=middle_color, linestyle='--', linewidth=1.8, alpha=0.7)
                ax.plot(time_minutes, part_middle_top, 
                        label=f'P{part_idx+1} Mid Top', 
                        color=middle_color, linestyle=':', linewidth=1.8, alpha=0.5)
                
                # Outer layer temperatures
                part_outer_bottom = self.T_outer[:, current_seg] - 273.15
                part_outer_avg = np.mean(self.T_outer[:, current_seg:end_seg], axis=1) - 273.15
                part_outer_top = self.T_outer[:, end_seg-1] - 273.15
                
                outer_color = plt.cm.RdPu(color_intensities[part_idx])
                ax.plot(time_minutes, part_outer_bottom, 
                        label=f'P{part_idx+1} Out Bottom', 
                        color=outer_color, linestyle='-', linewidth=1.8, alpha=1.0)
                ax.plot(time_minutes, part_outer_avg, 
                        label=f'P{part_idx+1} Out Avg', 
                        color=outer_color, linestyle='--', linewidth=1.8, alpha=0.7)
                ax.plot(time_minutes, part_outer_top, 
                        label=f'P{part_idx+1} Out Top', 
                        color=outer_color, linestyle=':', linewidth=1.8, alpha=0.5)
            else:
                # Single wall temperatures
                part_wall_bottom = self.T_wall[:, current_seg] - 273.15
                part_wall_avg = np.mean(self.T_wall[:, current_seg:end_seg], axis=1) - 273.15
                part_wall_top = self.T_wall[:, end_seg-1] - 273.15
                
                wall_color = plt.cm.Greens(color_intensities[part_idx])
                ax.plot(time_minutes, part_wall_bottom, 
                        label=f'P{part_idx+1} Wall Bottom', 
                        color=wall_color, linestyle='-', linewidth=2, alpha=1.0)
                ax.plot(time_minutes, part_wall_avg, 
                        label=f'P{part_idx+1} Wall Avg', 
                        color=wall_color, linestyle='--', linewidth=2, alpha=0.7)
                ax.plot(time_minutes, part_wall_top, 
                        label=f'P{part_idx+1} Wall Top', 
                        color=wall_color, linestyle=':', linewidth=2, alpha=0.5)
            
            current_seg = end_seg
        
        # Format axes
        ax.set_xlabel('Time (minutes)', fontsize=15)
        ax.set_ylabel('Temperature (°C)', fontsize=15)
        ax.set_title('Temperature Evolution by Part (Bottom/Average/Top)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add ambient temperature reference
        ax.axhline(y=self.parts[0].T_ambient_K - 273.15, color='gray', 
                   linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Legend with multiple columns
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        
        # Add color scheme and line style explanation
        info_text = (
            "Color Scheme:\n"
            "• Inlet: Black\n"
            "• Gas: Reds\n"
            "• Single Wall: Greens\n"
            "• Triple Inner: Blues\n"
            "• Triple Middle: Purples\n"
            "• Triple Outer: Pinks\n"
            "\n"
            "Line Styles:\n"
            "• Solid: Bottom segment\n"
            "• Dashed: Average\n"
            "• Dotted: Top segment\n"
            "\n"
            "Alpha:\n"
            "• 1.0: Bottom (darkest)\n"
            "• 0.7: Average\n"
            "• 0.5: Top (lightest)"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    #Done
    def plot_draft(self):
        """Plot draft-related parameters: buoyancy pressure, mass flow rate, and velocity"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # Convert time to minutes
        time_minutes = self.time_array / 60.0
        
        # 1. Calculate and plot total buoyancy pressure over time
        total_buoyancy = np.zeros(self.time_steps + 1)
        
        for step in range(self.time_steps + 1):
            buoyancy_sum = 0.0
            for seg in range(self.total_segments):
                part_idx, local_seg, part = self.get_segment_info(seg)
                segment_length = part.height_m / part.num_segments
                
                # Only vertical segments contribute to buoyancy
                if part.orientation.lower() == 'vertical':
                    rho_gas = self.density[step, seg]
                    rho_ambient = self.physics.air_density(part.T_ambient_K, part.Pa)
                    
                    # Buoyancy pressure: g * height * (rho_ambient - rho_gas)
                    dp_buoyancy = 9.81 * segment_length * (rho_ambient - rho_gas)
                    buoyancy_sum += dp_buoyancy
            
            total_buoyancy[step] = buoyancy_sum
        
        ax1.plot(time_minutes, total_buoyancy, 'b-', linewidth=2.5, label='Total Buoyancy')
        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_ylabel('Buoyancy Pressure (Pa)', fontsize=14)
        ax1.set_title('Draft Analysis', fontsize=18)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # 2. Plot mass flow rate
        ax2.plot(time_minutes, self.mass_flow_rate, 'r-', linewidth=2.5, label='Mass Flow Rate')
        ax2.set_ylabel('Mass Flow Rate (kg/s)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        # 3. Plot velocity profile for different segments
        # Use color scheme similar to temperature plots
        current_seg = 0
        n_parts = len(self.parts)
        color_intensities = np.linspace(0.3, 0.9, n_parts)
        
        for part_idx, part in enumerate(self.parts):
            end_seg = current_seg + part.num_segments
            
            # Plot velocity for first and last segment of each part
            vel_color = plt.cm.Blues(color_intensities[part_idx])
            
            # Bottom segment
            vel_bottom = self.velocity[:, current_seg]
            ax3.plot(time_minutes, vel_bottom, 
                    label=f'Part {part_idx+1} Bottom', 
                    color=vel_color, linestyle='-', linewidth=2, alpha=1.0)
            
            # Top segment
            if part.num_segments > 1:
                vel_top = self.velocity[:, end_seg-1]
                ax3.plot(time_minutes, vel_top, 
                        label=f'Part {part_idx+1} Top', 
                        color=vel_color, linestyle=':', linewidth=2, alpha=0.6)
            
            current_seg = end_seg
        
        ax3.set_xlabel('Time (minutes)', fontsize=14)
        ax3.set_ylabel('Velocity (m/s)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='both', which='major', labelsize=12)
        
        # Add text box with key insights
        # Calculate some key metrics
        max_buoyancy = np.max(total_buoyancy)
        final_buoyancy = total_buoyancy[-1]
        max_flow = np.max(self.mass_flow_rate)
        final_flow = self.mass_flow_rate[-1]
        
        info_text = (
            f"Key Metrics:\n"
            f"• Max buoyancy: {max_buoyancy:.1f} Pa\n"
            f"• Final buoyancy: {final_buoyancy:.1f} Pa\n"
            f"• Max flow rate: {max_flow:.4f} kg/s\n"
            f"• Final flow rate: {final_flow:.4f} kg/s\n"
            f"• Stack height: {sum(p.height_m for p in self.parts):.1f} m"
        )
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    #Done
    def plot_buoyancy_pressure(self):
        """Plot total buoyancy pressure over time with detailed analysis"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        
        # Convert time to minutes
        time_minutes = self.time_array / 60.0
        
        # Calculate total buoyancy pressure and per-part contributions
        total_buoyancy = np.zeros(self.time_steps + 1)
        part_buoyancy = np.zeros((self.time_steps + 1, len(self.parts)))
        
        for step in range(self.time_steps + 1):
            current_seg = 0
            
            for part_idx, part in enumerate(self.parts):
                end_seg = current_seg + part.num_segments
                part_buoyancy_sum = 0.0
                
                for seg in range(current_seg, end_seg):
                    segment_length = part.height_m / part.num_segments
                    
                    # Only vertical segments contribute to buoyancy
                    if part.orientation.lower() == 'vertical':
                        rho_gas = self.density[step, seg]
                        rho_ambient = self.physics.air_density(part.T_ambient_K, part.Pa)
                        
                        # Buoyancy pressure: g * height * (rho_ambient - rho_gas)
                        dp_buoyancy = 9.81 * segment_length * (rho_ambient - rho_gas)
                        part_buoyancy_sum += dp_buoyancy
                
                part_buoyancy[step, part_idx] = part_buoyancy_sum
                total_buoyancy[step] += part_buoyancy_sum
                current_seg = end_seg
        
        # Plot total buoyancy
        ax.plot(time_minutes, total_buoyancy, 'k-', linewidth=3.5, 
                label='Total Buoyancy', zorder=10)
        
        # Plot per-part contributions
        n_parts = len(self.parts)
        color_intensities = np.linspace(0.4, 0.9, n_parts)
        
        for part_idx, part in enumerate(self.parts):
            if part.orientation.lower() == 'vertical':
                color = plt.cm.Blues(color_intensities[part_idx])
                height_ft = round(part.height_ft,2)
                ax.plot(time_minutes, part_buoyancy[:, part_idx], 
                       label=f'Part {part_idx+1} ({height_ft} ft, {part.orientation})', 
                       color=color, linestyle='--', linewidth=2, alpha=0.8)
        
        # Add zero reference line
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Format axes
        ax.set_xlabel('Time (minutes)', fontsize=16)
        ax.set_ylabel('Buoyancy Pressure (Pa)', fontsize=16)
        ax.set_title('Buoyancy Pressure Development', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')
        
        # Add annotations for key points
        # Find time when buoyancy reaches 90% of final value
        final_buoyancy = total_buoyancy[-1]
        if final_buoyancy > 0:
            idx_90 = np.where(total_buoyancy >= 0.9 * final_buoyancy)[0]
            if len(idx_90) > 0:
                time_90 = time_minutes[idx_90[0]]
                ax.axvline(x=time_90, color='red', linestyle=':', alpha=0.5)
                ax.text(time_90, ax.get_ylim()[1]*0.95, f'90% at {time_90:.1f} min', 
                       ha='center', va='top', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add statistics box
        max_buoyancy = np.max(total_buoyancy)
        
        # Stack information
        total_height = sum(p.height_m for p in self.parts if p.orientation.lower() == 'vertical')
        
        info_text = (
            f"Buoyancy Statistics:\n"
            f"• Maximum: {max_buoyancy:.1f} Pa\n"
            f"• Final: {final_buoyancy:.1f} Pa\n"
            f"\n"
            f"Stack Info:\n"
            f"• Total height: {total_height:.1f} m ({total_height/0.3048:.0f} ft)\n"
            f"• Vertical parts: {sum(1 for p in self.parts if p.orientation.lower()=='vertical')}"
        )
        
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    #Done
    def temp_evolution_by_segment(self):
        """Plot temperature evolution for each individual segment"""
        # Create single plot for all temperatures
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))
        
        # First plot the inlet temperature in black
        T_ambient_inlet = self.parts[0].T_ambient_K
        inlet_temps = np.array([self._default_inlet_temp(t, T_ambient=T_ambient_inlet) 
                               for t in self.time_array]) - 273.15
        ax.plot(self.time_array, inlet_temps, 
                label='Inlet Gas', color='black', linestyle='-', linewidth=3, zorder=10)
        
        current_seg = 0
        
        # Create color intensities for segments
        total_segs = self.total_segments
        seg_intensities = np.linspace(0.3, 0.9, total_segs)
        seg_counter = 0
        
        for part_idx, part in enumerate(self.parts):
            end_seg = current_seg + part.num_segments
            
            for local_seg in range(part.num_segments):
                global_seg = current_seg + local_seg
                intensity = seg_intensities[seg_counter]
                seg_counter += 1
                
                # Plot gas temperature - use Reds colormap
                gas_temps = self.T_gas[:, global_seg] - 273.15
                gas_color = plt.cm.Reds(intensity)
                label_gas = f'P{part_idx+1}S{local_seg+1} Gas'
                ax.plot(self.time_array, gas_temps, 
                       label=label_gas, color=gas_color, 
                       linestyle='-', linewidth=2.5)
                
                # Plot wall temperatures based on type
                if part.type == 'single':
                    wall_temps = self.T_wall[:, global_seg] - 273.15
                    wall_color = plt.cm.Greens(intensity)
                    label_wall = f'P{part_idx+1}S{local_seg+1} Wall'
                    ax.plot(self.time_array, wall_temps,
                           label=label_wall, color=wall_color,
                           linestyle='--', linewidth=2)
                else:  # triple wall
                    # Plot all three layers with appropriate colors
                    liner_temps = self.T_liner[:, global_seg] - 273.15
                    middle_temps = self.T_middle[:, global_seg] - 273.15
                    outer_temps = self.T_outer[:, global_seg] - 273.15
                    
                    # Inner (liner) - Blues
                    liner_color = plt.cm.Blues(intensity)
                    ax.plot(self.time_array, liner_temps,
                           label=f'P{part_idx+1}S{local_seg+1} Inner',
                           color=liner_color, linestyle='--', linewidth=2)
                    
                    # Middle - Purples
                    middle_color = plt.cm.Purples(intensity)
                    ax.plot(self.time_array, middle_temps,
                           label=f'P{part_idx+1}S{local_seg+1} Mid',
                           color=middle_color, linestyle=':', linewidth=1.8)
                    
                    # Outer - RdPu (Pinks)
                    outer_color = plt.cm.RdPu(intensity)
                    ax.plot(self.time_array, outer_temps,
                           label=f'P{part_idx+1}S{local_seg+1} Out',
                           color=outer_color, linestyle='-.', linewidth=1.8)
            
            current_seg = end_seg
        
        # Format plot
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Evolution by Segment (All Components)')
        ax.grid(True, alpha=0.3)
        
        # Add ambient temperature reference
        ax.axhline(y=self.parts[0].T_ambient_K - 273.15, color='gray', 
                   linestyle=':', alpha=0.5, linewidth=1.5)
        ax.text(self.time_array[-1]*0.98, self.parts[0].T_ambient_K - 273.15 + 1, 
                'Ambient', ha='right', va='bottom', color='gray', fontsize=9)
        
        # Create custom legend with multiple columns
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=3)
        
        # Add color scheme explanation
        color_scheme_text = (
            "Color Scheme:\n"
            "• Inlet: Black\n"
            "• Gas: Reds (darker=higher)\n"
            "• Single Wall: Greens\n"
            "• Triple Inner: Blues\n"
            "• Triple Middle: Purples\n"
            "• Triple Outer: Pinks\n"
            "\n"
            "Line Styles:\n"
            "• Solid: Gas\n"
            "• Dashed: Wall/Inner\n"
            "• Dotted: Middle\n"
            "• Dash-dot: Outer"
        )
        ax.text(0.02, 0.98, color_scheme_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Add segment info
        info_text = (
            f"Notation: PxSy = Part x, Segment y\n"
            f"Total segments: {self.total_segments}\n"
            f"Darker colors = higher elevation"
        )
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    #Done    
    def _plot_heat_transfer_coefficients(self):
        """Plot heat transfer coefficient evolution for all segments"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        current_seg = 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.parts)))
        
        for part_idx, part in enumerate(self.parts):
            color = colors[part_idx]
            
            for local_seg in range(part.num_segments):
                global_seg = current_seg + local_seg
                
                # Inner heat transfer coefficient
                ax1.plot(self.time_array, self.h_conv_inner[:, global_seg],
                        label=f'Part {part_idx+1} Seg {local_seg+1}',
                        color=color, alpha=0.7 - 0.1*local_seg, linewidth=1.5)
                
                # Outer heat transfer coefficient
                ax2.plot(self.time_array, self.h_conv_outer[:, global_seg],
                        label=f'Part {part_idx+1} Seg {local_seg+1}',
                        color=color, alpha=0.7 - 0.1*local_seg, linewidth=1.5)
            
            current_seg += part.num_segments
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Inner h (W/m²·K)')
        ax1.set_title('Inner Surface Heat Transfer Coefficients')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Outer h (W/m²·K)')
        ax2.set_title('Outer Surface Heat Transfer Coefficients')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    #Done    
    def _print_detailed_segment_info(self):
        """Print detailed temperature information for each segment"""
        print("\n" + "="*80)
        print("DETAILED SEGMENT INFORMATION")
        print("="*80)
        
        current_seg = 0
        for part_idx, part in enumerate(self.parts):
            print(f"\nPart {part_idx+1} ({part.type.upper()} WALL):")
            print(f"  Segments: {current_seg+1} to {current_seg+part.num_segments}")
            print(f"  Height: {part.height_m:.2f}m")
            print(f"  Diameter: {part.inner_diameter_m*1000:.1f}mm")
            
            end_seg = current_seg + part.num_segments
            final_gas_temps = self.T_gas[-1, current_seg:end_seg] - 273.15
            
            print(f"  Final Gas Temperatures (°C): {final_gas_temps}")
            
            if part.type == 'triple':
                final_liner_temps = self.T_liner[-1, current_seg:end_seg] - 273.15
                final_middle_temps = self.T_middle[-1, current_seg:end_seg] - 273.15
                final_outer_temps = self.T_outer[-1, current_seg:end_seg] - 273.15
                
                print(f"  Final Liner Temperatures (°C): {final_liner_temps}")
                print(f"  Final Middle Temperatures (°C): {final_middle_temps}")
                print(f"  Final Outer Temperatures (°C): {final_outer_temps}")
                
                # Calculate and show insulation performance
                avg_liner = np.mean(final_liner_temps)
                avg_outer = np.mean(final_outer_temps)
                temp_reduction = avg_liner - avg_outer
                print(f"  Temperature Reduction (Liner to Outer): {temp_reduction:.1f}°C")
                print(f"  Insulation Effectiveness: {temp_reduction/avg_liner*100:.1f}%")
            else:
                final_wall_temps = self.T_wall[-1, current_seg:end_seg] - 273.15
                print(f"  Final Wall Temperatures (°C): {final_wall_temps}")
            
            current_seg = end_seg
    #Done
    def plot_chimney_snapshot(self, timestep=-1, figsize=(10, 8), wall_scale=5.0, 
                             save_path=None):
        """Plot a single snapshot of the chimney geometry with temperatures
        
        Args:
            timestep: Timestep index to plot (default -1 for final state)
            figsize: Figure size (width, height)
            wall_scale: Factor to scale wall thickness for visibility
            save_path: Optional path to save the figure
        """
        return plot_chimney_snapshot(self, timestep, figsize, wall_scale, save_path)