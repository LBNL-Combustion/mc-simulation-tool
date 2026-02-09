"""
Coupled flow-thermal solver with iterative convergence
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .flow.elbows import base_K
from .mass_flow import calculate_leakage_mass_flow, calculate_accel_pressure_loss
from .diagnostics_recorder import DiagnosticsRecorder

@dataclass
class MomentumConfig:
    """Master configuration for momentum/pressure physics features

    When distributed_draft_model is True, all sub-features auto-enable.
    Individual features can be toggled independently when master is False.
    """
    distributed_draft_model: bool = False  # Master switch - enables all features
    use_distributed_leakage: bool = False  # Leakage through pipe joints
    use_accel_loss: bool = False           # Acceleration/deceleration pressure loss
    friction_method: str = "current"       # "current" or "colebrook"
    N_leak_iter: int = 3                   # Iterations for leakage convergence
    leak_relaxation: float = 0.5           # Relaxation factor for leakage updates
    C_accel: float = 1.0                   # Acceleration loss coefficient

    # --- Flow model ---
    # "quasi_steady" (default): algebraic mdot_ss with first-order lag (current)
    # "unsteady_momentum": ODE-based L_eff/A * d(mdot)/dt = dp_avail - dp_loss
    flow_model: str = "quasi_steady"

    # --- Gas energy model ---
    # "current" (default): existing relaxation-based gas temperature update
    # "plugflow_ntu": NTU/effectiveness model per segment
    #   T_out = T_in - effectiveness * (T_in - T_wall)
    #   effectiveness = 1 - exp(-UA / (mdot * cp))
    #   Ensures energy conservation: Q_gas = Q_wall per segment per timestep.
    gas_energy_model: str = "current"


@dataclass
class ConvergenceMetrics:
    """Metrics for tracking convergence"""
    temperature_residual: float = 0.0
    flow_residual: float = 0.0
    pressure_residual: float = 0.0
    iterations: int = 0
    converged: bool = False

class CoupledFlowThermalSolver:
    """
    Iteratively coupled solver for flow and thermal equations
    
    WHY THIS EXISTS:
    Temperature affects density -> affects buoyancy -> affects flow rate
    Flow rate affects convection -> affects heat transfer -> affects temperature
    
    This circular dependency requires iteration to find a consistent solution
    where flow and temperature are in balance.
    """
    
    def __init__(self, tolerance: float = 1e-3, max_iterations: int = 5,
                 relaxation_factor: float = 0.3,
                 momentum_config: Optional[MomentumConfig] = None,
                 diagnostics: Optional[DiagnosticsRecorder] = None):
        """
        Initialize coupled solver

        Args:
            tolerance: Convergence tolerance for residuals (relative change < tolerance)
            max_iterations: Maximum coupling iterations per timestep (more = accurate but slow)
            relaxation_factor: Under-relaxation factor for stability (0-1)
                              Lower values = more stable but slower convergence
                              Higher values = faster but may oscillate
            momentum_config: Configuration for momentum physics features (leakage, accel loss)
            diagnostics: Optional DiagnosticsRecorder for budget tracking (default off)
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.relaxation_factor = relaxation_factor
        self.logger = logging.getLogger(__name__)

        # Diagnostics recorder (off by default)
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticsRecorder(enable=False)

        # Momentum physics configuration
        if momentum_config is None:
            self.momentum_config = MomentumConfig()
        else:
            self.momentum_config = momentum_config

        # Apply master switch
        if self.momentum_config.distributed_draft_model:
            self.momentum_config.use_distributed_leakage = True
            self.momentum_config.use_accel_loss = True

        # Adaptive relaxation parameters
        self.min_relaxation = 0.2
        self.max_relaxation = 0.5
        
    def calculate_residuals(self, current: Dict, previous: Dict) -> Tuple[float, float, float]:
        """
        Calculate normalized residuals between iterations
        
        Residual = relative change between iterations
        Small residual means solution is converging
        
        Args:
            current: Current iteration values
            previous: Previous iteration values
            
        Returns:
            Tuple of (temperature_residual, flow_residual, pressure_residual)
        """
        # TEMPERATURE RESIDUAL
        # How much did temperatures change? (vector norm for all segments)
        T_current = current['temperatures']
        T_previous = previous['temperatures']
        T_residual = np.linalg.norm(T_current - T_previous) / (np.linalg.norm(T_current) + 1e-10)
        
        # FLOW RESIDUAL  
        # How much did mass flow rate change? (scalar)
        mdot_current = current['mass_flow']
        mdot_previous = previous['mass_flow']
        flow_residual = abs(mdot_current - mdot_previous) / (abs(mdot_current) + 1e-10)
        
        # PRESSURE RESIDUAL
        # How much did pressure distribution change? (vector norm)
        P_current = current['pressures']
        P_previous = previous['pressures']
        P_residual = np.linalg.norm(P_current - P_previous) / (np.linalg.norm(P_current) + 1e-10)
        
        return T_residual, flow_residual, P_residual
    
    def apply_relaxation(self, new_value: np.ndarray, old_value: np.ndarray) -> np.ndarray:
        """
        Apply under-relaxation for numerical stability
        
        Instead of using the full new value, we blend it with the old value.
        This prevents oscillations but slows convergence.
        
        relaxed = α * new + (1-α) * old
        where α is the relaxation factor (0 < α < 1)
        
        Args:
            new_value: Newly calculated value
            old_value: Previous iteration value
            
        Returns:
            Relaxed value
        """
        return self.relaxation_factor * new_value + (1 - self.relaxation_factor) * old_value
    
    def solve_coupled_timestep(self, chimney, step: int, T_inlet: float, dt: float) -> ConvergenceMetrics:
        """
        Solve a single timestep with flow-thermal coupling
        
        This is the MAIN COUPLING ALGORITHM that iterates between:
        1. Update densities from temperatures
        2. Calculate buoyancy from densities  
        3. Solve momentum equation for flow
        4. Update heat transfer with new flow
        5. Calculate new temperatures
        6. Check convergence and repeat if needed
        
        Args:
            chimney: Chimney object with state and methods
            step: Current time step index
            T_inlet: Inlet temperature (K)
            dt: Time step size (s)
            
        Returns:
            Convergence metrics
        """
        metrics = ConvergenceMetrics()
        
        # SPECIAL HANDLING FOR STARTUP
        # In first few timesteps, flow is establishing from rest
        # Sequential approach is more stable during this transient
        if step < 1:
            self._solve_sequential(chimney, step, T_inlet)
            metrics.converged = True
            metrics.iterations = 1
            return metrics
        
        # Store initial state for residual calculation
        previous_state = {
            'temperatures': chimney.T_gas[step-1].copy(),
            'mass_flow': chimney.mass_flow_rate[step-1],
            'pressures': chimney.pressure[step-1].copy()
        }
        
        # Initialize current state from previous timestep
        current_state = {
            'temperatures': previous_state['temperatures'].copy(),
            'mass_flow': previous_state['mass_flow'],
            'pressures': previous_state['pressures'].copy()
        }
        
        # Set inlet boundary condition
        current_state['temperatures'][0] = T_inlet
        
        # DISABLED stability detection to ensure full accuracy matches baseline
        # Always use full iterations for maximum accuracy
        actual_max_iterations = self.max_iterations
        
        # Adaptive relaxation
        current_relaxation = self.relaxation_factor
        
        # MAIN COUPLING ITERATIONS
        # Each iteration refines the solution to be more consistent
        for iteration in range(actual_max_iterations):
            # Store state at start of iteration for convergence check
            # Only copy if we need to check convergence (not on last iteration)
            if iteration < actual_max_iterations - 1:
                iter_start_state = {
                    'temperatures': current_state['temperatures'].copy(),
                    'mass_flow': current_state['mass_flow'],
                    'pressures': current_state['pressures'].copy()
                }
            else:
                # Last iteration - no need to copy for convergence check
                iter_start_state = current_state
            
            # STEP 1: UPDATE DENSITIES
            # Gas density depends on temperature (ideal gas law)
            densities = self._update_densities(chimney, current_state['temperatures'])
            
            # STEP 2: CALCULATE BUOYANCY
            # Density differences create pressure that drives flow
            buoyancy_pressure = self._calculate_buoyancy(chimney, densities)
            
            # STEP 3: SOLVE MOMENTUM EQUATION
            # Balance buoyancy vs friction to find flow rate
            new_mass_flow, new_pressures = self._solve_momentum_improved(
                chimney, current_state['pressures'], densities, 
                buoyancy_pressure, current_state['mass_flow'], dt
            )
            
            # STEP 4: UPDATE VELOCITIES
            # Convert mass flow to velocity in each segment
            velocities = self._calculate_velocities(chimney, new_mass_flow, densities)
            
            # 5. Solve energy equation with updated flow
            new_temperatures = self._solve_energy_improved(
                chimney, step, current_state['temperatures'], 
                velocities, new_mass_flow, dt
            )
            
            # Apply adaptive relaxation
            if iteration > 2:
                # Check if residuals are decreasing
                T_res_prev = metrics.temperature_residual
                if T_res_prev > 0 and metrics.temperature_residual > T_res_prev:
                    # Residuals increasing, reduce relaxation
                    current_relaxation = max(self.min_relaxation, current_relaxation * 0.8)
                else:
                    # Residuals decreasing, can increase relaxation
                    current_relaxation = min(self.max_relaxation, current_relaxation * 1.1)
            
            # Apply relaxation with current factor
            current_state['temperatures'] = (current_relaxation * new_temperatures + 
                                           (1 - current_relaxation) * current_state['temperatures'])
            current_state['mass_flow'] = (current_relaxation * new_mass_flow + 
                                        (1 - current_relaxation) * current_state['mass_flow'])
            current_state['pressures'] = (current_relaxation * new_pressures + 
                                        (1 - current_relaxation) * current_state['pressures'])
            
            # Check convergence
            T_res, flow_res, P_res = self.calculate_residuals(current_state, iter_start_state)
            
            max_residual = max(T_res, flow_res, P_res)
            
            # Update metrics
            metrics.temperature_residual = T_res
            metrics.flow_residual = flow_res
            metrics.pressure_residual = P_res
            metrics.iterations = iteration + 1
            
            # Check for runaway temperatures
            T_max = np.max(current_state['temperatures'])
            if T_max > 1073.15:  # 800°C
                # Reduce relaxation to stabilize (warning removed)
                current_relaxation = max(0.1, current_relaxation * 0.5)
            
            if max_residual < self.tolerance:
                metrics.converged = True
                break
            
            # Check for stalled convergence
            if iteration > 5 and max_residual > 0.1:
                # Accept current state even if not fully converged
                self.logger.debug(f"Accepting partial convergence at iteration {iteration+1}")
                break
        
        # Update chimney state with final values
        chimney.T_gas[step] = current_state['temperatures']
        chimney.mass_flow_rate[step] = current_state['mass_flow']
        chimney.pressure[step] = current_state['pressures']
        chimney.density[step] = densities
        chimney.velocity[step] = velocities
        
        return metrics
    
    def _update_densities(self, chimney, temperatures: np.ndarray) -> np.ndarray:
        """Update gas densities based on temperatures"""
        densities = np.zeros_like(temperatures)
        
        # Single loop - no need to cache for such a simple operation
        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            densities[seg] = chimney.physics.air_density(temperatures[seg], part.Pa)
        
        return densities
    
    def _calculate_buoyancy(self, chimney, densities: np.ndarray) -> np.ndarray:
        """Calculate buoyancy pressure differences"""
        buoyancy = np.zeros_like(densities)
        
        # Cache ambient densities by part (they don't change)
        ambient_density_cache = {}
        
        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            segment_height = chimney.segment_lengths[seg]
            
            # Ambient density - cache by part since it's the same for all segments in a part
            if part_idx not in ambient_density_cache:
                ambient_density_cache[part_idx] = chimney.physics.air_density(part.T_ambient_K, part.Pa)
            rho_ambient = ambient_density_cache[part_idx]
            
            # Buoyancy pressure difference
            buoyancy[seg] = chimney.physics.GRAVITY * segment_height * (rho_ambient - densities[seg])
        
        return buoyancy

    def _calculate_inlet_restriction_resistance(self, chimney, rho_inlet: float) -> float:
        """Calculate quadratic resistance coefficient for inlet restriction

        Derives R_door such that ΔP_door = R_door × ṁ²

        Starting from the orifice model:
            ΔP = K × 0.5 × ρ × v²
            where v = ṁ/(ρ × A_eff)

        Substituting:
            ΔP = K × 0.5 × ρ × (ṁ/(ρ × A_eff))²
               = K × 0.5 × ρ × ṁ² / (ρ² × A_eff²)
               = K × ṁ² / (2 × ρ × A_eff²)

        Therefore:
            R_door = K / (2 × ρ × A_eff²)

        This resistance is added to total_resistance, ensuring consistent
        pressure-flow relationships and avoiding numerical issues from
        dividing by small areas.

        Args:
            chimney: Chimney object with inlet_restriction config
            rho_inlet: Gas density at inlet [kg/m³] - use inlet density for consistency

        Returns:
            Resistance coefficient R_door [Pa·s²/kg²]
        """
        # Check if feature enabled
        if not chimney.inlet_restriction.enable:
            return 0.0

        # Effective area already includes smooth leakage floor:
        # A_eff = sqrt(A_eff_raw² + A_leak²)
        # This ensures finite resistance even when door is fully closed
        A_eff = chimney.inlet_restriction.effective_area_m2

        # K_equiv = 1/Cd² (computed during initialization)
        K = chimney.inlet_restriction.K_equiv

        # R_door = K / (2 × ρ × A_eff²)
        # Using inlet density for consistency (flow enters at inlet conditions)
        R_door = K / (2.0 * rho_inlet * A_eff**2)

        # Add stove body resistance (fuel bed, baffles, flue collar)
        # This uses PIPE AREA (not door area) since it's inside the stove
        # R_stove = K_stove / (2 × ρ × A_pipe²)
        R_stove = 0.0
        K_stove = getattr(chimney.inlet_restriction, 'K_stove', 0.0)
        if K_stove > 0:
            A_pipe = chimney.parts[0].inner_area_m2  # Use first part's area
            R_stove = K_stove / (2.0 * rho_inlet * A_pipe**2)

        return R_door + R_stove

    def _calculate_catalytic_bypass_loss(self, chimney, mass_flow_rate: float,
                                         densities: np.ndarray) -> float:
        """Calculate pressure drop from catalytic bypass damper

        Models the bypass/catalyst as a variable resistance based on bypass state.
        When bypass is open: low K-loss (gases bypass catalyst)
        When bypass is closed: higher K-loss (gases flow through catalyst honeycomb)

        Args:
            chimney: Chimney object with catalytic_bypass config
            mass_flow_rate: Current mass flow rate [kg/s]
            densities: Array of gas densities at each segment [kg/m³]

        Returns:
            Pressure drop [Pa] - always positive (resistance opposes flow)
        """
        # Check if feature enabled
        if not chimney.catalytic_bypass.enable:
            return 0.0

        # Handle zero/very small flow
        if abs(mass_flow_rate) < 1e-10:
            return 0.0

        # Get segment index where catalyst is located
        seg_idx = chimney.catalytic_bypass.segment_index

        # Get segment properties
        part_idx, local_seg, part = chimney.get_segment_info(seg_idx)
        area = part.inner_area_m2

        # Gas density at catalyst location
        rho = densities[seg_idx]

        # Velocity at catalyst location
        velocity = abs(mass_flow_rate) / (rho * area)

        # Pressure drop from bypass/catalyst
        # ΔP = K × (1/2) × ρ × v²
        K = chimney.catalytic_bypass.K_effective
        dp_bypass = K * 0.5 * rho * velocity**2

        # Return positive value (resistance opposes flow)
        return dp_bypass

    # def _solve_momentum(self, chimney, pressures: np.ndarray, densities: np.ndarray,
    #                    buoyancy: np.ndarray, mass_flow_old: float) -> Tuple[float, np.ndarray]:
    #     """
    #     Solve momentum equation for mass flow rate
    #     Uses simplified approach - could be enhanced with full momentum equation
    #     """
    #     # Calculate total pressure drop through system
    #     total_buoyancy = np.sum(buoyancy)
        
    #     # Friction pressure drops
    #     total_friction = 0.0
    #     for seg in range(chimney.total_segments):
    #         part_idx, local_seg, part = chimney.get_segment_info(seg)
            
    #         # Segment properties
    #         area = part.inner_area_m2
    #         diameter = part.inner_diameter_m
    #         length = part.height_m / part.num_segments
    #         velocity = mass_flow_old / (densities[seg] * area)
            
    #         # Reynolds number
    #         viscosity = chimney.physics.air_viscosity(chimney.T_gas[chimney.time_step-1, seg])
    #         Re = densities[seg] * abs(velocity) * diameter / viscosity
            
    #         # Friction factor
    #         if Re < 2300:
    #             f = 64 / max(Re, 1)
    #         else:
    #             f = 0.316 / Re**0.25
            
    #         # Pressure drop
    #         dp_friction = f * (length / diameter) * 0.5 * densities[seg] * velocity**2
    #         total_friction += dp_friction
        
    #     # Net driving pressure
    #     dp_net = total_buoyancy - total_friction
        
    #     # Update mass flow (simplified - could use more sophisticated approach)
    #     # This uses a simple proportional update
    #     if dp_net > 0:
    #         mass_flow_new = mass_flow_old * (1 + 0.1 * dp_net / (total_friction + 1))
    #     else:
    #         mass_flow_new = mass_flow_old * (1 + 0.1 * dp_net / (total_friction + 1))
        
    #     # Ensure positive flow
    #     mass_flow_new = max(mass_flow_new, 0.001)
        
    #     # Calculate pressure distribution (simplified)
    #     new_pressures = np.zeros_like(pressures)
    #     new_pressures[0] = 101325  # Atmospheric at inlet
        
    #     for seg in range(1, chimney.total_segments):
    #         # Pressure change due to buoyancy and friction
    #         new_pressures[seg] = new_pressures[seg-1] - total_friction/chimney.total_segments + buoyancy[seg-1]
        
    #     return mass_flow_new, new_pressures
    
    def _calculate_velocities(self, chimney, mass_flow: float, densities: np.ndarray) -> np.ndarray:
        """Calculate velocities from mass flow rate"""
        velocities = np.zeros_like(densities)
        
        # Single loop, no unnecessary caching
        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            velocities[seg] = mass_flow / (densities[seg] * part.inner_area_m2)
        
        return velocities
    
    # def _solve_energy(self, chimney, step: int, temperatures: np.ndarray,
    #                  velocities: np.ndarray, mass_flow: float, dt: float) -> np.ndarray:
    #     """
    #     Solve energy equation with current flow field
    #     This is a simplified version - the actual implementation would call
    #     the comprehensive heat transfer model
    #     """
    #     new_temperatures = temperatures.copy()
        
    #     # For each segment, calculate heat transfer and update temperature
    #     for seg in range(chimney.total_segments):
    #         if seg == 0:
    #             # Inlet boundary condition
    #             continue
            
    #         part_idx, local_seg, part = chimney.get_segment_info(seg)
            
    #         # Simplified energy balance
    #         # In practice, this would call comprehensive_heat_transfer
    #         # Here we just show the structure
            
    #         # Advection from upstream
    #         T_upstream = new_temperatures[seg-1]
            
    #         # Heat loss to walls (simplified)
    #         T_wall = chimney.T_wall[step-1, seg] if step > 0 else part.T_ambient_K
    #         h_conv = 50.0  # Simplified - would calculate properly
            
    #         # Surface area
    #         length = part.height_m / part.num_segments
    #         area = np.pi * part.inner_diameter_m * length
            
    #         # Heat loss
    #         q_loss = h_conv * area * (temperatures[seg] - T_wall)
            
    #         # Temperature change
    #         cp = chimney.physics.air_specific_heat(temperatures[seg])
    #         volume = part.inner_area_m2 * length
    #         mass = chimney.density[step-1, seg] * volume if step > 0 else part.Pa / (287 * temperatures[seg]) * volume
            
    #         # Energy balance
    #         dT_dt = (mass_flow * cp * (T_upstream - temperatures[seg]) - q_loss) / (mass * cp)
            
    #         # Update temperature
    #         new_temperatures[seg] = temperatures[seg] + dT_dt * dt
        
    #     return new_temperatures
    
    def _solve_sequential(self, chimney, step: int, T_inlet: float):
        """
        Solve using sequential approach (for early timesteps)
        """
        # Just call the original sequential methods
        chimney._solve_comprehensive_heat_transfer(step, T_inlet)
        chimney._update_flow_and_pressure(step)
    
    def _solve_momentum_improved(self, chimney, pressures: np.ndarray, densities: np.ndarray,
                                buoyancy: np.ndarray, mass_flow_old: float, dt: float) -> Tuple[float, np.ndarray]:
        """
        Improved momentum solver with quadratic resistance model

        Uses consistent resistance-based approach:
            dp_drive = R_total × ṁ²
            ṁ_steady = sign(dp_drive) × sqrt(|dp_drive| / R_total)

        All resistances (pipe friction, inlet restriction, catalytic bypass)
        are combined into R_total, avoiding the need for special "decay" modes.
        """
        # Total driving force from buoyancy
        dp_drive = np.sum(buoyancy)

        # Add wind cap boundary effect (if enabled)
        # Negative deltaP (suction) aids draft → adds to driving force
        # Positive deltaP (pressurization) hurts draft → reduces driving force
        Pa_outlet = chimney.parts[-1].Pa if len(chimney.parts) > 0 else 101325.0
        T_outlet = chimney.parts[-1].T_ambient_K if len(chimney.parts) > 0 else 294.15
        deltaP_wind = chimney.get_wind_cap_deltaP(chimney.time_step, Pa_outlet, T_outlet)
        dp_drive = dp_drive - deltaP_wind

        # Apply room depressurization (reduces available draft, can cause backdraft)
        # Positive room_pressure_offset means room is depressurized (below outdoor pressure)
        # This reduces the pressure difference driving flow through the chimney
        room_depress = getattr(chimney, 'room_pressure_offset_Pa', 0.0)
        dp_drive = dp_drive - room_depress

        # Backflow risk warning
        if dp_drive < 0 and abs(deltaP_wind) > 0.1:
            print(f"[WIND CAP WARNING] step={chimney.time_step} Net driving head negative "
                  f"(stack={np.sum(buoyancy):.1f} Pa + wind={-deltaP_wind:.1f} Pa < 0): backflow risk")

        # Cache segment info to avoid repeated lookups
        segment_cache = {seg: chimney.get_segment_info(seg) for seg in range(chimney.total_segments)}

        # ============================================================
        # CALCULATE TOTAL RESISTANCE (R_total)
        # All resistances are combined: R_total = R_pipe + R_door + R_bypass
        # ============================================================
        total_resistance = 0.0
        avg_density = np.mean(densities)
        rho_inlet = densities[0]

        # Cache friction factors and viscosities to avoid recalculation
        friction_factors = {}
        viscosity_cache = {}

        # --- 1. Pipe segment resistance ---
        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = segment_cache[seg]

            # Segment properties
            area = part.inner_area_m2
            diameter = part.inner_diameter_m
            length = chimney.segment_lengths[seg]

            # Local velocity (for Re calculation)
            velocity = mass_flow_old / (densities[seg] * area) if abs(mass_flow_old) > 1e-10 else 0.1

            # Viscosity at local temperature
            T_seg = chimney.T_gas[chimney.time_step-1, seg] if chimney.time_step > 0 else part.T_ambient_K + 10
            viscosity = chimney.physics.air_viscosity(T_seg)
            viscosity_cache[seg] = viscosity

            # Reynolds number
            Re = densities[seg] * abs(velocity) * diameter / viscosity
            Re = max(Re, 100)  # Floor at 100 to avoid extreme f values

            # Friction factor
            if Re < 2300:
                f = 64 / Re
            else:
                relative_roughness = part.roughness_m / diameter
                if relative_roughness < 1e-6:
                    f = 0.316 / Re**0.25  # Blasius (smooth)
                else:
                    f = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/Re**0.9))**2  # Swamee-Jain

            friction_factors[seg] = f

            # Minor losses - use actual elbow K values from segment_elbows (Feb 2 fix)
            K_minor = 0.0
            elbow_key = (part_idx, local_seg)
            if hasattr(chimney, 'segment_elbows') and elbow_key in chimney.segment_elbows:
                for elbow_spec in chimney.segment_elbows[elbow_key]:
                    K_elbow = base_K(elbow_spec) * elbow_spec.vane_factor
                    K_minor += K_elbow

            # Segment resistance: R_seg = (f*L/D + K_minor) × ρ / (2 × A²)
            K_total = f * length / diameter + K_minor
            R_seg = K_total * densities[seg] / (2 * area**2)
            total_resistance += R_seg

        # --- 2. Inlet restriction resistance ---
        # R_door = K / (2 × ρ_inlet × A_eff²)
        # This is added to total_resistance (NOT subtracted from dp_drive)
        R_door = self._calculate_inlet_restriction_resistance(chimney, rho_inlet)
        total_resistance += R_door

        # --- 3. Catalytic bypass resistance ---
        # Convert dp_bypass to resistance: R_bypass ≈ dp_bypass / mdot²
        # (Keeping existing approach for now, but as resistance contribution)
        if abs(mass_flow_old) > 1e-10:
            dp_catalytic = self._calculate_catalytic_bypass_loss(chimney, mass_flow_old, densities)
            R_bypass = dp_catalytic / (mass_flow_old**2) if dp_catalytic > 0 else 0.0
            total_resistance += R_bypass

        # --- 4. Distributed Leakage (VENT-II Style) ---
        # Calculate leakage at each segment based on local gauge pressure
        # Uses iterative approach with relaxation for stability
        # Leakage IN (negative P_gauge) adds mass, leakage OUT (positive P_gauge) removes mass
        total_leakage_mdot = 0.0
        leakage_enabled = (
            self.momentum_config.use_distributed_leakage or
            (hasattr(chimney, 'leakage_config') and chimney.leakage_config.enable)
        )

        if leakage_enabled and hasattr(chimney, 'segment_leakage_area_m2'):
            # Get iteration parameters from config
            N_leak_iter = self.momentum_config.N_leak_iter
            leak_relax = self.momentum_config.leak_relaxation

            # Iterative leakage calculation for convergence
            for leak_iter in range(N_leak_iter):
                new_leakage_mdot = 0.0

                for seg in range(chimney.total_segments):
                    A_leak = chimney.segment_leakage_area_m2[seg]
                    if A_leak > 0:
                        part_idx, local_seg, part = segment_cache[seg]
                        # Estimate gauge pressure from previous iteration
                        P_gauge = pressures[seg] - part.Pa
                        rho_ambient = chimney.physics.air_density(part.T_ambient_K, part.Pa)
                        rho_gas = densities[seg]

                        # Get Cd from chimney config or use default
                        Cd_leak = 0.61
                        if hasattr(chimney, 'leakage_config') and chimney.leakage_config.enable:
                            Cd_leak = chimney.leakage_config.Cd_leak

                        m_leak = calculate_leakage_mass_flow(
                            P_gauge=P_gauge,
                            A_leak_m2=A_leak,
                            Cd_leak=Cd_leak,
                            rho_ambient=rho_ambient,
                            rho_gas=rho_gas
                        )
                        new_leakage_mdot += m_leak

                # Apply relaxation for stability
                total_leakage_mdot = (leak_relax * new_leakage_mdot +
                                      (1 - leak_relax) * total_leakage_mdot)

        # --- 5. Acceleration Loss ---
        # Pressure cost when gas speeds up due to heating (density decrease)
        dp_accel_total = 0.0
        if self.momentum_config.use_accel_loss and abs(mass_flow_old) > 1e-10:
            C_accel = self.momentum_config.C_accel
            for seg in range(1, chimney.total_segments):
                part_idx, local_seg, part = segment_cache[seg]
                area = part.inner_area_m2

                # Upstream and downstream velocities
                v_upstream = mass_flow_old / (densities[seg-1] * area)
                v_downstream = mass_flow_old / (densities[seg] * area)

                dp_accel = calculate_accel_pressure_loss(
                    rho_upstream=densities[seg-1],
                    v_upstream=v_upstream,
                    rho_downstream=densities[seg],
                    v_downstream=v_downstream,
                    C_accel=C_accel
                )
                dp_accel_total += dp_accel

            # Acceleration loss reduces available driving pressure
            dp_drive = dp_drive - dp_accel_total

        # Ensure minimum resistance to prevent division by zero
        total_resistance = max(total_resistance, 1e-6)

        # ============================================================
        # MOMENTUM BALANCE
        # ============================================================
        system_length = sum(chimney.segment_lengths)
        typical_area = chimney.parts[0].inner_area_m2

        if self.momentum_config.flow_model == "unsteady_momentum":
            # --- Unsteady momentum ODE ---
            # L_eff / A * d(mdot)/dt = dp_available - dp_losses(mdot)
            #
            # dp_losses = R_total * mdot * |mdot|  (quadratic, sign-preserving)
            #
            # IMPORTANT: Always integrate from the *previous timestep* mdot,
            # NOT from the Picard iteration's current estimate. The Picard loop
            # re-evaluates dp_drive and R_total at updated T/rho, but the ODE
            # starting point is fixed for the timestep.
            #
            # Semi-implicit Euler (unconditionally stable):
            #   mdot_new = (mdot_prev + C * dp_drive) / (1 + C * R_total * |mdot_prev|)
            # where C = dt * A / L_eff

            L_eff = system_length  # effective inertial length
            A_eff_mom = typical_area  # representative cross-section

            # Previous timestep mdot (fixed ODE starting point)
            step_idx = chimney.time_step
            mdot_prev_timestep = chimney.mass_flow_rate[step_idx - 1] if step_idx > 0 else 0.0

            C_mom = dt * A_eff_mom / L_eff  # [s * m^2 / m] = [m*s]
            numerator = mdot_prev_timestep + C_mom * dp_drive
            denominator = 1.0 + C_mom * total_resistance * abs(mdot_prev_timestep)
            mass_flow_new = numerator / denominator

        else:
            # --- Quasi-steady (default): algebraic mdot_ss + first-order lag ---
            # mdot_ss = sign(dp_drive) * sqrt(|dp_drive| / R_total)
            mass_flow_magnitude = np.sqrt(abs(dp_drive) / total_resistance)
            mass_flow_steady = np.sign(dp_drive) * mass_flow_magnitude

            # Handle case where dp_drive ~ 0
            if abs(dp_drive) < 1e-10:
                mass_flow_steady = 0.0

            # First-order lag toward steady state
            tau = avg_density * system_length / (total_resistance * (abs(mass_flow_steady) + 1e-6))
            tau = max(tau, dt)  # At least one timestep

            # Relaxation factor (limit to prevent overshooting)
            alpha = min(dt / (tau + dt), 0.5)
            mass_flow_new = mass_flow_old + alpha * (mass_flow_steady - mass_flow_old)

        # Apply leakage adjustment to mass flow
        # Positive leakage = air leaking IN (adds mass) → increases outlet flow
        # Negative leakage = air leaking OUT (loses mass) → decreases outlet flow
        mass_flow_new = mass_flow_new + total_leakage_mdot

        # Calculate geometry-based bounds
        max_velocity = 50.0  # m/s - reasonable upper limit
        max_mass_flow_magnitude = max_velocity * avg_density * typical_area

        # Allow both positive and negative flow, but limit magnitude
        mass_flow_new = np.clip(mass_flow_new, -max_mass_flow_magnitude, max_mass_flow_magnitude)
        
        # Update pressure distribution
        new_pressures = np.zeros_like(pressures)
        new_pressures[0] = 101325  # Reference pressure at inlet
        
        for seg in range(1, chimney.total_segments):
            # Pressure change = buoyancy - friction loss
            part_idx, local_seg, part = segment_cache[seg]  # Use cached segment info
            velocity = mass_flow_new / (densities[seg] * part.inner_area_m2)
            
            # Recalculate friction factor with new velocity
            # (Reynolds number changes with new mass flow)
            # Use pre-computed segment length
            length = chimney.segment_lengths[seg]
            diameter = part.inner_diameter_m
            # Use cached viscosity (temperature hasn't changed within this timestep)
            viscosity = viscosity_cache.get(seg)
            if viscosity is None:  # Fallback if not cached
                T_seg = chimney.T_gas[chimney.time_step-1, seg] if chimney.time_step > 0 else part.T_ambient_K + 10
                viscosity = chimney.physics.air_viscosity(T_seg)
            Re_new = densities[seg] * abs(velocity) * diameter / viscosity
            Re_new = max(Re_new, 10)
            
            # Use same friction factor calculation but with new Re
            if Re_new < 2300:
                f = 64 / Re_new
            else:
                # For turbulent, can often reuse if Re didn't change much
                # But for accuracy, recalculate with new velocity
                relative_roughness = part.roughness_m / diameter
                if relative_roughness < 1e-6:
                    f = 0.316 / Re_new**0.25
                else:
                    f = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/Re_new**0.9))**2
            
            # Pressure drop from friction
            dp_friction = f * (length / diameter) * 0.5 * densities[seg] * velocity**2

            # Fitting losses (Feb 2 fix) - add to pressure drop
            dp_minor = 0.0
            elbow_key = (part_idx, local_seg)
            if hasattr(chimney, 'segment_elbows') and elbow_key in chimney.segment_elbows:
                for elbow_spec in chimney.segment_elbows[elbow_key]:
                    K_elbow = base_K(elbow_spec) * elbow_spec.vane_factor
                    dp_minor += K_elbow * 0.5 * densities[seg] * velocity**2

            # Net pressure change = buoyancy - friction - fitting losses
            dp_segment = buoyancy[seg-1] - dp_friction - dp_minor
            new_pressures[seg] = new_pressures[seg-1] + dp_segment

        # --- Diagnostics: record momentum budget ---
        if self.diagnostics.enable:
            # Compute itemized loss components for diagnostics
            dp_friction_total = 0.0
            dp_fittings_total = 0.0
            for seg in range(chimney.total_segments):
                part_idx, local_seg, part = segment_cache[seg]
                area = part.inner_area_m2
                diameter = part.inner_diameter_m
                length_s = chimney.segment_lengths[seg]
                vel = mass_flow_new / (densities[seg] * area) if abs(mass_flow_new) > 1e-10 else 0.0
                f_seg = friction_factors.get(seg, 0.01)
                dp_friction_total += f_seg * (length_s / diameter) * 0.5 * densities[seg] * vel**2
                elbow_key = (part_idx, local_seg)
                if hasattr(chimney, 'segment_elbows') and elbow_key in chimney.segment_elbows:
                    for elbow_spec in chimney.segment_elbows[elbow_key]:
                        K_elbow = base_K(elbow_spec) * elbow_spec.vane_factor
                        dp_fittings_total += K_elbow * 0.5 * densities[seg] * vel**2

            # Inlet and stove resistance as dp
            dp_inlet_diag = R_door * mass_flow_new**2 if abs(mass_flow_new) > 1e-10 else 0.0
            dp_stove_diag = 0.0
            K_stove_val = getattr(chimney.inlet_restriction, 'K_stove', 0.0) if hasattr(chimney, 'inlet_restriction') else 0.0
            if K_stove_val > 0 and abs(mass_flow_new) > 1e-10:
                A_pipe = chimney.parts[0].inner_area_m2
                R_stove_diag = K_stove_val / (2.0 * rho_inlet * A_pipe**2)
                dp_stove_diag = R_stove_diag * mass_flow_new**2

            dp_bypass_diag = 0.0
            if abs(mass_flow_new) > 1e-10:
                dp_bypass_diag = self._calculate_catalytic_bypass_loss(chimney, mass_flow_new, densities)

            self.diagnostics.record_momentum_budget(
                step=chimney.time_step,
                time_s=chimney.current_time,
                mdot=mass_flow_new,
                velocity_mean=float(np.mean(np.abs(mass_flow_new / (densities * chimney.parts[0].inner_area_m2)))),
                rho_mean=float(avg_density),
                dp_buoyancy=float(np.sum(buoyancy)),
                dp_wind_cap=float(deltaP_wind),
                dp_room=float(room_depress),
                dp_friction=dp_friction_total,
                dp_fittings=dp_fittings_total,
                dp_inlet=dp_inlet_diag,
                dp_stove=dp_stove_diag,
                dp_bypass=dp_bypass_diag,
                dp_accel=dp_accel_total,
                dp_leakage=0.0,  # leakage modifies mdot, not dp directly
            )

        return mass_flow_new, new_pressures
    
    def _solve_energy_improved(self, chimney, step: int, temperatures: np.ndarray,
                              velocities: np.ndarray, mass_flow: float, dt: float) -> np.ndarray:
        """
        Improved energy solver that calls comprehensive heat transfer
        """
        new_temperatures = temperatures.copy()
        
        # Set inlet
        new_temperatures[0] = temperatures[0]  # Already set by boundary condition
        
        # Cache segment info to avoid repeated lookups
        segment_cache = {seg: chimney.get_segment_info(seg) for seg in range(chimney.total_segments)}
        
        # For each segment, use comprehensive heat transfer
        for seg in range(chimney.total_segments):
            # Note: Don't skip segment 0 for wall temperature updates
            # The inlet gas temperature is fixed, but walls still need heating
            
            part_idx, local_seg, part = segment_cache[seg]
            # Use pre-computed segment length
            segment_length = chimney.segment_lengths[seg]
            
            # Get previous wall temperatures
            if part.type == 'single':
                prev_T_wall = chimney.T_wall[step-1, seg] if step > 0 else part.T_ambient_K
                if prev_T_wall <= 0:
                    prev_T_wall = part.T_ambient_K
            else:  # triple
                prev_T_liner = chimney.T_liner[step-1, seg] if step > 0 else part.T_ambient_K
                prev_T_middle = chimney.T_middle[step-1, seg] if step > 0 else part.T_ambient_K
                prev_T_outer = chimney.T_outer[step-1, seg] if step > 0 else part.T_ambient_K
                if prev_T_liner <= 0:
                    prev_T_liner = part.T_ambient_K
                    prev_T_middle = part.T_ambient_K
                    prev_T_outer = part.T_ambient_K
            
            # Prepare segment data
            segment_data = {
                'gas_temperature': temperatures[seg],
                'ambient_temperature': part.T_ambient_K,
                'pressure': chimney.pressure[step-1, seg] if step > 0 else part.Pa,
                'velocity': velocities[seg],
                'diameter': part.inner_diameter_m,
                'length': segment_length,
                'orientation': part.orientation,
                'has_fittings': part.has_fittings,
                'humidity_ratio': part.humidity_ratio,
                'wall_type': part.type,
                'segment_id': seg,
                'mass_flow_rate': mass_flow
            }
            
            # Add wall temperatures
            if part.type == 'single':
                segment_data['wall_temperature'] = prev_T_wall
                segment_data['wall_properties'] = {
                    'density': part.density_w_kg_m3,
                    'specific_heat': part.cp_w_J_kg_K,
                    'thickness': part.thickness_w_m,
                    'thermal_conductivity': part.k_w_W_m_K,
                    'emissivity': part.epsilon_w
                }
            else:  # triple
                segment_data['liner_temperature'] = prev_T_liner
                segment_data['middle_temperature'] = prev_T_middle
                segment_data['outer_temperature'] = prev_T_outer
                segment_data['wall_properties'] = {
                    'liner': {
                        'density': part.density_inner_kg_m3,
                        'specific_heat': part.cp_inner_J_kg_K,
                        'thickness': part.thickness_inner_m,
                        'thermal_conductivity': part.k_inner_W_m_K,
                        'emissivity': part.epsilon_inner
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
                        'thermal_conductivity': part.k_outer_W_m_K,
                        'emissivity': part.epsilon_outer
                    }
                }
            
            # Call comprehensive heat transfer for proper physics
            # This ensures we use the full model, not simplified estimates

            # Get upstream temperature for advection - UPWIND SCHEME BASED ON FLOW DIRECTION
            # This allows proper handling of both normal draft and backdraft
            if mass_flow >= 0:  # UPWARD FLOW (normal draft)
                if seg == 0:
                    # Bottom inlet: use inlet boundary condition (from solve_coupled_timestep T_inlet arg)
                    T_upstream = temperatures[0]  # Already set to T_inlet at line 151
                else:
                    # Upstream is segment below
                    T_upstream = temperatures[seg-1]
            else:  # DOWNWARD FLOW (backdraft, mass_flow < 0)
                if seg == chimney.total_segments - 1:
                    # Top inlet: outdoor/ambient air entering from above
                    T_upstream = part.T_ambient_K
                else:
                    # Upstream is segment above
                    T_upstream = temperatures[seg+1]
            
            # Calculate segment heat transfer using comprehensive model
            # This will handle convection, radiation, condensation properly
            ht_results = chimney.comprehensive_ht.calculate_segment_heat_transfer(segment_data, dt)
            
            # CRITICAL FIX: Store wall temperature results back to chimney object
            # This was missing and causing -273.1°C temperatures
            if part.type == 'single':
                T_wall_new = ht_results.get('wall_temperature_new', part.T_ambient_K)
                if T_wall_new <= 0:
                    T_wall_new = part.T_ambient_K
                chimney.T_wall[step, seg] = T_wall_new
                # For single wall, outer = wall temp, but don't overwrite triple wall arrays
                chimney.T_outer[step, seg] = T_wall_new
                # Don't set T_liner and T_middle for single wall - leave them as initialized
            else:  # triple
                T_liner_new = ht_results.get('liner_temperature_new', part.T_ambient_K)
                T_middle_new = ht_results.get('middle_temperature_new', part.T_ambient_K)
                T_outer_new = ht_results.get('outer_temperature_new', part.T_ambient_K)
                
                if T_liner_new <= 0:
                    T_liner_new = part.T_ambient_K
                    T_middle_new = part.T_ambient_K
                    T_outer_new = part.T_ambient_K
                    
                chimney.T_liner[step, seg] = T_liner_new
                chimney.T_middle[step, seg] = T_middle_new
                chimney.T_outer[step, seg] = T_outer_new
                chimney.T_wall[step, seg] = T_liner_new
            
            # Extract the convection heat transfer
            h_inner = ht_results['h_inner']
            h_outer = ht_results['h_outer']
            area = np.pi * part.inner_diameter_m * segment_length
            
            # Store heat transfer coefficients
            chimney.h_conv_inner[step, seg] = h_inner
            chimney.h_conv_outer[step, seg] = h_outer
            
            # Get wall temperature based on wall type (use newly computed values)
            if part.type == 'single':
                T_wall = T_wall_new
            else:  # triple
                T_wall = T_liner_new
            
            # Gas properties
            cp = chimney.physics.air_specific_heat(temperatures[seg])
            gas_volume = part.inner_area_m2 * segment_length
            gas_mass = chimney.density[step-1, seg] * gas_volume if step > 0 else part.Pa / (287 * temperatures[seg]) * gas_volume
            
            # Energy balance for gas control volume
            # CRITICAL PHYSICS: When flow stops (T_gas → T_ambient), heat transfer must also stop

            # Determine if this segment is an inlet boundary (depends on flow direction)
            is_inlet_boundary = False
            if mass_flow >= 0:  # Normal upward flow
                is_inlet_boundary = (seg == 0)  # Bottom is inlet
            else:  # Backdraft (downward flow)
                is_inlet_boundary = (seg == chimney.total_segments - 1)  # Top is inlet

            if abs(mass_flow) > 1e-6 and not is_inlet_boundary:
                # Flowing case with advection (skip inlet boundaries)

                if self.momentum_config.gas_energy_model == "plugflow_ntu":
                    # --- Plug-flow NTU/effectiveness model ---
                    # Treat segment as a heat exchanger with:
                    #   hot side = gas flowing through
                    #   cold side = wall at T_wall (inner surface)
                    #
                    # UA = h_inner * A_inner (convective conductance)
                    # effectiveness = 1 - exp(-UA / (mdot * cp))
                    # T_out = T_in - effectiveness * (T_in - T_wall)
                    #
                    # Energy removed from gas = h_inner * A * effectiveness * (T_in - T_wall)
                    # This equals the wall source term for energy conservation.

                    UA = h_inner * area
                    mdot_cp = abs(mass_flow) * cp
                    NTU = UA / (mdot_cp + 1e-10)
                    effectiveness = 1.0 - np.exp(-NTU)

                    # Include condensation latent heat as additional "heat loss"
                    q_latent = ht_results.get('latent_heat_rate', 0)

                    # Gas outlet temperature for this segment
                    T_out_ntu = T_upstream - effectiveness * (T_upstream - T_wall)

                    # Adjust for latent heat (small correction)
                    if q_latent != 0 and mdot_cp > 1e-6:
                        T_out_ntu -= q_latent / mdot_cp

                    new_temperatures[seg] = T_out_ntu

                else:
                    # --- Current model (default) ---
                    # Relaxation-based gas temperature update

                    # Heat loss to walls (positive when gas loses heat)
                    q_loss_to_walls = h_inner * area * (temperatures[seg] - T_wall)

                    # Advection terms
                    residence_time = gas_mass / abs(mass_flow)

                    # Include condensation latent heat if present
                    q_latent = ht_results.get('latent_heat_rate', 0)
                    q_total_loss = q_loss_to_walls + q_latent

                    # Stable formulation: new temperature based on upstream and heat loss
                    T_steady = T_upstream - q_total_loss * residence_time / (gas_mass * cp)

                    # Bound T_steady to physical range: gas temperature must stay
                    # between upstream and wall temperatures.  Without this, small
                    # |mass_flow| near flow reversal causes residence_time to blow
                    # up and T_steady to diverge to +/-infinity.
                    T_bound_lo = min(T_upstream, T_wall) - 1.0
                    T_bound_hi = max(T_upstream, T_wall) + 1.0
                    T_steady = max(T_steady, T_bound_lo)
                    T_steady = min(T_steady, T_bound_hi)

                    # Apply relaxation toward steady state
                    tau_thermal = gas_mass * cp / (abs(mass_flow) * cp + h_inner * area)
                    alpha = dt / (tau_thermal + dt)
                    new_temperatures[seg] = temperatures[seg] + alpha * (T_steady - temperatures[seg])
            elif is_inlet_boundary:
                # Inlet boundary - temperature is fixed by boundary condition
                if mass_flow >= 0:
                    # Normal draft: bottom inlet temperature set by appliance/fire
                    new_temperatures[seg] = temperatures[seg]  # Already set to T_inlet
                else:
                    # Backdraft: top inlet temperature is outdoor air
                    new_temperatures[seg] = part.T_ambient_K
            else:
                # No/low flow case - gas temperature equilibrates with walls
                tau = gas_mass * cp / (h_inner * area + 1e-6)
                dT_dt = (T_wall - temperatures[seg]) / tau
                new_temperatures[seg] = temperatures[seg] + dT_dt * dt
            
            # Enforce physical bounds for non-inlet segments
            # (inlet boundary temperature is set by the fire/appliance and may be very hot)
            if not is_inlet_boundary:
                T_min_physical = part.T_ambient_K
                T_max_physical = 1073.15  # 800°C max

                # Apply bounds
                new_temperatures[seg] = max(new_temperatures[seg], T_min_physical)
                new_temperatures[seg] = min(new_temperatures[seg], T_max_physical)

        # --- Diagnostics: record energy budget ---
        if self.diagnostics.enable:
            # Compute bulk energy flows
            cp_in = chimney.physics.air_specific_heat(temperatures[0])
            cp_out = chimney.physics.air_specific_heat(new_temperatures[-1])
            Q_adv = abs(mass_flow) * (cp_in * temperatures[0] - cp_out * new_temperatures[-1])

            Q_wall = 0.0
            Q_lat = 0.0
            for seg in range(chimney.total_segments):
                p_idx, l_seg, prt = segment_cache[seg]
                seg_len = chimney.segment_lengths[seg]
                area_s = np.pi * prt.inner_diameter_m * seg_len
                h_i = chimney.h_conv_inner[step, seg] if step > 0 else 10.0
                if prt.type == 'single':
                    Tw = chimney.T_wall[step, seg]
                else:
                    Tw = chimney.T_liner[step, seg]
                Q_wall += h_i * area_s * (new_temperatures[seg] - Tw)

            self.diagnostics.record_energy_budget(
                step=chimney.time_step,
                time_s=chimney.current_time,
                Q_advection=Q_adv,
                Q_wall_total=Q_wall,
                Q_latent=Q_lat,
                T_inlet=float(new_temperatures[0]),
                T_outlet=float(new_temperatures[-1]),
                T_gas_mean=float(np.mean(new_temperatures)),
                T_wall_mean=float(np.mean(chimney.T_wall[step])),
            )

        return new_temperatures