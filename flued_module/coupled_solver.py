"""
Coupled flow-thermal solver with iterative convergence
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

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
                 relaxation_factor: float = 0.3):
        """
        Initialize coupled solver
        
        Args:
            tolerance: Convergence tolerance for residuals (relative change < tolerance)
            max_iterations: Maximum coupling iterations per timestep (more = accurate but slow)
            relaxation_factor: Under-relaxation factor for stability (0-1)
                              Lower values = more stable but slower convergence
                              Higher values = faster but may oscillate
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.relaxation_factor = relaxation_factor
        self.logger = logging.getLogger(__name__)
        
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
        if step < 5:
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
        
        # Adaptive relaxation
        current_relaxation = self.relaxation_factor
        
        # MAIN COUPLING ITERATIONS
        # Each iteration refines the solution to be more consistent
        for iteration in range(self.max_iterations):
            # Store state at start of iteration for convergence check
            iter_start_state = {
                'temperatures': current_state['temperatures'].copy(),
                'mass_flow': current_state['mass_flow'],
                'pressures': current_state['pressures'].copy()
            }
            
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
                self.logger.warning(f"Temperature exceeding limits: {T_max-273.15:.0f}°C")
                # Reduce relaxation to stabilize
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
        
        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            densities[seg] = chimney.physics.air_density(temperatures[seg], part.Pa)
        
        return densities
    
    def _calculate_buoyancy(self, chimney, densities: np.ndarray) -> np.ndarray:
        """Calculate buoyancy pressure differences"""
        buoyancy = np.zeros_like(densities)
        
        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            segment_height = part.height_m / part.num_segments
            
            # Ambient density
            rho_ambient = chimney.physics.air_density(part.T_ambient_K, part.Pa)
            
            # Buoyancy pressure difference
            buoyancy[seg] = chimney.physics.GRAVITY * segment_height * (rho_ambient - densities[seg])
        
        return buoyancy
    
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
        Improved momentum solver with better physics
        """
        # Total driving force
        total_buoyancy = np.sum(buoyancy)
        
        # Calculate resistance more accurately
        total_resistance = 0.0
        avg_density = np.mean(densities)
        
        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            
            # Segment properties
            area = part.inner_area_m2
            diameter = part.inner_diameter_m
            length = part.height_m / part.num_segments
            
            # Local velocity
            velocity = mass_flow_old / (densities[seg] * area)
            
            # Viscosity at local temperature
            T_seg = chimney.T_gas[chimney.time_step-1, seg] if chimney.time_step > 0 else part.T_ambient_K + 10
            viscosity = chimney.physics.air_viscosity(T_seg)
            
            # Reynolds number
            Re = densities[seg] * abs(velocity) * diameter / viscosity
            Re = max(Re, 10)  # Avoid very low Re
            
            # Friction factor including roughness effects
            if Re < 2300:
                f = 64 / Re
            else:
                # Colebrook-White equation for rough pipes
                # Use iterative solution or approximation
                relative_roughness = part.roughness_m / diameter
                if relative_roughness < 1e-6:
                    # Smooth pipe (Blasius)
                    f = 0.316 / Re**0.25
                else:
                    # Rough pipe - Moody approximation
                    f = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/Re**0.9))**2
            
            # Add minor losses if fittings present
            K_minor = 0.5 if part.has_fittings else 0.0
            
            # Total resistance for segment
            K_total = f * length / diameter + K_minor
            resistance = K_total * densities[seg] / (2 * area**2)
            total_resistance += resistance
        
        # Cache friction factors calculated above to avoid recalculation
        friction_factors = {}
        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            T_seg = chimney.T_gas[chimney.time_step-1, seg] if chimney.time_step > 0 else part.T_ambient_K + 10
            viscosity = chimney.physics.air_viscosity(T_seg)
            velocity = mass_flow_old / (densities[seg] * part.inner_area_m2)
            Re = densities[seg] * abs(velocity) * part.inner_diameter_m / viscosity
            Re = max(Re, 10)
            
            if Re < 2300:
                friction_factors[seg] = 64 / Re
            else:
                relative_roughness = part.roughness_m / part.inner_diameter_m
                if relative_roughness < 1e-6:
                    friction_factors[seg] = 0.316 / Re**0.25
                else:
                    friction_factors[seg] = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/Re**0.9))**2
        
        # Momentum balance: dp_buoyancy = resistance * mdot^2
        # Solve for new mass flow
        if total_buoyancy > 0 and total_resistance > 0:
            # mdot = sqrt(dp_buoyancy / resistance)
            mass_flow_steady = np.sqrt(total_buoyancy / total_resistance)
            
            # Apply inertia (first-order response)
            # Modified time constant to reduce oscillations
            tau = avg_density * chimney.parts[0].height_m / (total_resistance * (mass_flow_steady + 1e-6))
            # Limit alpha to prevent overshooting
            alpha = min(dt / (tau + dt), 0.5)
            mass_flow_new = mass_flow_old + alpha * (mass_flow_steady - mass_flow_old)
        else:
            # No buoyancy = no flow! This is fundamental physics
            # Flow decays to zero with time constant
            tau_decay = avg_density * chimney.parts[0].height_m / (2 * total_resistance * (mass_flow_old + 1e-6))
            mass_flow_new = mass_flow_old * np.exp(-dt / tau_decay)
        
        # Ensure reasonable bounds (but allow zero flow!)
        mass_flow_new = np.clip(mass_flow_new, 0.0, 1.0)
        
        # Update pressure distribution
        new_pressures = np.zeros_like(pressures)
        new_pressures[0] = 101325  # Atmospheric at inlet
        
        for seg in range(1, chimney.total_segments):
            # Pressure change = buoyancy - friction loss
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            velocity = mass_flow_new / (densities[seg] * part.inner_area_m2)
            
            # Use cached friction factor
            f = friction_factors.get(seg, 0.02)  # Default if not found
            
            # Pressure drop from friction
            length = part.height_m / part.num_segments
            diameter = part.inner_diameter_m
            dp_friction = f * (length / diameter) * 0.5 * densities[seg] * velocity**2
            
            # Net pressure change = buoyancy - friction
            dp_segment = buoyancy[seg-1] - dp_friction
            new_pressures[seg] = new_pressures[seg-1] + dp_segment
        
        return mass_flow_new, new_pressures
    
    def _solve_energy_improved(self, chimney, step: int, temperatures: np.ndarray,
                              velocities: np.ndarray, mass_flow: float, dt: float) -> np.ndarray:
        """
        Improved energy solver that calls comprehensive heat transfer
        """
        new_temperatures = temperatures.copy()
        
        # Set inlet
        new_temperatures[0] = temperatures[0]  # Already set by boundary condition
        
        # For each segment, use comprehensive heat transfer
        for seg in range(chimney.total_segments):
            # Note: Don't skip segment 0 for wall temperature updates
            # The inlet gas temperature is fixed, but walls still need heating
            
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            segment_length = part.height_m / part.num_segments
            
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
            
            # Get upstream temperature for advection
            if seg == 0:
                T_upstream = temperatures[0]  # Inlet temperature
            else:
                T_upstream = temperatures[seg-1]  # Use current iteration's upstream temp
            
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
            
            if mass_flow > 1e-6 and seg > 0:  # Flowing case (skip inlet)
                # Energy in: mass_flow * cp * T_upstream
                # Energy out: mass_flow * cp * T_current + q_loss_to_walls + q_latent
                # Storage: gas_mass * cp * dT/dt
                
                # Heat loss to walls (positive when gas loses heat)
                q_loss_to_walls = h_inner * area * (temperatures[seg] - T_wall)
                
                # Advection terms
                # Use upwind differencing for stability
                residence_time = gas_mass / mass_flow  # Time gas spends in segment
                
                # Temperature change due to advection and heat loss
                # dT/dt = (T_in - T_out)/tau - Q_loss/(m*cp)
                # For steady flow: T_out ≈ T_in - Q_loss*tau/(m*cp)
                
                # Include condensation latent heat if present
                q_latent = ht_results.get('latent_heat_rate', 0)
                q_total_loss = q_loss_to_walls + q_latent
                
                # Stable formulation: new temperature based on upstream and heat loss
                T_steady = T_upstream - q_total_loss * residence_time / (gas_mass * cp)
                
                # Apply relaxation toward steady state
                tau_thermal = gas_mass * cp / (mass_flow * cp + h_inner * area)
                alpha = dt / (tau_thermal + dt)
                new_temperatures[seg] = temperatures[seg] + alpha * (T_steady - temperatures[seg])
            elif seg == 0:
                # Inlet segment - temperature is fixed
                new_temperatures[seg] = temperatures[seg]
            else:
                # No/low flow case or after flow case is handled
                # Gas temperature approaches wall temperature
                if mass_flow <= 1e-6:
                    tau = gas_mass * cp / (h_inner * area)
                    dT_dt = (T_wall - temperatures[seg]) / tau
                    new_temperatures[seg] = temperatures[seg] + dT_dt * dt
            
            # Enforce physical bounds for all segments except inlet
            if seg > 0:
                T_min_physical = part.T_ambient_K
                T_max_physical = 1073.15  # 800°C max
                
                # Apply bounds
                new_temperatures[seg] = max(new_temperatures[seg], T_min_physical)
                new_temperatures[seg] = min(new_temperatures[seg], T_max_physical)
            
        return new_temperatures