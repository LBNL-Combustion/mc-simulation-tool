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
from ._perf_counters import COUNTERS as _PC  # TEMP: profiling

@dataclass
class MomentumConfig:
    """Master configuration for momentum/pressure physics features."""
    distributed_draft_model: bool = False  # Master switch - enables all features
    use_distributed_leakage: bool = False
    use_accel_loss: bool = False
    friction_method: str = "current"       # "current" or "colebrook"
    N_leak_iter: int = 3
    leak_relaxation: float = 0.5
    C_accel: float = 1.0

    # "quasi_steady" (default) or "unsteady_momentum" (ODE-based)
    flow_model: str = "quasi_steady"

    # "current" (default) or "plugflow_ntu" (NTU/effectiveness per segment)
    gas_energy_model: str = "current"


@dataclass
class ConvergenceMetrics:
    """Per-timestep convergence metrics."""
    temperature_residual: float = 0.0
    flow_residual: float = 0.0
    pressure_residual: float = 0.0
    iterations: int = 0
    converged: bool = False

class CoupledFlowThermalSolver:
    """Iteratively coupled solver for flow and thermal equations (Picard iteration)."""
    
    def __init__(self, tolerance: float = 1e-3, max_iterations: int = 5,
                 relaxation_factor: float = 0.3,
                 alpha_cap: float = 0.5,
                 momentum_config: Optional[MomentumConfig] = None,
                 diagnostics: Optional[DiagnosticsRecorder] = None):
        """
        Args:
            tolerance: Convergence tolerance (relative change)
            max_iterations: Max coupling iterations per timestep
            relaxation_factor: Under-relaxation factor (0-1)
            alpha_cap: Max fraction of steady-state response per timestep (0-1)
            momentum_config: Momentum physics feature configuration
            diagnostics: Optional DiagnosticsRecorder
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.relaxation_factor = relaxation_factor
        self.alpha_cap = alpha_cap
        self.logger = logging.getLogger(__name__)

        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticsRecorder(enable=False)

        if momentum_config is None:
            self.momentum_config = MomentumConfig()
        else:
            self.momentum_config = momentum_config

        if self.momentum_config.distributed_draft_model:
            self.momentum_config.use_distributed_leakage = True
            self.momentum_config.use_accel_loss = True

        self.min_relaxation = 0.2
        self.max_relaxation = 0.5
        
    def calculate_residuals(self, current: Dict, previous: Dict) -> Tuple[float, float, float]:
        """Calculate normalized residuals between Picard iterations."""
        T_current = current['temperatures']
        T_previous = previous['temperatures']
        T_residual = np.linalg.norm(T_current - T_previous) / (np.linalg.norm(T_current) + 1e-10)

        mdot_current = current['mass_flow']
        mdot_previous = previous['mass_flow']
        flow_residual = abs(mdot_current - mdot_previous) / (abs(mdot_current) + 1e-10)

        P_current = current['pressures']
        P_previous = previous['pressures']
        P_residual = np.linalg.norm(P_current - P_previous) / (np.linalg.norm(P_current) + 1e-10)
        
        return T_residual, flow_residual, P_residual
    
    def apply_relaxation(self, new_value: np.ndarray, old_value: np.ndarray) -> np.ndarray:
        """Apply under-relaxation: relaxed = alpha * new + (1-alpha) * old."""
        return self.relaxation_factor * new_value + (1 - self.relaxation_factor) * old_value
    
    def solve_coupled_timestep(self, chimney, step: int, T_inlet: float, dt: float) -> ConvergenceMetrics:
        """Solve a single timestep with Picard flow-thermal coupling."""
        metrics = ConvergenceMetrics()
        
        # Sequential approach is more stable during startup transient
        if step < 1:
            self._solve_sequential(chimney, step, T_inlet)
            metrics.converged = True
            metrics.iterations = 1
            return metrics
        
        previous_state = {
            'temperatures': chimney.T_gas[step-1].copy(),
            'mass_flow': chimney.mass_flow_rate[step-1],
            'pressures': chimney.pressure[step-1].copy()
        }
        
        current_state = {
            'temperatures': previous_state['temperatures'].copy(),
            'mass_flow': previous_state['mass_flow'],
            'pressures': previous_state['pressures'].copy()
        }
        
        current_state['temperatures'][0] = T_inlet

        actual_max_iterations = self.max_iterations
        current_relaxation = self.relaxation_factor

        for iteration in range(actual_max_iterations):
            if iteration < actual_max_iterations - 1:
                iter_start_state = {
                    'temperatures': current_state['temperatures'].copy(),
                    'mass_flow': current_state['mass_flow'],
                    'pressures': current_state['pressures'].copy()
                }
            else:
                iter_start_state = current_state

            densities = self._update_densities(chimney, current_state['temperatures'])
            buoyancy_pressure = self._calculate_buoyancy(chimney, densities)

            new_mass_flow, new_pressures = self._solve_momentum_improved(
                chimney, current_state['pressures'], densities, 
                buoyancy_pressure, current_state['mass_flow'], dt
            )
            
            velocities = self._calculate_velocities(chimney, new_mass_flow, densities)

            new_temperatures = self._solve_energy_improved(
                chimney, step, current_state['temperatures'], 
                velocities, new_mass_flow, dt
            )
            
            # Adaptive relaxation
            if iteration > 2:
                T_res_prev = metrics.temperature_residual
                if T_res_prev > 0 and metrics.temperature_residual > T_res_prev:
                    current_relaxation = max(self.min_relaxation, current_relaxation * 0.8)
                else:
                    current_relaxation = min(self.max_relaxation, current_relaxation * 1.1)

            current_state['temperatures'] = (current_relaxation * new_temperatures + 
                                           (1 - current_relaxation) * current_state['temperatures'])
            current_state['mass_flow'] = (current_relaxation * new_mass_flow + 
                                        (1 - current_relaxation) * current_state['mass_flow'])
            current_state['pressures'] = (current_relaxation * new_pressures + 
                                        (1 - current_relaxation) * current_state['pressures'])
            
            T_res, flow_res, P_res = self.calculate_residuals(current_state, iter_start_state)
            max_residual = max(T_res, flow_res, P_res)

            metrics.temperature_residual = T_res
            metrics.flow_residual = flow_res
            metrics.pressure_residual = P_res
            metrics.iterations = iteration + 1
            
            T_max = np.max(current_state['temperatures'])
            if T_max > 1073.15:  # 800 degC - reduce relaxation to stabilize
                current_relaxation = max(0.1, current_relaxation * 0.5)
            
            if max_residual < self.tolerance:
                metrics.converged = True
                break
            
            if iteration > 5 and max_residual > 0.1:
                self.logger.debug(f"Accepting partial convergence at iteration {iteration+1}")
                break
        
        _PC['picard_iterations'] += metrics.iterations  # TEMP: profiling

        # ----------------------------------------------------------------
        # Reverse-flow ignition correction
        # ----------------------------------------------------------------
        # In 1D upwind advection, reverse flow (mdot<0) prevents hot inlet
        # gas from propagating upward.  The solver can lock into reverse
        # flow even when the fire's driving potential clearly exceeds the
        # opposing room depressurization.  This correction detects that
        # condition and re-solves with a positive initial guess so that
        # the forward-flow solution can be found.
        #
        # The driving potential is computed over the first chimney part
        # (stove-adjacent section) only, using T_inlet as the hypothetical
        # gas temperature.  This represents the available draft if the
        # fire's heat could establish in that section — which the 1D
        # reverse-flow advection cannot otherwise capture.
        # ----------------------------------------------------------------
        MARGIN_FORWARD_PA = 1.0

        if current_state['mass_flow'] < 0:
            T_amb_part0 = chimney.parts[0].T_ambient_K
            if T_inlet > T_amb_part0 + 20:
                dp_drive_pot = self._compute_fire_driving_potential_first_part(
                    chimney, T_inlet)
                dp_room = getattr(chimney, 'room_pressure_offset_Pa', 0.0)

                if dp_drive_pot > dp_room + MARGIN_FORWARD_PA:
                    # Seed first-part gas temperatures conservatively
                    n_part0 = chimney.parts[0].num_segments
                    for seg in range(n_part0):
                        T_local_amb = chimney.parts[0].T_ambient_K
                        T_seed = 0.8 * T_inlet + 0.2 * T_local_amb
                        current_state['temperatures'][seg] = max(
                            current_state['temperatures'][seg], T_seed)

                    # Positive mass-flow initial guess from driving potential
                    densities_new = self._update_densities(
                        chimney, current_state['temperatures'])
                    buoyancy_new = self._calculate_buoyancy(chimney, densities_new)

                    rho_inlet_new = densities_new[0]
                    R_door = self._calculate_inlet_restriction_resistance(
                        chimney, rho_inlet_new)
                    R_total_approx = R_door + 1e-3  # small floor
                    for seg in range(chimney.total_segments):
                        pidx, lseg, part = chimney.get_segment_info(seg)
                        R_total_approx += 0.02 * (chimney.segment_lengths[seg]
                            / part.inner_diameter_m) / (
                            2.0 * densities_new[seg] * part.inner_area_m2**2)

                    dp_net = max(dp_drive_pot - dp_room, 0.1)
                    mdot_guess = np.sqrt(dp_net / R_total_approx)
                    current_state['mass_flow'] = mdot_guess

                    # Re-run Picard with positive initial guess
                    for re_iter in range(self.max_iterations):
                        densities_new = self._update_densities(
                            chimney, current_state['temperatures'])
                        buoyancy_new = self._calculate_buoyancy(
                            chimney, densities_new)
                        new_mf, new_pr = self._solve_momentum_improved(
                            chimney, current_state['pressures'],
                            densities_new, buoyancy_new,
                            current_state['mass_flow'], dt)
                        vels = self._calculate_velocities(
                            chimney, new_mf, densities_new)
                        new_temps = self._solve_energy_improved(
                            chimney, step, current_state['temperatures'],
                            vels, new_mf, dt)

                        current_state['temperatures'] = (
                            current_relaxation * new_temps
                            + (1 - current_relaxation)
                            * current_state['temperatures'])
                        current_state['mass_flow'] = (
                            current_relaxation * new_mf
                            + (1 - current_relaxation)
                            * current_state['mass_flow'])
                        current_state['pressures'] = (
                            current_relaxation * new_pr
                            + (1 - current_relaxation)
                            * current_state['pressures'])

                        if abs(new_mf - current_state['mass_flow']) / (
                                abs(new_mf) + 1e-10) < self.tolerance:
                            break

                    densities = self._update_densities(
                        chimney, current_state['temperatures'])
                    velocities = self._calculate_velocities(
                        chimney, current_state['mass_flow'], densities)

        chimney.T_gas[step] = current_state['temperatures']
        chimney.mass_flow_rate[step] = current_state['mass_flow']
        chimney.pressure[step] = current_state['pressures']
        chimney.density[step] = densities
        chimney.velocity[step] = velocities

        return metrics
    
    def _compute_fire_driving_potential_first_part(self, chimney, T_inlet: float) -> float:
        """Driving potential if the first chimney part were filled with gas at T_inlet.

        This is NOT the actual buoyancy in the column — it is the driving
        pressure that the fire COULD produce in the stove-adjacent section
        if its heat propagated freely.  Used only during reverse-flow
        conditions to decide whether the fire is strong enough to
        overcome room depressurization and establish forward draft.

        Only the first part (typically the single-wall black pipe directly
        above the stove) is considered.  The rest of the chimney is
        assumed to remain at ambient temperature, contributing zero net
        buoyancy.

        Returns:
            Driving potential in Pa (positive = upward draft).
        """
        g = chimney.physics.GRAVITY
        part0 = chimney.parts[0]
        n_seg0 = part0.num_segments
        dp = 0.0
        for seg in range(n_seg0):
            rho_amb = chimney.physics.air_density(chimney.parts[-1].T_ambient_K, part0.Pa)
            rho_hot = chimney.physics.air_density(T_inlet, part0.Pa)
            dp += (rho_amb - rho_hot) * g * chimney.segment_vertical_heights[seg]
        return dp

    def _update_densities(self, chimney, temperatures: np.ndarray) -> np.ndarray:
        """Update gas densities from ideal gas law."""
        densities = np.zeros_like(temperatures)

        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            densities[seg] = chimney.physics.air_density(temperatures[seg], part.Pa)
        
        return densities
    
    def _calculate_buoyancy(self, chimney, densities: np.ndarray) -> np.ndarray:
        """Calculate buoyancy pressure difference per segment."""
        buoyancy = np.zeros_like(densities)

        ambient_density_cache = {}

        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            segment_height = chimney.segment_vertical_heights[seg]

            if part_idx not in ambient_density_cache:
                ambient_density_cache[part_idx] = chimney.physics.air_density(chimney.parts[-1].T_ambient_K, part.Pa)
            rho_ambient = ambient_density_cache[part_idx]

            buoyancy[seg] = chimney.physics.GRAVITY * segment_height * (rho_ambient - densities[seg])
        
        return buoyancy

    def _calculate_inlet_restriction_resistance(self, chimney, rho_inlet: float) -> float:
        """Quadratic resistance R_door [Pa*s^2/kg^2] from inlet restriction.

        R_door = K / (2 * rho * A_eff^2), added to total_resistance.
        """
        if not chimney.inlet_restriction.enable:
            return 0.0

        # A_eff includes leakage floor to keep resistance finite when door is closed
        A_eff = chimney.inlet_restriction.effective_area_m2
        K = chimney.inlet_restriction.K_equiv  # 1/Cd^2
        # the air going through the door gap is room air, not flue gas
        part0 = chimney.parts[0]
        rho_room = chimney.physics.air_density(part0.T_ambient_K, part0.Pa)
        R_door = K / (2.0 * rho_room * A_eff**2)

        # Stove body resistance (fuel bed, baffles, flue collar) uses pipe area
        R_stove = 0.0
        K_stove = getattr(chimney.inlet_restriction, 'K_stove', 0.0)
        if K_stove > 0:
            A_pipe = chimney.parts[0].inner_area_m2  # Use first part's area
            R_stove = K_stove / (2.0 * rho_inlet * A_pipe**2)

        return R_door + R_stove

    def _calculate_catalytic_bypass_loss(self, chimney, mass_flow_rate: float,
                                         densities: np.ndarray) -> float:
        """Pressure drop [Pa] from catalytic bypass damper (K-loss model)."""
        if not chimney.catalytic_bypass.enable:
            return 0.0

        if abs(mass_flow_rate) < 1e-10:
            return 0.0

        seg_idx = chimney.catalytic_bypass.segment_index
        part_idx, local_seg, part = chimney.get_segment_info(seg_idx)
        area = part.inner_area_m2
        rho = densities[seg_idx]
        velocity = abs(mass_flow_rate) / (rho * area)

        # dP = K * 0.5 * rho * v^2
        K = chimney.catalytic_bypass.K_effective
        dp_bypass = K * 0.5 * rho * velocity**2

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
        """Convert mass flow rate to velocity per segment."""
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
        """Sequential solve for early timesteps."""
        chimney._solve_comprehensive_heat_transfer(step, T_inlet)
        chimney._update_flow_and_pressure(step)
    
    def _solve_momentum_improved(self, chimney, pressures: np.ndarray, densities: np.ndarray,
                                buoyancy: np.ndarray, mass_flow_old: float, dt: float) -> Tuple[float, np.ndarray]:
        """Momentum solver: mdot_ss = sign(dp) * sqrt(|dp| / R_total)."""
        dp_drive = np.sum(buoyancy)

        Pa_outlet = chimney.parts[-1].Pa if len(chimney.parts) > 0 else 101325.0
        T_outlet = chimney.parts[-1].T_ambient_K if len(chimney.parts) > 0 else 294.15
        deltaP_wind = chimney.get_wind_cap_deltaP(chimney.time_step, Pa_outlet, T_outlet)
        dp_drive = dp_drive - deltaP_wind

        # Positive room_pressure_offset = room depressurized below outdoor, reduces draft
        room_depress = getattr(chimney, 'room_pressure_offset_Pa', 0.0)
        dp_drive = dp_drive - room_depress

        if dp_drive < 0 and abs(deltaP_wind) > 0.1:
            print(f"[WIND CAP WARNING] step={chimney.time_step} Net driving head negative "
                  f"(stack={np.sum(buoyancy):.1f} Pa + wind={-deltaP_wind:.1f} Pa < 0): backflow risk")

        segment_cache = {seg: chimney.get_segment_info(seg) for seg in range(chimney.total_segments)}

        # === TOTAL RESISTANCE (R_total = R_pipe + R_door + R_bypass) ===
        total_resistance = 0.0
        avg_density = np.mean(densities)
        rho_inlet = densities[0]

        friction_factors = {}
        viscosity_cache = {}

        # --- 1. Pipe segment resistance ---
        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = segment_cache[seg]
            area = part.inner_area_m2
            diameter = part.inner_diameter_m
            length = chimney.segment_lengths[seg]

            velocity = mass_flow_old / (densities[seg] * area) if abs(mass_flow_old) > 1e-10 else 0.1

            T_seg = chimney.T_gas[chimney.time_step-1, seg] if chimney.time_step > 0 else part.T_ambient_K + 10
            viscosity = chimney.physics.air_viscosity(T_seg)
            viscosity_cache[seg] = viscosity

            Re = densities[seg] * abs(velocity) * diameter / viscosity
            Re = max(Re, 100)  # floor to avoid extreme f values

            if Re < 2300:
                f = 64 / Re
            else:
                relative_roughness = part.roughness_m / diameter
                if relative_roughness < 1e-6:
                    f = 0.316 / Re**0.25  # Blasius (smooth)
                else:
                    f = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/Re**0.9))**2  # Swamee-Jain

            friction_factors[seg] = f

            K_minor = 0.0
            elbow_key = (part_idx, local_seg)
            if hasattr(chimney, 'segment_elbows') and elbow_key in chimney.segment_elbows:
                for elbow_spec in chimney.segment_elbows[elbow_key]:
                    K_elbow = base_K(elbow_spec) * elbow_spec.vane_factor
                    K_minor += K_elbow

            K_total = f * length / diameter + K_minor
            # Mass-flow quadratic resistance: dp = R * mdot^2
            # From dp = (fL/D + K) * 0.5 * rho * v^2  with v = mdot/(rho*A):
            #   dp = (fL/D + K) * mdot^2 / (2 * rho * A^2)
            R_seg = K_total / (2 * densities[seg] * area**2)
            total_resistance += R_seg

        # --- 2. Inlet restriction resistance ---
        R_door = self._calculate_inlet_restriction_resistance(chimney, rho_inlet)
        total_resistance += R_door

        # --- 3. Catalytic bypass resistance ---
        if abs(mass_flow_old) > 1e-10:
            dp_catalytic = self._calculate_catalytic_bypass_loss(chimney, mass_flow_old, densities)
            R_bypass = dp_catalytic / (mass_flow_old**2) if dp_catalytic > 0 else 0.0
            total_resistance += R_bypass

        # --- 3b. Exit velocity head loss ---
        K_exit = getattr(chimney, 'K_exit', 0.0)
        R_exit = 0.0
        if K_exit > 0:
            last_seg = chimney.total_segments - 1
            part_idx_exit, _, part_exit = segment_cache[last_seg]
            rho_exit = densities[last_seg]
            A_exit = part_exit.inner_area_m2
            R_exit = K_exit / (2.0 * rho_exit * A_exit**2)
            total_resistance += R_exit

        # --- 4. Distributed Leakage (VENT-II style) ---
        total_leakage_mdot = 0.0
        leakage_enabled = (
            self.momentum_config.use_distributed_leakage or
            (hasattr(chimney, 'leakage_config') and chimney.leakage_config.enable)
        )

        if leakage_enabled and hasattr(chimney, 'segment_leakage_area_m2'):
            N_leak_iter = self.momentum_config.N_leak_iter
            leak_relax = self.momentum_config.leak_relaxation

            for leak_iter in range(N_leak_iter):
                new_leakage_mdot = 0.0

                for seg in range(chimney.total_segments):
                    A_leak = chimney.segment_leakage_area_m2[seg]
                    if A_leak > 0:
                        part_idx, local_seg, part = segment_cache[seg]
                        P_gauge = pressures[seg] - part.Pa
                        rho_ambient = chimney.physics.air_density(part.T_ambient_K, part.Pa)
                        rho_gas = densities[seg]

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

                total_leakage_mdot = (leak_relax * new_leakage_mdot +
                                      (1 - leak_relax) * total_leakage_mdot)

        # --- 5. Acceleration loss (density decrease from heating) ---
        dp_accel_total = 0.0
        if self.momentum_config.use_accel_loss and abs(mass_flow_old) > 1e-10:
            C_accel = self.momentum_config.C_accel
            for seg in range(1, chimney.total_segments):
                part_idx, local_seg, part = segment_cache[seg]
                area = part.inner_area_m2
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

            dp_drive = dp_drive - dp_accel_total

        total_resistance = max(total_resistance, 1e-6)

        # === MOMENTUM BALANCE ===
        system_length = sum(chimney.segment_lengths)
        typical_area = chimney.parts[0].inner_area_m2

        if self.momentum_config.flow_model == "unsteady_momentum":
            # Semi-implicit Euler on L_eff/A * d(mdot)/dt = dp_drive - R*mdot*|mdot|
            # Always integrates from previous-timestep mdot (not Picard estimate)
            L_eff = system_length
            A_eff_mom = typical_area

            step_idx = chimney.time_step
            mdot_prev_timestep = chimney.mass_flow_rate[step_idx - 1] if step_idx > 0 else 0.0

            C_mom = dt * A_eff_mom / L_eff  # [m*s]
            numerator = mdot_prev_timestep + C_mom * dp_drive
            denominator = 1.0 + C_mom * total_resistance * abs(mdot_prev_timestep)
            mass_flow_new = numerator / denominator

        else:
            # Quasi-steady: algebraic mdot_ss + first-order lag
            mass_flow_magnitude = np.sqrt(abs(dp_drive) / total_resistance)
            mass_flow_steady = np.sign(dp_drive) * mass_flow_magnitude

            if abs(dp_drive) < 1e-10:
                mass_flow_steady = 0.0

            tau = avg_density * system_length / (total_resistance * (abs(mass_flow_steady) + 1e-6))
            tau = max(tau, dt)

            alpha = min(dt / (tau + dt), self.alpha_cap)
            mass_flow_new = mass_flow_old + alpha * (mass_flow_steady - mass_flow_old)

        mass_flow_new = mass_flow_new + total_leakage_mdot

        max_velocity = 50.0  # m/s
        max_mass_flow_magnitude = max_velocity * avg_density * typical_area

        mass_flow_new = np.clip(mass_flow_new, -max_mass_flow_magnitude, max_mass_flow_magnitude)

        new_pressures = np.zeros_like(pressures)
        # The gas pressure at the chimney inlet is below atmospheric by the inlet
        # orifice pressure drop.  Setting new_pressures[0] = Pa (atmospheric) gives
        # P_gauge > 0 in all segments (exfiltration everywhere), which is wrong.
        # With the correct boundary the lower segments are at negative P_gauge
        # (cold-air infiltration) and only upper segments cross into exfiltration,
        # matching the physical stack underpressure profile.
        dp_inlet_drop = R_door * (mass_flow_new ** 2) if abs(mass_flow_new) > 1e-10 else 0.0
        new_pressures[0] = chimney.parts[0].Pa - dp_inlet_drop
        
        for seg in range(1, chimney.total_segments):
            part_idx, local_seg, part = segment_cache[seg]
            velocity = mass_flow_new / (densities[seg] * part.inner_area_m2)
            
            length = chimney.segment_lengths[seg]
            diameter = part.inner_diameter_m
            viscosity = viscosity_cache.get(seg)
            if viscosity is None:
                T_seg = chimney.T_gas[chimney.time_step-1, seg] if chimney.time_step > 0 else part.T_ambient_K + 10
                viscosity = chimney.physics.air_viscosity(T_seg)
            Re_new = densities[seg] * abs(velocity) * diameter / viscosity
            Re_new = max(Re_new, 10)
            
            if Re_new < 2300:
                f = 64 / Re_new
            else:
                relative_roughness = part.roughness_m / diameter
                if relative_roughness < 1e-6:
                    f = 0.316 / Re_new**0.25
                else:
                    f = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/Re_new**0.9))**2
            
            dp_friction = f * (length / diameter) * 0.5 * densities[seg] * velocity**2

            dp_minor = 0.0
            elbow_key = (part_idx, local_seg)
            if hasattr(chimney, 'segment_elbows') and elbow_key in chimney.segment_elbows:
                for elbow_spec in chimney.segment_elbows[elbow_key]:
                    K_elbow = base_K(elbow_spec) * elbow_spec.vane_factor
                    dp_minor += K_elbow * 0.5 * densities[seg] * velocity**2

            dp_segment = buoyancy[seg] - dp_friction - dp_minor
            new_pressures[seg] = new_pressures[seg-1] + dp_segment

        if self.diagnostics.enable:
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

            dp_exit_diag = R_exit * mass_flow_new**2 if R_exit > 0 and abs(mass_flow_new) > 1e-10 else 0.0

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
                dp_exit=dp_exit_diag,
                dp_accel=dp_accel_total,
                dp_leakage=0.0,  # leakage modifies mdot, not dp directly
            )

        return mass_flow_new, new_pressures
    
    def _solve_energy_improved(self, chimney, step: int, temperatures: np.ndarray,
                              velocities: np.ndarray, mass_flow: float, dt: float) -> np.ndarray:
        """Energy solver using comprehensive heat transfer model."""
        new_temperatures = temperatures.copy()
        new_temperatures[0] = temperatures[0]

        segment_cache = {seg: chimney.get_segment_info(seg) for seg in range(chimney.total_segments)}

        for seg in range(chimney.total_segments):
            # Don't skip seg 0: gas T is fixed but walls still need heating
            part_idx, local_seg, part = segment_cache[seg]
            segment_length = chimney.segment_lengths[seg]

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
            
            x_from_part_start = (local_seg + 0.5) * segment_length

            segment_data = {
                'gas_temperature': temperatures[seg],
                'ambient_temperature': part.T_ambient_K,
                'pressure': chimney.pressure[step-1, seg] if step > 0 else part.Pa,
                'velocity': velocities[seg],
                'diameter': part.inner_diameter_m,
                'length': segment_length,
                'x_from_part_start': x_from_part_start,
                'orientation': part.orientation,
                'part_height': part.height_m,
                'has_fittings': part.has_fittings,
                'humidity_ratio': part.humidity_ratio,
                'wall_type': part.type,
                'segment_id': seg,
                'mass_flow_rate': mass_flow
            }

            if hasattr(chimney, 'thermal_extra_length') and chimney.thermal_extra_length[seg] > 0:
                segment_data['thermal_length'] = segment_length + chimney.thermal_extra_length[seg]
            if hasattr(chimney, 'thermal_extra_wall_Cj') and chimney.thermal_extra_wall_Cj[seg] > 0:
                segment_data['extra_wall_Cj'] = chimney.thermal_extra_wall_Cj[seg]

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
            
            # Upwind scheme for advection (handles both normal draft and backdraft)
            if mass_flow >= 0:
                if seg == 0:
                    T_upstream = temperatures[0]
                else:
                    T_upstream = temperatures[seg-1]
            else:
                if seg == chimney.total_segments - 1:
                    T_upstream = part.T_ambient_K
                else:
                    T_upstream = temperatures[seg+1]
                # Note: reverse-flow ignition correction is handled at the
                # timestep level (after Picard convergence), not here in the
                # energy solver advection scheme.
            
            ht_results = chimney.comprehensive_ht.calculate_segment_heat_transfer(segment_data, dt)

            if part.type == 'single':
                T_wall_new = ht_results.get('wall_temperature_new', part.T_ambient_K)
                if T_wall_new <= 0:
                    T_wall_new = part.T_ambient_K
                chimney.T_wall[step, seg] = T_wall_new
                chimney.T_outer[step, seg] = T_wall_new
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
            
            h_inner = ht_results['h_inner']
            h_outer = ht_results['h_outer']
            thermal_len = segment_data.get('thermal_length', segment_length)  # includes fitting contribution
            area = np.pi * part.inner_diameter_m * thermal_len
            
            chimney.h_conv_inner[step, seg] = h_inner
            chimney.h_conv_outer[step, seg] = h_outer

            if part.type == 'single':
                T_wall = T_wall_new
            else:  # triple
                T_wall = T_liner_new
            
            cp = chimney.physics.air_specific_heat(temperatures[seg])
            gas_volume = part.inner_area_m2 * segment_length
            gas_mass = chimney.density[step-1, seg] * gas_volume if step > 0 else part.Pa / (287 * temperatures[seg]) * gas_volume
            
            is_inlet_boundary = False
            if mass_flow >= 0:
                is_inlet_boundary = (seg == 0)
            else:
                is_inlet_boundary = (seg == chimney.total_segments - 1)

            if abs(mass_flow) > 1e-6 and not is_inlet_boundary:

                if self.momentum_config.gas_energy_model == "plugflow_ntu":
                    # NTU/effectiveness: T_out = T_in - eff * (T_in - T_wall)
                    UA = h_inner * area
                    mdot_cp = abs(mass_flow) * cp
                    NTU = UA / (mdot_cp + 1e-10)
                    effectiveness = 1.0 - np.exp(-NTU)

                    q_latent = ht_results.get('latent_heat_rate', 0)
                    T_out_ntu = T_upstream - effectiveness * (T_upstream - T_wall)

                    if q_latent != 0 and mdot_cp > 1e-6:
                        T_out_ntu -= q_latent / mdot_cp

                    new_temperatures[seg] = T_out_ntu

                else:
                    # Relaxation-based gas temperature update (default model)
                    q_loss_to_walls = h_inner * area * (temperatures[seg] - T_wall)
                    residence_time = gas_mass / abs(mass_flow)
                    q_latent = ht_results.get('latent_heat_rate', 0)
                    q_total_loss = q_loss_to_walls + q_latent

                    T_steady = T_upstream - q_total_loss * residence_time / (gas_mass * cp)

                    # Bound to [T_wall, T_upstream] range: prevents divergence when
                    # |mass_flow| is small near flow reversal (residence_time -> inf)
                    T_bound_lo = min(T_upstream, T_wall) - 1.0
                    T_bound_hi = max(T_upstream, T_wall) + 1.0
                    T_steady = max(T_steady, T_bound_lo)
                    T_steady = min(T_steady, T_bound_hi)

                    tau_thermal = gas_mass * cp / (abs(mass_flow) * cp + h_inner * area)
                    alpha = dt / (tau_thermal + dt)
                    new_temperatures[seg] = temperatures[seg] + alpha * (T_steady - temperatures[seg])
            elif is_inlet_boundary:
                if mass_flow >= 0:
                    new_temperatures[seg] = temperatures[seg]
                else:
                    new_temperatures[seg] = part.T_ambient_K
            else:
                # No/low flow: gas equilibrates with walls
                tau = gas_mass * cp / (h_inner * area + 1e-6)
                dT_dt = (T_wall - temperatures[seg]) / tau
                new_temperatures[seg] = temperatures[seg] + dT_dt * dt
            
            # Inlet boundary T is set by fire/appliance and may exceed these bounds
            if not is_inlet_boundary:
                T_min_physical = part.T_ambient_K
                T_max_physical = 1073.15  # 800 degC
                new_temperatures[seg] = max(new_temperatures[seg], T_min_physical)
                new_temperatures[seg] = min(new_temperatures[seg], T_max_physical)

        if self.diagnostics.enable:
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