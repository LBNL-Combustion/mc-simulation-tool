"""
Adaptive triple wall solver that uses sub-stepping only when needed.
This avoids the need for tiny global timesteps.

Inter-layer radiation is included by default (consistent with the stable
solver). A linearized radiation conductance is used so that the explicit
Gauss-Seidel sub-stepping remains stable.

Config flag: include_interlayer_radiation (default True).
"""
import numpy as np
from math import pi, log, ceil
from typing import Dict, Tuple, Optional

STEFAN_BOLTZMANN = 5.67e-8  # W/(m^2 K^4)


class AdaptiveTripleWallSolver:
    """
    Solves triple wall heat transfer with adaptive sub-stepping
    for numerical stability without slowing down the entire simulation.

    Now includes inter-layer radiation between concentric cylinders,
    matching the physics in StableTripleWallSolver.
    """

    def __init__(self, include_interlayer_radiation: bool = True):
        """
        Args:
            include_interlayer_radiation: If True (default), include radiation
                between liner-middle and middle-outer layers. If False,
                reverts to conduction-only (legacy behavior).
        """
        self.include_interlayer_radiation = include_interlayer_radiation
        # Cache for precomputed geometry constants
        self._geometry_cache: Dict[int, Dict] = {}

    def _get_geometry_constants(self, geometry: Dict, properties: Dict) -> Dict:
        """
        Get or compute cached geometry constants for a segment.
        These values are constant for a given segment and don't change
        during simulation.
        """
        # Create a cache key from geometry, material properties, and radiation flag
        cache_key = (
            geometry['inner_diameter'],
            geometry['length'],
            properties['liner'].thickness,
            properties['middle'].thickness,
            properties['outer'].thickness,
            properties['liner'].density,
            properties['middle'].density,
            properties['outer'].density,
            self.include_interlayer_radiation,
        )
        cache_key_hash = hash(cache_key)

        if cache_key_hash in self._geometry_cache:
            return self._geometry_cache[cache_key_hash]

        # Compute all geometry constants once
        r_gas = geometry['inner_diameter'] / 2
        length = geometry['length']

        r_1 = r_gas + properties['liner'].thickness
        r_2 = r_1 + properties['middle'].thickness
        r_3 = r_2 + properties['outer'].thickness

        # Surface areas
        A_gas = 2 * pi * r_gas * length
        A_1 = 2 * pi * r_1 * length
        A_2 = 2 * pi * r_2 * length
        A_3 = 2 * pi * r_3 * length

        # Volumes
        V_1 = pi * length * (r_1**2 - r_gas**2)
        V_2 = pi * length * (r_2**2 - r_1**2)
        V_3 = pi * length * (r_3**2 - r_2**2)

        # Thermal masses
        C_1 = properties['liner'].density * properties['liner'].specific_heat * V_1
        C_2 = properties['middle'].density * properties['middle'].specific_heat * V_2
        C_3 = properties['outer'].density * properties['outer'].specific_heat * V_3

        # Conduction resistances (cylindrical)
        R_12 = log(r_2 / r_1) / (2 * pi * properties['middle'].thermal_conductivity * length)
        R_23 = log(r_3 / r_2) / (2 * pi * properties['outer'].thermal_conductivity * length)

        # --- Inter-layer radiation resistances ---
        # Using the concentric-cylinder network:
        #   R_rad = (1-eps_i)/(eps_i*A_i) + 1/A_i + (1-eps_o)/(eps_o*A_o)
        # for the pair (r_1, r_2) and (r_2, r_3).
        R_rad_12 = float('inf')
        R_rad_23 = float('inf')

        if self.include_interlayer_radiation:
            eps_liner = getattr(properties['liner'], 'emissivity', 0.0)
            eps_middle = getattr(properties['middle'], 'emissivity', 0.9)
            eps_outer = getattr(properties['outer'], 'emissivity', 0.0)

            if eps_liner > 0 and eps_middle > 0:
                R_rad_12 = (
                    (1.0 - eps_liner) / (eps_liner * A_1 + 1e-30)
                    + 1.0 / (A_1 + 1e-30)
                    + (1.0 - eps_middle) / (eps_middle * A_2 + 1e-30)
                )
            if eps_middle > 0 and eps_outer > 0:
                R_rad_23 = (
                    (1.0 - eps_middle) / (eps_middle * A_2 + 1e-30)
                    + 1.0 / (A_2 + 1e-30)
                    + (1.0 - eps_outer) / (eps_outer * A_3 + 1e-30)
                )

        # Time constants for stability calculation
        tau_1 = (properties['liner'].density * properties['liner'].specific_heat *
                 properties['liner'].thickness**2 / properties['liner'].thermal_conductivity)
        tau_2 = (properties['middle'].density * properties['middle'].specific_heat *
                 properties['middle'].thickness**2 / properties['middle'].thermal_conductivity)
        tau_3 = (properties['outer'].density * properties['outer'].specific_heat *
                 properties['outer'].thickness**2 / properties['outer'].thermal_conductivity)

        # Required timestep for stability (semi-implicit allows larger factor than 0.1)
        dt_required = 0.25 * min(tau_1, tau_2, tau_3)

        constants = {
            'A_gas': A_gas,
            'A_3': A_3,
            'C_1': C_1,
            'C_2': C_2,
            'C_3': C_3,
            'R_12': R_12,
            'R_23': R_23,
            'R_rad_12': R_rad_12,
            'R_rad_23': R_rad_23,
            'dt_required': dt_required,
        }

        self._geometry_cache[cache_key_hash] = constants
        return constants

    def solve_with_adaptive_dt(self, T_gas: float, T_ambient: float,
                               h_inner: float, h_outer: float,
                               T_liner_old: float, T_middle_old: float, T_outer_old: float,
                               geometry: Dict, properties: Dict, dt_global: float) -> Tuple[float, float, float]:
        """
        Solve triple wall with adaptive sub-stepping.

        Args:
            dt_global: Global simulation timestep (can be large)

        Returns:
            Tuple of (T_liner_new, T_middle_new, T_outer_new)
        """
        # Get precomputed constants
        consts = self._get_geometry_constants(geometry, properties)
        dt_required = consts['dt_required']

        # If global timestep is small enough, use it directly
        if dt_global <= dt_required:
            return self._solve_single_step_fast(
                T_gas, T_ambient, h_inner, h_outer,
                T_liner_old, T_middle_old, T_outer_old,
                consts, dt_global
            )

        # Otherwise, use sub-stepping
        n_substeps = int(ceil(dt_global / dt_required))
        dt_sub = dt_global / n_substeps

        # Initialize temperatures
        T_liner = T_liner_old
        T_middle = T_middle_old
        T_outer = T_outer_old

        # Sub-step through time
        for _ in range(n_substeps):
            T_liner, T_middle, T_outer = self._solve_single_step_fast(
                T_gas, T_ambient, h_inner, h_outer,
                T_liner, T_middle, T_outer,
                consts, dt_sub
            )

        return T_liner, T_middle, T_outer

    def _solve_single_step_fast(self, T_gas: float, T_ambient: float,
                                h_inner: float, h_outer: float,
                                T_liner_old: float, T_middle_old: float, T_outer_old: float,
                                consts: Dict, dt: float) -> Tuple[float, float, float]:
        """
        Fast single timestep solver using precomputed constants.
        Includes inter-layer radiation via linearized conductance.
        Uses Python min/max instead of np.clip for scalar values.
        """
        # Extract precomputed constants
        A_gas = consts['A_gas']
        A_3 = consts['A_3']
        C_1 = consts['C_1']
        C_2 = consts['C_2']
        C_3 = consts['C_3']
        R_12 = consts['R_12']
        R_23 = consts['R_23']
        R_rad_12 = consts['R_rad_12']
        R_rad_23 = consts['R_rad_23']

        # --- Linearized radiation conductance ---
        # Q_rad = sigma * (T_hot^4 - T_cold^4) / R_rad
        #       ~ 4*sigma*T_avg^3 * (T_hot - T_cold) / R_rad
        # G_rad = 4*sigma*T_avg^3 / R_rad  [W/K]
        # This linearization is stable for explicit stepping.
        G_rad_12 = 0.0
        G_rad_23 = 0.0

        if R_rad_12 < 1e30:
            T_avg_12 = 0.5 * (T_liner_old + T_middle_old)
            G_rad_12 = 4.0 * STEFAN_BOLTZMANN * T_avg_12**3 / R_rad_12

        if R_rad_23 < 1e30:
            T_avg_23 = 0.5 * (T_middle_old + T_outer_old)
            G_rad_23 = 4.0 * STEFAN_BOLTZMANN * T_avg_23**3 / R_rad_23

        # Liner energy balance
        Q_gas_liner = h_inner * A_gas * (T_gas - T_liner_old)
        Q_liner_middle_cond = (T_liner_old - T_middle_old) / R_12
        Q_liner_middle_rad = G_rad_12 * (T_liner_old - T_middle_old)
        Q_liner_middle = Q_liner_middle_cond + Q_liner_middle_rad
        dT1_dt = (Q_gas_liner - Q_liner_middle) / C_1
        T_liner_new = T_liner_old + dT1_dt * dt

        # Middle layer energy balance (Gauss-Seidel: use new liner temp)
        Q_middle_from_liner_cond = (T_liner_new - T_middle_old) / R_12
        Q_middle_from_liner_rad = G_rad_12 * (T_liner_new - T_middle_old)
        Q_middle_from_liner = Q_middle_from_liner_cond + Q_middle_from_liner_rad
        Q_middle_outer_cond = (T_middle_old - T_outer_old) / R_23
        Q_middle_outer_rad = G_rad_23 * (T_middle_old - T_outer_old)
        Q_middle_outer = Q_middle_outer_cond + Q_middle_outer_rad
        dT2_dt = (Q_middle_from_liner - Q_middle_outer) / C_2
        T_middle_new = T_middle_old + dT2_dt * dt

        # Outer layer energy balance (Gauss-Seidel: use new middle temp)
        Q_outer_from_middle_cond = (T_middle_new - T_outer_old) / R_23
        Q_outer_from_middle_rad = G_rad_23 * (T_middle_new - T_outer_old)
        Q_outer_from_middle = Q_outer_from_middle_cond + Q_outer_from_middle_rad
        Q_outer_ambient = h_outer * A_3 * (T_outer_old - T_ambient)
        dT3_dt = (Q_outer_from_middle - Q_outer_ambient) / C_3
        T_outer_new = T_outer_old + dT3_dt * dt

        # Apply physical bounds using Python min/max (faster than np.clip for scalars)
        T_liner_new = max(T_ambient, min(T_gas, T_liner_new))
        T_middle_new = max(T_ambient, min(T_liner_new, T_middle_new))
        T_outer_new = max(T_ambient, min(T_middle_new, T_outer_new))

        return T_liner_new, T_middle_new, T_outer_new

    def _solve_single_step(self, T_gas: float, T_ambient: float,
                           h_inner: float, h_outer: float,
                           T_liner_old: float, T_middle_old: float, T_outer_old: float,
                           geometry: Dict, properties: Dict, dt: float) -> Tuple[float, float, float]:
        """
        Legacy single timestep solver (kept for compatibility).
        Use _solve_single_step_fast with precomputed constants for better performance.
        """
        consts = self._get_geometry_constants(geometry, properties)
        return self._solve_single_step_fast(
            T_gas, T_ambient, h_inner, h_outer,
            T_liner_old, T_middle_old, T_outer_old,
            consts, dt
        )
