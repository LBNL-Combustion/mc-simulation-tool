"""
Adaptive triple wall solver that uses sub-stepping only when needed
This avoids the need for tiny global timesteps
"""
import numpy as np
from typing import Dict, Tuple

class AdaptiveTripleWallSolver:
    """
    Solves triple wall heat transfer with adaptive sub-stepping
    for numerical stability without slowing down the entire simulation
    """
    
    def __init__(self):
        self.STEFAN_BOLTZMANN = 5.67e-8
        
    def solve_with_adaptive_dt(self, T_gas: float, T_ambient: float,
                              h_inner: float, h_outer: float,
                              T_liner_old: float, T_middle_old: float, T_outer_old: float,
                              geometry: Dict, properties: Dict, dt_global: float) -> Tuple[float, float, float]:
        """
        Solve triple wall with adaptive sub-stepping
        
        Args:
            dt_global: Global simulation timestep (can be large)
            
        Returns:
            Tuple of (T_liner_new, T_middle_new, T_outer_new)
        """
        # Calculate the required timestep for stability
        # Based on the thinnest layer with smallest thermal mass
        
        # Extract geometry
        r_gas = geometry['inner_diameter'] / 2
        length = geometry['length']
        
        # Calculate volumes for each layer
        r_1 = r_gas + properties['liner'].thickness
        r_2 = r_1 + properties['middle'].thickness
        r_3 = r_2 + properties['outer'].thickness
        
        V_1 = np.pi * length * (r_1**2 - r_gas**2)
        V_2 = np.pi * length * (r_2**2 - r_1**2)
        V_3 = np.pi * length * (r_3**2 - r_2**2)
        
        # Thermal masses
        C_1 = properties['liner'].density * properties['liner'].specific_heat * V_1
        C_2 = properties['middle'].density * properties['middle'].specific_heat * V_2
        C_3 = properties['outer'].density * properties['outer'].specific_heat * V_3
        
        # Estimate time constants based on conduction
        # tau = rho * cp * L^2 / k
        tau_1 = (properties['liner'].density * properties['liner'].specific_heat * 
                properties['liner'].thickness**2 / properties['liner'].thermal_conductivity)
        tau_2 = (properties['middle'].density * properties['middle'].specific_heat * 
                properties['middle'].thickness**2 / properties['middle'].thermal_conductivity)
        tau_3 = (properties['outer'].density * properties['outer'].specific_heat * 
                properties['outer'].thickness**2 / properties['outer'].thermal_conductivity)
        
        # Required timestep for stability (conservative)
        dt_required = 0.1 * min(tau_1, tau_2, tau_3)
        
        # If global timestep is small enough, use it directly
        if dt_global <= dt_required:
            return self._solve_single_step(
                T_gas, T_ambient, h_inner, h_outer,
                T_liner_old, T_middle_old, T_outer_old,
                geometry, properties, dt_global
            )
        
        # Otherwise, use sub-stepping
        n_substeps = int(np.ceil(dt_global / dt_required))
        dt_sub = dt_global / n_substeps
        
        # Initialize temperatures
        T_liner = T_liner_old
        T_middle = T_middle_old
        T_outer = T_outer_old
        
        # Sub-step through time
        for i in range(n_substeps):
            T_liner, T_middle, T_outer = self._solve_single_step(
                T_gas, T_ambient, h_inner, h_outer,
                T_liner, T_middle, T_outer,
                geometry, properties, dt_sub
            )
        
        return T_liner, T_middle, T_outer
    
    def _solve_single_step(self, T_gas: float, T_ambient: float,
                          h_inner: float, h_outer: float,
                          T_liner_old: float, T_middle_old: float, T_outer_old: float,
                          geometry: Dict, properties: Dict, dt: float) -> Tuple[float, float, float]:
        """
        Solve a single timestep using semi-implicit method
        """
        # Extract geometry
        r_gas = geometry['inner_diameter'] / 2
        length = geometry['length']
        
        r_1 = r_gas + properties['liner'].thickness
        r_2 = r_1 + properties['middle'].thickness
        r_3 = r_2 + properties['outer'].thickness
        
        # Surface areas
        A_gas = 2 * np.pi * r_gas * length
        A_1 = 2 * np.pi * r_1 * length
        A_2 = 2 * np.pi * r_2 * length
        A_3 = 2 * np.pi * r_3 * length
        
        # Volumes
        V_1 = np.pi * length * (r_1**2 - r_gas**2)
        V_2 = np.pi * length * (r_2**2 - r_1**2)
        V_3 = np.pi * length * (r_3**2 - r_2**2)
        
        # Thermal masses
        C_1 = properties['liner'].density * properties['liner'].specific_heat * V_1
        C_2 = properties['middle'].density * properties['middle'].specific_heat * V_2
        C_3 = properties['outer'].density * properties['outer'].specific_heat * V_3
        
        # Conduction resistances
        R_12 = np.log(r_2/r_1) / (2 * np.pi * properties['middle'].thermal_conductivity * length)
        R_23 = np.log(r_3/r_2) / (2 * np.pi * properties['outer'].thermal_conductivity * length)
        
        # Use semi-implicit update for each layer
        # This is stable for larger timesteps than explicit methods
        
        # Liner energy balance
        Q_gas_liner = h_inner * A_gas * (T_gas - T_liner_old)
        Q_liner_middle = (T_liner_old - T_middle_old) / R_12
        dT1_dt = (Q_gas_liner - Q_liner_middle) / C_1
        T_liner_new = T_liner_old + dT1_dt * dt
        
        # Middle layer energy balance
        Q_middle_from_liner = (T_liner_new - T_middle_old) / R_12  # Use new liner temp
        Q_middle_outer = (T_middle_old - T_outer_old) / R_23
        dT2_dt = (Q_middle_from_liner - Q_middle_outer) / C_2
        T_middle_new = T_middle_old + dT2_dt * dt
        
        # Outer layer energy balance
        Q_outer_from_middle = (T_middle_new - T_outer_old) / R_23  # Use new middle temp
        Q_outer_ambient = h_outer * A_3 * (T_outer_old - T_ambient)
        dT3_dt = (Q_outer_from_middle - Q_outer_ambient) / C_3
        T_outer_new = T_outer_old + dT3_dt * dt
        
        # Apply physical bounds
        T_liner_new = np.clip(T_liner_new, T_ambient, T_gas)
        T_middle_new = np.clip(T_middle_new, T_ambient, T_liner_new)
        T_outer_new = np.clip(T_outer_new, T_ambient, T_middle_new)
        
        return T_liner_new, T_middle_new, T_outer_new