"""
Stable triple-wall heat transfer solver with proper scaling and radiation
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass 
class WallLayerProperties:
    """Properties for a single wall layer"""
    thickness: float  # m
    thermal_conductivity: float  # W/(m·K)
    density: float  # kg/m³
    specific_heat: float  # J/(kg·K)
    emissivity: float = 0.0  # For radiation

class StableTripleWallSolver:
    """
    Numerically stable solver for triple-wall heat transfer
    
    Triple walls have 5 thermal nodes:
    1. Inner surface (gas side)
    2. Outer surface of inner wall
    3. Inner surface of middle layer
    4. Outer surface of middle layer  
    5. Outer surface (ambient side)
    
    The challenge: very different thermal resistances (metal vs insulation)
    can create ill-conditioned matrices. This solver uses scaling techniques
    to maintain numerical stability.
    """
    
    STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴) - for radiation calculations
    
    def __init__(self):
        pass
    
    def calculate_view_factors(self, r_inner: float, r_outer: float) -> Tuple[float, float]:
        """
        Calculate radiation view factors for concentric cylinders

        Args:
            r_inner: Inner radius (m)
            r_outer: Outer radius (m)

        Returns:
            Tuple of (F_inner_to_outer, F_outer_to_inner)
        """
        # For concentric cylinders, ALL radiation from inner surface hits outer
        F_inner_to_outer = 1.0
        
        # Reciprocity relation: A_i * F_ij = A_j * F_ji
        # Since A = 2πrL, ratio of areas = ratio of radii
        F_outer_to_inner = (r_inner / r_outer) * F_inner_to_outer
        
        return F_inner_to_outer, F_outer_to_inner
    
    def radiation_resistance(self, r_inner: float, r_outer: float,
                           emiss_inner: float, emiss_outer: float,
                           length: float) -> float:
        """
        Calculate radiation resistance between concentric cylinders

        Args:
            r_inner: Inner radius (m)
            r_outer: Outer radius (m)
            emiss_inner: Inner surface emissivity (0-1)
            emiss_outer: Outer surface emissivity (0-1)
            length: Cylinder length (m)

        Returns:
            Radiation resistance (K/W)
        """
        A_inner = 2 * np.pi * r_inner * length
        A_outer = 2 * np.pi * r_outer * length

        # Three resistances in series: emissivity(inner), geometric, emissivity(outer)
        R1 = (1 - emiss_inner) / (emiss_inner * A_inner)
        R2 = 1 / A_inner
        R3 = (1 - emiss_outer) / (emiss_outer * A_outer)
        
        R_rad = R1 + R2 + R3
        
        return R_rad
    
    def scale_system(self, K_matrix: np.ndarray, C_matrix: np.ndarray,
                    Q_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale system matrices to improve conditioning.

        Metal vs insulation conductances differ by ~1000x; scaling normalizes
        them to similar magnitudes for numerical stability.

        Args:
            K_matrix: Conductance matrix (W/K)
            C_matrix: Capacitance matrix (J/K)
            Q_vector: Heat source vector (W)

        Returns:
            Tuple of (scaled_K, scaled_C, scaled_Q, scaling_factors)
        """
        n = len(Q_vector)

        # Geometric mean of K and C diagonals balances conductance/capacitance scales
        scale_factors = np.ones(n)
        for i in range(n):
            if K_matrix[i,i] > 0 and C_matrix[i,i] > 0 and np.isfinite(K_matrix[i,i]):
                scale_factors[i] = np.sqrt(K_matrix[i,i] * C_matrix[i,i])
            elif C_matrix[i,i] > 0:
                scale_factors[i] = C_matrix[i,i]

        S = np.diag(1.0 / scale_factors)
        S_inv = np.diag(scale_factors)

        K_scaled = S @ K_matrix @ S_inv
        C_scaled = S @ C_matrix @ S_inv
        Q_scaled = S @ Q_vector
        
        return K_scaled, C_scaled, Q_scaled, scale_factors
    
    def solve_triple_wall_stable(self, T_gas: float, T_ambient: float,
                                h_inner: float, h_outer: float,
                                T_liner_old: float, T_middle_old: float, T_outer_old: float,
                                geometry: Dict, properties: Dict, dt: float) -> Tuple[float, float, float]:
        """
        Solve triple-wall heat transfer with stable numerical method
        
        Args:
            T_gas: Gas temperature (K)
            T_ambient: Ambient temperature (K)
            h_inner: Inner convection coefficient (W/(m²·K))
            h_outer: Outer convection coefficient (W/(m²·K))
            T_liner_old: Previous liner temperature (K)
            T_middle_old: Previous middle temperature (K)
            T_outer_old: Previous outer temperature (K)
            geometry: Geometry dictionary with radii and length
            properties: Properties dictionary for each layer
            dt: Time step (s)
            
        Returns:
            Tuple of (T_liner_new, T_middle_new, T_outer_new)
        """
        r_gas = geometry['inner_diameter'] / 2
        r_1 = r_gas + properties['liner'].thickness
        r_2 = r_1 + properties['middle'].thickness  
        r_3 = r_2 + properties['outer'].thickness
        length = geometry['length']
        
        if r_gas <= 0 or length <= 0:
            print(f"ERROR: Invalid geometry - r_gas={r_gas}, length={length}")
            raise ValueError("Invalid geometry")
        
        if r_1 <= r_gas or r_2 <= r_1 or r_3 <= r_2:
            print(f"ERROR: Invalid radii - r_gas={r_gas}, r_1={r_1}, r_2={r_2}, r_3={r_3}")
            raise ValueError("Invalid radii progression")
        
        # Surface areas
        A_gas = 2 * np.pi * r_gas * length
        A_1 = 2 * np.pi * r_1 * length
        A_2 = 2 * np.pi * r_2 * length
        A_3 = 2 * np.pi * r_3 * length
        
        # Conduction resistances (cylindrical)
        k_1 = properties['liner'].thermal_conductivity
        k_2 = properties['middle'].thermal_conductivity
        k_3 = properties['outer'].thermal_conductivity
        
        if k_1 <= 0 or k_2 <= 0 or k_3 <= 0:
            print(f"ERROR: Invalid thermal conductivities - k_1={k_1}, k_2={k_2}, k_3={k_3}")
            raise ValueError("Thermal conductivity must be positive")
        
        R_cond_1 = np.log(r_1 / r_gas) / (2 * np.pi * k_1 * length)
        R_cond_2 = np.log(r_2 / r_1) / (2 * np.pi * k_2 * length)
        R_cond_3 = np.log(r_3 / r_2) / (2 * np.pi * k_3 * length)
        
        # Convection resistances
        if h_inner <= 0:
            print(f"WARNING: h_inner = {h_inner} <= 0, using minimum value")
            h_inner = 1.0  # Minimum value for natural convection
        if h_outer <= 0:
            print(f"WARNING: h_outer = {h_outer} <= 0, using minimum value")
            h_outer = 1.0  # Minimum value for natural convection
            
        R_conv_inner = 1 / (h_inner * A_gas)
        R_conv_outer = 1 / (h_outer * A_3)

        # Series: convection + liner conduction
        R_gas_to_liner = R_conv_inner + R_cond_1

        # Series: outer conduction + convection
        R_middle_to_ambient = R_cond_3 + R_conv_outer

        # Radiation resistances between layers
        if hasattr(properties['liner'], 'emissivity') and hasattr(properties['middle'], 'emissivity'):
            R_rad_1_2 = self.radiation_resistance(r_1, r_2, 
                                                 properties['liner'].emissivity,
                                                 properties['middle'].emissivity, length)
        else:
            R_rad_1_2 = float('inf')  # No radiation
            
        if hasattr(properties['middle'], 'emissivity') and hasattr(properties['outer'], 'emissivity'):
            R_rad_2_3 = self.radiation_resistance(r_2, r_3,
                                                 properties['middle'].emissivity,
                                                 properties['outer'].emissivity, length)
        else:
            R_rad_2_3 = float('inf')  # No radiation
        
        # Thermal capacitances
        V_1 = np.pi * length * (r_1**2 - r_gas**2)
        V_2 = np.pi * length * (r_2**2 - r_1**2)
        V_3 = np.pi * length * (r_3**2 - r_2**2)
        
        C_1 = properties['liner'].density * properties['liner'].specific_heat * V_1
        C_2 = properties['middle'].density * properties['middle'].specific_heat * V_2
        C_3 = properties['outer'].density * properties['outer'].specific_heat * V_3
        # Add absorbed fitting steel mass to innermost wall node
        C_1 += geometry.get('extra_wall_Cj', 0.0)

        # Conductance matrix [T_liner, T_middle, T_outer]
        K = np.zeros((3, 3))
        
        eps = 1e-10  # regularization to prevent singular matrices

        # CRITICAL: Convection + liner conduction are in SERIES for gas-to-liner path
        # G_total = 1/(R1 + R2), NOT G1 + G2
        G_gas_to_liner = 1/(R_gas_to_liner + eps)  # Combined: gas convection + liner conduction
        G_cond_2 = 1/(R_cond_2 + eps)              # Insulation conduction (liner to middle)
        G_cond_3 = 1/(R_cond_3 + eps)              # Outer wall conduction (middle to outer)
        G_conv_outer = 1/(R_conv_outer + eps)      # Outer convection (outer to ambient)
        G_rad_1_2 = 1/(R_rad_1_2 + eps) if R_rad_1_2 < float('inf') else 0
        G_rad_2_3 = 1/(R_rad_2_3 + eps) if R_rad_2_3 < float('inf') else 0

        # Liner node
        K[0, 0] = G_gas_to_liner + G_cond_2 + G_rad_1_2
        K[0, 1] = -(G_cond_2 + G_rad_1_2)
        K[0, 2] = 0

        # Middle node
        K[1, 0] = -(G_cond_2 + G_rad_1_2)
        K[1, 1] = G_cond_2 + G_rad_1_2 + G_cond_3 + G_rad_2_3
        K[1, 2] = -(G_cond_3 + G_rad_2_3)

        # Outer node
        K[2, 0] = 0
        K[2, 1] = -(G_cond_3 + G_rad_2_3)
        K[2, 2] = G_cond_3 + G_rad_2_3 + G_conv_outer

        C = np.diag([C_1, C_2, C_3])

        # Source vector (external boundary terms only)
        Q = np.array([
            G_gas_to_liner * T_gas,
            0,
            G_conv_outer * T_ambient
        ])
        
        T_old = np.array([T_liner_old, T_middle_old, T_outer_old])

        K_scaled, C_scaled, Q_scaled, scale_factors = self.scale_system(K, C, Q)
        T_old_scaled = T_old / scale_factors
        
        # Time integration using implicit Euler (unconditionally stable)
        # (C/dt + K) * T_new = C/dt * T_old + Q
        
        A_matrix = C_scaled / dt + K_scaled
        b_vector = C_scaled @ T_old_scaled / dt + Q_scaled
        
        try:
            if np.any(np.isinf(A_matrix)) or np.any(np.isnan(A_matrix)):
                print("\nERROR: Matrix contains inf/nan values!")
                print("\nDiagnosing the problem...")

                print(f"\nInputs:")
                print(f"  T_gas = {T_gas:.1f} K ({T_gas-273.15:.1f}°C)")
                print(f"  T_ambient = {T_ambient:.1f} K ({T_ambient-273.15:.1f}°C)")
                print(f"  h_inner = {h_inner:.2f} W/(m²·K)")
                print(f"  h_outer = {h_outer:.2f} W/(m²·K)")
                
                print(f"\nResistances:")
                print(f"  R_conv_inner = {R_conv_inner:.6f}")
                print(f"  R_conv_outer = {R_conv_outer:.6f}")
                print(f"  R_cond_1 = {R_cond_1:.6f}")
                print(f"  R_cond_2 = {R_cond_2:.6f}")
                print(f"  R_cond_3 = {R_cond_3:.6f}")
                print(f"  R_rad_1_2 = {R_rad_1_2}")
                print(f"  R_rad_2_3 = {R_rad_2_3}")
                
                print(f"\nConductances (G = 1/R):")
                print(f"  G_conv_inner = {G_conv_inner:.6f}")
                print(f"  G_conv_outer = {G_conv_outer:.6f}")
                print(f"  G_cond_1 = {G_cond_1:.6f}")
                print(f"  G_cond_2 = {G_cond_2:.6f}")
                print(f"  G_cond_3 = {G_cond_3:.6f}")
                print(f"  G_rad_1_2 = {G_rad_1_2:.6f}")
                print(f"  G_rad_2_3 = {G_rad_2_3:.6f}")
                
                print(f"\nCapacitances:")
                print(f"  C_1 (liner) = {C_1:.2f} J/K")
                print(f"  C_2 (middle) = {C_2:.2f} J/K")
                print(f"  C_3 (outer) = {C_3:.2f} J/K")
                
                print(f"\nTime constants:")
                tau_1 = C_1 / (K[0,0] + 1e-10) if K[0,0] > 0 else float('inf')
                tau_2 = C_2 / (K[1,1] + 1e-10) if K[1,1] > 0 else float('inf')
                tau_3 = C_3 / (K[2,2] + 1e-10) if K[2,2] > 0 else float('inf')
                print(f"  tau_1 = {tau_1:.3f} s")
                print(f"  tau_2 = {tau_2:.3f} s")
                print(f"  tau_3 = {tau_3:.3f} s")
                print(f"  dt = {dt} s")
                print(f"  dt/tau_min = {dt/min(tau_1, tau_2, tau_3):.2f}")
                
                print(f"\nK matrix diagonal:")
                print(f"  K[0,0] = {K[0,0]}")
                print(f"  K[1,1] = {K[1,1]}")
                print(f"  K[2,2] = {K[2,2]}")
                
                print(f"\nA_matrix info:")
                print(f"  Contains inf: {np.any(np.isinf(A_matrix))}")
                print(f"  Contains nan: {np.any(np.isnan(A_matrix))}")
                print(f"  Max value: {np.nanmax(A_matrix[np.isfinite(A_matrix)]) if np.any(np.isfinite(A_matrix)) else 'All inf/nan'}")
                
                raise np.linalg.LinAlgError("Matrix contains inf/nan")
            
            try:
                cond = np.linalg.cond(A_matrix)
                if cond > 1e10:
                    print(f"Warning: Poor matrix conditioning = {cond:.2e}")
                    diag_C = np.diag(C)
                    diag_K = np.diag(K)
                    print(f"  Capacitances: {diag_C}")
                    print(f"  Conductances: {diag_K}")
                    finite_K = diag_K[np.isfinite(diag_K)]
                    if len(finite_K) > 0:
                        print(f"  dt = {dt}, min(C)/max(finite_K) = {np.min(diag_C)/np.max(finite_K):.3e}")
            except:
                pass
            
            T_new_scaled = np.linalg.solve(A_matrix, b_vector)
            T_new = T_new_scaled * scale_factors
            
        except np.linalg.LinAlgError:
            print("Warning: Matrix singular, using stabilized explicit update")
            
            # Replace inf (disabled radiation) with zero for explicit fallback
            K_finite = K.copy()
            K_finite[np.isinf(K_finite)] = 0.0
            K_finite[np.isnan(K_finite)] = 0.0
            
            C_diag = np.diag(C)
            dT_dt = np.zeros(3)
            for i in range(3):
                if C_diag[i] > 0:
                    net_heat = 0

                    if i == 0:  # Liner gets heat from gas
                        net_heat += G_conv_inner * (T_gas - T_old[0])
                    elif i == 2:  # Outer loses heat to ambient
                        net_heat += G_conv_outer * (T_ambient - T_old[2])
                    
                    for j in range(3):
                        if i != j and K_finite[i,j] != 0:
                            net_heat += K_finite[i,j] * (T_old[j] - T_old[i])
                    
                    dT_dt[i] = net_heat / C_diag[i]
            
            max_dT = 10.0  # Maximum 10K change per timestep
            dT_dt = np.clip(dT_dt, -max_dT/dt, max_dT/dt)
            T_new = T_old + dT_dt * dt
        
        # Physical bounds: passive system, monotonic T_gas >= T_liner >= T_middle >= T_outer >= T_ambient
        T_new[0] = np.clip(T_new[0], T_ambient, max(T_gas, T_ambient))  # Liner
        T_new[1] = np.clip(T_new[1], T_ambient, max(T_new[0], T_ambient))  # Middle
        T_new[2] = np.clip(T_new[2], T_ambient, max(T_new[1], T_ambient))  # Outer
        
        return T_new[0], T_new[1], T_new[2]
    
    # def linearize_radiation(self, T_ref: float, emissivity: float) -> float:
    #     """
    #     Linearize radiation heat transfer coefficient around reference temperature
        
    #     Args:
    #         T_ref: Reference temperature for linearization (K)
    #         emissivity: Surface emissivity
            
    #     Returns:
    #         Linearized radiation coefficient (W/(m²·K))
    #     """
    #     return 4 * emissivity * self.STEFAN_BOLTZMANN * T_ref**3