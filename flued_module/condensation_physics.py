"""
Physics-based condensation model with proper mass balance
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CondensationState:
    """State variables for condensation tracking"""
    film_thickness: float = 0.0  # m
    film_mass: float = 0.0  # kg
    condensation_rate: float = 0.0  # kg/s
    evaporation_rate: float = 0.0  # kg/s
    net_rate: float = 0.0  # kg/s
    latent_heat_rate: float = 0.0  # W

class CondensationPhysics:
    """
    Physics-based condensation model that ensures mass balance
    Uses mole fraction and saturation pressure approach
    """
    
    # Physical constants
    R_UNIVERSAL = 8314.0  # J/(kmol·K)
    R_WATER = 461.5  # J/(kg·K) - specific gas constant for water vapor
    R_AIR = 287.0  # J/(kg·K) - specific gas constant for dry air
    MW_WATER = 18.015  # kg/kmol - molecular weight of water
    MW_AIR = 28.97  # kg/kmol - molecular weight of dry air
    H_FG_REF = 2.501e6  # J/kg - latent heat at 0°C
    CP_WATER_VAPOR = 1859.0  # J/(kg·K) - specific heat of water vapor
    CP_LIQUID_WATER = 4186.0  # J/(kg·K) - specific heat of liquid water
    RHO_LIQUID_WATER = 1000.0  # kg/m³ - density of liquid water
    
    def __init__(self):
        """Initialize condensation physics calculator"""
        pass
    
    def saturation_pressure(self, T_K: float) -> float:
        """
        Calculate saturation pressure of water vapor using accurate correlation
        
        Args:
            T_K: Temperature in Kelvin
            
        Returns:
            Saturation pressure in Pa
        """
        T_C = T_K - 273.15
        
        # Bound temperature to prevent overflow
        if T_C < -50:  # Below -50°C, vapor pressure is negligible
            return 1e-10  # Essentially zero
        elif T_C < 0:
            # Ice - use Tetens formula with bounds to prevent overflow
            # Limit the exponent to prevent overflow
            exponent = min(22.452 * T_C / (272.55 + T_C), 20.0)  # exp(20) is safe
            return 611.657 * np.exp(exponent)
        elif T_C < 100:
            # Water - use Buck equation (accurate for 0-100°C)
            return 611.21 * np.exp((18.678 - T_C/234.5) * T_C / (257.14 + T_C))
        else:
            # Steam - use Antoine equation
            log10_p_kPa = 3.55959 - 643.748 / (T_C - 198.043)
            return 10**(log10_p_kPa) * 1000  # Convert kPa to Pa
    
    def latent_heat(self, T_K: float) -> float:
        """
        Calculate latent heat of vaporization as function of temperature
        
        Args:
            T_K: Temperature in Kelvin
            
        Returns:
            Latent heat in J/kg
        """
        T_C = T_K - 273.15
        # Watson correlation
        if T_C < 0:
            return self.H_FG_REF  # Use reference value for ice
        else:
            # Temperature-dependent latent heat
            return self.H_FG_REF + (self.CP_WATER_VAPOR - self.CP_LIQUID_WATER) * T_C
    
    def mole_fraction_to_humidity_ratio(self, x_water: float) -> float:
        """
        Convert water vapor mole fraction to humidity ratio
        
        Args:
            x_water: Mole fraction of water vapor
            
        Returns:
            Humidity ratio (kg water / kg dry air)
        """
        if x_water >= 1.0:
            return float('inf')
        return x_water * self.MW_WATER / ((1 - x_water) * self.MW_AIR)
    
    def humidity_ratio_to_mole_fraction(self, W: float) -> float:
        """
        Convert humidity ratio to water vapor mole fraction
        
        Args:
            W: Humidity ratio (kg water / kg dry air)
            
        Returns:
            Mole fraction of water vapor
        """
        return W * self.MW_AIR / (W * self.MW_AIR + self.MW_WATER)
    
    def mass_transfer_coefficient(self, h_heat: float, T_gas: float, P_total: float) -> float:
        """
        Calculate mass transfer coefficient using heat-mass transfer analogy
        
        Args:
            h_heat: Heat transfer coefficient (W/(m²·K))
            T_gas: Gas temperature (K)
            P_total: Total pressure (Pa)
            
        Returns:
            Mass transfer coefficient (kg/(m²·s))
        """
        # Lewis number for air-water vapor ≈ 1
        Lewis = 1.0
        
        # Mass transfer coefficient from Chilton-Colburn analogy
        h_mass = h_heat / (self.CP_WATER_VAPOR * Lewis**(2/3))
        
        # Convert to proper units
        return h_mass * self.MW_WATER / (self.R_UNIVERSAL * T_gas)
    
    def condensation_rate(self, T_gas: float, T_wall: float, P_total: float, 
                         W_gas: float, h_heat: float, area: float) -> Tuple[float, float]:
        """
        Calculate condensation rate using physics-based approach
        
        Args:
            T_gas: Gas temperature (K)
            T_wall: Wall surface temperature (K)
            P_total: Total pressure (Pa)
            W_gas: Gas humidity ratio (kg water / kg dry air)
            h_heat: Heat transfer coefficient (W/(m²·K))
            area: Surface area (m²)
            
        Returns:
            Tuple of (condensation_rate kg/s, latent_heat_rate W)
        """
        # Saturation pressure at wall temperature
        P_sat_wall = self.saturation_pressure(T_wall)
        
        # Partial pressure of water vapor in bulk gas
        x_water_gas = self.humidity_ratio_to_mole_fraction(W_gas)
        P_vapor_gas = P_total * x_water_gas
        
        # Saturation humidity ratio at wall with bounds
        # Prevent divide by zero when P_sat_wall approaches P_total
        if P_sat_wall >= 0.99 * P_total:
            W_sat_wall = 100.0  # Very high humidity ratio (essentially all water)
        else:
            W_sat_wall = 0.622 * P_sat_wall / (P_total - P_sat_wall)
        
        # Check if condensation occurs
        if P_vapor_gas <= P_sat_wall or T_wall >= T_gas:
            return 0.0, 0.0
        
        # Mass transfer coefficient
        h_mass = self.mass_transfer_coefficient(h_heat, T_gas, P_total)
        
        # Driving force for condensation (humidity ratio difference)
        delta_W = W_gas - W_sat_wall
        
        if delta_W <= 0:
            return 0.0, 0.0
        
        # Condensation mass flow rate
        m_dot_condensation = h_mass * area * delta_W
        
        # Latent heat at film temperature
        T_film = (T_gas + T_wall) / 2
        h_fg = self.latent_heat(T_film)
        
        # Latent heat rate
        Q_latent = m_dot_condensation * h_fg
        
        return m_dot_condensation, Q_latent
    
    def evaporation_rate(self, T_wall: float, T_gas: float, P_total: float,
                        W_gas: float, h_heat: float, area: float, 
                        film_mass: float) -> Tuple[float, float]:
        """
        Calculate evaporation rate from existing condensate film
        
        Args:
            T_wall: Wall surface temperature (K)
            T_gas: Gas temperature (K)
            P_total: Total pressure (Pa)
            W_gas: Gas humidity ratio (kg water / kg dry air)
            h_heat: Heat transfer coefficient (W/(m²·K))
            area: Surface area (m²)
            film_mass: Existing film mass (kg)
            
        Returns:
            Tuple of (evaporation_rate kg/s, latent_heat_rate W)
        """
        # No evaporation if no film
        if film_mass <= 0:
            return 0.0, 0.0
        
        # Saturation pressure at wall temperature
        P_sat_wall = self.saturation_pressure(T_wall)
        
        # Saturation humidity ratio at wall with bounds
        # Prevent divide by zero when P_sat_wall approaches P_total
        if P_sat_wall >= 0.99 * P_total:
            W_sat_wall = 100.0  # Very high humidity ratio (essentially all water)
        else:
            W_sat_wall = 0.622 * P_sat_wall / (P_total - P_sat_wall)
        
        # Check if evaporation occurs
        if W_gas >= W_sat_wall:
            return 0.0, 0.0
        
        # Mass transfer coefficient
        h_mass = self.mass_transfer_coefficient(h_heat, T_gas, P_total)
        
        # Driving force for evaporation
        delta_W = W_sat_wall - W_gas
        
        # Maximum evaporation rate
        m_dot_evaporation = h_mass * area * delta_W
        
        # Limit by available film mass (can't evaporate more than exists)
        dt = 0.1  # Assume small timestep for rate limiting
        m_dot_evaporation = min(m_dot_evaporation, film_mass / dt)
        
        # Latent heat at film temperature
        T_film = (T_gas + T_wall) / 2
        h_fg = self.latent_heat(T_film)
        
        # Latent heat rate (negative for evaporation - removes heat)
        Q_latent = -m_dot_evaporation * h_fg
        
        return m_dot_evaporation, Q_latent
    
    def film_heat_transfer_modification(self, h_base: float, film_thickness: float,
                                      k_liquid: float = 0.6) -> float:
        """
        Modify heat transfer coefficient due to condensate film
        
        Args:
            h_base: Base heat transfer coefficient without film (W/(m²·K))
            film_thickness: Condensate film thickness (m)
            k_liquid: Thermal conductivity of liquid water (W/(m·K))
            
        Returns:
            Modified heat transfer coefficient (W/(m²·K))
        """
        if film_thickness <= 0:
            return h_base
        
        # Film resistance
        R_film = film_thickness / k_liquid
        
        # Total resistance
        R_total = 1/h_base + R_film
        
        # Modified heat transfer coefficient
        return 1 / R_total
    
    def update_condensation_state(self, state: CondensationState, 
                                 T_gas: float, T_wall: float, P_total: float,
                                 W_gas: float, h_heat: float, area: float,
                                 dt: float) -> CondensationState:
        """
        Update condensation state with proper mass balance
        
        Args:
            state: Current condensation state
            T_gas: Gas temperature (K)
            T_wall: Wall surface temperature (K)
            P_total: Total pressure (Pa)
            W_gas: Gas humidity ratio (kg water / kg dry air)
            h_heat: Heat transfer coefficient (W/(m²·K))
            area: Surface area (m²)
            dt: Time step (s)
            
        Returns:
            Updated condensation state
        """
        # Calculate condensation rate
        m_dot_cond, Q_cond = self.condensation_rate(
            T_gas, T_wall, P_total, W_gas, h_heat, area
        )
        
        # Calculate evaporation rate
        m_dot_evap, Q_evap = self.evaporation_rate(
            T_wall, T_gas, P_total, W_gas, h_heat, area, state.film_mass
        )
        
        # Net mass change
        net_mass_rate = m_dot_cond - m_dot_evap
        
        # Update film mass with mass balance
        new_film_mass = max(0, state.film_mass + net_mass_rate * dt)
        
        # Update film thickness
        if area > 0:
            new_film_thickness = new_film_mass / (self.RHO_LIQUID_WATER * area)
        else:
            new_film_thickness = 0.0
        
        # Cap film thickness at reasonable maximum
        max_thickness = 0.001  # 1 mm maximum
        if new_film_thickness > max_thickness:
            new_film_thickness = max_thickness
            new_film_mass = new_film_thickness * self.RHO_LIQUID_WATER * area
            # Excess water drains away (could track separately)
        
        # Total latent heat effect
        net_latent_heat = Q_cond + Q_evap  # Q_evap is negative
        
        # Return updated state
        return CondensationState(
            film_thickness=new_film_thickness,
            film_mass=new_film_mass,
            condensation_rate=m_dot_cond,
            evaporation_rate=m_dot_evap,
            net_rate=net_mass_rate,
            latent_heat_rate=net_latent_heat
        )
    
    def gas_humidity_change(self, W_gas: float, m_dot_water: float, 
                           m_dot_gas: float, V_gas: float, rho_gas: float,
                           dt: float) -> float:
        """
        Calculate change in gas humidity ratio due to condensation/evaporation
        
        Args:
            W_gas: Current gas humidity ratio (kg water / kg dry air)
            m_dot_water: Net water mass flow rate (kg/s) - negative for condensation
            m_dot_gas: Dry gas mass flow rate (kg/s)
            V_gas: Gas volume (m³)
            rho_gas: Gas density (kg/m³)
            dt: Time step (s)
            
        Returns:
            New gas humidity ratio
        """
        # Mass of dry air in volume
        m_dry_air = V_gas * rho_gas / (1 + W_gas)
        
        # Current water mass in gas
        m_water_current = m_dry_air * W_gas
        
        # Change in water mass (opposite sign - condensation removes from gas)
        dm_water = -m_dot_water * dt
        
        # New water mass
        m_water_new = max(0, m_water_current + dm_water)
        
        # New humidity ratio
        if m_dry_air > 0:
            W_new = m_water_new / m_dry_air
        else:
            W_new = W_gas
        
        return W_new