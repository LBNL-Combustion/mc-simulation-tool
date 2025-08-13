"""
Physical properties and constants for flue gas and air calculations
"""
import numpy as np

class PhysicalProperties:
    """Physical properties and correlations for flue gas and air
    
    This class provides fundamental fluid properties needed for heat transfer
    and flow calculations. All correlations are for air, which is a good
    approximation for flue gas in most applications.
    """
    
    # PHYSICAL CONSTANTS
    R_UNIVERSAL = 8314.0  # Universal gas constant J/(kmol·K) - for any ideal gas
    R_AIR = 287.0  # Specific gas constant for air J/(kg·K) - R_universal / M_air
    GRAVITY = 9.81  # Gravitational acceleration m/s² - for buoyancy calculations
    
    # STANDARD CONDITIONS (for reference)
    T_STD = 273.15  # Standard temperature in K (0°C)
    P_STD = 101325.0  # Standard pressure in Pa (1 atm)
    
    @staticmethod
    def air_density(temperature_K, pressure_Pa=P_STD):
        """Calculate air density using ideal gas law
        
        Ideal gas law: P = ρ * R * T
        Rearranged: ρ = P / (R * T)
        
        Density decreases with temperature (hot air rises!)
        This is what drives buoyancy in chimneys.
        
        Args:
            temperature_K: Temperature in Kelvin
            pressure_Pa: Pressure in Pascals (default is atmospheric)
            
        Returns:
            Density in kg/m³
        """
        return pressure_Pa / (PhysicalProperties.R_AIR * temperature_K)
    
    @staticmethod
    def air_viscosity(temperature_K):
        """Calculate air dynamic viscosity using Sutherland's formula
        
        Viscosity represents internal friction in the fluid.
        For gases, viscosity INCREASES with temperature (opposite of liquids).
        
        Sutherland's formula accounts for this temperature dependence
        based on kinetic theory of gases.
        
        Args:
            temperature_K: Temperature in Kelvin
            
        Returns:
            Dynamic viscosity in Pa·s (Pascal-seconds)
        """
        mu_ref = 1.716e-5  # Reference viscosity at 273.15 K (Pa·s)
        T_ref = 273.15  # Reference temperature (K)
        S = 110.4  # Sutherland constant for air (K)
        
        # Sutherland's formula: accounts for molecular interactions
        return mu_ref * ((temperature_K / T_ref) ** 1.5) * ((T_ref + S) / (temperature_K + S))
    
    @staticmethod
    def air_thermal_conductivity(temperature_K):
        """Calculate air thermal conductivity
        
        Thermal conductivity: how well the material conducts heat.
        Higher k = better heat conduction.
        
        For gases, thermal conductivity increases with temperature
        due to increased molecular motion.
        
        Args:
            temperature_K: Temperature in Kelvin
            
        Returns:
            Thermal conductivity in W/(m·K) (Watts per meter-Kelvin)
        """
        # Empirical correlation for air (valid from ~100K to ~1000K)
        return 2.624e-3 * temperature_K ** 1.5 / (temperature_K + 245.4 * 10 ** (-12 / temperature_K))
    
    @staticmethod
    def air_specific_heat(temperature_K):
        """Calculate air specific heat at constant pressure
        
        Specific heat (cp): energy needed to raise 1 kg by 1 K.
        For air, cp varies slightly with temperature.
        
        This polynomial fit is accurate for typical chimney temperatures.
        
        Args:
            temperature_K: Temperature in Kelvin
            
        Returns:
            Specific heat in J/(kg·K) (Joules per kilogram-Kelvin)
        """
        # Third-order polynomial fit for air cp
        # Valid roughly from 200K to 800K
        T = temperature_K
        return 1030.5 - 0.19975 * T + 4.2775e-4 * T**2 - 2.4223e-7 * T**3
    
    @staticmethod
    def nusselt_natural_convection_vertical_cylinder(rayleigh, height_diameter_ratio):
        """Calculate Nusselt number for natural convection on a vertical cylinder
        
        Args:
            rayleigh: Rayleigh number
            height_diameter_ratio: H/D ratio
            
        Returns:
            Nusselt number (dimensionless)
        """
        if rayleigh < 1e9:
            return 0.59 * rayleigh ** 0.25
        else:
            return 0.1 * rayleigh ** (1/3)
    
    @staticmethod
    def friction_factor_smooth_tube(reynolds):
        """Calculate friction factor for smooth tubes
        
        Args:
            reynolds: Reynolds number
            
        Returns:
            Friction factor (dimensionless)
        """
        if reynolds < 2300:  # Laminar
            return 64 / reynolds
        else:  # Turbulent - Blasius equation for smooth pipes
            return 0.316 / reynolds ** 0.25