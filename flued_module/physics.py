"""
Physical properties and constants for flue gas and air calculations
"""

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
    P_STD = 98457.0  # Atmospheric pressure at lab elevation (Pa)
    
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
    def ceramic_fiber_thermal_conductivity(temperature_K, k_ref=0.04):
        """Calculate temperature-dependent thermal conductivity of ceramic fiber insulation

        Ceramic fiber blanket thermal conductivity increases significantly with temperature
        due to increased radiation heat transfer through the porous structure.

        Based on typical ceramic fiber blanket data (density ~100 kg/m³):
        - At 20°C (293K): k ≈ 0.04 W/m·K
        - At 200°C (473K): k ≈ 0.07 W/m·K
        - At 400°C (673K): k ≈ 0.12 W/m·K
        - At 600°C (873K): k ≈ 0.18 W/m·K

        Correlation: k(T) = k_ref + alpha * (T - T_ref) + beta * (T - T_ref)^2
        where T_ref = 293K and coefficients fit to manufacturer data.

        Args:
            temperature_K: Insulation mean temperature in Kelvin
            k_ref: Reference thermal conductivity at 20°C (default 0.04 W/m·K)

        Returns:
            Thermal conductivity in W/(m·K)
        """
        T_ref = 293.15  # Reference temperature (20°C) in Kelvin

        # Temperature difference from reference
        dT = max(0, temperature_K - T_ref)  # Only increase k for T > T_ref

        # Coefficients based on ceramic fiber blanket data
        # Linear: accounts for conduction increase
        # Quadratic: accounts for radiation contribution at high temps
        alpha = 1.8e-4   # Linear coefficient (W/m·K per K)
        beta = 1.5e-7    # Quadratic coefficient (W/m·K per K²)

        k = k_ref + alpha * dT + beta * dT**2

        # Cap at reasonable maximum (very high temps)
        return min(k, 0.30)