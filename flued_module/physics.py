"""
Physical properties and constants for flue gas and air calculations
"""

class PhysicalProperties:
    """Physical properties and correlations for flue gas and air"""
    
    # PHYSICAL CONSTANTS
    R_UNIVERSAL = 8314.0  # Universal gas constant J/(kmol·K) - for any ideal gas
    R_AIR = 287.0  # Specific gas constant for air J/(kg·K) - R_universal / M_air
    GRAVITY = 9.81  # Gravitational acceleration m/s² - for buoyancy calculations
    
    # STANDARD CONDITIONS (for reference)
    T_STD = 273.15  # Standard temperature in K (0°C)
    P_STD = 101325.0  # Standard atmospheric pressure at sea level (Pa)

    @staticmethod
    def pressure_from_elevation(elevation_m):
        """Atmospheric pressure from site elevation via the ISA barometric formula.

        Args:
            elevation_m: Elevation above sea level in meters. Valid for the
                troposphere (h < 11 km).

        Returns:
            Ambient pressure in Pa.
        """
        P0 = 101325.0   # Pa, sea level
        T0 = 288.15     # K
        L = 0.0065      # K/m, lapse rate
        EXP = 5.2561    # g*M/(R*L)
        return P0 * (1.0 - L * elevation_m / T0) ** EXP

    @staticmethod
    def air_density(temperature_K, pressure_Pa=P_STD):
        """Air density from ideal gas law.

        Args:
            temperature_K: Temperature in Kelvin
            pressure_Pa: Pressure in Pascals (default is atmospheric)

        Returns:
            Density in kg/m³
        """
        return pressure_Pa / (PhysicalProperties.R_AIR * temperature_K)
    
    @staticmethod
    def air_viscosity(temperature_K):
        """Air dynamic viscosity via Sutherland's formula.

        Args:
            temperature_K: Temperature in Kelvin

        Returns:
            Dynamic viscosity in Pa·s
        """
        mu_ref = 1.716e-5  # Reference viscosity at 273.15 K (Pa·s)
        T_ref = 273.15  # Reference temperature (K)
        S = 110.4  # Sutherland constant for air (K)

        return mu_ref * ((temperature_K / T_ref) ** 1.5) * ((T_ref + S) / (temperature_K + S))
    
    @staticmethod
    def air_thermal_conductivity(temperature_K):
        """Air thermal conductivity, empirical correlation.

        Args:
            temperature_K: Temperature in Kelvin

        Returns:
            Thermal conductivity in W/(m·K)
        """
        # Valid from ~100K to ~1000K
        return 2.624e-3 * temperature_K ** 1.5 / (temperature_K + 245.4 * 10 ** (-12 / temperature_K))
    
    @staticmethod
    def air_specific_heat(temperature_K):
        """Air cp, polynomial fit.

        Args:
            temperature_K: Temperature in Kelvin

        Returns:
            Specific heat in J/(kg·K)
        """
        # Valid roughly from 200K to 800K
        T = temperature_K
        return 1030.5 - 0.19975 * T + 4.2775e-4 * T**2 - 2.4223e-7 * T**3

    @staticmethod
    def ceramic_fiber_thermal_conductivity(temperature_K, k_ref=0.04):
        """Ceramic fiber blanket k(T), quadratic fit to manufacturer data (~100 kg/m³).

        Reference values:
        - 293K: 0.04, 473K: 0.07, 673K: 0.12, 873K: 0.18 W/m·K

        Args:
            temperature_K: Insulation mean temperature in Kelvin
            k_ref: Reference thermal conductivity at 20°C (default 0.04 W/m·K)

        Returns:
            Thermal conductivity in W/(m·K)
        """
        T_ref = 293.15  # Reference temperature (20°C) in Kelvin

        dT = max(0, temperature_K - T_ref)

        alpha = 1.8e-4   # Linear coefficient (W/m·K per K)
        beta = 1.5e-7    # Quadratic coefficient (W/m·K per K²)

        k = k_ref + alpha * dT + beta * dT**2

        return min(k, 0.30)  # cap at 0.30 W/m·K