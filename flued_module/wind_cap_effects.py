"""
Wind-induced cap pressure effects for chimney outlet boundary

Models aerodynamic pressure at the top termination caused by ambient wind
flowing past the cap. Negative Cp (suction) aids draft; positive Cp
(pressurization) reduces draft.

Physics: ΔP_wind = Cp * (0.5 * ρ * V²) * G
where Cp depends on cap geometry, exposure, and incidence angle.
"""

from dataclasses import dataclass
from typing import Union

@dataclass
class WindCapConfig:
    """Configuration for wind cap boundary effect"""
    enable: bool = False
    wind_speed_m_s: Union[float, list] = 0.0
    wind_direction_deg: float = 0.0
    air_temp_K: float = 294.15
    cap_type: str = "open"
    roof_exposure: str = "open"
    orientation: str = "vertical"
    gust_factor: float = 1.0
    Cp_override: Union[float, None] = None
    update_mode: str = "dynamic"   # "dynamic" or "static"

# Baseline pressure coefficients from literature
# Negative = suction (aids draft), Positive = pressurization (hurts draft)
BASELINE_CP = {
    "open":     -0.10,   # Mild suction from shear over lip
    "rain_cap": +0.05,   # Blockage causes weak pressurization
    "shrouded": -0.20,   # Sheltering shroud creates modest suction
    "cone":     -0.30,   # Diffuser/venturi-like geometry
    "venturi":  -0.50,   # Strong suction from venturi effect
    "H_cap":    -0.05,   # Near-neutral, resists down-draft
}

# Exposure scalar reduces wind effect in sheltered locations
EXPOSURE_SCALAR = {
    "open":      1.00,   # Full wind exposure
    "suburban":  0.85,   # Partial sheltering
    "sheltered": 0.70,   # Significant sheltering
}

# Gas constant for air
R_AIR = 287.0  # J/(kg·K)


def air_density(Pa: float = 98457.0, T_K: float = 294.15) -> float:
    """
    Calculate air density from ideal gas law

    Args:
        Pa: Atmospheric pressure (Pa)
        T_K: Air temperature (K)

    Returns:
        Air density (kg/m³)
    """
    return Pa / (R_AIR * T_K)


def incidence_factor(theta_deg: float) -> float:
    """
    Incidence angle correction factor for directional effects

    For vertical stacks, azimuthal sensitivity is weak. Currently returns 1.0.
    Hook for future implementation: F(θ) = 1 - 0.15*sin²(θ)

    Args:
        theta_deg: Wind direction angle (degrees)

    Returns:
        Incidence factor (dimensionless, typically near 1.0)
    """
    # Placeholder: vertical stacks weakly depend on wind azimuth
    return 1.0


def cap_Cp(cap_type: str, exposure: str, theta_deg: float,
           Cp_override: Union[float, None]) -> float:
    """
    Calculate effective pressure coefficient for cap

    Combines baseline Cp (geometry), exposure scalar (sheltering),
    and incidence factor (angle). Clamps to ±0.8 for safety.

    Args:
        cap_type: Cap geometry type (e.g., "venturi", "rain_cap")
        exposure: Site exposure ("open", "suburban", "sheltered")
        theta_deg: Wind direction angle (degrees)
        Cp_override: If provided, bypasses calculation and uses this value

    Returns:
        Effective pressure coefficient (dimensionless)
    """
    if Cp_override is not None:
        return Cp_override

    Cp0 = BASELINE_CP.get(cap_type, -0.10)
    E = EXPOSURE_SCALAR.get(exposure, 1.0)
    F = incidence_factor(theta_deg)

    Cp = Cp0 * E * F

    # Safety clamp: |Cp| ≤ 0.8
    return max(min(Cp, 0.8), -0.8)


def wind_cap_deltaP(Pa: float, T_out_K: float, V_wind: float,
                    cap_type: str, exposure: str, theta_deg: float,
                    gust_factor: float = 1.0,
                    Cp_override: Union[float, None] = None) -> float:
    """
    Calculate wind-induced pressure at chimney cap

    Physics: ΔP = Cp * q * G
    where q = 0.5 * ρ * V² (dynamic pressure)
          Cp = pressure coefficient (geometry, exposure, angle)
          G = gust factor

    Sign convention:
        Negative ΔP → suction → aids draft (increases driving head)
        Positive ΔP → pressurization → hurts draft (reduces driving head)

    Args:
        Pa: Ambient pressure at outlet (Pa)
        T_out_K: Ambient air temperature (K)
        V_wind: Wind speed (m/s)
        cap_type: Cap geometry type
        exposure: Site exposure
        theta_deg: Wind direction angle (degrees)
        gust_factor: Gust multiplier (default 1.0)
        Cp_override: Optional Cp override

    Returns:
        Wind pressure effect ΔP_wind (Pa)
        Negative = suction (helpful), Positive = pressurization (harmful)
    """
    # Input clamping for safety
    V_wind = max(V_wind, 0.0)  # No negative wind speeds
    T_out_K = max(min(T_out_K, 350.0), 180.0)  # Clamp temperature to reasonable range

    rho = air_density(Pa, T_out_K)

    # Dynamic pressure
    q = 0.5 * rho * (V_wind ** 2)

    # Effective pressure coefficient
    Cp = cap_Cp(cap_type, exposure, theta_deg, Cp_override)

    # Wind-induced pressure
    return Cp * q * gust_factor


# Validation and example usage
if __name__ == "__main__":
    print("="*70)
    print("WIND CAP EFFECTS - VALIDATION TESTS")
    print("="*70)

    # Test case: venturi cap, 5 m/s wind
    Pa = 98457.0
    T = 294.15
    V = 5.0

    print("\nReference Case:")
    print(f"  Ambient: {Pa:.0f} Pa, {T:.1f} K")
    print(f"  Wind speed: {V:.1f} m/s")

    rho = air_density(Pa, T)
    q = 0.5 * rho * V**2
    print(f"  Air density: {rho:.3f} kg/m³")
    print(f"  Dynamic pressure q: {q:.2f} Pa")

    print("\nCap Type Effects:")
    for cap in ["venturi", "cone", "shrouded", "open", "H_cap", "rain_cap"]:
        dP = wind_cap_deltaP(Pa, T, V, cap, "open", 0.0)
        sign = "suction" if dP < 0 else "pressurization"
        effect = "aids draft" if dP < 0 else "hurts draft"
        Cp = cap_Cp(cap, "open", 0.0, None)
        print(f"  {cap:12s}: Cp={Cp:+.2f}, dP={dP:+6.2f} Pa ({sign:15s}, {effect})")

    print("\nExposure Effects (venturi cap):")
    for exp in ["open", "suburban", "sheltered"]:
        dP = wind_cap_deltaP(Pa, T, V, "venturi", exp, 0.0)
        Cp = cap_Cp("venturi", exp, 0.0, None)
        print(f"  {exp:10s}: Cp={Cp:+.2f}, dP={dP:+6.2f} Pa")

    print("\nWind Speed Sensitivity (venturi, open):")
    for V_test in [0, 2, 5, 10, 15]:
        dP = wind_cap_deltaP(Pa, T, V_test, "venturi", "open", 0.0)
        print(f"  V={V_test:2d} m/s: dP={dP:+7.2f} Pa")

    print("\nSign Convention Validation:")
    dP_venturi = wind_cap_deltaP(Pa, T, V, "venturi", "open", 0.0)
    dP_rain = wind_cap_deltaP(Pa, T, V, "rain_cap", "open", 0.0)

    assert dP_venturi < 0, "Venturi should create suction (negative ΔP)"
    assert dP_rain > 0, "Rain cap should create pressurization (positive ΔP)"
    assert abs(dP_venturi) > abs(dP_rain), "Venturi should have larger magnitude"

    print("  [OK] Venturi creates suction (aids draft)")
    print("  [OK] Rain cap creates pressurization (hurts draft)")
    print("  [OK] Magnitudes correct")

    print("\n" + "="*70)
    print("ALL VALIDATION TESTS PASSED")
    print("="*70)
