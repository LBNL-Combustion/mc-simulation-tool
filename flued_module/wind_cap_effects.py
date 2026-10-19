"""
Wind-induced cap pressure effects for chimney outlet boundary

Models aerodynamic pressure at the top termination caused by ambient wind
flowing past the cap. Negative Cp (suction) aids draft; positive Cp
(pressurization) reduces draft.

Physics: ΔP_wind = Cp * F_flow * (0.5 * ρ * V²) * G
where Cp is the blocked-flue wind pressure coefficient of the termination
and F_flow reduces it when the flue is flowing.

Source for the coefficients: Haysom, J.C. and Swinton, M.C. (1987). The
Influence of Termination Configuration on the Flow Performance of Flues.
Canada Mortgage and Housing Corporation, Ottawa (wind tunnel tests by RWDI).
Haysom defines the coefficient as the rise in draft inside the flue divided
by the wind dynamic pressure, so their positive values are negative here.
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
    flue_flow_factor: float = 1.0  # 1.0 = blocked flue (start-up); see FLUE_FLOW_FACTOR

# Blocked-flue wind pressure coefficients, 0 deg wind, Haysom & Swinton (1987) Table 1A
# Negative = suction (aids draft), Positive = pressurization (hurts draft)
BASELINE_CP = {
    "open":          -1.626,  # open masonry chimney
    "rain_cap":      -0.728,  # A-vent, high-temperature cap for factory-built solid-fuel chimneys (Class A)
    "b_vent":        -0.503,  # B-vent (gas vent) cap
    "aerocowl":      -0.580,  # Aerocowl
    "cap_and_grill": -0.958,  # masonry cap and grill
}

# Cap flow loss K (velocity heads of total pressure lost at the termination, including
# the exit velocity head), Haysom & Swinton (1987) Table 2. Pass as top_boundary['K_exit'].
CAP_K_EXIT = {
    "open":          0.71,
    "rain_cap":      1.05,
    "b_vent":        1.38,
    "aerocowl":      3.58,
    "cap_and_grill": 1.00,
}

# With the flue flowing, the wind pressure across the cap drops (Haysom & Swinton 1987, Fig. 7,
# 10 km/h wind): about 0.7 of the blocked value for the A-vent and 0.5 to 0.6 for the B-vent.
# Haysom recommend the blocked value for low-flow start-up and the unblocked value for a full burn.
# Pass as top_boundary['flue_flow_factor']; the default 1.0 is the blocked flue.
FLUE_FLOW_FACTOR = {
    "rain_cap": 0.7,
    "b_vent":   0.55,
}

# Exposure scalar reduces wind effect in sheltered locations (not from Haysom; no source)
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
    Haysom Table 1A gives the coefficients at -45 to +45 deg; only 0 deg is used here.

    Args:
        theta_deg: Wind direction angle (degrees)

    Returns:
        Incidence factor (dimensionless, typically near 1.0)
    """
    return 1.0


def cap_Cp(cap_type: str, exposure: str, theta_deg: float,
           Cp_override: Union[float, None]) -> float:
    """
    Calculate effective pressure coefficient for cap

    Combines baseline Cp (geometry), exposure scalar (sheltering),
    and incidence factor (angle). Clamps to ±2.0 for safety
    (the largest measured value is 1.626, open chimney).

    Args:
        cap_type: Cap geometry type (a key of BASELINE_CP)
        exposure: Site exposure ("open", "suburban", "sheltered")
        theta_deg: Wind direction angle (degrees)
        Cp_override: If provided, bypasses calculation and uses this value

    Returns:
        Effective pressure coefficient (dimensionless)
    """
    if Cp_override is not None:
        return Cp_override

    if cap_type not in BASELINE_CP:
        raise ValueError(f"Unknown cap_type '{cap_type}'; use one of {sorted(BASELINE_CP)} "
                         f"or set Cp_override")
    Cp0 = BASELINE_CP[cap_type]
    E = EXPOSURE_SCALAR.get(exposure, 1.0)
    F = incidence_factor(theta_deg)

    Cp = Cp0 * E * F

    return max(min(Cp, 2.0), -2.0)


def wind_cap_deltaP(Pa: float, T_out_K: float, V_wind: float,
                    cap_type: str, exposure: str, theta_deg: float,
                    gust_factor: float = 1.0,
                    Cp_override: Union[float, None] = None,
                    flue_flow_factor: float = 1.0) -> float:
    """
    Calculate wind-induced pressure at chimney cap

    Physics: ΔP = Cp * F_flow * q * G
    where q = 0.5 * ρ * V² (dynamic pressure)
          Cp = blocked-flue pressure coefficient (geometry, exposure, angle)
          F_flow = reduction with the flue flowing (1.0 = blocked)
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
        flue_flow_factor: Multiplier for a flowing flue (default 1.0)

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
    return Cp * flue_flow_factor * q * gust_factor


# Validation and example usage
if __name__ == "__main__":
    print("="*70)
    print("WIND CAP EFFECTS - VALIDATION TESTS")
    print("="*70)

    # Test case: 10 km/h wind, as in Haysom Table 1B
    Pa = 101325.0
    T = 273.15  # Table 1B values imply q of about 4.95 Pa at 10 km/h, i.e. air near 0 C
    V = 10.0 / 3.6

    print("\nReference Case:")
    print(f"  Ambient: {Pa:.0f} Pa, {T:.1f} K")
    print(f"  Wind speed: {V:.2f} m/s")

    rho = air_density(Pa, T)
    q = 0.5 * rho * V**2
    print(f"  Air density: {rho:.3f} kg/m³")
    print(f"  Dynamic pressure q: {q:.2f} Pa")

    print("\nCap Type Effects (blocked flue; Haysom Table 1B, 0 deg, 10 km/h, for comparison):")
    table_1b = {"rain_cap": 3.6, "b_vent": 2.5, "aerocowl": 2.9, "open": 8.1, "cap_and_grill": 4.8}
    for cap in BASELINE_CP:
        dP = wind_cap_deltaP(Pa, T, V, cap, "open", 0.0)
        print(f"  {cap:14s}: Cp={cap_Cp(cap, 'open', 0.0, None):+.3f}, dP={dP:+6.2f} Pa "
              f"(Haysom {-table_1b[cap]:+.1f} Pa), K_exit={CAP_K_EXIT[cap]:.2f}")

    print("\nRain cap with the flue flowing (factor 0.7):")
    dP = wind_cap_deltaP(Pa, T, V, "rain_cap", "open", 0.0, flue_flow_factor=0.7)
    print(f"  dP={dP:+6.2f} Pa")

    print("\nSign Convention Validation:")
    for cap in BASELINE_CP:
        dP = wind_cap_deltaP(Pa, T, V, cap, "open", 0.0)
        assert dP < 0, f"{cap}: wind across the cap should add draft (negative ΔP)"
        assert abs(dP + table_1b[cap]) < 0.15, f"{cap}: {dP:.2f} Pa does not match Haysom Table 1B"
    print("  [OK] Every cap creates suction (aids draft), as measured by Haysom & Swinton (1987)")
    print("  [OK] Magnitudes match Haysom Table 1B within 0.15 Pa")

    print("\n" + "="*70)
    print("ALL VALIDATION TESTS PASSED")
    print("="*70)
