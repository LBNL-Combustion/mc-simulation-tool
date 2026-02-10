"""
Quick validation tests for wind cap effects

Tests the physics module in isolation without running full simulations.
Verifies sign conventions, exposure scalars, and static/dynamic behavior.
"""

from flued_module.wind_cap_effects import wind_cap_deltaP, cap_Cp, BASELINE_CP, EXPOSURE_SCALAR

def test_signs():
    """Test that sign convention is correct"""
    Pa, T = 101325.0, 294.0
    V = 5.0

    # Venturi should create suction (negative ΔP, aids draft)
    dP_venturi = wind_cap_deltaP(Pa, T, V, "venturi", "open", 0.0)
    assert dP_venturi < 0.0, f"Venturi should create suction, got {dP_venturi}"

    # Rain cap should create pressurization (positive ΔP, hurts draft)
    dP_rain = wind_cap_deltaP(Pa, T, V, "rain_cap", "open", 0.0)
    assert dP_rain > 0.0, f"Rain cap should create pressurization, got {dP_rain}"

    print("✓ Sign convention test passed")
    print(f"  Venturi: {dP_venturi:.2f} Pa (suction, aids draft)")
    print(f"  Rain cap: {dP_rain:.2f} Pa (pressurization, hurts draft)")


def test_exposure_scalar():
    """Test that exposure reduces wind effect"""
    Pa, T, V = 101325.0, 294.0, 5.0

    # Open exposure should have largest magnitude
    dP_open = wind_cap_deltaP(Pa, T, V, "venturi", "open", 0.0)
    dP_suburban = wind_cap_deltaP(Pa, T, V, "venturi", "suburban", 0.0)
    dP_sheltered = wind_cap_deltaP(Pa, T, V, "venturi", "sheltered", 0.0)

    # All should be negative (suction) but sheltered should have smallest magnitude
    assert dP_open < 0 and dP_suburban < 0 and dP_sheltered < 0
    assert abs(dP_open) > abs(dP_suburban) > abs(dP_sheltered), \
        "Exposure should reduce wind effect magnitude"

    print("✓ Exposure scalar test passed")
    print(f"  Open:      {dP_open:.2f} Pa")
    print(f"  Suburban:  {dP_suburban:.2f} Pa")
    print(f"  Sheltered: {dP_sheltered:.2f} Pa")


def test_magnitude_ordering():
    """Test that different cap types have expected magnitude ordering"""
    Pa, T, V = 101325.0, 294.0, 5.0

    caps_negative = ["venturi", "cone", "shrouded", "open", "H_cap"]
    caps_positive = ["rain_cap"]

    results = {}
    for cap in caps_negative + caps_positive:
        dP = wind_cap_deltaP(Pa, T, V, cap, "open", 0.0)
        results[cap] = dP

    # Venturi should have largest suction
    assert results["venturi"] < results["cone"] < results["shrouded"]
    assert results["venturi"] < -5.0, "Venturi should create substantial suction"

    # Rain cap should be positive but small
    assert 0 < results["rain_cap"] < 5.0

    print("✓ Magnitude ordering test passed")
    for cap, dP in results.items():
        sign = "suction" if dP < 0 else "pressurization"
        print(f"  {cap:12s}: {dP:+6.2f} Pa ({sign})")


def test_wind_speed_quadratic():
    """Test that pressure varies quadratically with wind speed"""
    Pa, T = 101325.0, 294.0

    dP_2 = wind_cap_deltaP(Pa, T, 2.0, "venturi", "open", 0.0)
    dP_4 = wind_cap_deltaP(Pa, T, 4.0, "venturi", "open", 0.0)
    dP_8 = wind_cap_deltaP(Pa, T, 8.0, "venturi", "open", 0.0)

    # Should scale as V², so 4x wind = 16x pressure
    ratio_2_to_4 = abs(dP_4 / dP_2)
    ratio_4_to_8 = abs(dP_8 / dP_4)

    assert 3.5 < ratio_2_to_4 < 4.5, f"2x wind should give ~4x pressure, got {ratio_2_to_4}"
    assert 3.5 < ratio_4_to_8 < 4.5, f"2x wind should give ~4x pressure, got {ratio_4_to_8}"

    print("✓ Quadratic wind speed test passed")
    print(f"  V=2 m/s: {dP_2:.2f} Pa")
    print(f"  V=4 m/s: {dP_4:.2f} Pa (ratio: {ratio_2_to_4:.2f})")
    print(f"  V=8 m/s: {dP_8:.2f} Pa (ratio: {ratio_4_to_8:.2f})")


def test_gust_factor():
    """Test that gust factor multiplies pressure correctly"""
    Pa, T, V = 101325.0, 294.0, 5.0

    dP_1 = wind_cap_deltaP(Pa, T, V, "venturi", "open", 0.0, gust_factor=1.0)
    dP_15 = wind_cap_deltaP(Pa, T, V, "venturi", "open", 0.0, gust_factor=1.5)

    ratio = abs(dP_15 / dP_1)
    assert 1.45 < ratio < 1.55, f"Gust factor 1.5 should give 1.5x pressure, got {ratio}"

    print("✓ Gust factor test passed")
    print(f"  G=1.0: {dP_1:.2f} Pa")
    print(f"  G=1.5: {dP_15:.2f} Pa (ratio: {ratio:.2f})")


def test_Cp_override():
    """Test that Cp_override bypasses table lookup"""
    Pa, T, V = 101325.0, 294.0, 5.0

    # With override, should use exact Cp regardless of cap_type
    Cp_custom = -0.75
    dP_override = wind_cap_deltaP(Pa, T, V, "open", "open", 0.0, Cp_override=Cp_custom)

    # Calculate expected: q = 0.5 * rho * V^2, dP = Cp * q
    from flued_module.wind_cap_effects import air_density
    rho = air_density(Pa, T)
    q = 0.5 * rho * V**2
    expected = Cp_custom * q

    assert abs(dP_override - expected) < 0.01, f"Cp_override not working, got {dP_override}, expected {expected}"

    print("✓ Cp_override test passed")
    print(f"  Custom Cp={Cp_custom}, ΔP={dP_override:.2f} Pa (expected {expected:.2f} Pa)")


def test_safety_clamps():
    """Test that extreme inputs are clamped safely"""
    Pa, T = 101325.0, 294.0

    # Negative wind speed should be clamped to zero
    dP_neg_wind = wind_cap_deltaP(Pa, T, -10.0, "venturi", "open", 0.0)
    assert dP_neg_wind == 0.0, "Negative wind speed should be clamped to zero"

    # Extreme temperature should be clamped
    dP_hot = wind_cap_deltaP(Pa, 500.0, 5.0, "venturi", "open", 0.0)  # 500K clamped to 350K
    dP_cold = wind_cap_deltaP(Pa, 100.0, 5.0, "venturi", "open", 0.0)  # 100K clamped to 180K
    assert dP_hot < 0 and dP_cold < 0, "Extreme temperatures should still give valid results"

    # Cp should be clamped to ±0.8 (unless override is provided)
    Cp_venturi = cap_Cp("venturi", "open", 0.0, None)
    assert -0.8 <= Cp_venturi <= 0.8, f"Cp should be clamped to ±0.8, got {Cp_venturi}"

    print("✓ Safety clamps test passed")
    print(f"  Negative wind → ΔP={dP_neg_wind:.2f} Pa (should be 0)")
    print(f"  Extreme temps handled correctly")
    print(f"  Cp clamped to [{Cp_venturi:.2f}] within ±0.8")


if __name__ == "__main__":
    print("="*70)
    print("WIND CAP EFFECTS - QUICK VALIDATION TESTS")
    print("="*70)
    print()

    test_signs()
    print()

    test_exposure_scalar()
    print()

    test_magnitude_ordering()
    print()

    test_wind_speed_quadratic()
    print()

    test_gust_factor()
    print()

    test_Cp_override()
    print()

    test_safety_clamps()
    print()

    print("="*70)
    print("ALL TESTS PASSED")
    print("="*70)
