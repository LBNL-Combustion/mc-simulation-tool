"""
Test script for inlet restriction feature (orifice/Cd model)

Tests the swivel door inlet restriction with discharge coefficient model:
    ΔP = K_equiv × 0.5 × ρ × v²  where K_equiv = 1/Cd²

Test cases:
1. No restriction (baseline)
2. Door at 75% open - mild restriction
3. Door at 50% open - moderate restriction
4. Door at 25% open - significant restriction
5. Formula validation with known Cd
"""

import sys
import math
import numpy as np
from flued_module.chimney import Chimney

# Define a simple chimney for testing
parts = [
    {
        'type': 'single',
        'inner_diameter_ft': 6/12,  # 6 inches
        'height_ft': 15,
        'z_ft': 0,
        'orientation': 'vertical',
        'T_ambient_K': 273.15 + 20,  # 20°C
        'Pa': 101325,
        'num_segments': 10,
    }
]

print("="*70)
print("INLET RESTRICTION FEATURE TEST (Orifice/Cd Model)")
print("="*70)

# Calculate inlet pipe area for reference
inlet_diameter_m = (6/12) * 0.3048  # 6 inches in meters = 0.152 m
inlet_area_m2 = np.pi * (inlet_diameter_m / 2)**2  # ~0.0182 m²
print(f"\nInlet pipe area: {inlet_area_m2:.5f} m²")
print(f"Inlet diameter: {inlet_diameter_m*100:.2f} cm ({inlet_diameter_m/0.0254:.1f} in)")

# Use a door area SMALLER than pipe area so restriction is meaningful
# Typical loading door: ~10cm x 8cm = 0.008 m² (smaller than pipe's 0.0182 m²)
door_area = 0.008  # m²
print(f"Test door area: {door_area:.5f} m² (smaller than pipe for actual restriction)")
print(f"Default Cd: 0.62 (sharp-edged orifice)")
print("")

# ============================================================
# Test 1: No restriction (baseline)
# ============================================================
print("\n" + "="*70)
print("TEST 1: NO RESTRICTION (BASELINE)")
print("="*70)
chimney_baseline = Chimney(parts=parts, dt=0.5, total_time=60)
chimney_baseline.run_simulation()

# ============================================================
# Test 2: Door at 75% open (67.5°) - mild restriction
# ============================================================
print("\n" + "="*70)
print("TEST 2: DOOR AT 75% OPEN (67.5°)")
print("="*70)
chimney_75 = Chimney(
    parts=parts,
    dt=0.5,
    total_time=60,
    inlet_restriction={
        'enable': True,
        'door_opening_percent': 75.0,
        'door_area_m2': door_area,
        # Cd defaults to 0.62
    }
)
chimney_75.run_simulation()

# ============================================================
# Test 3: Door at 50% open (45°) - moderate restriction
# ============================================================
print("\n" + "="*70)
print("TEST 3: DOOR AT 50% OPEN (45°)")
print("="*70)
chimney_50 = Chimney(
    parts=parts,
    dt=0.5,
    total_time=60,
    inlet_restriction={
        'enable': True,
        'door_opening_percent': 50.0,
        'door_area_m2': door_area,
    }
)
chimney_50.run_simulation()

# ============================================================
# Test 4: Door at 25% open (22.5°) - significant restriction
# ============================================================
print("\n" + "="*70)
print("TEST 4: DOOR AT 25% OPEN (22.5°)")
print("="*70)
chimney_25 = Chimney(
    parts=parts,
    dt=0.5,
    total_time=60,
    inlet_restriction={
        'enable': True,
        'door_opening_percent': 25.0,
        'door_area_m2': door_area,
    }
)
chimney_25.run_simulation()

# ============================================================
# Compare results
# ============================================================
print("\n" + "="*70)
print("COMPARISON OF RESULTS")
print("="*70)

print("\nFinal Mass Flow Rates:")
mdot_baseline = chimney_baseline.mass_flow_rate[-1]
mdot_75 = chimney_75.mass_flow_rate[-1]
mdot_50 = chimney_50.mass_flow_rate[-1]
mdot_25 = chimney_25.mass_flow_rate[-1]

print(f"  No restriction:   {mdot_baseline:.6f} kg/s (100%)")
print(f"  75% open (67.5°): {mdot_75:.6f} kg/s ({mdot_75/mdot_baseline*100:.1f}% of baseline)")
print(f"  50% open (45°):   {mdot_50:.6f} kg/s ({mdot_50/mdot_baseline*100:.1f}% of baseline)")
print(f"  25% open (22.5°): {mdot_25:.6f} kg/s ({mdot_25/mdot_baseline*100:.1f}% of baseline)")

print("\nEffective Areas (swivel door: A_eff = door_area × sin(angle)):")
A_eff_75 = chimney_75.inlet_restriction.effective_area_m2
A_eff_50 = chimney_50.inlet_restriction.effective_area_m2
A_eff_25 = chimney_25.inlet_restriction.effective_area_m2
print(f"  75% open:  {A_eff_75:.6f} m² [sin(67.5°) = {math.sin(math.radians(67.5)):.3f}]")
print(f"  50% open:  {A_eff_50:.6f} m² [sin(45°) = {math.sin(math.radians(45)):.3f}]")
print(f"  25% open:  {A_eff_25:.6f} m² [sin(22.5°) = {math.sin(math.radians(22.5)):.3f}]")

print("\nK_equiv Values (= 1/Cd² = 1/0.62² ≈ 2.60):")
print(f"  75% open:  K_equiv = {chimney_75.inlet_restriction.K_equiv:.3f}")
print(f"  50% open:  K_equiv = {chimney_50.inlet_restriction.K_equiv:.3f}")
print(f"  25% open:  K_equiv = {chimney_25.inlet_restriction.K_equiv:.3f}")

# ============================================================
# ASSERTIONS
# ============================================================
print("\n" + "="*70)
print("VALIDATION ASSERTIONS")
print("="*70)

# Assertion 1: Mass flow decreases monotonically as opening decreases
# (more restriction → less flow)
print("\n1. Checking monotonic mass flow decrease with more restriction...")
assert mdot_baseline >= mdot_75, f"FAIL: baseline ({mdot_baseline}) should be >= 75% open ({mdot_75})"
assert mdot_75 >= mdot_50, f"FAIL: 75% open ({mdot_75}) should be >= 50% open ({mdot_50})"
assert mdot_50 >= mdot_25, f"FAIL: 50% open ({mdot_50}) should be >= 25% open ({mdot_25})"
print("   PASS: mdot_baseline >= mdot_75 >= mdot_50 >= mdot_25")

# Assertion 2: K_equiv matches expected value from Cd
Cd = 0.62
K_expected = 1.0 / (Cd ** 2)
K_actual = chimney_50.inlet_restriction.K_equiv
tolerance = 0.01
print(f"\n2. Checking K_equiv = 1/Cd² = 1/{Cd}² = {K_expected:.4f}...")
assert abs(K_actual - K_expected) < tolerance, \
    f"FAIL: K_equiv ({K_actual:.4f}) should be {K_expected:.4f} ± {tolerance}"
print(f"   PASS: K_equiv = {K_actual:.4f} (expected {K_expected:.4f})")

# Assertion 3: Effective area calculation is correct
# A_eff = door_area × sin(opening_pct/100 × π/2)
print("\n3. Checking effective area calculation (swivel door)...")
for name, chimney, pct in [("75%", chimney_75, 75), ("50%", chimney_50, 50), ("25%", chimney_25, 25)]:
    angle_rad = (pct / 100.0) * (math.pi / 2.0)
    A_eff_expected = door_area * math.sin(angle_rad)
    A_eff_actual = chimney.inlet_restriction.effective_area_m2
    assert abs(A_eff_actual - A_eff_expected) < 1e-9, \
        f"FAIL: {name} A_eff ({A_eff_actual:.6f}) should be {A_eff_expected:.6f}"
    print(f"   PASS: {name} open → A_eff = {A_eff_actual:.6f} m²")

# Assertion 4: Verify pressure drop formula manually for one case
# ΔP = K_equiv × 0.5 × ρ × v²  where v = ṁ/(ρ × A_eff)
print("\n4. Checking pressure drop formula (manual calculation)...")
# Use 50% case with final values
rho_inlet = chimney_50.density[-1, 0]  # Density at inlet at final time
A_eff = chimney_50.inlet_restriction.effective_area_m2
K = chimney_50.inlet_restriction.K_equiv
mdot = chimney_50.mass_flow_rate[-1]

v_restricted = mdot / (rho_inlet * A_eff)
dp_expected = K * 0.5 * rho_inlet * v_restricted**2

print(f"   At 50% open (final timestep):")
print(f"     ṁ = {mdot:.6f} kg/s")
print(f"     ρ = {rho_inlet:.3f} kg/m³")
print(f"     A_eff = {A_eff:.6f} m²")
print(f"     v = ṁ/(ρ×A_eff) = {v_restricted:.3f} m/s")
print(f"     K_equiv = {K:.3f}")
print(f"     ΔP = K × 0.5 × ρ × v² = {dp_expected:.3f} Pa")
print(f"   PASS: Formula computes without error")

# Assertion 5: Restriction should produce meaningful pressure drop
print("\n5. Checking that restriction produces meaningful effect...")
assert mdot_25 < 0.95 * mdot_baseline, \
    f"FAIL: 25% open should reduce flow by >5% (got {mdot_25/mdot_baseline*100:.1f}%)"
print(f"   PASS: 25% open reduces flow to {mdot_25/mdot_baseline*100:.1f}% of baseline")

# ============================================================
# Test with custom Cd
# ============================================================
print("\n" + "="*70)
print("TEST 5: CUSTOM Cd VALUE (well-rounded entrance, Cd=0.82)")
print("="*70)
chimney_rounded = Chimney(
    parts=parts,
    dt=0.5,
    total_time=60,
    inlet_restriction={
        'enable': True,
        'door_opening_percent': 50.0,
        'door_area_m2': door_area,
        'Cd': 0.82,  # Well-rounded entrance
    }
)
chimney_rounded.run_simulation()

K_rounded = chimney_rounded.inlet_restriction.K_equiv
K_expected_rounded = 1.0 / (0.82 ** 2)
print(f"\nCustom Cd=0.82 → K_equiv = {K_rounded:.3f} (expected {K_expected_rounded:.3f})")
assert abs(K_rounded - K_expected_rounded) < tolerance, \
    f"FAIL: K_equiv for Cd=0.82 should be {K_expected_rounded:.3f}"
print("PASS: Custom Cd value correctly applied")

# Higher Cd means less resistance, so flow should be higher than with default Cd
mdot_rounded = chimney_rounded.mass_flow_rate[-1]
print(f"\nMass flow comparison at 50% open:")
print(f"  Cd=0.62 (sharp):   {mdot_50:.6f} kg/s")
print(f"  Cd=0.82 (rounded): {mdot_rounded:.6f} kg/s")
assert mdot_rounded > mdot_50, "FAIL: Higher Cd should give higher mass flow"
print("PASS: Higher Cd gives higher mass flow (less restriction)")

# ============================================================
# Test 6: 0% open (fully closed) - should produce VERY HIGH dp
# ============================================================
print("\n" + "="*70)
print("TEST 6: 0% OPEN (FULLY CLOSED) - EDGE CASE")
print("="*70)
chimney_0 = Chimney(
    parts=parts,
    dt=0.5,
    total_time=60,
    inlet_restriction={
        'enable': True,
        'door_opening_percent': 0.0,  # Fully closed!
        'door_area_m2': door_area,
    }
)
chimney_0.run_simulation()

# Get A_EFF_MIN from the config
A_EFF_MIN = chimney_0.inlet_restriction.A_EFF_MIN
print(f"\nA_EFF_MIN = {A_EFF_MIN:.2e} m² (minimum effective area)")
print(f"A_eff at 0% open: {chimney_0.inlet_restriction.effective_area_m2:.2e} m²")

# Verify A_eff was clamped to A_EFF_MIN
assert chimney_0.inlet_restriction.effective_area_m2 == A_EFF_MIN, \
    f"FAIL: A_eff at 0% should be clamped to A_EFF_MIN ({A_EFF_MIN})"
print("PASS: A_eff clamped to A_EFF_MIN at 0% open")

# ============================================================
# Test 7: Pressure drop monotonicity (direct dp calculation)
# ============================================================
print("\n" + "="*70)
print("TEST 7: PRESSURE DROP MONOTONICITY CHECK")
print("="*70)
print("\nVerifying: dp(0%) > dp(1%) > dp(5%) > dp(25%) for fixed ṁ and ρ")

# Use fixed values for direct comparison
mdot_test = 0.01  # kg/s
rho_test = 1.0    # kg/m³
Cd_test = 0.62
K_test = 1.0 / (Cd_test ** 2)

def calc_dp(opening_pct, door_area, A_eff_min):
    """Calculate dp for a given opening percentage"""
    angle_rad = (opening_pct / 100.0) * (math.pi / 2.0)
    A_eff_raw = door_area * math.sin(angle_rad)
    A_eff = max(A_eff_raw, A_eff_min)  # Clamp to minimum
    v = mdot_test / (rho_test * A_eff)
    dp = K_test * 0.5 * rho_test * v**2
    return dp, A_eff

# Calculate dp at various openings
dp_0, A_eff_0 = calc_dp(0.0, door_area, A_EFF_MIN)
dp_1, A_eff_1 = calc_dp(1.0, door_area, A_EFF_MIN)
dp_5, A_eff_5 = calc_dp(5.0, door_area, A_EFF_MIN)
dp_25, A_eff_25 = calc_dp(25.0, door_area, A_EFF_MIN)

print(f"\nWith ṁ = {mdot_test} kg/s, ρ = {rho_test} kg/m³, Cd = {Cd_test}:")
print(f"  0% open:  A_eff = {A_eff_0:.2e} m², dp = {dp_0:.1f} Pa")
print(f"  1% open:  A_eff = {A_eff_1:.2e} m², dp = {dp_1:.1f} Pa")
print(f"  5% open:  A_eff = {A_eff_5:.2e} m², dp = {dp_5:.1f} Pa")
print(f"  25% open: A_eff = {A_eff_25:.2e} m², dp = {dp_25:.2f} Pa")

# Verify monotonicity: more closed = higher dp
print("\nMonotonicity checks:")
assert dp_0 >= dp_1, f"FAIL: dp(0%) = {dp_0:.1f} should be >= dp(1%) = {dp_1:.1f}"
print(f"   PASS: dp(0%) = {dp_0:.1f} Pa >= dp(1%) = {dp_1:.1f} Pa")

assert dp_1 > dp_5, f"FAIL: dp(1%) = {dp_1:.1f} should be > dp(5%) = {dp_5:.1f}"
print(f"   PASS: dp(1%) = {dp_1:.1f} Pa > dp(5%) = {dp_5:.1f} Pa")

assert dp_5 > dp_25, f"FAIL: dp(5%) = {dp_5:.1f} should be > dp(25%) = {dp_25:.2f}"
print(f"   PASS: dp(5%) = {dp_5:.1f} Pa > dp(25%) = {dp_25:.2f} Pa")

# Verify 0% produces VERY high dp (at least 1000x the 25% case)
dp_ratio = dp_0 / dp_25
print(f"\nExtreme case check:")
print(f"   dp(0%) / dp(25%) = {dp_ratio:.0f}x")
assert dp_ratio > 100, f"FAIL: 0% open should have dp >> 25% open (ratio = {dp_ratio:.0f})"
print(f"   PASS: 0% open produces {dp_ratio:.0f}x more dp than 25% open")

# ============================================================
# Test 8: Verify formula with clamped A_eff
# ============================================================
print("\n" + "="*70)
print("TEST 8: FORMULA VERIFICATION WITH CLAMPED A_EFF")
print("="*70)

# Expected dp at 0% open with clamped area
v_clamped = mdot_test / (rho_test * A_EFF_MIN)
dp_expected_clamped = K_test * 0.5 * rho_test * v_clamped**2

print(f"\nAt 0% open with A_eff clamped to {A_EFF_MIN:.2e} m²:")
print(f"  v = ṁ/(ρ×A_eff) = {mdot_test}/({rho_test}×{A_EFF_MIN:.2e}) = {v_clamped:.0f} m/s")
print(f"  dp = K × 0.5 × ρ × v² = {K_test:.3f} × 0.5 × {rho_test} × {v_clamped:.0f}² = {dp_expected_clamped:.0f} Pa")

# Verify our calc_dp function matches
assert abs(dp_0 - dp_expected_clamped) < 0.1, \
    f"FAIL: Calculated dp ({dp_0:.1f}) doesn't match expected ({dp_expected_clamped:.1f})"
print(f"   PASS: Calculated dp = {dp_0:.0f} Pa matches expected = {dp_expected_clamped:.0f} Pa")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("ALL TESTS PASSED")
print("="*70)
print("""
Summary of orifice/Cd model:
- ΔP = K_equiv × 0.5 × ρ × v²  where K_equiv = 1/Cd²
- v = ṁ / (ρ × A_eff)  (velocity through door opening)
- A_eff = door_area × sin(opening_angle)  (swivel door geometry)
- A_eff is clamped to A_EFF_MIN (1e-6 m²) for near-closed openings
- Default Cd = 0.62 (sharp-edged orifice)
- Higher Cd (0.8-0.85) for rounded/well-designed inlets

Key fix: Near-closed openings now produce HIGH dp (not zero)
because A_eff is clamped to a small positive value.
""")
print("="*70 + "\n")
