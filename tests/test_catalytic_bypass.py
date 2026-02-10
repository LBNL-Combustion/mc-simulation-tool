"""
Test script for catalytic bypass damper feature

Tests the catalytic bypass with:
1. No bypass (baseline)
2. Bypass OPEN (startup mode - low resistance)
3. Bypass CLOSED (catalyst engaged - higher resistance)
4. Partial bypass (50% open)
"""

import sys
import numpy as np
from flued_module.chimney import Chimney

# Define a simple chimney for testing
# Assume catalyst is in segment 5 (middle of the chimney)
parts = [
    {
        'type': 'single',
        'inner_diameter_ft': 6/12,  # 6 inches
        'height_ft': 15,
        'z_ft': 0,
        'orientation': 'vertical',
        'T_ambient_K': 273.15 + 20,  # 20C
        'Pa': 101325,
        'num_segments': 10,
    }
]

print("="*70)
print("CATALYTIC BYPASS DAMPER FEATURE TEST")
print("="*70)

# Calculate some reference values
inlet_diameter_m = (6/12) * 0.3048  # 6 inches in meters
inlet_area_m2 = np.pi * (inlet_diameter_m / 2)**2
print(f"\nChimney Configuration:")
print(f"  Inner diameter: {inlet_diameter_m*100:.2f} cm")
print(f"  Flow area: {inlet_area_m2:.5f} m2")
print(f"  Total segments: 10")
print(f"  Catalyst location: segment 5 (middle)")
print("")

# Test 1: No bypass (baseline)
print("\n" + "="*70)
print("TEST 1: NO BYPASS (BASELINE)")
print("="*70)
chimney_baseline = Chimney(parts=parts, dt=0.5, total_time=60)
chimney_baseline.run_simulation()

# Test 2: Bypass OPEN (startup mode)
print("\n" + "="*70)
print("TEST 2: BYPASS OPEN (startup mode - low resistance)")
print("="*70)
chimney_open = Chimney(
    parts=parts,
    dt=0.5,
    total_time=60,
    catalytic_bypass={
        'enable': True,
        'bypass_state': 'open',
        'segment_index': 5,
        'K_bypass_open': 0.5,
        'K_catalyst_closed': 3.0
    }
)
chimney_open.run_simulation()

# Test 3: Bypass CLOSED (catalyst engaged)
print("\n" + "="*70)
print("TEST 3: BYPASS CLOSED (catalyst engaged - higher resistance)")
print("="*70)
chimney_closed = Chimney(
    parts=parts,
    dt=0.5,
    total_time=60,
    catalytic_bypass={
        'enable': True,
        'bypass_state': 'closed',
        'segment_index': 5,
        'K_bypass_open': 0.5,
        'K_catalyst_closed': 3.0
    }
)
chimney_closed.run_simulation()

# Test 4: Partial bypass (50% open)
print("\n" + "="*70)
print("TEST 4: PARTIAL BYPASS (50% open)")
print("="*70)
chimney_partial = Chimney(
    parts=parts,
    dt=0.5,
    total_time=60,
    catalytic_bypass={
        'enable': True,
        'bypass_state': 50,  # 50% open (numeric value)
        'segment_index': 5,
        'K_bypass_open': 0.5,
        'K_catalyst_closed': 3.0
    }
)
chimney_partial.run_simulation()

# Compare results
print("\n" + "="*70)
print("COMPARISON OF RESULTS")
print("="*70)

print("\nFinal Mass Flow Rates:")
print(f"  No bypass (baseline):  {chimney_baseline.mass_flow_rate[-1]:.6f} kg/s")
print(f"  Bypass OPEN:           {chimney_open.mass_flow_rate[-1]:.6f} kg/s "
      f"({chimney_open.mass_flow_rate[-1]/chimney_baseline.mass_flow_rate[-1]*100:.1f}% of baseline)")
print(f"  Bypass CLOSED:         {chimney_closed.mass_flow_rate[-1]:.6f} kg/s "
      f"({chimney_closed.mass_flow_rate[-1]/chimney_baseline.mass_flow_rate[-1]*100:.1f}% of baseline)")
print(f"  Bypass 50% open:       {chimney_partial.mass_flow_rate[-1]:.6f} kg/s "
      f"({chimney_partial.mass_flow_rate[-1]/chimney_baseline.mass_flow_rate[-1]*100:.1f}% of baseline)")

print("\nFinal Outlet Temperatures:")
print(f"  No bypass (baseline):  {chimney_baseline.T_gas[-1, -1] - 273.15:.2f}C")
print(f"  Bypass OPEN:           {chimney_open.T_gas[-1, -1] - 273.15:.2f}C "
      f"(+{chimney_open.T_gas[-1, -1] - chimney_baseline.T_gas[-1, -1]:.2f}C)")
print(f"  Bypass CLOSED:         {chimney_closed.T_gas[-1, -1] - 273.15:.2f}C "
      f"(+{chimney_closed.T_gas[-1, -1] - chimney_baseline.T_gas[-1, -1]:.2f}C)")
print(f"  Bypass 50% open:       {chimney_partial.T_gas[-1, -1] - 273.15:.2f}C "
      f"(+{chimney_partial.T_gas[-1, -1] - chimney_baseline.T_gas[-1, -1]:.2f}C)")

print("\nEffective K-Loss Coefficients:")
print(f"  No bypass:     K = 0 (no catalyst modeled)")
print(f"  Bypass OPEN:   K = {chimney_open.catalytic_bypass.K_effective:.3f}")
print(f"  Bypass CLOSED: K = {chimney_closed.catalytic_bypass.K_effective:.3f}")
print(f"  Bypass 50%:    K = {chimney_partial.catalytic_bypass.K_effective:.3f}")

print("\n" + "="*70)
print("OBSERVATIONS:")
print("="*70)
print("1. Bypass OPEN has lowest resistance -> highest mass flow")
print("2. Bypass CLOSED has highest resistance -> lowest mass flow")
print("3. Partial bypass (50%) falls between open and closed")
print("4. Lower flow -> higher temperatures (less cooling)")
print("5. K-loss interpolates linearly between open and closed states")
print("="*70)

print("\n" + "="*70)
print("ADDITIONAL TEST: Verify bypass_fraction values")
print("="*70)
print(f"  Bypass OPEN:   bypass_fraction = {chimney_open.catalytic_bypass.bypass_fraction:.2f} (expected 1.0)")
print(f"  Bypass CLOSED: bypass_fraction = {chimney_closed.catalytic_bypass.bypass_fraction:.2f} (expected 0.0)")
print(f"  Bypass 50%:    bypass_fraction = {chimney_partial.catalytic_bypass.bypass_fraction:.2f} (expected 0.5)")

# Verify expected behavior
all_pass = True

# Check that bypass OPEN has highest flow
if chimney_open.mass_flow_rate[-1] < chimney_closed.mass_flow_rate[-1]:
    print("\nFAIL: Bypass OPEN should have higher flow than CLOSED")
    all_pass = False

# Check that bypass CLOSED has lowest flow (among bypass cases)
if chimney_closed.mass_flow_rate[-1] > chimney_open.mass_flow_rate[-1]:
    print("\nFAIL: Bypass CLOSED should have lower flow than OPEN")
    all_pass = False

# Check that partial is between open and closed
if not (chimney_closed.mass_flow_rate[-1] <= chimney_partial.mass_flow_rate[-1] <= chimney_open.mass_flow_rate[-1]):
    print("\nFAIL: Partial bypass should have flow between OPEN and CLOSED")
    all_pass = False

# Check K values
if chimney_open.catalytic_bypass.K_effective != 0.5:
    print(f"\nFAIL: K_effective for OPEN should be 0.5, got {chimney_open.catalytic_bypass.K_effective}")
    all_pass = False

if chimney_closed.catalytic_bypass.K_effective != 3.0:
    print(f"\nFAIL: K_effective for CLOSED should be 3.0, got {chimney_closed.catalytic_bypass.K_effective}")
    all_pass = False

expected_K_partial = 3.0 + 0.5 * (0.5 - 3.0)  # = 1.75
if abs(chimney_partial.catalytic_bypass.K_effective - expected_K_partial) > 0.001:
    print(f"\nFAIL: K_effective for 50% should be {expected_K_partial}, got {chimney_partial.catalytic_bypass.K_effective}")
    all_pass = False

print("="*70)
if all_pass:
    print("\nAll tests PASSED!")
else:
    print("\nSome tests FAILED - check output above")
print("="*70)
