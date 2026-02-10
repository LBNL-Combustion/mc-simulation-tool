"""
Quick validation test for spiral corrugation implementation
Tests the preprocessing functions without running full simulation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flued_module.corrugation_effects import (
    SpiralCorrugationProperties,
    compute_spiral_corrugation_effects,
    compute_spiral_factor
)
import numpy as np

print("="*80)
print("SPIRAL CORRUGATION PREPROCESSING TEST")
print("="*80)

# Test case: 4-inch diameter pipe with typical spiral corrugation
diameter_m = 4 * 0.0254  # 4 inches = 0.1016 m
length_m = 7.5 * 0.3048 / 10  # 7.5 feet divided into 10 segments

# Create spiral corrugation properties
spiral_props = SpiralCorrugationProperties(
    depth_m=0.003,           # 3mm depth
    pitch_m=0.180,           # 180mm pitch
    num_starts=4,            # Quad-start spiral
    base_roughness_m=1e-6,   # 1 micron base
    has_lock_seam=True,
    seam_protrusion_m=0.002,
    seam_width_m=0.008,
    K_seam_per_crossing=0.05,
    K_lock_seam=0.03
)

# Test helix angle calculation
beta_rad = spiral_props.compute_helix_angle(diameter_m)
beta_deg = spiral_props.compute_helix_angle_deg(diameter_m)

print(f"\nTest 1: Helix Angle Computation")
print(f"  Diameter: {diameter_m*1000:.1f} mm")
print(f"  Pitch: {spiral_props.pitch_m*1000:.1f} mm")
print(f"  Helix angle β: {beta_deg:.2f}° ({beta_rad:.4f} rad)")
print(f"  Expected: ~29° for these parameters")

# Test spiral factor
c_spiral = compute_spiral_factor(beta_rad)
print(f"\nTest 2: Spiral Factor Computation")
print(f"  c_spiral(β={beta_deg:.1f}°): {c_spiral:.3f}")
print(f"  Range: 0.2 (β=90°) to 0.5 (β=0°)")

# Test full preprocessing
effects = compute_spiral_corrugation_effects(
    diameter_m=diameter_m,
    length_m=length_m,
    spiral_props=spiral_props,
    representative_Re=10000
)

print(f"\nTest 3: Full Preprocessing")
print(f"  Effective roughness (corrugation): {effects['effective_roughness_m']*1e6:.2f} µm")
print(f"  Number of seam crossings: {effects['n_seams']:.2f}")
print(f"  Total K from seams: {effects['K_total_seams']:.4f}")
print(f"  Lock seam K: {effects['K_lock_seam']:.4f}")
print(f"  Equivalent roughness (total): {effects['equivalent_roughness_m']*1e6:.1f} µm")
print(f"\nFriction Factors at Re=10000:")
print(f"  f_corrugation: {effects['friction_factor_corrugation']:.5f}")
print(f"  f_seams: {effects['friction_factor_seams']:.5f}")
print(f"  f_total: {effects['friction_factor_total']:.5f}")

# Validation checks
print(f"\nValidation Checks:")
checks_passed = 0
checks_total = 0

# Check 1: Helix angle in reasonable range
checks_total += 1
if 10 <= beta_deg <= 45:
    print(f"  ✓ Helix angle in reasonable range (10-45°)")
    checks_passed += 1
else:
    print(f"  ✗ Helix angle outside expected range: {beta_deg:.1f}°")

# Check 2: Spiral factor in valid range
checks_total += 1
if 0.2 <= c_spiral <= 0.5:
    print(f"  ✓ Spiral factor in valid range (0.2-0.5)")
    checks_passed += 1
else:
    print(f"  ✗ Spiral factor outside expected range: {c_spiral:.3f}")

# Check 3: Effective roughness > base roughness
checks_total += 1
if effects['effective_roughness_m'] > spiral_props.base_roughness_m:
    print(f"  ✓ Effective roughness > base roughness")
    checks_passed += 1
else:
    print(f"  ✗ Effective roughness not increased")

# Check 4: Equivalent roughness >= effective roughness
checks_total += 1
if effects['equivalent_roughness_m'] >= effects['effective_roughness_m']:
    print(f"  ✓ Equivalent roughness >= effective roughness (seams add drag)")
    checks_passed += 1
else:
    print(f"  ✗ Equivalent roughness inconsistent")

# Check 5: Friction factor total = corrugation + seams
checks_total += 1
f_sum = effects['friction_factor_corrugation'] + effects['friction_factor_seams']
if abs(effects['friction_factor_total'] - f_sum) < 1e-6:
    print(f"  ✓ Friction factors sum correctly")
    checks_passed += 1
else:
    print(f"  ✗ Friction factor sum mismatch")

# Check 6: Seam crossings reasonable
checks_total += 1
expected_crossings = (length_m * spiral_props.num_starts) / spiral_props.pitch_m
if abs(effects['n_seams'] - expected_crossings) < 0.01:
    print(f"  ✓ Seam crossings computed correctly")
    checks_passed += 1
else:
    print(f"  ✗ Seam crossing calculation error")

print(f"\nPassed {checks_passed}/{checks_total} validation checks")

if checks_passed == checks_total:
    print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
else:
    print(f"\n✗✗✗ {checks_total - checks_passed} TESTS FAILED ✗✗✗")

print("="*80)
