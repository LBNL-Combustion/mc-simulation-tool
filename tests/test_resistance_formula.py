"""
Diagnostic script to verify resistance formula bug in coupled solver

The Darcy-Weisbach equation is:
    dp = f × (L/D) × (ρ × v²) / 2

With v = ṁ / (ρ × A), this becomes:
    dp = f × (L/D) × ṁ² / (2 × ρ × A²)

So: R = f × (L/D) / (2 × ρ × A²)  -- density in DENOMINATOR

But coupled_solver.py line 652 has:
    R_seg = K_total * densities[seg] / (2 * area**2)  -- density in NUMERATOR

This is WRONG by a factor of ρ²!
"""

import math

# Typical chimney parameters
D = 0.127  # 5" diameter in meters
A = math.pi * (D/2)**2  # Cross-sectional area
L = 5.2  # 17 ft chimney height in meters
f = 0.02  # Typical friction factor for turbulent flow
K_minor = 0.5  # Minor losses

# Typical gas properties
rho_hot = 0.4  # Hot gas density at ~600K
rho_cold = 1.2  # Ambient air density at 300K

# Total K factor
K_total = f * L / D + K_minor
print(f"K_total = {K_total:.4f}")

# OLD WRONG formula (before fix)
R_wrong = K_total * rho_hot / (2 * A**2)
print(f"\nOLD WRONG formula (before fix):")
print(f"  R = K_total × ρ / (2 × A²) = {R_wrong:.2e} Pa·s²/kg²")

# FIXED formula (after correction)
R_correct = K_total / (2 * rho_hot * A**2)
print(f"\nFIXED formula (after correction):")
print(f"  R = K_total / (2 × ρ × A²) = {R_correct:.2e} Pa·s²/kg²")

# Ratio
ratio = R_correct / R_wrong
print(f"\nRatio (correct/wrong) = {ratio:.2f}")
print(f"This is 1/ρ² = 1/{rho_hot}² = {1/rho_hot**2:.2f}")

# Effect on mass flow
# dp = R × ṁ²  →  ṁ = sqrt(dp/R)
# Typical buoyancy pressure
dp_drive = 9.81 * L * (rho_cold - rho_hot)
print(f"\nBuoyancy driving pressure: {dp_drive:.1f} Pa")

mdot_wrong = math.sqrt(dp_drive / R_wrong)
mdot_correct = math.sqrt(dp_drive / R_correct)

print(f"\nMass flow with OLD WRONG R: {mdot_wrong:.4f} kg/s")
print(f"Mass flow with FIXED R: {mdot_correct:.4f} kg/s")
print(f"Ratio (wrong/correct) = {mdot_wrong/mdot_correct:.2f}")

# Velocities
v_wrong = mdot_wrong / (rho_hot * A)
v_correct = mdot_correct / (rho_hot * A)
print(f"\nVelocity with OLD WRONG R: {v_wrong:.1f} m/s")
print(f"Velocity with FIXED R: {v_correct:.1f} m/s")

# Sanity check: experimental data shows velocities around 1-2 m/s
print(f"\n=== Expected Improvement After Fix ===")
print(f"Sequential solver: ~1.6-1.9 m/s")
print(f"Coupled solver BEFORE fix: ~24-25 m/s (too high)")
print(f"Coupled solver AFTER fix: should be ~{v_correct:.1f} m/s")
print(f"\nNote: There may still be some remaining difference between")
print(f"sequential and coupled, but the 12x error should be reduced to ~{v_correct/1.8:.1f}x")
