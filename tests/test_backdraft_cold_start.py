"""
Test Case 4: Cold Start Backdraft (Simple)
Tests backdraft in a cold chimney with cold gas and warmer outdoor conditions.

Configuration:
- Cold gas in chimney (colder than ambient)
- No fire
- Expected: Slight backdraft due to inverted temperature gradient

Expected behavior:
- Negative or near-zero flow (cold gas is denser, sinks)
- Minimal velocities
- May be unstable near zero flow
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from flued_module import Chimney
import numpy as np

# COLD START TEST
# Chimney gas is colder than ambient (inverted buoyancy)

parts = [{
    'type': 'single',
    'inner_diameter_ft': 6/12,
    'height_ft': 15.0,
    'z_m': 0,

    # Wall properties
    'thickness_w_m': 0.00076,
    'density_w_kg_m3': 7850,
    'cp_w_J_kg_K': 460,
    'k_w_W_m_K': 50,
    'epsilon_w': 0.8,

    # Operating conditions - warmer ambient than indoor
    'T_ambient_K': 293.15,            # Warm outside (20°C)
    'Pa': 101325,
    'humidity_ratio': 0.005,

    'orientation': 'vertical',
    'num_segments': 10,
    'roughness_m': 4.5e-5,
}]

print("="*70)
print("TEST 4: COLD START BACKDRAFT")
print("="*70)
print("Configuration:")
print("  - Cold gas in chimney (15°C)")
print("  - Warmer ambient outside (20°C)")
print("  - No fire, no wind")
print("  - Expected: Weak backdraft due to density inversion")
print()

# Create chimney
chimney = Chimney(parts=parts, dt=0.5, total_time=200, use_coupled_solver=True)

# Custom inlet temperature: keep it cold (below ambient)
def cold_start_inlet(time):
    return 288.15  # 15°C - colder than ambient (20°C)

chimney._default_inlet_temp = cold_start_inlet

# Run simulation
print("Starting cold start simulation...")
chimney.run_simulation(save_interval=1)

# VALIDATION CHECKS
print("\n" + "="*70)
print("VALIDATION RESULTS:")
print("="*70)

# Check 1: Flow should be negative or very weak
min_flow = np.min(chimney.mass_flow_rate)
max_flow = np.max(chimney.mass_flow_rate)
final_flow = chimney.mass_flow_rate[-1]

print(f"✓ Mass flow range: {min_flow:.4f} to {max_flow:.4f} kg/s")
print(f"✓ Final mass flow: {final_flow:.4f} kg/s")

if final_flow < 0:
    print("  ✓ PASS: Backdraft occurred (cold gas sinks)")
elif abs(final_flow) < 0.001:
    print("  ~ Near-zero flow (stalled)")
else:
    print("  ⚠️  Unexpected positive flow with inverted buoyancy")

# Check 2: Density check
# Cold gas should be denser than warm ambient
final_density_bottom = chimney.density[-1, 0]
T_ambient = parts[0]['T_ambient_K']
rho_ambient = 101325 / (287 * T_ambient)

print(f"\nDensity comparison:")
print(f"  Gas density (bottom): {final_density_bottom:.3f} kg/m³")
print(f"  Ambient density: {rho_ambient:.3f} kg/m³")

if final_density_bottom > rho_ambient:
    print("  ✓ PASS: Cold gas is denser than ambient (inverted buoyancy)")
else:
    print("  ⚠️  Gas should be denser than ambient")

# Check 3: Temperature profile
print(f"\nFinal temperature profile:")
for seg in [0, chimney.total_segments//2, chimney.total_segments-1]:
    T = chimney.T_gas[-1, seg] - 273.15
    location = ["Bottom", "Middle", "Top"][seg // max(1, chimney.total_segments//2)]
    print(f"  {location:6s}: {T:.1f}°C")

print(f"  Ambient: {T_ambient - 273.15:.1f}°C")

# Check 4: Backdraft detection
if chimney.backdraft_detected:
    print(f"\n✓ Backdraft detected at: {chimney.backdraft_start_time:.1f}s")
else:
    # May not detect if flow starts at zero and goes negative
    print(f"\n~ Backdraft flag: {chimney.backdraft_detected}")
    print("  (May not trigger if starting from zero flow)")

print("\n" + "="*70)
print("TEST 4 COMPLETE")
print("="*70)
print("\nNote: This test demonstrates inverted buoyancy backdraft.")
print("Flow may be very weak due to small temperature difference.")
