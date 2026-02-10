"""
Test Case 1: Normal Draft Validation
This test ensures that the backdraft-enabled solver still works correctly
for normal upward draft conditions (hot fire, standard operation).

Expected behavior:
- Mass flow should remain positive throughout
- Results should closely match the pre-backdraft implementation
- No backdraft warnings should be printed
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from flued_module import Chimney
import numpy as np

# NORMAL OPERATION TEST
# Hot fire with good draft conditions

parts = [{
    # Single-wall steel chimney
    'type': 'single',
    'inner_diameter_ft': 6/12,      # 6 inch diameter
    'height_ft': 15.0,               # 15 feet tall
    'z_m': 0,

    # Wall properties (thin steel)
    'thickness_w_m': 0.00076,        # 0.76mm steel
    'density_w_kg_m3': 7850,         # Steel density
    'cp_w_J_kg_K': 460,              # Steel specific heat
    'k_w_W_m_K': 50,                 # Steel conductivity
    'epsilon_w': 0.8,                # Emissivity

    # Operating conditions
    'T_ambient_K': 273.15,           # Cold outside (0°C) → strong buoyancy
    'Pa': 101325,
    'humidity_ratio': 0.005,

    # Geometry
    'orientation': 'vertical',
    'num_segments': 10,
    'roughness_m': 4.5e-5,           # Smooth steel
}]

print("="*70)
print("TEST 1: NORMAL DRAFT VALIDATION")
print("="*70)
print("Configuration:")
print("  - Hot fire (exponential rise to ~600K)")
print("  - Cold ambient (0°C)")
print("  - 15 ft chimney, 6 inch diameter")
print("  - Expected: Strong upward draft throughout")
print()

# Run simulation
chimney = Chimney(parts=parts, dt=0.5, total_time=300, use_coupled_solver=True)
chimney.run_simulation(save_interval=1)

# VALIDATION CHECKS
print("\n" + "="*70)
print("VALIDATION RESULTS:")
print("="*70)

# Check 1: All mass flows should be positive
min_flow = np.min(chimney.mass_flow_rate)
max_flow = np.max(chimney.mass_flow_rate)
all_positive = np.all(chimney.mass_flow_rate >= 0)

print(f"✓ Mass flow range: {min_flow:.4f} to {max_flow:.4f} kg/s")
print(f"✓ All flows positive: {all_positive}")

if not all_positive:
    print("  ⚠️  WARNING: Negative flow detected in normal draft case!")
    negative_steps = np.where(chimney.mass_flow_rate < 0)[0]
    print(f"  Negative at timesteps: {negative_steps}")
else:
    print("  ✓ PASS: No backdraft occurred (as expected)")

# Check 2: No backdraft detection
print(f"✓ Backdraft detected flag: {chimney.backdraft_detected}")
if chimney.backdraft_detected:
    print("  ⚠️  WARNING: Backdraft flag set in normal operation!")
else:
    print("  ✓ PASS: No backdraft flag (as expected)")

# Check 3: Final steady-state values
final_flow = chimney.mass_flow_rate[-1]
final_T_inlet = chimney.T_gas[-1, 0]
final_T_outlet = chimney.T_gas[-1, -1]

print(f"\nFinal steady-state values:")
print(f"  Inlet temperature: {final_T_inlet - 273.15:.1f}°C")
print(f"  Outlet temperature: {final_T_outlet - 273.15:.1f}°C")
print(f"  Mass flow rate: {final_flow:.4f} kg/s")
print(f"  Draft strength: {'STRONG' if final_flow > 0.01 else 'WEAK'}")

# Check 4: Temperature gradient (should decrease upward due to heat loss)
T_gradient = chimney.T_gas[-1, -1] - chimney.T_gas[-1, 0]
print(f"\nTemperature drop (inlet to outlet): {-T_gradient:.1f}°C")
if T_gradient < 0:
    print("  ✓ PASS: Temperature decreases upward (normal heat loss)")
else:
    print("  ⚠️  WARNING: Temperature increases upward (unexpected)")

print("\n" + "="*70)
print("TEST 1 COMPLETE")
print("="*70)
