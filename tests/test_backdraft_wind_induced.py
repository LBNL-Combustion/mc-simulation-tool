"""
Test Case 2: Wind-Induced Backdraft
Tests backdraft caused by strong adverse wind pressure at the chimney outlet.

Configuration:
- No fire (cold chimney, T_inlet = T_ambient)
- Strong wind with rain cap (positive pressure at outlet)
- Expected: Downward flow (backdraft) due to wind pressure

Expected behavior:
- Mass flow should become negative (backdraft)
- Cold outdoor air flows downward through chimney
- Backdraft detection should trigger
- Top segment temperature ≈ ambient (outdoor air inlet)
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from flued_module import Chimney
import numpy as np

# WIND-INDUCED BACKDRAFT TEST

parts = [{
    'type': 'single',
    'inner_diameter_ft': 6/12,       # 6 inch diameter
    'height_ft': 15.0,                # 15 feet tall
    'z_m': 0,

    # Wall properties
    'thickness_w_m': 0.00076,
    'density_w_kg_m3': 7850,
    'cp_w_J_kg_K': 460,
    'k_w_W_m_K': 50,
    'epsilon_w': 0.8,

    # Operating conditions - NO FIRE, just ambient temperature
    'T_ambient_K': 283.15,            # Mild outside (10°C)
    'Pa': 101325,
    'humidity_ratio': 0.005,

    'orientation': 'vertical',
    'num_segments': 10,
    'roughness_m': 4.5e-5,
}]

# ADVERSE WIND CONFIGURATION
# Rain cap creates positive pressure (pressurizes outlet)
# This opposes upward draft and can cause backdraft
top_boundary = {
    'enable_wind_cap_effect': True,
    'wind_speed_m_s': 15.0,           # Strong wind (15 m/s ≈ 33 mph)
    'cap_type': 'rain_cap',           # Positive Cp = +0.05 (pressurizes)
    'roof_exposure': 'open',          # Full wind exposure
    'wind_cap_update': 'static'       # Constant wind
}

print("="*70)
print("TEST 2: WIND-INDUCED BACKDRAFT")
print("="*70)
print("Configuration:")
print("  - NO FIRE (T_inlet = T_ambient = 10°C)")
print("  - Strong wind: 15 m/s (33 mph)")
print("  - Rain cap (pressurizes outlet)")
print("  - Expected: Downward flow (backdraft)")
print()

# Create chimney with custom inlet temperature function (no heating)
chimney = Chimney(parts=parts, dt=0.5, total_time=300, use_coupled_solver=True,
                  top_boundary=top_boundary)

# Override inlet temperature to be constant (no fire)
def cold_inlet(time):
    return parts[0]['T_ambient_K']  # No heating

chimney._default_inlet_temp = cold_inlet

# Run simulation
print("Starting simulation with adverse wind...")
chimney.run_simulation(save_interval=1)

# VALIDATION CHECKS
print("\n" + "="*70)
print("VALIDATION RESULTS:")
print("="*70)

# Check 1: Flow should be negative (backdraft)
min_flow = np.min(chimney.mass_flow_rate)
max_flow = np.max(chimney.mass_flow_rate)
final_flow = chimney.mass_flow_rate[-1]

print(f"✓ Mass flow range: {min_flow:.4f} to {max_flow:.4f} kg/s")
print(f"✓ Final mass flow: {final_flow:.4f} kg/s")

if final_flow < 0:
    print("  ✓ PASS: Backdraft occurred (negative flow)")
else:
    print("  ⚠️  WARNING: Expected negative flow, got positive!")

# Check 2: Backdraft detection should trigger
print(f"\n✓ Backdraft detected: {chimney.backdraft_detected}")
if chimney.backdraft_detected:
    print(f"  ✓ Backdraft started at: {chimney.backdraft_start_time:.1f}s")
    print(f"  ✓ Flow reversal count: {chimney.flow_reversal_count}")
else:
    print("  ⚠️  WARNING: Backdraft should have been detected!")

# Check 3: Temperature profile
# With backdraft, cold outdoor air enters at top and flows down
print(f"\nTemperature profile (should show cold air entering at top):")
final_T_bottom = chimney.T_gas[-1, 0]
final_T_top = chimney.T_gas[-1, -1]
T_ambient_C = parts[0]['T_ambient_K'] - 273.15

print(f"  Bottom (outlet): {final_T_bottom - 273.15:.1f}°C")
print(f"  Top (inlet): {final_T_top - 273.15:.1f}°C")
print(f"  Ambient: {T_ambient_C:.1f}°C")

if abs(final_T_top - parts[0]['T_ambient_K']) < 5:
    print("  ✓ PASS: Top temperature ≈ ambient (outdoor air inlet)")
else:
    print(f"  ⚠️  Top temperature differs from ambient by {abs(final_T_top - parts[0]['T_ambient_K']):.1f}K")

# Check 4: Velocity profile (should be negative throughout)
final_velocities = chimney.velocity[-1, :]
all_negative = np.all(final_velocities <= 0)
print(f"\n✓ All velocities negative: {all_negative}")
if all_negative:
    print("  ✓ PASS: Downward flow throughout chimney")
else:
    positive_segs = np.where(final_velocities > 0)[0]
    print(f"  ⚠️  Some segments have positive velocity: {positive_segs}")

print("\n" + "="*70)
print("TEST 2 COMPLETE")
print("="*70)
