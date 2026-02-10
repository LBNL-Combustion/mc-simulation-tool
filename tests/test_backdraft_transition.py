"""
Test Case 3: Backdraft → Recovery Transition
Tests a scenario where backdraft occurs initially, then normal draft is restored.

Configuration:
- Start with cold chimney and mild conditions (weak draft or backdraft)
- Apply fire after some delay
- Expected: Initial backdraft or stalled flow → recovery to normal draft

Expected behavior:
- Initially: weak or negative flow
- After fire starts: flow becomes positive (draft restored)
- Backdraft detection should trigger for both transitions
- Smooth transition without oscillations
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from flued_module import Chimney
import numpy as np

# TRANSITION TEST: Cold Start → Fire → Recovery

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

    # Operating conditions
    'T_ambient_K': 290.15,            # Moderate outside (17°C)
    'Pa': 101325,
    'humidity_ratio': 0.005,

    'orientation': 'vertical',
    'num_segments': 10,
    'roughness_m': 4.5e-5,
}]

# Wind creates initial backdraft
top_boundary = {
    'enable_wind_cap_effect': True,
    'wind_speed_m_s': 12.0,           # Moderate wind
    'cap_type': 'rain_cap',
    'roof_exposure': 'open',
    'wind_cap_update': 'static'
}

print("="*70)
print("TEST 3: BACKDRAFT → RECOVERY TRANSITION")
print("="*70)
print("Configuration:")
print("  - Start cold, moderate wind causes initial backdraft")
print("  - Fire starts and ramps up gradually")
print("  - Expected: Backdraft → Normal draft recovery")
print()

# Create chimney
chimney = Chimney(parts=parts, dt=0.5, total_time=600, use_coupled_solver=True,
                  top_boundary=top_boundary)

# Custom inlet temperature: cold for first 100s, then fire starts
def delayed_fire_inlet(time):
    T_ambient = parts[0]['T_ambient_K']

    if time < 100:
        # Cold start period - no fire
        return T_ambient
    else:
        # Fire starts at t=100s
        time_since_ignition = time - 100
        T_max = 650  # K - hot fire
        tau = 60     # seconds - heating time constant
        return T_ambient + (T_max - T_ambient) * (1 - np.exp(-time_since_ignition / tau))

chimney._default_inlet_temp = delayed_fire_inlet

# Run simulation
print("Starting transition simulation...")
print("  Phase 1 (0-100s): Cold start with adverse wind")
print("  Phase 2 (100-600s): Fire ignites and heats up")
chimney.run_simulation(save_interval=1)

# VALIDATION CHECKS
print("\n" + "="*70)
print("VALIDATION RESULTS:")
print("="*70)

# Check 1: Flow should transition from negative to positive
flow_at_50s = chimney.mass_flow_rate[int(50/chimney.dt)]
flow_at_200s = chimney.mass_flow_rate[int(200/chimney.dt)]
flow_final = chimney.mass_flow_rate[-1]

print("Mass flow evolution:")
print(f"  At t=50s (cold period): {flow_at_50s:.4f} kg/s")
print(f"  At t=200s (after fire): {flow_at_200s:.4f} kg/s")
print(f"  Final (t=600s): {flow_final:.4f} kg/s")

transition_occurred = (flow_at_50s < 0 and flow_final > 0)
if transition_occurred:
    print("  ✓ PASS: Flow transitioned from backdraft to normal draft")
else:
    print("  ⚠️  Expected transition from negative to positive flow")

# Check 2: Backdraft detection
print(f"\n✓ Backdraft detected: {chimney.backdraft_detected}")
if chimney.backdraft_detected:
    print(f"  ✓ Backdraft started at: {chimney.backdraft_start_time:.1f}s")
    if chimney.backdraft_end_time:
        print(f"  ✓ Draft restored at: {chimney.backdraft_end_time:.1f}s")
        duration = chimney.backdraft_end_time - chimney.backdraft_start_time
        print(f"  ✓ Backdraft duration: {duration:.1f}s")
    print(f"  ✓ Total flow reversals: {chimney.flow_reversal_count}")
else:
    print("  ⚠️  WARNING: Backdraft should have been detected!")

# Check 3: Smooth transition (no rapid oscillations)
# Count how many times flow changes sign
flow_signs = np.sign(chimney.mass_flow_rate[1:])  # Skip initial zero
sign_changes = np.sum(np.abs(np.diff(flow_signs)) > 0)

print(f"\n✓ Flow sign changes: {sign_changes}")
if sign_changes <= 2:  # Allow one reversal in each direction
    print("  ✓ PASS: Smooth transition (no oscillations)")
else:
    print(f"  ⚠️  WARNING: Multiple flow reversals ({sign_changes}) - may indicate oscillation")

# Check 4: Final state is stable positive flow
if flow_final > 0.001:
    print(f"\n✓ PASS: Final state is stable normal draft ({flow_final:.4f} kg/s)")
else:
    print(f"\n⚠️  WARNING: Final flow is weak or negative ({flow_final:.4f} kg/s)")

# Check 5: Temperature evolution
T_inlet_50s = chimney.T_gas[int(50/chimney.dt), 0]
T_inlet_final = chimney.T_gas[-1, 0]
print(f"\nInlet temperature evolution:")
print(f"  At t=50s: {T_inlet_50s - 273.15:.1f}°C (cold)")
print(f"  Final: {T_inlet_final - 273.15:.1f}°C (hot fire)")

# Plot mass flow over time for visual inspection
print("\nMass flow time series (every 50s):")
for t in [0, 50, 100, 150, 200, 300, 400, 500, 600]:
    step = int(t / chimney.dt)
    if step < len(chimney.mass_flow_rate):
        flow = chimney.mass_flow_rate[step]
        direction = "⬆ UPWARD" if flow >= 0 else "⬇ DOWNWARD"
        print(f"  t={t:3d}s: {flow:+.4f} kg/s  {direction}")

print("\n" + "="*70)
print("TEST 3 COMPLETE")
print("="*70)
