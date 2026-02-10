"""
Test script to verify optimizations don't change results
"""
from flued_module import Chimney
import time

# Your exact configuration from the baseline
parts = [
    {
        'type': 'single',
        'inner_diameter_ft': 6/12,
        'height_ft': 10,
        'z_ft': 0,
        'thickness_w_m': 0.00076,
        'density_w_kg_m3': 7850,
        'cp_w_J_kg_K': 460,
        'k_w_W_m_K': 50,
        'epsilon_w': 0.8,
        'T_ambient_K': 294.15,
        'Pa': 101325,
        'film_w': 5.08e-5,
        'orientation': 'vertical',
        'num_segments': 3,
    },
    {
        'type': 'single',
        'inner_diameter_ft': 6/12,
        'height_ft': 10,
        'z_ft': 10,
        'thickness_w_m': 0.00076,
        'density_w_kg_m3': 7850,
        'cp_w_J_kg_K': 460,
        'k_w_W_m_K': 50,
        'epsilon_w': 0.8,
        'T_ambient_K': 294.15,
        'Pa': 101325,
        'film_w': 5.08e-5,
        'orientation': 'vertical',
        'num_segments': 3,
    },
    {
        'type': 'single',
        'inner_diameter_ft': 6/12,
        'height_ft': 10,
        'z_ft': 20,
        'thickness_w_m': 0.00076,
        'density_w_kg_m3': 7850,
        'cp_w_J_kg_K': 460,
        'k_w_W_m_K': 50,
        'epsilon_w': 0.8,
        'T_ambient_K': 294.15,
        'Pa': 101325,
        'film_w': 5.08e-5,
        'orientation': 'vertical',
        'num_segments': 3,
    }
]

# Your exact parameters
print("Testing with your exact configuration:")
print("dt=1s, total_time=240s (testing first 4 minutes), use_coupled_solver=True")
print("9 segments total (3 per part)")

chimney = Chimney(parts=parts, dt=1, total_time=240, use_coupled_solver=True)

# Time the simulation
start_time = time.time()
print("\nRunning simulation...")
chimney.run_simulation(save_interval=1)
elapsed = time.time() - start_time

print(f"\nSimulation completed in {elapsed:.1f} seconds")

# Check results at your baseline times
baseline_times = [60, 120, 180, 240]

print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)

for t in baseline_times:
    if t <= chimney.total_time:
        step = int(t / chimney.dt)
        gas_temps = chimney.T_gas[step] - 273.15
        wall_temps = chimney.T_wall[step] - 273.15
        
        print(f"\nAt t={t}s:")
        print(f"  Gas:  {gas_temps.min():.1f} - {gas_temps.max():.1f}°C")
        print(f"  Wall: {wall_temps.min():.1f} - {wall_temps.max():.1f}°C")
        print(f"  Mass flow: {chimney.mass_flow_rate[step]:.6f} kg/s")
        
        # Show your baseline for comparison
        if t == 60:
            print("  YOUR BASELINE: Gas: 81.7-107.8°C, Wall: 52.6-61.8°C")
        elif t == 120:
            print("  YOUR BASELINE: Gas: 148.3-178.9°C, Wall: 98.8-127.3°C")

# Performance metrics
print("\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)
print(f"Time per timestep: {elapsed/240:.3f} seconds")
print(f"Projected time for 1200s simulation: {elapsed * 1200/240:.1f} seconds")