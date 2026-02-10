"""
Example: Chimney with Tee Junction

Demonstrates how to use a tee junction in the chimney configuration.
Tees are treated as 90° fittings with appropriate loss coefficients.

NOTE: This models single-path flow. The tee is treated as a 90° turn
into the branch. Flow splitting (Y-junctions) is not modeled.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flued_module.chimney import Chimney

print("="*80)
print("TEE JUNCTION EXAMPLE")
print("="*80)

# Configuration: Vertical section → Tee → Horizontal section → Vertical section
parts = [
    # Part 0: Vertical rise from appliance
    {
        'type': 'single',
        'inner_diameter_ft': 4/12,  # 4 inches
        'height_ft': 3.0,
        'z_m': 0,
        'orientation': 'vertical',
        'T_ambient_K': 294.15,
        'Pa': 101325,
        'num_segments': 5,
        'thickness_w_m': 0.00076,
        'density_w_kg_m3': 7850,
        'cp_w_J_kg_K': 460,
        'k_w_W_m_K': 50,
        'epsilon_w': 0.8,
        'film_w': 5.08e-5,
    },
    # Part 1: Horizontal offset through wall (connected via tee)
    {
        'type': 'single',
        'inner_diameter_ft': 4/12,
        'height_ft': 2.0,  # Horizontal length
        'z_m': 0.914,  # 3 ft up
        'orientation': 'horizontal',
        'T_ambient_K': 294.15,
        'Pa': 101325,
        'num_segments': 3,
        'thickness_w_m': 0.00076,
        'density_w_kg_m3': 7850,
        'cp_w_J_kg_K': 460,
        'k_w_W_m_K': 50,
        'epsilon_w': 0.8,
        'film_w': 5.08e-5,
    },
    # Part 2: Vertical section to outlet
    {
        'type': 'single',
        'inner_diameter_ft': 4/12,
        'height_ft': 4.0,
        'z_m': 0.914,  # Same z as horizontal section
        'orientation': 'vertical',
        'T_ambient_K': 294.15,
        'Pa': 101325,
        'num_segments': 5,
        'thickness_w_m': 0.00076,
        'density_w_kg_m3': 7850,
        'cp_w_J_kg_K': 460,
        'k_w_W_m_K': 50,
        'epsilon_w': 0.8,
        'film_w': 5.08e-5,
    }
]

# Define fittings: Tee between part 0 and 1, Elbow between part 1 and 2
elbows = [
    # Tee junction: vertical to horizontal (branch flow)
    {
        'between_segments': (0, 1),
        'elbow_type': 'tee',           # ← TEE JUNCTION
        # angle_deg automatically set to 90
        # R_over_D automatically set to 1.0
        # K ≈ 1.0 (branch flow loss)
    },
    # Standard 90° elbow: horizontal to vertical
    {
        'between_segments': (1, 2),
        'angle_deg': 90,
        'R_over_D': 1.5,
        'elbow_type': 'long_radius'
    }
]

print("\nConfiguration:")
print("  Part 0: 3.0 ft vertical")
print("  → TEE (90° branch)")
print("  Part 1: 2.0 ft horizontal")
print("  → ELBOW (90° turn)")
print("  Part 2: 4.0 ft vertical")
print("\nTotal system height: 7.0 ft")

# Run simulation
chimney = Chimney(
    parts=parts,
    elbows=elbows,
    elbow_method='K_sum',
    dt=1.0,
    total_time=60,
    use_coupled_solver=True
)

print("\nRunning simulation...")
chimney.run_simulation()

# Get results
summary = chimney.get_summary_data()

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nFinal mass flow rate: {summary['final_mass_flow_rate_kg_s']:.6f} kg/s")
print(f"Final gas temp (top): {summary['final_gas_temp_range_C'][1]:.2f} °C")
print(f"Final wall temp (top): {summary['final_wall_temp_range_C'][1]:.2f} °C")

print("\nFitting Losses:")
print(f"  Tee junction: K ≈ 1.0 (branch flow, 90° turn)")
print(f"  90° elbow: K ≈ 0.4 (long radius)")
print(f"  Total additional losses from fittings accounted for in simulation")

print("\n" + "="*80)
print("NOTES:")
print("="*80)
print("  - Tee is treated as 90° branch flow (single path)")
print("  - Flow splitting (Y-junction) is NOT modeled")
print("  - Main flow path is: appliance → tee branch → horizontal → elbow → vertical → outlet")
print("  - Loss coefficient K ≈ 1.0 for tee branch flow")
print("  - To model through-flow (straight through tee), use elbow_type='long_radius'")
print("  - with angle_deg=0 or very small angle")
