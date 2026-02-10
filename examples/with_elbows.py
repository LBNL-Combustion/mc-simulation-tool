"""
Example demonstrating the new physics-based elbow loss calculations.
"""

# Unit conversion helpers
FT = 0.3048  # feet to meters
IN = 0.0254  # inches to meters

# Common 4-inch diameter flue
D_m = 4 * IN

# Define chimney parts
parts = [
    {  # Part 0: Single wall vertical section
        "type": "single",
        "inner_diameter_m": D_m,
        "height_m": 7.5 * FT,
        "orientation": "vertical",
        "T_ambient_K": 294.15,
        "Pa": 101325,
        "num_segments": 5,
        "thickness_w_m": 0.00076,
        "density_w_kg_m3": 7850,
        "cp_w_J_kg_K": 460,
        "k_w_W_m_K": 50,
        "film_w": 5.08e-5,
        "epsilon_w": 0.8,
        "humidity_ratio": 0.01,
        "has_fittings": False,  # We'll use explicit elbows instead
    },
    {  # Part 1: Single wall horizontal section
        "type": "single",
        "inner_diameter_m": D_m,
        "height_m": 2.0 * FT,  # Length for horizontal
        "orientation": "horizontal",
        "T_ambient_K": 294.15,
        "Pa": 101325,
        "num_segments": 2,
        "thickness_w_m": 0.00076,
        "density_w_kg_m3": 7850,
        "cp_w_J_kg_K": 460,
        "k_w_W_m_K": 50,
        "film_w": 5.08e-5,
        "epsilon_w": 0.8,
        "humidity_ratio": 0.01,
        "has_fittings": False,
    },
    {  # Part 2: Triple wall vertical section (outdoor)
        "type": "triple",
        "inner_diameter_m": D_m,
        "height_m": 5.0 * FT,
        "orientation": "vertical",
        "T_ambient_K": 273.15,  # Colder outside
        "Pa": 101325,
        "num_segments": 4,
        # Triple wall layer properties
        "thickness_inner_m": 0.00076,
        "density_inner_kg_m3": 7850,
        "cp_inner_J_kg_K": 460,
        "k_inner_W_m_K": 50,
        "epsilon_inner": 0.8,
        "air_gap_inner_m": 0.0127,
        "thickness_middle_m": 0.00076,
        "density_middle_kg_m3": 7850,
        "cp_middle_J_kg_K": 460,
        "k_middle_W_m_K": 50,
        "air_gap_outer_m": 0.0127,
        "thickness_outer_m": 0.00076,
        "density_outer_kg_m3": 7850,
        "cp_outer_J_kg_K": 460,
        "k_outer_W_m_K": 50,
        "epsilon_outer": 0.8,
        "humidity_ratio": 0.01,
        "has_fittings": False,
    },
]

# Define explicit elbows at junctions between parts
elbows = [
    {
        "type": "elbow",
        "between_segments": [0, 1],  # Junction between part 0 (vertical) and part 1 (horizontal)
        "angle_deg": 90,
        "R_over_D": 1.5,
        "elbow_type": "long_radius",
        "smoothness": "commercial_steel",
        "vane_factor": 1.0,  # No guide vanes
    },
    {
        "type": "elbow",
        "between_segments": [1, 2],  # Junction between part 1 (horizontal) and part 2 (vertical)
        "angle_deg": 90,
        "R_over_D": 1.5,
        "elbow_type": "long_radius",
        "smoothness": "commercial_steel",
        "vane_factor": 1.0,
    },
]

def run_example():
    """Run the example simulation with elbows."""
    from flued_module import Chimney
    
    print("=" * 60)
    print("CHIMNEY SIMULATION WITH PHYSICS-BASED ELBOW LOSSES")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Flue diameter: {D_m*1000:.1f} mm ({D_m/IN:.1f} inches)")
    print(f"  Total height: {sum(p['height_m'] for p in parts)/FT:.1f} ft")
    print(f"  Number of elbows: {len(elbows)}")
    print()
    
    # Create chimney with explicit elbows
    chimney = Chimney(
        parts=parts,
        elbows=elbows,
        elbow_method="equivalent_length",  # Use equivalent length method
        dt=0.5,
        total_time=600,  # 10 minute simulation
        use_coupled_solver=True,
    )
    
    print("\nElbow configuration processed:")
    for i, elbow in enumerate(elbows):
        print(f"  Elbow {i+1}: {elbow['angle_deg']}° {elbow['elbow_type']} "
              f"(R/D={elbow['R_over_D']}) between parts {elbow['between_segments']}")
    
    print("\nRunning simulation...")
    chimney.run_simulation()
    
    # Display results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    
    # Final conditions
    final_time_idx = -1
    print(f"\nFinal conditions at t={chimney.total_time}s:")
    print(f"  Mass flow rate: {chimney.mass_flow_rate[final_time_idx]*1000:.2f} g/s")
    print(f"  Outlet temperature: {chimney.T_gas[final_time_idx, -1]-273.15:.1f}°C")
    print(f"  Total pressure drop: {chimney.pressure_drop[final_time_idx]:.1f} Pa")
    
    # Temperature profile
    print("\nTemperature profile (final time):")
    for part_idx, part in enumerate(chimney.parts):
        seg_start = sum(p.num_segments for p in chimney.parts[:part_idx])
        seg_end = seg_start + part.num_segments - 1
        T_in = chimney.T_gas[final_time_idx, seg_start] - 273.15
        T_out = chimney.T_gas[final_time_idx, seg_end] - 273.15
        print(f"  Part {part_idx} ({part.orientation}, {part.type}): "
              f"inlet={T_in:.1f}°C, outlet={T_out:.1f}°C")
    
    print("\nSimulation complete!")
    return chimney

def compare_with_legacy():
    """Compare new elbow method with legacy has_fittings approach."""
    from flued_module import Chimney
    
    print("\n" + "=" * 60)
    print("COMPARISON: NEW ELBOWS vs LEGACY has_fittings")
    print("=" * 60)
    
    # Legacy approach using has_fittings
    parts_legacy = parts.copy()
    parts_legacy[0]["has_fittings"] = True  # Add fitting to first part
    
    print("\nRunning legacy simulation (has_fittings=True)...")
    chimney_legacy = Chimney(
        parts=parts_legacy,
        dt=0.5,
        total_time=60,  # Short simulation for comparison
        use_coupled_solver=False,
    )
    chimney_legacy.run_simulation()
    
    # New approach with explicit elbows
    print("\nRunning new simulation (explicit elbows)...")
    chimney_new = Chimney(
        parts=parts,
        elbows=elbows[:1],  # Just first elbow for fair comparison
        elbow_method="K_sum",
        dt=0.5,
        total_time=60,
        use_coupled_solver=False,
    )
    chimney_new.run_simulation()
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    final_idx = -1
    print(f"\nLegacy (has_fittings):")
    print(f"  Pressure drop: {chimney_legacy.pressure_drop[final_idx]:.1f} Pa")
    print(f"  Mass flow: {chimney_legacy.mass_flow_rate[final_idx]*1000:.2f} g/s")
    
    print(f"\nNew (explicit elbows):")
    print(f"  Pressure drop: {chimney_new.pressure_drop[final_idx]:.1f} Pa")
    print(f"  Mass flow: {chimney_new.mass_flow_rate[final_idx]*1000:.2f} g/s")
    
    print("\nNote: Differences are expected as the new method uses")
    print("physics-based calculations rather than simple multipliers.")

if __name__ == "__main__":
    # Run the main example
    chimney = run_example()
    
    # Optional: Compare with legacy approach
    # compare_with_legacy()
    
    # Optional: Access the results for further analysis
    # import matplotlib.pyplot as plt
    # plt.plot(chimney.time_history, chimney.mass_flow_rate)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Mass flow rate (kg/s)")
    # plt.show()